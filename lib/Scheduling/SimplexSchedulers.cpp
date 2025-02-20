//===- SimplexSchedulers.cpp - Linear programming-based schedulers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of linear programming-based schedulers with a built-in simplex
// solver.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "simplex-schedulers"

using namespace circt;
using namespace circt::scheduling;

using llvm::dbgs;
using llvm::format;

namespace {

/// This class provides a framework to model certain scheduling problems as
/// lexico-parametric linear programs (LP), which are then solved with an
/// extended version of the dual simplex algorithm.
///
/// The approach is described in:
///  [1] B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///      Instruction Scheduling", PRISM 1994.22, 1994.
///  [2] B. D. de Dinechin, "Fast Modulo Scheduling Under the Simplex Scheduling
///      Framework", PRISM 1995.01, 1995.
///
/// Resource-free scheduling problems (called "central problems" in the papers)
/// have an *integer* linear programming formulation with a totally unimodular
/// constraint matrix. Such ILPs can however be solved optimally in polynomial
/// time with a (non-integer) LP solver (such as the simplex algorithm), as the
/// LP solution is guaranteed to be integer. Note that this is the same idea as
/// used by SDC-based schedulers.
class SimplexSchedulerBase {
protected:
  /// The objective is to minimize the start time of this operation.
  Operation *lastOp;

  /// S is part of a mechanism to assign fixed values to the LP variables.
  int parameterS;

  /// T represents the initiation interval (II). Its minimally-feasible value is
  /// computed by the algorithm.
  int parameterT;

  /// The simplex tableau is the algorithm's main data structure.
  /// The dashed parts always contain the zero respectively the identity matrix,
  /// and therefore are not stored explicitly.
  ///
  ///                        ◀───nColumns────▶
  ///           nParameters────┐
  ///                        ◀─┴─▶
  ///                       ┌─────┬───────────┬ ─ ─ ─ ─ ┐
  ///                      ▲│. . .│. . ... . .│    0        ▲
  ///           nObjectives││. . .│. . ... . .│         │   │
  ///                      ▼│. . .│. . ... . .│             │
  ///                       ├─────┼───────────┼ ─ ─ ─ ─ ┤   │
  ///  firstConstraintRow > │. . .│. . ... . .│1            │nRows
  ///                       │. . .│. . ... . .│  1      │   │
  ///                       │. . .│. . ... . .│    1        │
  ///                       │. . .│. . ... . .│      1  │   │
  ///                       │. . .│. . ... . .│        1    ▼
  ///                       └─────┴───────────┴ ─ ─ ─ ─ ┘
  ///       parameter1Column ^
  ///         parameterSColumn ^
  ///           parameterTColumn ^
  ///  firstNonBasicVariableColumn ^
  ///                              ─────────── ──────────
  ///                       nonBasicVariables   basicVariables
  SmallVector<SmallVector<int>> tableau;

  /// During the pivot operation, one column in the elided part of the tableau
  /// is modified; this vector temporarily catches the changes.
  SmallVector<int> implicitBasicVariableColumnVector;

  /// The linear program models the operations' start times as variables, which
  /// we identify here as 0, ..., |ops|-1.
  /// Additionally, for each dependence (precisely, the inequality modeling the
  /// precedence constraint), a slack variable is required; these are identified
  /// as |ops|, ..., |ops|+|deps|-1.
  ///
  /// This vector stores the numeric IDs of non-basic variables. A variable's
  /// index *i* in this vector corresponds to the tableau *column*
  /// `firstNonBasicVariableColumn`+*i*.
  SmallVector<unsigned> nonBasicVariables;

  /// This vector store the numeric IDs of basic variables. A variable's index
  /// *i* in this vector corresponds to the tableau *row*
  /// `firstConstraintRow`+*i*.
  SmallVector<unsigned> basicVariables;

  /// Used to conveniently retrieve an operation's start time variable. The
  /// alternative would be to find the op's index in the problem's list of
  /// operations.
  DenseMap<Operation *, unsigned> startTimeVariables;

  /// This vector keeps track of the current locations (i.e. row or column) of
  /// a start time variable in the tableau. We encode column numbers as positive
  /// integers, and row numbers as negative integers. We do not track the slack
  /// variables.
  SmallVector<int> startTimeLocations;

  /// Non-basic variables can be "frozen" to a specific value, which prevents
  /// them from being pivoted into basis again.
  DenseMap<unsigned, unsigned> frozenVariables;

  /// Number of rows in the tableau = |obj| + |deps|.
  unsigned nRows;
  /// Number of explicitly stored columns in the tableau = |params| + |ops|.
  unsigned nColumns;

  // Number of objective rows.
  unsigned nObjectives;
  /// All other rows encode linear constraints.
  unsigned &firstConstraintRow = nObjectives;

  // Number of parameters (fixed for now).
  static constexpr unsigned nParameters = 3;
  /// The first column corresponds to the always-one "parameter" in u = (1,S,T).
  static constexpr unsigned parameter1Column = 0;
  /// The second column corresponds to the variable-freezing parameter S.
  static constexpr unsigned parameterSColumn = 1;
  /// The third column corresponds to the parameter T, i.e. the current II.
  static constexpr unsigned parameterTColumn = 2;
  /// All other (explicitly stored) columns represent non-basic variables.
  static constexpr unsigned firstNonBasicVariableColumn = nParameters;

  virtual Problem &getProblem() = 0;
  virtual bool fillObjectiveRow(SmallVector<int> &row, unsigned obj);
  virtual void fillConstraintRow(SmallVector<int> &row,
                                 Problem::Dependence dep);
  void buildTableau();

  int getParametricConstant(unsigned row);
  SmallVector<int> getObjectiveVector(unsigned column);
  Optional<unsigned> findDualPivotRow();
  Optional<unsigned> findDualPivotColumn(unsigned pivotRow,
                                         bool allowPositive = false);
  Optional<unsigned> findPrimalPivotColumn();
  Optional<unsigned> findPrimalPivotRow(unsigned pivotColumn);
  void multiplyRow(unsigned row, int factor);
  void addMultipleOfRow(unsigned sourceRow, int factor, unsigned targetRow);
  void pivot(unsigned pivotRow, unsigned pivotColumn);
  LogicalResult solveTableau();
  LogicalResult restoreDualFeasibility();
  bool isInBasis(unsigned startTimeVariable);
  unsigned freeze(unsigned startTimeVariable, unsigned timeStep);
  void translate(unsigned column, int factor1, int factorS, int factorT);
  LogicalResult scheduleAt(unsigned startTimeVariable, unsigned timeStep);
  void moveBy(unsigned startTimeVariable, unsigned amount);
  unsigned getStartTime(unsigned startTimeVariable);

  void dumpTableau();

public:
  explicit SimplexSchedulerBase(Operation *lastOp) : lastOp(lastOp) {}
  virtual ~SimplexSchedulerBase() = default;
  virtual LogicalResult schedule() = 0;
};

/// This class solves the basic, acyclic `Problem`.
class SimplexScheduler : public SimplexSchedulerBase {
private:
  Problem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SimplexScheduler(Problem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}

  LogicalResult schedule() override;
};

/// This class solves the resource-free `CyclicProblem`.  The optimal initiation
/// interval (II) is determined as a side product of solving the parametric
/// problem, and corresponds to the "RecMII" (= recurrence-constrained minimum
/// II) usually considered as one component in the lower II bound used by modulo
/// schedulers.
class CyclicSimplexScheduler : public SimplexSchedulerBase {
private:
  CyclicProblem &prob;

protected:
  Problem &getProblem() override { return prob; }
  void fillConstraintRow(SmallVector<int> &row,
                         Problem::Dependence dep) override;

public:
  CyclicSimplexScheduler(CyclicProblem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves acyclic, resource-constrained `SharedOperatorsProblem` with
// a simplified version of the iterative heuristic presented in [2].
class SharedOperatorsSimplexScheduler : public SimplexSchedulerBase {
private:
  SharedOperatorsProblem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SharedOperatorsSimplexScheduler(SharedOperatorsProblem &prob,
                                  Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves the `ModuloProblem` using the iterative heuristic presented
// in [2].
class ModuloSimplexScheduler : public CyclicSimplexScheduler {
private:
  struct MRT {
    ModuloSimplexScheduler &sched;

    using TableType = SmallDenseMap<unsigned, DenseSet<Operation *>>;
    using ReverseTableType = SmallDenseMap<Operation *, unsigned>;
    SmallDenseMap<Problem::OperatorType, TableType> tables;
    SmallDenseMap<Problem::OperatorType, ReverseTableType> reverseTables;

    explicit MRT(ModuloSimplexScheduler &sched) : sched(sched) {}
    LogicalResult enter(Operation *op, unsigned timeStep);
    void release(Operation *op);
  };

  ModuloProblem &prob;
  SmallVector<unsigned> asapTimes, alapTimes;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;

protected:
  Problem &getProblem() override { return prob; }
  enum { OBJ_LATENCY = 0, OBJ_AXAP /* i.e. either ASAP or ALAP */ };
  bool fillObjectiveRow(SmallVector<int> &row, unsigned obj) override;
  void updateMargins();
  void incrementII();
  void scheduleOperation(Operation *n);

public:
  ModuloSimplexScheduler(ModuloProblem &prob, Operation *lastOp)
      : CyclicSimplexScheduler(prob, lastOp), prob(prob), mrt(*this) {}
  LogicalResult schedule() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SimplexSchedulerBase
//===----------------------------------------------------------------------===//

bool SimplexSchedulerBase::fillObjectiveRow(SmallVector<int> &row,
                                            unsigned obj) {
  assert(obj == 0);
  // Minimize start time of user-specified last operation.
  row[startTimeLocations[startTimeVariables[lastOp]]] = 1;
  return false;
}

void SimplexSchedulerBase::fillConstraintRow(SmallVector<int> &row,
                                             Problem::Dependence dep) {
  auto &prob = getProblem();
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();
  unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
  row[parameter1Column] = -latency; // note the negation
  if (src != dst) { // note that these coefficients just zero out in self-arcs.
    row[startTimeLocations[startTimeVariables[src]]] = 1;
    row[startTimeLocations[startTimeVariables[dst]]] = -1;
  }
}

void SimplexSchedulerBase::buildTableau() {
  auto &prob = getProblem();

  // The initial tableau is constructed so that operations' start time variables
  // are out of basis, whereas all slack variables are in basis. We will number
  // them accordingly.
  unsigned var = 0;

  // Assign column and variable numbers to the operations' start times.
  for (auto *op : prob.getOperations()) {
    nonBasicVariables.push_back(var);
    startTimeVariables[op] = var;
    startTimeLocations.push_back(firstNonBasicVariableColumn + var);
    ++var;
  }

  // one column for each parameter (1,S,T), and for all operations
  nColumns = nParameters + nonBasicVariables.size();

  // Helper to grow both the tableau and the implicit column vector.
  auto addRow = [&]() -> SmallVector<int> & {
    implicitBasicVariableColumnVector.push_back(0);
    return tableau.emplace_back(nColumns, 0);
  };

  // Set up the objective rows.
  nObjectives = 0;
  bool hasMoreObjectives;
  do {
    auto &objRowVec = addRow();
    hasMoreObjectives = fillObjectiveRow(objRowVec, nObjectives);
    ++nObjectives;
  } while (hasMoreObjectives);

  // Now set up rows/constraints for the dependences.
  for (auto *op : prob.getOperations()) {
    for (auto &dep : prob.getDependences(op)) {
      auto &consRowVec = addRow();
      fillConstraintRow(consRowVec, dep);
      basicVariables.push_back(var);
      ++var;
    }
  }

  // one row per objective + one row per dependence
  nRows = tableau.size();
}

int SimplexSchedulerBase::getParametricConstant(unsigned row) {
  auto &rowVec = tableau[row];
  // Compute the dot-product ~B[row] * u between the constant matrix and the
  // parameter vector.
  return rowVec[parameter1Column] + rowVec[parameterSColumn] * parameterS +
         rowVec[parameterTColumn] * parameterT;
}

SmallVector<int> SimplexSchedulerBase::getObjectiveVector(unsigned column) {
  SmallVector<int> objVec;
  // Extract the column vector C^T[column] from the cost matrix.
  for (unsigned obj = 0; obj < nObjectives; ++obj)
    objVec.push_back(tableau[obj][column]);
  return objVec;
}

Optional<unsigned> SimplexSchedulerBase::findDualPivotRow() {
  // Find the first row in which the parametric constant is negative.
  for (unsigned row = firstConstraintRow; row < nRows; ++row)
    if (getParametricConstant(row) < 0)
      return row;

  return None;
}

Optional<unsigned>
SimplexSchedulerBase::findDualPivotColumn(unsigned pivotRow,
                                          bool allowPositive) {
  SmallVector<int> maxQuot(nObjectives, std::numeric_limits<int>::min());
  Optional<unsigned> pivotCol;

  // Look for non-zero entries in the constraint matrix (~A part of the
  // tableau). If multiple candidates exist, take the one corresponding to the
  // lexicographical maximum (over the objective rows) of the quotients:
  //   tableau[<objective row>][col] / pivotCand
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    if (frozenVariables.count(
            nonBasicVariables[col - firstNonBasicVariableColumn]))
      continue;

    int pivotCand = tableau[pivotRow][col];
    // Only negative candidates bring us closer to the optimal solution.
    // However, when freezing variables to a certain value, we accept that the
    // value of the objective function degrades.
    if (pivotCand < 0 || (allowPositive && pivotCand > 0)) {
      // The constraint matrix has only {-1, 0, 1} entries by construction.
      assert(pivotCand * pivotCand == 1);

      SmallVector<int> quot;
      for (unsigned obj = 0; obj < nObjectives; ++obj)
        quot.push_back(tableau[obj][col] / pivotCand);

      if (std::lexicographical_compare(maxQuot.begin(), maxQuot.end(),
                                       quot.begin(), quot.end())) {
        maxQuot = quot;
        pivotCol = col;
      }
    }
  }

  return pivotCol;
}

Optional<unsigned> SimplexSchedulerBase::findPrimalPivotColumn() {
  // Find the first lexico-negative column in the cost matrix.
  SmallVector<int> zeroVec(nObjectives, 0);
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    if (frozenVariables.count(
            nonBasicVariables[col - firstNonBasicVariableColumn]))
      continue;

    SmallVector<int> objVec = getObjectiveVector(col);
    if (std::lexicographical_compare(objVec.begin(), objVec.end(),
                                     zeroVec.begin(), zeroVec.end()))
      return col;
  }

  return None;
}

Optional<unsigned>
SimplexSchedulerBase::findPrimalPivotRow(unsigned pivotColumn) {
  int minQuot = std::numeric_limits<int>::max();
  Optional<unsigned> pivotRow;

  // Look for positive entries in the constraint matrix (~A part of the
  // tableau). If multiple candidates exist, take the one corresponding to the
  // minimum of the quotient:
  //   parametricConstant(row) / pivotCand
  for (unsigned row = firstConstraintRow; row < nRows; ++row) {
    int pivotCand = tableau[row][pivotColumn];
    if (pivotCand > 0) {
      // The constraint matrix has only {-1, 0, 1} entries by construction.
      assert(pivotCand == 1);
      int quot = getParametricConstant(row) / pivotCand;
      if (quot < minQuot) {
        minQuot = quot;
        pivotRow = row;
      }
    }
  }

  return pivotRow;
}

void SimplexSchedulerBase::multiplyRow(unsigned row, int factor) {
  assert(factor != 0);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[row][col] *= factor;
  // Also multiply the corresponding entry in the temporary column vector.
  implicitBasicVariableColumnVector[row] *= factor;
}

void SimplexSchedulerBase::addMultipleOfRow(unsigned sourceRow, int factor,
                                            unsigned targetRow) {
  assert(factor != 0 && sourceRow != targetRow);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[targetRow][col] += tableau[sourceRow][col] * factor;
  // Again, perform row operation on the temporary column vector as well.
  implicitBasicVariableColumnVector[targetRow] +=
      implicitBasicVariableColumnVector[sourceRow] * factor;
}

/// The pivot operation applies elementary row operations to the tableau in
/// order to make the \p pivotColumn (corresponding to a non-basic variable) a
/// unit vector (only the \p pivotRow'th entry is 1). Then, a basis exchange is
/// performed: the non-basic variable is swapped with the basic variable
/// associated with the pivot row.
void SimplexSchedulerBase::pivot(unsigned pivotRow, unsigned pivotColumn) {
  // The implicit columns are part of an identity matrix.
  implicitBasicVariableColumnVector[pivotRow] = 1;

  int pivotElem = tableau[pivotRow][pivotColumn];
  // The constraint matrix has only {-1, 0, 1} entries by construction.
  assert(pivotElem * pivotElem == 1);
  // Make `tableau[pivotRow][pivotColumn]` := 1
  multiplyRow(pivotRow, 1 / pivotElem);

  for (unsigned row = 0; row < nRows; ++row) {
    if (row == pivotRow)
      continue;

    int elem = tableau[row][pivotColumn];
    if (elem == 0)
      continue; // nothing to do

    // Make `tableau[row][pivotColumn]` := 0.
    addMultipleOfRow(pivotRow, -elem, row);
  }

  // Swap the pivot column with the implicitly constructed column vector.
  // We really only need to copy in one direction here, as the former pivot
  // column is a unit vector, which is not stored explicitly.
  for (unsigned row = 0; row < nRows; ++row) {
    tableau[row][pivotColumn] = implicitBasicVariableColumnVector[row];
    implicitBasicVariableColumnVector[row] = 0; // Reset for next pivot step.
  }

  // Look up numeric IDs of variables involved in this pivot operation.
  unsigned &nonBasicVar =
      nonBasicVariables[pivotColumn - firstNonBasicVariableColumn];
  unsigned &basicVar = basicVariables[pivotRow - firstConstraintRow];

  // Keep track of where start time variables are; ignore slack variables.
  if (nonBasicVar < startTimeLocations.size())
    startTimeLocations[nonBasicVar] = -pivotRow; // ...going into basis.
  if (basicVar < startTimeLocations.size())
    startTimeLocations[basicVar] = pivotColumn; // ...going out of basis.

  // Record the swap in the variable lists.
  std::swap(nonBasicVar, basicVar);
}

LogicalResult SimplexSchedulerBase::solveTableau() {
  // "Solving" technically means perfoming dual pivot steps until primal
  // feasibility is reached, i.e. the parametric constants in all constraint
  // rows are non-negative.
  while (auto pivotRow = findDualPivotRow()) {
    // Look for pivot elements.
    if (auto pivotCol = findDualPivotColumn(*pivotRow)) {
      pivot(*pivotRow, *pivotCol);
      continue;
    }

    // If we did not find a pivot column, then the entire row contained only
    // positive entries, and the problem is in principle infeasible. However, if
    // the entry in the `parameterTColumn` is positive, we can make the LP
    // feasible again by increasing the II.
    int entry1Col = tableau[*pivotRow][parameter1Column];
    int entryTCol = tableau[*pivotRow][parameterTColumn];
    if (entryTCol > 0) {
      // The negation of `entry1Col` is not in the paper. I think this is an
      // oversight, because `entry1Col` certainly is negative (otherwise the row
      // would not have been a valid pivot row), and without the negation, the
      // new II would be negative.
      assert(entry1Col < 0);
      parameterT = (-entry1Col - 1) / entryTCol + 1;

      LLVM_DEBUG(dbgs() << "Increased II to " << parameterT << '\n');

      continue;
    }

    // Otherwise, the linear program is infeasible.
    return failure();
  }

  // Optimal solution found!
  return success();
}

LogicalResult SimplexSchedulerBase::restoreDualFeasibility() {
  // Dual feasibility requires that all columns in the cost matrix are
  // non-lexico-negative. This property may be violated after changing the order
  // of the objective rows, and can be restored by performing primal pivot
  // steps.
  while (auto pivotCol = findPrimalPivotColumn()) {
    // Look for pivot elements.
    if (auto pivotRow = findPrimalPivotRow(*pivotCol)) {
      pivot(*pivotRow, *pivotCol);
      continue;
    }

    // Otherwise, the linear program is unbounded.
    return failure();
  }

  // Optimal solution found!
  return success();
}

bool SimplexSchedulerBase::isInBasis(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());
  int loc = startTimeLocations[startTimeVariable];
  if (-loc >= (int)firstConstraintRow)
    return true;
  if (loc >= (int)firstNonBasicVariableColumn)
    return false;
  llvm_unreachable("Invalid variable location");
}

unsigned SimplexSchedulerBase::freeze(unsigned startTimeVariable,
                                      unsigned timeStep) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  // Mark variable.
  frozenVariables[startTimeVariable] = timeStep;

  if (!isInBasis(startTimeVariable))
    // That's all for non-basic variables.
    return startTimeLocations[startTimeVariable];

  // We need to pivot this variable one out of basis.
  unsigned pivotRow = -startTimeLocations[startTimeVariable];

  // Here, positive pivot elements can be considered as well, hence finding a
  // suitable column should not fail.
  auto pivotCol = findDualPivotColumn(pivotRow, /* allowPositive= */ true);
  assert(pivotCol);

  // Perform the exchange.
  pivot(pivotRow, *pivotCol);

  // `startTimeVariable` is now represented by `pivotCol`.
  return *pivotCol;
}

void SimplexSchedulerBase::translate(unsigned column, int factor1, int factorS,
                                     int factorT) {
  for (unsigned row = 0; row < nRows; ++row) {
    auto &rowVec = tableau[row];
    int elem = rowVec[column];
    if (elem == 0)
      continue;

    rowVec[parameter1Column] += -elem * factor1;
    rowVec[parameterSColumn] += -elem * factorS;
    rowVec[parameterTColumn] += -elem * factorT;
  }
}

LogicalResult SimplexSchedulerBase::scheduleAt(unsigned startTimeVariable,
                                               unsigned timeStep) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  // Freeze variable and translate its column by parameter S.
  unsigned frozenCol = freeze(startTimeVariable, timeStep);
  translate(frozenCol, /* factor1= */ 0, /* factorS= */ 1, /* factorT= */ 0);

  // Temporarily set S to the desired value, and attempt to solve.
  parameterS = timeStep;
  auto solved = solveTableau();
  parameterS = 0;

  if (failed(solved)) {
    // The LP is infeasible with the new constraint. We could try other values
    // for S, but for now, we just roll back and signal failure to the driver.
    translate(frozenCol, /* factor1= */ 0, /* factorS= */ -1, /* factorT= */ 0);
    frozenVariables.erase(startTimeVariable);
    auto solvedAfterRollback = solveTableau();
    assert(succeeded(solvedAfterRollback));
    (void)solvedAfterRollback;
    return failure();
  }

  // Translate S by the other parameter(s). For acyclic problems, this means
  // setting `factor1` to `timeStep`. For cyclic problems, we perform a modulo
  // decomposition: S = `factor1` + `factorT` * T, with `factor1` < T.
  //
  // This translation does not change the values of the parametric constants,
  // hence we do not need to solve the tableau again.
  //
  // Note: I added a negation of the factors here, which is not mentioned in the
  // paper's text, but apparently used in the example. Without it, the intended
  // effect, i.e. making the S-column all-zero again, is not achieved.
  if (parameterT == 0)
    translate(parameterSColumn, /* factor1= */ -timeStep, /* factorS= */ 1,
              /* factorT= */ 0);
  else
    translate(parameterSColumn, /* factor1= */ -(timeStep % parameterT),
              /* factorS= */ 1,
              /* factorT= */ -(timeStep / parameterT));

  return success();
}

void SimplexSchedulerBase::moveBy(unsigned startTimeVariable, unsigned amount) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(frozenVariables.count(startTimeVariable));

  // Bookkeeping.
  frozenVariables[startTimeVariable] += amount;

  // Moving an already frozen variable means translating it by the desired
  // amount, and solving the tableau to restore primal feasibility...
  translate(startTimeLocations[startTimeVariable], /* factor1= */ amount,
            /* factorS= */ 0, /* factorT= */ 0);

  // ... however, we typically batch-move multiple operations (otherwise, the
  // tableau may become infeasible on intermediate steps), so actually defer
  // solving to the caller.
}

unsigned SimplexSchedulerBase::getStartTime(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());

  if (!isInBasis(startTimeVariable))
    // Non-basic variables that are not already fixed to a specific time step
    // are 0 at the end of the simplex algorithm.
    return frozenVariables.lookup(startTimeVariable);

  // For the variables currently in basis, we look up the solution in the
  // tableau.
  return getParametricConstant(-startTimeLocations[startTimeVariable]);
}

void SimplexSchedulerBase::dumpTableau() {
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  for (unsigned i = 0; i < nRows; ++i) {
    if (i == firstConstraintRow) {
      for (unsigned j = 0; j < nColumns; ++j) {
        if (j == firstNonBasicVariableColumn)
          dbgs() << "-+";
        dbgs() << "----";
      }
      dbgs() << '\n';
    }
    for (unsigned j = 0; j < nColumns; ++j) {
      if (j == firstNonBasicVariableColumn)
        dbgs() << " |";
      dbgs() << format(" %3d", tableau[i][j]);
    }
    if (i >= firstConstraintRow)
      dbgs() << format(" |< %2d", basicVariables[i - firstConstraintRow]);
    dbgs() << '\n';
  }
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  dbgs() << format(" %3d %3d %3d | ", 1, parameterS, parameterT);
  for (unsigned j = firstNonBasicVariableColumn; j < nColumns; ++j)
    dbgs() << format(" %2d^",
                     nonBasicVariables[j - firstNonBasicVariableColumn]);
  dbgs() << '\n';
}

//===----------------------------------------------------------------------===//
// SimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult SimplexScheduler::schedule() {
  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Optimal solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// CyclicSimplexScheduler
//===----------------------------------------------------------------------===//

void CyclicSimplexScheduler::fillConstraintRow(SmallVector<int> &row,
                                               Problem::Dependence dep) {
  SimplexSchedulerBase::fillConstraintRow(row, dep);
  if (auto dist = prob.getDistance(dep))
    row[parameterTColumn] = *dist;
}

LogicalResult CyclicSimplexScheduler::schedule() {
  parameterS = 0;
  parameterT = 1;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Optimal solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// SharedOperatorsSimplexScheduler
//===----------------------------------------------------------------------===//

static bool isLimited(Operation *op, SharedOperatorsProblem &prob) {
  return prob.getLimit(*prob.getLinkedOperatorType(op)).getValueOr(0) > 0;
}

LogicalResult SharedOperatorsSimplexScheduler::schedule() {
  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  LLVM_DEBUG(dbgs() << "After solving resource-free problem:\n"; dumpTableau());

  // The *heuristic* part of this scheduler starts here:
  // We will now *choose* start times for operations using a shared operator
  // type, in a way that respects the allocation limits, and consecutively solve
  // the LP with these added constraints. The individual LPs are still solved to
  // optimality (meaning: the start times of the "last" operation is still
  // optimal w.r.t. the already fixed operations), however the heuristic choice
  // means we cannot guarantee the optimality for the overall problem.

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  SmallVector<Operation *> limitedOps;
  for (auto *op : ops)
    if (isLimited(op, prob))
      limitedOps.push_back(op);

  // Build a priority list of the limited operations.
  //
  // We sort by the resource-free start times to produce a topological order of
  // the operations. Better priority functions are known, but require computing
  // additional properties, e.g. ASAP and ALAP times for mobility, or graph
  // analysis for height. Assigning operators (=resources) in this order at
  // least ensures that the (acyclic!) problem remains feasible throughout the
  // process.
  //
  // TODO: Implement more sophisticated priority function.
  std::stable_sort(limitedOps.begin(), limitedOps.end(),
                   [&](Operation *a, Operation *b) {
                     return getStartTime(startTimeVariables[a]) <
                            getStartTime(startTimeVariables[b]);
                   });

  // Store the number of operations using an operator type in a particular time
  // step.
  SmallDenseMap<Problem::OperatorType, SmallDenseMap<unsigned, unsigned>>
      reservationTable;

  for (auto *op : limitedOps) {
    auto opr = *prob.getLinkedOperatorType(op);
    unsigned limit = prob.getLimit(opr).getValueOr(0);
    assert(limit > 0);

    // Find the first time step (beginning at the current start time in the
    // partial schedule) in which an operator instance is available.
    unsigned startTimeVar = startTimeVariables[op];
    unsigned candTime = getStartTime(startTimeVar);
    while (reservationTable[opr].lookup(candTime) == limit)
      ++candTime;

    // Fix the start time. As explained above, this cannot make the problem
    // infeasible.
    auto fixed = scheduleAt(startTimeVar, candTime);
    assert(succeeded(fixed));
    (void)fixed;

    // Record the operator use.
    ++reservationTable[opr][candTime];

    LLVM_DEBUG(dbgs() << "After scheduling " << startTimeVar
                      << " to t=" << candTime << ":\n";
               dumpTableau());
  }

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Feasible solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloSimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult ModuloSimplexScheduler::MRT::enter(Operation *op,
                                                 unsigned timeStep) {
  auto opr = *sched.prob.getLinkedOperatorType(op);
  auto lim = *sched.prob.getLimit(opr);
  assert(lim > 0);

  auto &revTab = reverseTables[opr];
  assert(!revTab.count(op));

  unsigned slot = timeStep % sched.parameterT;
  auto &cell = tables[opr][slot];
  if (cell.size() < lim) {
    cell.insert(op);
    revTab[op] = slot;
    return success();
  }
  return failure();
}

void ModuloSimplexScheduler::MRT::release(Operation *op) {
  auto opr = *sched.prob.getLinkedOperatorType(op);
  auto &revTab = reverseTables[opr];
  auto it = revTab.find(op);
  assert(it != revTab.end());
  tables[opr][it->second].erase(op);
  revTab.erase(it);
}

bool ModuloSimplexScheduler::fillObjectiveRow(SmallVector<int> &row,
                                              unsigned obj) {
  switch (obj) {
  case OBJ_LATENCY:
    // Minimize start time of user-specified last operation.
    row[startTimeLocations[startTimeVariables[lastOp]]] = 1;
    return true;
  case OBJ_AXAP:
    // Minimize sum of start times of all-but-the-last operation.
    for (auto *op : getProblem().getOperations())
      if (op != lastOp)
        row[startTimeLocations[startTimeVariables[op]]] = 1;
    return false;
  default:
    llvm_unreachable("Unsupported objective requested");
  }
}

void ModuloSimplexScheduler::updateMargins() {
  // Assumption: current secondary objective is "ASAP".
  // Negate the objective row once to effectively maximize the sum of start
  // times, which yields the "ALAP" times after solving the tableau. Then,
  // negate it again to restore the "ASAP" objective, and store these times as
  // well.
  for (auto *axapTimes : {&alapTimes, &asapTimes}) {
    multiplyRow(OBJ_AXAP, -1);
    // This should not fail for a feasible tableau.
    auto dualFeasRestored = restoreDualFeasibility();
    auto solved = solveTableau();
    assert(succeeded(dualFeasRestored) && succeeded(solved));
    (void)dualFeasRestored, (void)solved;

    for (unsigned stv = 0; stv < startTimeLocations.size(); ++stv)
      (*axapTimes)[stv] = getStartTime(stv);
  }
}

void ModuloSimplexScheduler::incrementII() {
  // Account for the shift in the frozen start times that will be caused by
  // increasing `parameterT`: Assuming decompositions of
  //   t = phi * II + tau  and  t' = phi * (II + 1) + tau,
  // so the required shift is
  //   t' - t = phi = floordiv(t / II)
  for (auto &kv : frozenVariables) {
    unsigned &frozenTime = kv.getSecond();
    frozenTime += frozenTime / parameterT;
  }

  // Increment the parameter.
  ++parameterT;
}

void ModuloSimplexScheduler::scheduleOperation(Operation *n) {
  unsigned stvN = startTimeVariables[n];

  // Get current state of the LP, and determine range of alternative times
  // guaranteed to be feasible.
  unsigned stN = getStartTime(stvN);
  unsigned lbN = (unsigned)std::max<int>(asapTimes[stvN], stN - parameterT + 1);
  unsigned ubN = (unsigned)std::min<int>(alapTimes[stvN], lbN + parameterT - 1);

  LLVM_DEBUG(dbgs() << "Attempting to schedule at t=" << stN << ", or in ["
                    << lbN << ", " << ubN << "]: " << *n << '\n');

  SmallVector<unsigned> candTimes;
  candTimes.push_back(stN);
  for (unsigned ct = lbN; ct <= ubN; ++ct)
    if (ct != stN)
      candTimes.push_back(ct);

  for (unsigned ct : candTimes)
    if (succeeded(mrt.enter(n, ct))) {
      auto fixedN = scheduleAt(stvN, stN);
      assert(succeeded(fixedN));
      (void)fixedN;
      LLVM_DEBUG(dbgs() << "Success at t=" << stN << " " << *n << '\n');
      return;
    }

  // As a last resort, increase II to make room for the op. De Dinechin's
  // Theorem 1 lays out conditions/guidelines to transform the current partial
  // schedule for II to a valid one for a larger II'.

  LLVM_DEBUG(dbgs() << "Incrementing II to " << (parameterT + 1)
                    << " to resolve resource conflict for " << *n << '\n');

  // Note that the approach below is much simpler than in the paper
  // because of the fully-pipelined operators. In our case, it's always
  // sufficient to increment the II by one.

  // Decompose start time.
  unsigned phiN = stN / parameterT;
  unsigned tauN = stN % parameterT;

  // Keep track whether the following moves free at least one operator
  // instance in the slot desired by the current op - then it can stay there.
  unsigned deltaN = 1;

  // We're going to revisit the current partial schedule.
  SmallVector<Operation *> moved;
  for (Operation *j : scheduled) {
    unsigned stvJ = startTimeVariables[j];
    unsigned stJ = getStartTime(stvJ);
    unsigned phiJ = stJ / parameterT;
    unsigned tauJ = stJ % parameterT;
    unsigned deltaJ = 0;

    // To actually resolve the resource conflicts, we move operations that are
    // "preceded" (cf. de Dinechin's ≺ relation) one slot to the right.
    if (tauN < tauJ || (tauN == tauJ && phiN > phiJ) ||
        (tauN == tauJ && phiN == phiJ && stvN < stvJ)) {
      // TODO: Replace the last condition with a proper graph analysis.

      deltaJ = 1;
      moved.push_back(j);
      if (tauN == tauJ)
        deltaN = 0;
    }

    // Apply the move to the tableau.
    moveBy(stvJ, deltaJ);
  }

  // Finally, increment the II.
  incrementII();
  auto solved = solveTableau();
  assert(succeeded(solved));
  (void)solved;

  // Re-enter moved operations into their new slots.
  for (auto *m : moved)
    mrt.release(m);
  for (auto *m : moved) {
    auto enteredM = mrt.enter(m, getStartTime(startTimeVariables[m]));
    assert(succeeded(enteredM));
    (void)enteredM;
  }

  // Finally, schedule the operation. Adding `phiN` accounts for the implicit
  // shift caused by incrementing the II; cf. `incrementII()`.
  auto fixedN = scheduleAt(stvN, stN + phiN + deltaN);
  auto enteredN = mrt.enter(n, tauN + deltaN);
  assert(succeeded(fixedN) && succeeded(enteredN));
  (void)fixedN, (void)enteredN;
}

LogicalResult ModuloSimplexScheduler::schedule() {
  parameterS = 0;
  parameterT = 1;
  buildTableau();
  asapTimes.resize(startTimeLocations.size());
  alapTimes.resize(startTimeLocations.size());

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  for (auto *op : ops)
    if (isLimited(op, prob))
      unscheduled.push_back(op);

  // Main loop: Iteratively fix limited operations to time steps.
  while (!unscheduled.empty()) {
    // Update ASAP/ALAP times.
    updateMargins();

    // Heuristically (here: least amount of slack) pick the next operation to
    // schedule.
    auto *opIt =
        std::min_element(unscheduled.begin(), unscheduled.end(),
                         [&](Operation *opA, Operation *opB) {
                           auto stvA = startTimeVariables[opA];
                           auto stvB = startTimeVariables[opB];
                           auto slackA = alapTimes[stvA] - asapTimes[stvA];
                           auto slackB = alapTimes[stvB] - asapTimes[stvB];
                           return slackA < slackB;
                         });
    Operation *op = *opIt;
    unscheduled.erase(opIt);

    scheduleOperation(op);
    scheduled.push_back(op);
  }

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult scheduling::scheduleSimplex(Problem &prob, Operation *lastOp) {
  SimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(CyclicProblem &prob,
                                          Operation *lastOp) {
  CyclicSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(SharedOperatorsProblem &prob,
                                          Operation *lastOp) {
  SharedOperatorsSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ModuloProblem &prob,
                                          Operation *lastOp) {
  ModuloSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}
