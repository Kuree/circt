//===- CompileControl.cpp - Compile Control Pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Compile Control pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

/// A helper function to create constants in the HW dialect.
static hw::ConstantOp createConstant(OpBuilder &builder, Location loc,
                                     size_t bitWidth, size_t value) {
  return builder.create<hw::ConstantOp>(
      loc, APInt(bitWidth, value, /*unsigned=*/true));
}

/// Given some number of states, returns the necessary bit width
/// TODO(Calyx): Probably a better built-in operation?
static size_t getNecessaryBitWidth(size_t numStates) {
  APInt apNumStates(64, numStates);
  size_t log2 = apNumStates.ceilLogBase2();
  return log2 > 1 ? log2 : 1;
}

/// Creates a RegisterOp, with input and output port bit widths defined by
/// `width`.
static RegisterOp createRegister(OpBuilder &builder, ComponentOp &component,
                                 size_t width, StringRef name) {
  IRRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(component.getBody());
  return builder.create<RegisterOp>(component->getLoc(), name, width);
}

class CompileControlVisitor {
public:
  void dispatch(Operation *op, ComponentOp component) {
    TypeSwitch<Operation *>(op)
        .template Case<SeqOp, EnableOp>(
            [&](auto opNode) { visit(opNode, component); })
        .Default([&](auto) {
          op->emitError() << "Operation '" << op->getName()
                          << "' not supported for control compilation";
        });
  }

private:
  void visit(SeqOp seqOp, ComponentOp &component);
  void visit(EnableOp, ComponentOp &) {
    // nothing to do
  }
};

/// Generates a latency-insensitive FSM to realize a sequential operation.
/// This is done by initializing GroupGoOp values for the enabled groups in
/// the SeqOp, and then creating a new Seq GroupOp with the given FSM. Each
/// step in the FSM is guarded by the done operation of the group currently
/// being executed. After the group is complete, the FSM is incremented. This
/// SeqOp is then replaced in the control with an Enable statement referring
/// to the new Seq GroupOp.
void CompileControlVisitor::visit(SeqOp seq, ComponentOp &component) {
  auto wires = component.getWiresOp();
  Block *wiresBody = wires.getBody();

  auto &seqOps = seq.getBody()->getOperations();
  if (!llvm::all_of(seqOps, [](auto &&op) { return isa<EnableOp>(op); })) {
    seq.emitOpError("should only contain EnableOps in this pass.");
    return;
  }

  // This should be the number of enable statements + 1 since this is the
  // maximum value the FSM register will reach.
  size_t fsmBitWidth = getNecessaryBitWidth(seqOps.size() + 1);

  OpBuilder builder(component->getRegion(0));
  auto fsmRegister = createRegister(builder, component, fsmBitWidth, "fsm");
  Value fsmIn = fsmRegister.in();
  Value fsmWriteEn = fsmRegister.write_en();
  Value fsmOut = fsmRegister.out();

  builder.setInsertionPointToStart(wiresBody);
  auto oneConstant = createConstant(builder, wires->getLoc(), 1, 1);

  // Create the new compilation group to replace this SeqOp.
  builder.setInsertionPointToEnd(wiresBody);
  auto seqGroup =
      builder.create<GroupOp>(wires->getLoc(), builder.getStringAttr("seq"));

  // Guarantees a unique SymbolName for the group.
  SymbolTable symTable(wires);
  symTable.insert(seqGroup);

  size_t fsmIndex = 0;
  SmallVector<Attribute, 8> compiledGroups;
  Value fsmNextState;
  seq.walk([&](EnableOp enable) {
    StringRef groupName = enable.groupName();
    compiledGroups.push_back(
        SymbolRefAttr::get(builder.getContext(), groupName));
    auto groupOp = symTable.lookup<GroupOp>(groupName);

    builder.setInsertionPoint(groupOp);
    auto fsmCurrentState =
        createConstant(builder, wires->getLoc(), fsmBitWidth, fsmIndex);

    // TODO(Calyx): Eventually, we should canonicalize the GroupDoneOp's guard
    // and source.
    auto guard = groupOp.getDoneOp().guard();
    auto source = groupOp.getDoneOp().src();
    auto doneOpValue =
        !guard ? source
               : builder.create<comb::AndOp>(wires->getLoc(), guard, source);

    // Build the Guard for the `go` signal of the current group being walked.
    // The group should begin when:
    // (1) the current step in the fsm is reached, and
    // (2) the done signal of this group is not high.
    auto eqCmp = builder.create<comb::ICmpOp>(
        wires->getLoc(), comb::ICmpPredicate::eq, fsmOut, fsmCurrentState);
    auto notDone =
        builder.create<comb::XorOp>(wires->getLoc(), doneOpValue, oneConstant);
    auto groupGoGuard =
        builder.create<comb::AndOp>(wires->getLoc(), eqCmp, notDone);

    // Guard for the `in` and `write_en` signal of the fsm register. These are
    // driven when the group has completed.
    builder.setInsertionPoint(seqGroup);
    auto groupDoneGuard =
        builder.create<comb::AndOp>(wires->getLoc(), eqCmp, doneOpValue);

    // Directly update the GroupGoOp of the current group being walked.
    auto goOp = groupOp.getGoOp();
    assert(goOp && "The Go Insertion pass should be run before this.");
    goOp->setOperands({oneConstant, groupGoGuard});

    // Add guarded assignments to the fsm register `in` and `write_en` ports.
    fsmNextState =
        createConstant(builder, wires->getLoc(), fsmBitWidth, fsmIndex + 1);
    builder.setInsertionPointToEnd(seqGroup.getBody());
    builder.create<AssignOp>(wires->getLoc(), fsmIn, fsmNextState,
                             groupDoneGuard);
    builder.create<AssignOp>(wires->getLoc(), fsmWriteEn, oneConstant,
                             groupDoneGuard);
    // Increment the fsm index for the next group.
    ++fsmIndex;
  });

  // Build the final guard for the new Seq group's GroupDoneOp. This is
  // defined by the fsm's final state.
  builder.setInsertionPoint(seqGroup);
  auto isFinalState = builder.create<comb::ICmpOp>(
      wires->getLoc(), comb::ICmpPredicate::eq, fsmOut, fsmNextState);

  // Insert the respective GroupDoneOp.
  builder.setInsertionPointToEnd(seqGroup.getBody());
  builder.create<GroupDoneOp>(seqGroup->getLoc(), oneConstant, isFinalState);

  // Add continuous wires to reset the `in` and `write_en` ports of the fsm
  // when the SeqGroup is finished executing.
  builder.setInsertionPointToEnd(wiresBody);
  auto zeroConstant = createConstant(builder, wires->getLoc(), fsmBitWidth, 0);
  builder.create<AssignOp>(wires->getLoc(), fsmIn, zeroConstant, isFinalState);
  builder.create<AssignOp>(wires->getLoc(), fsmWriteEn, oneConstant,
                           isFinalState);

  // Replace the SeqOp with an EnableOp.
  builder.setInsertionPoint(seq);
  builder.create<EnableOp>(
      seq->getLoc(), seqGroup.sym_name(),
      ArrayAttr::get(builder.getContext(), compiledGroups));

  seq->erase();
}

namespace {

struct CompileControlPass : public CompileControlBase<CompileControlPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void CompileControlPass::runOnOperation() {
  ComponentOp component = getOperation();
  CompileControlVisitor CompileControlVisitor;
  component.getControlOp().walk(
      [&](Operation *op) { CompileControlVisitor.dispatch(op, component); });

  // A post-condition of this pass is that all undefined GroupGoOps, created
  // in the Go Insertion pass, are now defined.
  component.getWiresOp().walk([&](UndefinedOp op) { op->erase(); });
}

std::unique_ptr<mlir::Pass> circt::calyx::createCompileControlPass() {
  return std::make_unique<CompileControlPass>();
}
