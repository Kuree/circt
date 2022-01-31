#define GET_OP_CLASSES

#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"

mlir::StringRef getSymOpName(mlir::Operation *symOp);

namespace circt::hw {
mlir::StringAttr getVerilogModuleNameAttr(mlir::Operation *module);
} // namespace circt::hw

namespace circt::debug {

struct HWDebugFile;
struct HWDebugScope;
class HWDebugContext;

void setEntryLocation(HWDebugScope &scope, const mlir::Location &location,
                      mlir::Operation *op = nullptr);

enum class HWDebugScopeType { None, Assign, Declare, Module, Block };

std::string toString(HWDebugScopeType type) {
  switch (type) {
  case HWDebugScopeType::None:
    return "none";
  case HWDebugScopeType::Assign:
    return "assign";
  case HWDebugScopeType::Declare:
    return "decl";
  case HWDebugScopeType::Module:
    return "module";
  case HWDebugScopeType::Block:
    return "block";
  default:
    llvm_unreachable("unknown type");
  }
}

struct HWDebugScope {
public:
  explicit HWDebugScope(HWDebugContext &context, mlir::Operation *op)
      : context(context), op(op) {}

  std::vector<HWDebugScope *> scopes;

  uint32_t line = 0;
  uint32_t column = 0;

  HWDebugScope *parent = nullptr;
  HWDebugFile *file = nullptr;

  HWDebugContext &context;

  mlir::Operation *op;

  // NOLINTNEXTLINE
  [[nodiscard]] virtual llvm::json::Value toJSON() const {
    auto res = getScopeJSON(true);
    return res;
  }

  [[nodiscard]] virtual HWDebugScopeType type() const {
    return scopes.empty() ? HWDebugScopeType::None : HWDebugScopeType::Block;
  }

protected:
  // NOLINTNEXTLINE
  [[nodiscard]] llvm::json::Object getScopeJSON(bool includeScope) const {
    llvm::json::Object res{{"line", line}};
    if (column > 0) {
      res["column"] = column;
    }
    res["type"] = toString(type());
    if (includeScope) {
      setScope(res);
    }
    return res;
  }

  // NOLINTNEXTLINE
  void setScope(llvm::json::Object &obj) const {
    llvm::json::Array array;
    array.reserve(scopes.size());
    for (auto const *scope : scopes) {
      if (scope)
        array.emplace_back(std::move(scope->toJSON()));
    }
    obj["scopes"] = std::move(array);
  }
};

struct HWDebugLineInfo : HWDebugScope {
  enum class LineType {
    None = static_cast<int>(HWDebugScopeType::None),
    Assign = static_cast<int>(HWDebugScopeType::Assign),
    Declare = static_cast<int>(HWDebugScopeType::Declare),
  };

  std::string condition;

  LineType lineType;

  HWDebugLineInfo(HWDebugContext &context, LineType type, mlir::Operation *op)
      : HWDebugScope(context, op), lineType(type) {}

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = getScopeJSON(false);
    if (!condition.empty()) {
      res["condition"] = condition;
    }
    return res;
  }

  [[nodiscard]] HWDebugScopeType type() const override {
    return static_cast<HWDebugScopeType>(lineType);
  }
};

struct HWDebugVarDef {
  std::string name;
  std::string value;
  // for how it's always RTL value
  bool rtl = true;

  [[nodiscard]] llvm::json::Value toJSON() const {
    return llvm::json::Object({{"name", name}, {"value", value}, {"rtl", rtl}});
  }
};

struct HWModuleInfo : public HWDebugScope {
public:
  // module names
  std::string name;

  std::vector<HWDebugVarDef> variables;
  std::map<std::string, std::string> instances;

  explicit HWModuleInfo(HWDebugContext &context, mlir::Operation *moduleOp)
      : HWDebugScope(context, moduleOp) {}

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = getScopeJSON(true);
    res["name"] = name;

    llvm::json::Array vars;
    vars.reserve(variables.size());
    for (auto const &varDef : variables) {
      vars.emplace_back(varDef.toJSON());
    }
    res["variables"] = std::move(vars);

    if (!instances.empty()) {
      llvm::json::Array insts;
      insts.reserve(instances.size());
      for (auto const &[n, def] : instances) {
        insts.emplace_back(llvm::json::Object{{"name", n}, {"module", def}});
      }
      res["instances"] = std::move(insts);
    }

    return res;
  }

  [[nodiscard]] HWDebugScopeType type() const override {
    return HWDebugScopeType::Module;
  }
};

struct HWDebugVarDeclareLineInfo : public HWDebugLineInfo {
  HWDebugVarDeclareLineInfo(HWDebugContext &context, mlir::Operation *op)
      : HWDebugLineInfo(context, LineType::Declare, op) {}

  HWDebugVarDef variable;

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = HWDebugLineInfo::toJSON();
    (*res.getAsObject())["variable"] = std::move(variable.toJSON());
    return res;
  }
};

struct HWDebugVarAssignLineInfo : public HWDebugLineInfo {
  // This also encodes mapping information
  HWDebugVarAssignLineInfo(HWDebugContext &context, mlir::Operation *op)
      : HWDebugLineInfo(context, LineType::Assign, op) {}

  HWDebugVarDef variable;

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = HWDebugLineInfo::toJSON();
    (*res.getAsObject())["variable"] = std::move(variable.toJSON());
    return res;
  }
};

struct HWDebugFile : HWDebugScope {
public:
  HWDebugFile(HWDebugContext &context, const std::string &filename)
      : HWDebugScope(context, nullptr), filename(filename) {}

  HWModuleInfo *addModule(HWModuleInfo *module, circt::hw::HWModuleOp op) {
    auto const *opPtr = op.getOperation();
    scopes.emplace_back(module);
    scopeMappings.emplace(opPtr, module);
    return module;
  }

  // NOLINTNEXTLINE
  HWDebugScope *getParentScope(::mlir::Operation *op) {
    if (!op)
      return nullptr;
    auto *parentOp = op->getParentOp();
    if (!parentOp) {
      return nullptr;
    }

    if (scopeMappings.find(parentOp) == scopeMappings.end()) {
      // need to create a scope entry for that, with type of None
      // then we need to create all the scopes up to the module
      auto *scope = getNormalScope(parentOp);
      scopes.emplace_back(scope);
      scopeMappings.emplace(parentOp, scope);
      (void)getParentScope(parentOp);
    }

    auto *ptr = scopeMappings.at(parentOp);

    return ptr;
  }

  [[nodiscard]] llvm::json::Value toJSON() const override {
    llvm::json::Object res{{"type", "file"}, {"filename", filename}};
    setScope(res);
    return res;
  }

private:
  std::string filename;
  // scope mapping
  std::unordered_map<const ::mlir::Operation *, HWDebugScope *> scopeMappings;

  HWDebugScope *getNormalScope(::mlir::Operation *op);
};

class HWDebugContext {
public:
  HWDebugFile *createFile(const std::string &filename) {
    if (files.find(filename) == files.end()) {
      files.emplace(filename, std::make_unique<HWDebugFile>(*this, filename));
    }
    return files.at(filename).get();
  }

  HWDebugFile *getFile(const std::string &filename) {
    if (files.find(filename) != files.end()) {
      return files.at(filename).get();
    }
    return nullptr;
  }

  [[nodiscard]] llvm::json::Value toJSON() const {
    llvm::json::Object res{{"generator", "circt"}};
    llvm::json::Array array;
    array.reserve(files.size());
    for (auto const &[_, file] : files) {
      array.emplace_back(std::move(file->toJSON()));
    }
    res["table"] = std::move(array);
    return res;
  }

  template <typename T, typename... Args>
  T *createScope(Args &&...args) {
    auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
    return reinterpret_cast<T *>(scopes.emplace_back(std::move(ptr)).get());
  }

private:
  std::unordered_map<std::string, std::unique_ptr<HWDebugFile>> files;
  std::vector<std::unique_ptr<HWDebugScope>> scopes;
};

HWDebugScope *HWDebugFile::getNormalScope(::mlir::Operation *op) {
  auto *ptr = context.createScope<HWDebugScope>(context, op);
  setEntryLocation(*ptr, op->getLoc());
  return ptr;
}

void setEntryLocation(HWDebugScope &scope, const mlir::Location &location,
                      mlir::Operation *op) {
  // need to get the containing module, as well as the line number
  // information
  auto const fileLoc = location.cast<::mlir::FileLineColLoc>();
  auto filename = fileLoc.getFilename();
  auto const line = fileLoc.getLine();
  auto const column = fileLoc.getColumn();

  if (op) {
    if (auto debugFilename = op->getAttr("hw.debug.filename")) {
      filename = debugFilename.cast<mlir::StringAttr>();
    }
  }

  scope.file = scope.context.createFile(filename.str());
  scope.line = line;
  scope.column = column;
}

class HWDebugBuilder {
public:
  HWDebugBuilder(HWDebugContext &context) : context(context) {}

  HWDebugFile *createFile(const std::string &filename) {
    return context.createFile(filename);
  }

  HWDebugVarDeclareLineInfo *createVarDeclaration(::mlir::Value value) {
    auto loc = value.getLoc();
    auto *op = value.getDefiningOp();
    auto *targetOp = getDebugOp(value, op);
    if (!targetOp)
      return nullptr;

    // need to get the containing module, as well as the line number
    // information
    auto *info = context.createScope<HWDebugVarDeclareLineInfo>(context, op);
    setEntryLocation(*info, loc);
    info->variable = createVarDef(targetOp);
    // add to scope
    auto *result = addToScope(info, op);
    return result;
  }

  HWDebugVarAssignLineInfo *createAssign(::mlir::Value value,
                                         ::mlir::Operation *op) {
    // only create assign if the target has frontend variable
    auto *targetOp = getDebugOp(value, op);
    if (!targetOp)
      return nullptr;

    auto loc = op->getLoc();

    auto *assign = context.createScope<HWDebugVarAssignLineInfo>(context, op);
    setEntryLocation(*assign, loc);

    assign->variable = createVarDef(targetOp);

    // add to scope
    auto *result = addToScope(assign, op);
    return result;
  }

  HWDebugVarDef createVarDef(::mlir::Operation *op) {
    // the OP has to have this attr. need to check before calling this function
    auto frontEndName =
        op->getAttr("hw.debug.name").cast<mlir::StringAttr>().str();
    auto rtlName = ::getSymOpName(op).str();
    HWDebugVarDef var{.name = frontEndName, .value = rtlName, .rtl = true};
    return var;
  }

  HWModuleInfo *createModule(circt::hw::HWModuleOp *op) {
    auto *info = context.createScope<HWModuleInfo>(context, op->getOperation());
    setEntryLocation(*info, op->getLoc(), op->getOperation());
    if (info->file) {
      return info->file->addModule(info, *op);
    }
    return nullptr;
  }

private:
  HWDebugContext &context;

  template <typename T>
  T *addToScope(T *info, ::mlir::Operation *op) {
    auto *scope = info->file->getParentScope(op);
    if (scope) {
      scope->scopes.emplace_back(info);
      return info;
    }
    return nullptr;
  }

  static mlir::Operation *getDebugOp(mlir::Value value, mlir::Operation *op) {
    auto *valueOP = value.getDefiningOp();
    if (valueOP && valueOP->hasAttr("hw.debug.name")) {
      return valueOP;
    }
    if (op && op->hasAttr("hw.debug.name")) {
      return op;
    }
    return nullptr;
  }
};

class DebugStmtVisitor : public circt::hw::StmtVisitor<DebugStmtVisitor>,
                         public circt::sv::Visitor<DebugStmtVisitor, void> {
public:
  DebugStmtVisitor(HWDebugBuilder &builder, HWModuleInfo *module)
      : builder(builder), module(module) {}

  void visitStmt(circt::hw::InstanceOp op) {
    auto instNameRef = ::getSymOpName(op);
    auto instNameStr = std::string(instNameRef.begin(), instNameRef.end());
    // need to find definition names
    auto *mod = op.getReferencedModule(nullptr);
    auto moduleNameStr = circt::hw::getVerilogModuleNameAttr(mod).str();
    module->instances.emplace(instNameStr, moduleNameStr);
  }

  void visitSV(circt::sv::RegOp op) {
    // we treat this as a generator variable
    // only generate if we have annotated in the frontend
    if (hasDebug(op)) {
      auto var = builder.createVarDef(op);
      module->variables.emplace_back(var);
    }
  }

  void visitSV(circt::sv::WireOp op) {
    if (hasDebug(op)) {
      auto var = builder.createVarDef(op);
      module->variables.emplace_back(var);
    }
  }

  // assignment
  // we only care about the target of the assignment
  void visitSV(circt::sv::AssignOp op) {
    if (!hasDebug(op))
      return;
    auto target = op.dest();
    builder.createAssign(target, op);
  }

  void visitSV(circt::sv::BPAssignOp op) {
    if (!hasDebug(op))
      return;
    auto target = op.dest();
    builder.createAssign(target, op);
  }

  void visitSV(circt::sv::PAssignOp op) {
    if (!hasDebug(op))
      return;
    auto target = op.dest();
    builder.createAssign(target, op);
  }

  void visitStmt(circt::hw::OutputOp op) {
    hw::HWModuleOp parent = op->getParentOfType<hw::HWModuleOp>();
    for (auto i = 0u; i < parent.getPorts().outputs.size(); i++) {
      auto operand = op.getOperand(i);
      builder.createAssign(operand, op);
    }
  }

  // visit blocks
  void visitSV(circt::sv::AlwaysOp op) { visitBlock(*op.getBodyBlock()); }

  void visitSV(circt::sv::AlwaysCombOp op) { visitBlock(*op.getBodyBlock()); }

  void visitSV(circt::sv::AlwaysFFOp op) {
    if (op.getResetBlock()) {
      visitBlock(*op.getResetBlock());
    }
    visitBlock(*op.getBodyBlock());
  }

  void visitSV(circt::sv::InitialOp op) { visitBlock(*op.getBodyBlock()); }

  // noop HW visit functions
  void visitStmt(circt::hw::ProbeOp) {}
  void visitStmt(circt::hw::TypedeclOp) {}

  void visitStmt(circt::hw::TypeScopeOp op) { visitBlock(*op.getBodyBlock()); }

  // noop SV visit functions
  void visitSV(circt::sv::ReadInOutOp) {}
  void visitSV(circt::sv::ArrayIndexInOutOp) {}
  void visitSV(circt::sv::VerbatimExprOp) {}
  void visitSV(circt::sv::VerbatimExprSEOp) {}
  void visitSV(circt::sv::IndexedPartSelectInOutOp) {}
  void visitSV(circt::sv::IndexedPartSelectOp) {}
  void visitSV(circt::sv::StructFieldInOutOp) {}
  void visitSV(circt::sv::ConstantXOp) {}
  void visitSV(circt::sv::ConstantZOp) {}
  void visitSV(circt::sv::LocalParamOp) {}
  void visitSV(circt::sv::XMROp) {}
  void visitSV(circt::sv::IfDefOp) {}
  void visitSV(circt::sv::IfDefProceduralOp) {}
  void visitSV(circt::sv::CaseZOp) {}
  void visitSV(circt::sv::ForceOp) {}
  void visitSV(circt::sv::ReleaseOp) {}
  void visitSV(circt::sv::AliasOp) {}
  void visitSV(circt::sv::FWriteOp) {}
  void visitSV(circt::sv::VerbatimOp) {}
  void visitSV(circt::sv::InterfaceOp) {}
  void visitSV(circt::sv::InterfaceSignalOp) {}
  void visitSV(circt::sv::InterfaceModportOp) {}
  void visitSV(circt::sv::InterfaceInstanceOp) {}
  void visitSV(circt::sv::GetModportOp) {}
  void visitSV(circt::sv::AssignInterfaceSignalOp) {}
  void visitSV(circt::sv::ReadInterfaceSignalOp) {}
  void visitSV(circt::sv::AssertOp) {}
  void visitSV(circt::sv::AssumeOp) {}
  void visitSV(circt::sv::CoverOp) {}
  void visitSV(circt::sv::AssertConcurrentOp) {}
  void visitSV(circt::sv::AssumeConcurrentOp) {}
  void visitSV(circt::sv::CoverConcurrentOp) {}
  void visitSV(circt::sv::BindOp) {}
  void visitSV(circt::sv::StopOp) {}
  void visitSV(circt::sv::FinishOp) {}
  void visitSV(circt::sv::ExitOp) {}
  void visitSV(circt::sv::FatalOp) {}
  void visitSV(circt::sv::ErrorOp) {}
  void visitSV(circt::sv::WarningOp) {}
  void visitSV(circt::sv::InfoOp) {}
  void visitSV(circt::sv::IfOp) {}

  // ignore invalid stuff
  void visitInvalidStmt(Operation *) {}
  void visitInvalidSV(Operation *) {}

  void dispatch(mlir::Operation *op) {
    dispatchStmtVisitor(op);
    dispatchSVVisitor(op);
  }

  void visitBlock(mlir::Block &block) {
    for (auto &op : block) {
      dispatch(&op);
    }
  }

private:
  HWDebugBuilder &builder;
  HWModuleInfo *module;

  bool hasDebug(mlir::Operation *op) {
    auto r = op && op->hasAttr("hw.debug.name");
    return r;
  }
};

void fixModuleFilename(circt::hw::HWModuleOp op) {
  if (op->hasAttr("hw.debug.filename")) return;
  auto getFilename = [](mlir::Operation *op) {
    return op->getLoc().cast<::mlir::FileLineColLoc>().getFilename();
  };
  auto filename = getFilename(op);
  // if the underlying name is different, add a name attribute to override the
  // filename
  for (auto &entry : *op.getBodyBlock()) {
    auto entryFilename = getFilename(&entry);
    if (entryFilename.str() != filename.str()) {
      // override the module filename
      // this happens in Firrtl, which doesn't encode location for module
      printf("fix it to :%s\n", entryFilename.str().c_str());
      op->setAttr("hw.debug.filename", entryFilename);
      // done
      return;
    }
  }
}

void exportDebugTable(mlir::ModuleOp moduleOp, const std::string &filename) {
  // collect all the files
  HWDebugContext context;
  HWDebugBuilder builder(context);
  for (auto &op : *moduleOp.getBody()) {
    mlir::TypeSwitch<mlir::Operation *>(&op).Case<circt::hw::HWModuleOp>(
        [&builder](circt::hw::HWModuleOp mod) {
          // fix the filename using heuristics
          fixModuleFilename(mod);
          // get verilog name
          auto defName = circt::hw::getVerilogModuleNameAttr(mod).str();
          if (defName.empty())
            return;
          auto *module = builder.createModule(&mod);
          module->name = defName;
          DebugStmtVisitor visitor(builder, module);
          auto *body = mod.getBodyBlock();
          visitor.visitBlock(*body);
        });
  }
  auto json = context.toJSON();
  std::error_code error;
  llvm::raw_fd_ostream os(filename, error);
  if (!error) {
    os << json;
  }
  os.close();
}

struct ExportDebugTablePass : public ::mlir::OperationPass<mlir::ModuleOp> {
  ExportDebugTablePass(std::string filename)
      : ::mlir::OperationPass<mlir::ModuleOp>(
            ::mlir::TypeID::get<ExportDebugTablePass>()),
        filename(filename) {}
  ExportDebugTablePass(const ExportDebugTablePass &other)
      : ::mlir::OperationPass<mlir::ModuleOp>(other), filename(other.filename) {
  }

  void runOnOperation() override {
    exportDebugTable(getOperation(), filename);
    markAllAnalysesPreserved();
  }

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("export-hgdb");
  }
  ::llvm::StringRef getArgument() const override { return "export-hgdb"; }

  ::llvm::StringRef getDescription() const override {
    return "Generate symbol table for HGDB";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ExportHGDB");
  }
  ::llvm::StringRef getName() const override { return "ExportHGDB"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<ExportDebugTablePass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<ExportDebugTablePass>(
        *static_cast<const ExportDebugTablePass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {

    registry.insert<circt::sv::SVDialect>();

    registry.insert<circt::comb::CombDialect>();

    registry.insert<circt::hw::HWDialect>();
  }

private:
  std::string filename;
};

std::unique_ptr<mlir::Pass> createExportHGDBPass(std::string filename) {
  return std::make_unique<ExportDebugTablePass>(std::move(filename));
}

} // namespace circt::debug