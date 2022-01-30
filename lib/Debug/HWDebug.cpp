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

void setEntryLocation(HWDebugScope &scope, const mlir::Location &location);

struct HWDebugScope {
public:
  explicit HWDebugScope(HWDebugContext &context) : context(context) {}

  std::vector<std::unique_ptr<HWDebugScope>> scopes;

  uint32_t line = 0;
  uint32_t column = 0;

  HWDebugScope *parent = nullptr;
  HWDebugFile *file = nullptr;

  HWDebugContext &context;

  // NOLINTNEXTLINE
  [[nodiscard]] virtual llvm::json::Value toJSON() const {
    // by default, it's block scope
    auto res = getScopeJSON(true);
    res["type"] = "block";
    return res;
  }

protected:
  // NOLINTNEXTLINE
  [[nodiscard]] llvm::json::Object getScopeJSON(bool includeScope) const {
    llvm::json::Object res{{"line", line}};
    if (column > 0) {
      res["column"] = column;
    }
    if (includeScope) {
      setScope(res);
    }
    return res;
  }

  void setScope(llvm::json::Object &obj) const {
    llvm::json::Array array;
    array.reserve(scopes.size());
    for (auto const &scope : scopes) {
      array.emplace_back(std::move(scope->toJSON()));
    }
    obj["scopes"] = std::move(array);
  }
};

struct HWDebugLineInfo : HWDebugScope {
  enum class LineType { None, Assign, Declare };

  std::string condition;

  LineType type;

  explicit HWDebugLineInfo(HWDebugContext &context)
      : HWDebugScope(context), type(LineType::None) {}
  HWDebugLineInfo(HWDebugContext &context, LineType type)
      : HWDebugScope(context), type(type) {}

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = getScopeJSON(false);
    std::string typeStr;
    switch (type) {
    case LineType::None:
      typeStr = "none";
      break;
    case LineType::Assign:
      typeStr = "assign";
      break;
    case LineType::Declare:
      typeStr = "decl";
      break;
    default:
      llvm_unreachable("unknown line type");
    }
    res["type"] = typeStr;

    if (!condition.empty()) {
      res["condition"] = condition;
    }

    return res;
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

  explicit HWModuleInfo(HWDebugContext &context) : HWDebugScope(context) {}

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
};

struct HWDebugVarDeclareLineInfo : public HWDebugLineInfo {
  HWDebugVarDeclareLineInfo(HWDebugContext &context)
      : HWDebugLineInfo(context, LineType::Declare) {}

  HWDebugVarDef variable;

  [[nodiscard]] llvm::json::Value toJSON() const override {
    auto res = HWDebugLineInfo::toJSON();
    (*res.getAsObject())["variable"] = std::move(variable.toJSON());
    return res;
  }
};

struct HWDebugVarAssignLineInfo : public HWDebugLineInfo {
  // This also encodes mapping information
  HWDebugVarAssignLineInfo(HWDebugContext &context)
      : HWDebugLineInfo(context, LineType::Assign) {}

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
      : HWDebugScope(context), filename(filename) {}

  HWModuleInfo *addModule(std::unique_ptr<HWModuleInfo> module,
                          circt::hw::HWModuleOp op) {
    auto const *opPtr = op.getOperation();
    auto &entry = scopes.emplace_back(std::move(module));
    auto *scope = entry.get();
    scopeMappings.emplace(opPtr, scope);
    return reinterpret_cast<HWModuleInfo *>(scope);
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
      auto scope = getNormalScope(parentOp);
      auto *scopePtr = scopes.emplace_back(std::move(scope)).get();
      scopeMappings.emplace(parentOp, scopePtr);
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

  std::unique_ptr<HWDebugScope> getNormalScope(::mlir::Operation *op) {
    auto ptr = std::make_unique<HWDebugScope>(context);
    setEntryLocation(*ptr, op->getLoc());
    return std::move(ptr);
  }
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

private:
  std::unordered_map<std::string, std::unique_ptr<HWDebugFile>> files;
};

void setEntryLocation(HWDebugScope &scope, const mlir::Location &location) {
  // need to get the containing module, as well as the line number
  // information
  auto const fileLoc = location.cast<::mlir::FileLineColLoc>();
  auto const filename = fileLoc.getFilename();
  auto const line = fileLoc.getLine();
  auto const column = fileLoc.getColumn();

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
    auto *targetOp = getDebugOp(value, value.getDefiningOp());
    if (!targetOp)
      return nullptr;

    // need to get the containing module, as well as the line number
    // information
    auto info = std::make_unique<HWDebugVarDeclareLineInfo>(context);
    setEntryLocation(*info, loc);
    auto *op = value.getDefiningOp();
    info->variable = createVarDef(targetOp);
    // add to scope
    auto *result = addToScope(std::move(info), op);
    return result;
  }

  HWDebugVarAssignLineInfo *createAssign(::mlir::Value value,
                                         ::mlir::Operation *op) {
    // only create assign if the target has frontend variable
    auto *targetOp = getDebugOp(value, op);
    if (!targetOp)
      return nullptr;

    auto loc = op->getLoc();

    auto assign = std::make_unique<HWDebugVarAssignLineInfo>(context);
    setEntryLocation(*assign, loc);

    assign->variable = createVarDef(targetOp);

    // add to scope
    auto *result = addToScope(std::move(assign), op);
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

  HWModuleInfo *createModule(const circt::hw::HWModuleOp &op) {
    auto info = std::make_unique<HWModuleInfo>(context);
    setEntryLocation(*info, op->getLoc());
    if (info->file) {
      auto *ptr = info.get();
      return ptr->file->addModule(std::move(info), op);
    }
    return nullptr;
  }

private:
  HWDebugContext &context;

  template <typename T>
  T *addToScope(std::unique_ptr<T> info, ::mlir::Operation *op) {
    auto *scope = info->file->getParentScope(op);
    if (scope) {
      auto &ptr = scope->scopes.emplace_back(std::move(info));
      return reinterpret_cast<T *>(ptr.get());
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

void exportDebugTable(mlir::ModuleOp moduleOp, const std::string &filename) {
  // collect all the files
  HWDebugContext context;
  HWDebugBuilder builder(context);
  for (auto &op : *moduleOp.getBody()) {
    mlir::TypeSwitch<mlir::Operation *>(&op).Case<circt::hw::HWModuleOp>(
        [&builder](circt::hw::HWModuleOp mod) {
          auto *module = builder.createModule(mod);
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