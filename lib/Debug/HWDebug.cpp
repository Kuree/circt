#define GET_OP_CLASSES

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Value.h"

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
};

struct HWDebugLineInfo : HWDebugScope {
  enum class LineType { None, Assign, Declare };

  std::string condition;

  LineType type;

  explicit HWDebugLineInfo(HWDebugContext &context)
      : HWDebugScope(context), type(LineType::None) {}
  HWDebugLineInfo(HWDebugContext &context, LineType type)
      : HWDebugScope(context), type(type) {}
};

struct HWVarDef {
  std::string name;
  std::string value;
  // for how it's always RTL value
  bool rtl = true;
};

struct HWModuleInfo : public HWDebugScope {
public:
  // module names
  std::string name;

  std::vector<HWVarDef> variables;
  std::map<std::string, std::string> instances;

  HWDebugFile *file = nullptr;

  explicit HWModuleInfo(HWDebugContext &context) : HWDebugScope(context) {}
};

struct HWDebugVarDeclareLineInfo : public HWDebugLineInfo {
  HWDebugVarDeclareLineInfo(HWDebugContext &context, ::mlir::Value value)
      : HWDebugLineInfo(context, LineType::Declare), value(value) {}

  ::mlir::Value value;
};

struct HWDebugVarAssignLineInfo : public HWDebugLineInfo {
  // This also encodes mapping information
  HWDebugVarAssignLineInfo(HWDebugContext &context, ::mlir::Value target)
      : HWDebugLineInfo(context, LineType::Assign), target(target) {}
  ::mlir::Value target;
};

struct HWDebugFile : HWDebugScope {
public:
  HWDebugFile(HWDebugContext &context, const std::string &filename)
      : HWDebugScope(context), filename(filename) {}

  void addModule(std::unique_ptr<HWModuleInfo> module,
                 circt::hw::HWModuleOp op) {
    auto const *opPtr = op.getOperation();
    scopeMappings.emplace(opPtr, std::move(module));
  }

  // NOLINTNEXTLINE
  HWDebugScope *getParentScope(::mlir::Operation *op) {
    if (!op)
      return nullptr;
    auto *parentOp = op->getParentOp();

    if (scopeMappings.find(parentOp) == scopeMappings.end()) {
      // need to create a scope entry for that, with type of None
      // then we need to create all the scopes up to the module
      auto scope = getNormalScope(parentOp);
      scopeMappings.emplace(parentOp, std::move(scope));
      (void)getParentScope(parentOp);
    }

    auto &ptr = scopeMappings.at(parentOp);

    return ptr.get();
  }

private:
  std::string filename;
  // scope mapping
  std::unordered_map<const ::mlir::Operation *, std::unique_ptr<HWDebugScope>>
      scopeMappings;

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

private:
  std::unordered_map<std::string, std::unique_ptr<HWDebugFile>> files;
};

void setEntryLocation(HWDebugScope &scope, const mlir::Location &location) {
  if (location.isa<::mlir::FileLineColLoc>()) {
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
}

class HWDebugBuilder {
public:
  HWDebugBuilder(HWDebugContext *context) : context(context) {}

  HWDebugFile *createFile(const std::string &filename) {
    return context->createFile(filename);
  }

  HWDebugVarDeclareLineInfo *createVarDeclaration(::mlir::Value value) {
    HWDebugVarDeclareLineInfo *result = nullptr;
    auto loc = value.getLoc();

    // need to get the containing module, as well as the line number
    // information
    auto info = std::make_unique<HWDebugVarDeclareLineInfo>(*context, value);
    setEntryLocation(*info, loc);
    auto *op = value.getDefiningOp();
    auto *scope = info->file->getParentScope(op);
    if (scope) {
      auto &ptr = scope->scopes.emplace_back(std::move(info));
      result = reinterpret_cast<HWDebugVarDeclareLineInfo *>(ptr.get());
    }

    return result;
  }

  HWModuleInfo *createModule(const circt::hw::HWModuleOp &op) {
    auto info = std::make_unique<HWModuleInfo>(*context);
    setEntryLocation(*info, op->getLoc());
    info->file->addModule(std::move(info), op);

    // need to create generator instances and variables
  }

private:
  HWDebugContext *context;
};

} // namespace circt::debug