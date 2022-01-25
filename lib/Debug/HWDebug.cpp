#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "circt/Dialect/HW/HW.h.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace circt::debug {

class HWDebugFile;

struct HWDebugScope {
public:
  HWDebugScope() = default;

  std::vector<std::unique_ptr<HWDebugScope>> scopes;

  HWDebugFile *file = nullptr;
  HWDebugScope *parent = nullptr;
};

struct HWDebugLineInfo : HWDebugScope {
  enum class LineType { None, Assign, Declare };

  uint32_t line;
  uint32_t column = 0;
  std::string condition;

  LineType type;

  HWDebugLineInfo() : type(LineType::None) {}
  HWDebugLineInfo(LineType type) : type(type) {}
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
};

struct HWDebugVarDeclareLineInfo : public HWDebugLineInfo {
  HWDebugVarDeclareLineInfo(::mlir::Value value)
      : HWDebugLineInfo(LineType::Declare), value(value) {}

  ::mlir::Value value;
};

struct HWDebugVarAssignLineInfo : public HWDebugLineInfo {
  // This also encodes mapping information
  HWDebugVarAssignLineInfo(::mlir::Value target) : target(target) {}
  ::mlir::Value target;
};

class HWDebugFile : HWDebugScope {
public:
  HWDebugFile(const std::string &filename) : filename(filename) {}

  void addModule(std::unique_ptr<HWModuleInfo> module,
                 circt::hw::HWModuleOp op) {
    moduleDefs.emplace_back(std::move(module));
  }

  HWDebugScope *getParentScope(::mlir::Operation *op) {
    (void)(op);
    return nullptr;
  }

private:
  std::string filename;
  std::vector<std::unique_ptr<HWModuleInfo>> moduleDefs;
  // scope mapping
  std::unordered_map<const ::mlir::Operation *, std::unique_ptr<HWDebugScope>>
      scopeMappings;
};

class HWDebugContext {
public:
  HWDebugFile *createFile(const std::string &filename) {
    if (files.find(filename) == files.end()) {
      files.emplace(filename, std::make_unique<HWDebugFile>(filename));
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

class HWDebugBuilder {
public:
  HWDebugBuilder(HWDebugContext *context) : context(context) {}

  HWDebugFile *createFile(const std::string &filename) {
    return context->createFile(filename);
  }

  HWDebugVarDeclareLineInfo *createVarDeclaration(::mlir::Value value) {
    HWDebugVarDeclareLineInfo *result = nullptr;
    auto loc = value.getLoc();
    if (loc.isa<::mlir::FileLineColLoc>()) {
      // need to get the containing module, as well as the line number
      // information
      auto const fileLoc = loc.cast<::mlir::FileLineColLoc>();
      auto const fileName = fileLoc.getFilename();
      auto const line = fileLoc.getLine();
      auto const column = fileLoc.getColumn();

      auto info = std::make_unique<HWDebugVarDeclareLineInfo>(value);
      info->file = context->createFile(fileName.str());
      info->line = line;
      info->column = column;
      auto *op = value.getDefiningOp();
      auto *scope = info->file->getParentScope(op);
      if (scope) {
        auto &ptr = scope->scopes.emplace_back(std::move(info));
        result = reinterpret_cast<HWDebugVarDeclareLineInfo *>(ptr.get());
      }
    }
    return result;
  }

private:
  HWDebugContext *context;
};

} // namespace circt::debug