//===- InstanceGraph.cpp - Instance Graph -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace firrtl;

InstanceRecord *InstanceGraphNode::recordInstance(InstanceOp instance,
                                                  InstanceGraphNode *target) {
  moduleInstances.emplace_back(instance, this, target);
  return &moduleInstances.back();
}

void InstanceGraphNode::recordUse(InstanceRecord *record) {
  moduleUses.push_back(record);
}

InstanceGraph::InstanceGraph(Operation *operation) {
  if (auto mod = dyn_cast<mlir::ModuleOp>(operation))
    for (auto &op : *mod.getBody())
      if ((operation = dyn_cast<CircuitOp>(&op)))
        break;

  auto circuitOp = cast<CircuitOp>(operation);

  // We insert the top level module first in to the node map.  Getting the node
  // here is enough to ensure that it is the first one added.
  getOrAddNode(circuitOp.name());

  for (auto &op : *circuitOp.getBody()) {
    if (auto extModule = dyn_cast<FExtModuleOp>(op)) {
      auto *currentNode = getOrAddNode(extModule.getName());
      currentNode->module = extModule;
    }
    if (auto module = dyn_cast<FModuleOp>(op)) {
      auto *currentNode = getOrAddNode(module.getName());
      currentNode->module = module;
      // Find all instance operations in the module body.
      module.body().walk([&](InstanceOp instanceOp) {
        // Add an edge to indicate that this module instantiates the target.
        auto *targetNode = getOrAddNode(instanceOp.moduleName());
        auto *instanceRecord =
            currentNode->recordInstance(instanceOp, targetNode);
        targetNode->recordUse(instanceRecord);
      });
    }
  }
}

InstanceGraphNode *InstanceGraph::getTopLevelNode() {
  // The graph always puts the top level module in the array first.
  if (!nodes.size())
    return nullptr;
  return &nodes[0];
}

FModuleLike InstanceGraph::getTopLevelModule() {
  return getTopLevelNode()->getModule();
}

InstanceGraphNode *InstanceGraph::lookup(StringRef name) {
  auto it = nodeMap.find(name);
  assert(it != nodeMap.end() && "Module not in InstanceGraph!");
  return &nodes[it->second];
}

InstanceGraphNode *InstanceGraph::lookup(Operation *op) {
  if (auto extModule = dyn_cast<FExtModuleOp>(op)) {
    return lookup(extModule.getName());
  }
  if (auto module = dyn_cast<FModuleOp>(op)) {
    return lookup(module.getName());
  }
  llvm_unreachable("Can only look up module operations.");
}

InstanceGraphNode *InstanceGraph::getOrAddNode(StringRef name) {
  // Try to insert an InstanceGraphNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto itAndInserted = nodeMap.try_emplace(name, 0);
  auto &index = itAndInserted.first->second;
  if (itAndInserted.second) {
    // This is a new node, we have to add an element to the NodeVec.
    nodes.emplace_back();
    // Store the node storage index in to the map.
    index = nodes.size() - 1;
    return &nodes.back();
  }
  return &nodes[index];
}

Operation *InstanceGraph::getReferencedModule(InstanceOp op) {
  return lookup(op.moduleName())->getModule();
}

void InstanceGraph::replaceInstance(InstanceOp inst, InstanceOp newInst) {
  assert(inst.moduleName() == newInst.moduleName() &&
         "Both instances must be targeting the same module");

  // Find the instance record of this instance.
  auto *node = lookup(inst.moduleName());
  auto it = llvm::find_if(node->uses(), [&](InstanceRecord *record) {
    return record->getInstance() == inst;
  });
  assert(it != node->uses_end() && "Instance of module not recorded in graph");

  // We can just replace the instance op in the InstanceRecord without updating
  // any instance lists.
  (*it)->instance = newInst;
}

bool InstanceGraph::isAncestor(FModuleLike child, FModuleOp parent) {
  DenseSet<InstanceGraphNode *> seen;
  SmallVector<InstanceGraphNode *> worklist;
  auto *cn = lookup(child);
  worklist.push_back(cn);
  seen.insert(cn);
  while (!worklist.empty()) {
    auto *node = worklist.back();
    worklist.pop_back();
    if (node->getModule() == parent)
      return true;
    for (auto *use : node->uses()) {
      auto *mod = use->getParent();
      if (!seen.count(mod)) {
        seen.insert(mod);
        worklist.push_back(mod);
      }
    }
  }
  return false;
}

ArrayRef<InstancePath> InstancePathCache::getAbsolutePaths(Operation *op) {
  assert((isa<FModuleOp, FExtModuleOp>(op))); // extra parens makes parser smile

  // If we have reached the circuit root, we're done.
  if (op == instanceGraph.getTopLevelNode()->getModule()) {
    static InstancePath empty{};
    return empty; // array with single empty path
  }

  // Fast path: hit the cache.
  auto cached = absolutePathsCache.find(op);
  if (cached != absolutePathsCache.end())
    return cached->second;

  // For each instance, collect the instance paths to its parent and append the
  // instance itself to each.
  SmallVector<InstancePath, 8> extendedPaths;
  for (auto inst : instanceGraph[op]->uses()) {
    auto instPaths = getAbsolutePaths(inst->getParent()->getModule());
    extendedPaths.reserve(instPaths.size());
    for (auto path : instPaths) {
      extendedPaths.push_back(appendInstance(path, inst->getInstance()));
    }
  }

  // Move the list of paths into the bump allocator for later quick retrieval.
  ArrayRef<InstancePath> pathList;
  if (!extendedPaths.empty()) {
    auto paths = allocator.Allocate<InstancePath>(extendedPaths.size());
    std::copy(extendedPaths.begin(), extendedPaths.end(), paths);
    pathList = ArrayRef<InstancePath>(paths, extendedPaths.size());
  }

  absolutePathsCache.insert({op, pathList});
  return pathList;
}

InstancePath InstancePathCache::appendInstance(InstancePath path,
                                               InstanceOp inst) {
  size_t n = path.size() + 1;
  auto newPath = allocator.Allocate<InstanceOp>(n);
  std::copy(path.begin(), path.end(), newPath);
  newPath[path.size()] = inst;
  return InstancePath(newPath, n);
}
