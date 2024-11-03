#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class Mem2RegPromoter {
private:
  // Record all stores used by each alloca
  DenseMap<AllocaInst *, std::vector<StoreInst *>> allocaStores;
  // Record all loads used by each alloca
  DenseMap<AllocaInst *, std::vector<LoadInst *>> allocaLoads;
  // Record phi nodes for each basic block
  DenseMap<BasicBlock *, DenseMap<AllocaInst *, PHINode *>> blockPhis;

  // Check if an alloca is promotable
  bool isPromotable(AllocaInst *AI) {
    // Only process basic type alloca
    if (!AI->getAllocatedType()->isFirstClassType())
      return false;

    // Ensure all uses are loads or stores
    for (User *U : AI->users()) {
      if (!(isa<LoadInst>(U) || isa<StoreInst>(U)))
        return false;
    }
    return true;
  }

  // Collect all load and store instructions
  void collectLoadAndStores(AllocaInst *AI) {
    for (User *U : AI->users()) {
      if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
        allocaLoads[AI].push_back(LI);
      } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
        allocaStores[AI].push_back(SI);
      }
    }
  }

  // Insert phi nodes at necessary points
  void insertPhiNodes(AllocaInst *AI, Function &F, DominatorTree &DT) {
    std::set<BasicBlock *> defBlocks;

    // Collect all defining points (basic blocks containing store instructions)
    for (StoreInst *SI : allocaStores[AI]) {
      defBlocks.insert(SI->getParent());
    }

    // For each load, find its dominance frontier and insert phi nodes
    std::set<BasicBlock *> phiBlocks;
    for (LoadInst *LI : allocaLoads[AI]) {
      BasicBlock *loadBB = LI->getParent();

      // Simplified: insert phi nodes at all merge points
      for (BasicBlock &BB : F) {
        if (pred_size(&BB) >= 2) { // Basic block with multiple predecessors
          bool hasDefiningPred = false;
          bool hasNonDefiningPred = false;

          for (BasicBlock *Pred : predecessors(&BB)) {
            if (defBlocks.count(Pred))
              hasDefiningPred = true;
            else
              hasNonDefiningPred = true;
          }

          if (hasDefiningPred && hasNonDefiningPred) {
            phiBlocks.insert(&BB);
          }
        }
      }
    }

    // Create phi nodes
    for (BasicBlock *BB : phiBlocks) {
      PHINode *phi = PHINode::Create(AI->getAllocatedType(), pred_size(BB),
                                     AI->getName() + ".phi", &BB->front());
      blockPhis[BB][AI] = phi;
    }
  }

  // Rewrite all loads and stores
  void rewriteLoadAndStores(AllocaInst *AI) {
    // Create a mapping to track the value each load should use
    DenseMap<LoadInst *, Value *> loadToValue;

    // First, find the store value for each load
    for (LoadInst *LI : allocaLoads[AI]) {
      // Find the most recent store before this load
      Value *storedValue = nullptr;
      for (StoreInst *SI : allocaStores[AI]) {
        if (SI->getParent() == LI->getParent()) {
          // If in the same basic block, check if the store is before the load
          bool foundStore = false;
          for (BasicBlock::iterator it = BasicBlock::iterator(LI);
               it != SI->getParent()->begin(); --it) {
            if (&*it == SI) {
              storedValue = SI->getValueOperand();
              foundStore = true;
              break;
            }
          }
          // Check the first instruction separately
          if (!foundStore && SI == &SI->getParent()->front()) {
            storedValue = SI->getValueOperand();
          }
        }
      }
      if (storedValue) {
        loadToValue[LI] = storedValue;
      }
    }

    // Replace all loads and delete them
    for (auto &pair : loadToValue) {
      LoadInst *LI = pair.first;
      Value *newValue = pair.second;
      LI->replaceAllUsesWith(newValue);
      LI->eraseFromParent();
    }

    // Delete all stores
    for (StoreInst *SI : allocaStores[AI]) {
      SI->eraseFromParent();
    }
  }

public:
  void promote(Function &F) {
    DominatorTree DT(F);

    // First, collect all promotable allocas before making any modifications
    std::vector<AllocaInst *> promotableAllocas;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
          if (isPromotable(AI)) {
            collectLoadAndStores(AI); // Collect info before any modifications
            promotableAllocas.push_back(AI);
          }
        }
      }
    }

    // Then process each alloca
    for (AllocaInst *AI : promotableAllocas) {
      if (!AI->getType()->isFirstClassType())
        continue; // Extra safety check

      insertPhiNodes(AI, F, DT);
      rewriteLoadAndStores(AI);

      // Only erase the alloca after all its users have been processed
      if (AI->use_empty()) {
        AI->eraseFromParent();
      }
    }

    outs() << "(mem2reg) Promoted " << promotableAllocas.size()
           << " allocas to registers in function: " << F.getName() << "\n";
  }
};

// main visitor function
void visitor(Function &F) {
  Mem2RegPromoter promoter;
  promoter.promote(F);
}

// new PM implementation
struct Mem2Reg : PassInfoMixin<Mem2Reg> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    visitor(F);
    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getMem2RegPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Mem2Reg1", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "mem2reg1") {
                    FPM.addPass(Mem2Reg());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getMem2RegPluginInfo();
}
