#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {

class LoopUnroller {
private:
  const unsigned UnrollFactor = 4; // Fixed unroll factor

  bool isSimpleCountingLoop(Loop *L) {
    // Only handle loops with single exit
    if (!L->getExitingBlock() || !L->getLoopLatch())
      return false;

    // Only handle loops with simple induction variable
    PHINode *IndVar = L->getCanonicalInductionVariable();
    if (!IndVar)
      return false;

    return true;
  }

  void unrollLoop(Loop *L, LoopInfo &LI) {
    BasicBlock *Header = L->getHeader();
    BasicBlock *Latch = L->getLoopLatch();
    BasicBlock *Exit = L->getExitBlock();

    if (!Header || !Latch || !Exit)
      return;

    // Clone the loop body UnrollFactor-1 times
    SmallVector<BasicBlock *, 8> NewBlocks;

    for (unsigned i = 1; i < UnrollFactor; ++i) {
      SmallVector<BasicBlock *, 8> LoopBlocks;
      for (BasicBlock *BB : L->blocks())
        LoopBlocks.push_back(BB);

      // Clone all blocks
      ValueToValueMapTy VMap;
      for (BasicBlock *BB : LoopBlocks) {
        BasicBlock *ClonedBB = CloneBasicBlock(BB, VMap, ".unroll." + Twine(i),
                                             BB->getParent());
        VMap[BB] = ClonedBB;
        NewBlocks.push_back(ClonedBB);
      }

      // Update the PHI nodes
      for (BasicBlock *BB : NewBlocks) {
        for (Instruction &I : *BB) {
          RemapInstruction(&I, VMap, RF_NoModuleLevelChanges);
        }
      }
    }

    // Update branch instructions
    BranchInst *LatchBr = dyn_cast<BranchInst>(Latch->getTerminator());
    if (LatchBr)
      LatchBr->setSuccessor(0, Exit);

    outs() << "Unrolled loop in function " << Header->getParent()->getName()
           << " by factor " << UnrollFactor << "\n";
  }

public:
  void unroll(Function &F, LoopInfo &LI) {
    bool Modified = false;

    for (Loop *L : LI) {
      if (isSimpleCountingLoop(L)) {
        unrollLoop(L, LI);
        Modified = true;
      }
    }
  }
};

struct LoopUnrollPass : PassInfoMixin<LoopUnrollPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
    LoopUnroller Unroller;
    Unroller.unroll(F, LI);
    return PreservedAnalyses::none();
  }

  static bool isRequired() { return true; }
};

} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getLoopUnrollPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopUnroll", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "loop-unroll") {
                    FPM.addPass(LoopUnrollPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLoopUnrollPluginInfo();
} 
