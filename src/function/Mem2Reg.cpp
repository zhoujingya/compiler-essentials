#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include <map>
#include <set>

using namespace llvm;

namespace {

class Mem2RegPromoter {
private:
    // 记录每个 alloca 的所有 store 指令
    std::map<AllocaInst*, std::vector<StoreInst*>> allocaStores;
    // 记录每个 alloca 的所有 load 指令
    std::map<AllocaInst*, std::vector<LoadInst*>> allocaLoads;
    // 记录每个基本块中的 phi 节点
    std::map<BasicBlock*, std::map<AllocaInst*, PHINode*>> blockPhis;

    // 检查 alloca 是否可以提升
    bool isPromotable(AllocaInst* AI) {
        // 只处理基本类型的 alloca
        if (!AI->getAllocatedType()->isFirstClassType())
            return false;

        // 确保所有使用都是 load 或 store
        for (User* U : AI->users()) {
            if (!(isa<LoadInst>(U) || isa<StoreInst>(U)))
                return false;
        }
        return true;
    }

    // 收集所有的 load 和 store 指令
    void collectLoadAndStores(AllocaInst* AI) {
        for (User* U : AI->users()) {
            if (LoadInst* LI = dyn_cast<LoadInst>(U)) {
                allocaLoads[AI].push_back(LI);
            } else if (StoreInst* SI = dyn_cast<StoreInst>(U)) {
                allocaStores[AI].push_back(SI);
            }
        }
    }

    // 在必要的位置插入 phi 节点
    void insertPhiNodes(AllocaInst* AI, Function& F, DominatorTree& DT) {
        std::set<BasicBlock*> defBlocks;

        // 收集所有定义点（store 指令所在的基本块）
        for (StoreInst* SI : allocaStores[AI]) {
            defBlocks.insert(SI->getParent());
        }

        // 对于每个 load，找到其支配边界并插入 phi 节点
        std::set<BasicBlock*> phiBlocks;
        for (LoadInst* LI : allocaLoads[AI]) {
            BasicBlock* loadBB = LI->getParent();

            // 简化版：在所有汇合点插入 phi 节点
            for (BasicBlock& BB : F) {
                if (pred_size(&BB) >= 2) {  // 有多个前驱的基本块
                    bool hasDefiningPred = false;
                    bool hasNonDefiningPred = false;

                    for (BasicBlock* Pred : predecessors(&BB)) {
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

        // 创建 phi 节点
        for (BasicBlock* BB : phiBlocks) {
            PHINode* phi = PHINode::Create(
                AI->getAllocatedType(),
                pred_size(BB),
                AI->getName() + ".phi",
                &BB->front()
            );
            blockPhis[BB][AI] = phi;
        }
    }

    // 重写所有的 load 和 store
    void rewriteLoadAndStores(AllocaInst* AI) {
        // 重写 store
        for (StoreInst* SI : allocaStores[AI]) {
            Value* storedVal = SI->getValueOperand();
            BasicBlock* BB = SI->getParent();

            // 更新相关的 phi 节点
            for (auto& entry : blockPhis) {
                BasicBlock* phiBB = entry.first;
                PHINode* phi = entry.second[AI];
                if (phi && BB == phiBB->getUniquePredecessor()) {
                    phi->addIncoming(storedVal, BB);
                }
            }

            SI->eraseFromParent();
        }

        // 重写 load
        for (LoadInst* LI : allocaLoads[AI]) {
            BasicBlock* BB = LI->getParent();
            Value* newVal = nullptr;

            // 检查是否有对应的 phi 节点
            auto it = blockPhis.find(BB);
            if (it != blockPhis.end() && it->second.count(AI)) {
                newVal = it->second[AI];
            } else {
                // 使用最近的定义
                // 这是一个简化版本，完整实现需要更复杂的变量定义追踪
                for (StoreInst* SI : allocaStores[AI]) {
                    if (SI->getParent() == BB) {
                        newVal = SI->getValueOperand();
                        break;
                    }
                }
            }

            if (newVal) {
                LI->replaceAllUsesWith(newVal);
                LI->eraseFromParent();
            }
        }
    }

public:
    void promote(Function& F) {
        DominatorTree DT(F);

        // 找到所有可提升的 alloca
        std::vector<AllocaInst*> promotableAllocas;
        for (auto& BB : F) {
            for (auto& I : BB) {
                if (AllocaInst* AI = dyn_cast<AllocaInst>(&I)) {
                    if (isPromotable(AI)) {
                        promotableAllocas.push_back(AI);
                    }
                }
            }
        }

        // 对每个可提升的 alloca 进行处理
        for (AllocaInst* AI : promotableAllocas) {
            // 1. 收集所有的 load 和 store
            collectLoadAndStores(AI);

            // 2. 插入必要的 phi 节点
            insertPhiNodes(AI, F, DT);

            // 3. 重写所有的 load 和 store
            rewriteLoadAndStores(AI);

            // 4. 删除原始的 alloca
            AI->eraseFromParent();
        }

        outs() << "(mem2reg) Promoted " << promotableAllocas.size()
               << " allocas to registers in function: " << F.getName() << "\n";
    }
};

// 主要的访问函数实现
void visitor(Function &F) {
    Mem2RegPromoter promoter;
    promoter.promote(F);
}

// 新的 PM 实现
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
