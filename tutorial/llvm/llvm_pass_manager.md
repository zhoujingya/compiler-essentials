## LLVM pass manager

* The `getAnalysisUsage()` function defines how a pass interacts with other passes

* Given `getAnalysisUsage(AnalysisUsage &AU)` for PassX:

  * AU.addRequired<PassY>()
  * PassY must be executed first
  * AU.addPreserved<PassY>()
  * PassY is still preserved by running PassX
  * AU.setPreservesAll()
  * PassX preserves all previous passes
  * AU.setPreservesCFG()
  * PassX might make changes, but not to the CFG
  * If nothing is specified, it is assumed that all previous passes are invalidated

DeadStoreElimination Pass:

```
void getAnalysisUsage(AnalysisUsage &AU) const override {
  AU.setPreservesCFG();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<MemoryDependenceAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<MemoryDependenceAnalysis>();
```
