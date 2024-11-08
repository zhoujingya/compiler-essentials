#ifndef LLVM_TOOLS_MY_EXEGESIS_UOPS_H
#define LLVM_TOOLS_MY_EXEGESIS_UOPS_H

#include "BenchmarkRunner.h"

namespace myexegesis {

class UopsBenchmarkRunner : public BenchmarkRunner {
public:
  ~UopsBenchmarkRunner() override;

private:
  const char *getDisplayName() const override;

  llvm::Expected<std::vector<llvm::MCInst>>
  createCode(const LLVMState &State, unsigned OpcodeIndex,
             unsigned NumRepetitions,
             const JitFunctionContext &Context) const override;

  std::vector<BenchmarkMeasure>
  runMeasurements(const LLVMState &State, const JitFunction &Function,
                  unsigned NumRepetitions) const override;
};

} // namespace exegesis

#endif
