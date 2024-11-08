#ifndef LLVM_TOOLS_MY_EXEGESIS_LATENCY_H
#define LLVM_TOOLS_MY_EXEGESIS_LATENCY_H

#include "BenchmarkRunner.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"

namespace myexegesis {

class LatencyBenchmarkRunner : public BenchmarkRunner {
public:
  ~LatencyBenchmarkRunner() override;

private:
  const char *getDisplayName() const override;

  Expected<std::vector<MCInst>>
  createCode(const LLVMState &State, unsigned OpcodeIndex,
             unsigned NumRepetitions,
             const JitFunctionContext &Context) const override;

  std::vector<BenchmarkMeasure>
  runMeasurements(const LLVMState &State, const JitFunction &Function,
                  unsigned NumRepetitions) const override;
};

} // namespace myexegesis

#endif