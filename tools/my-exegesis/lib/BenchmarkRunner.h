#include "LlvmState.h"
#include "BenchmarkResult.h"
#include "InMemoryAssembler.h"
#include "llvm/Support/Error.h"
#include "llvm/MC/MCInst.h"

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H
#define LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H

using namespace llvm;

namespace myexegesis {
// Common code for all benchmark modes.
class BenchmarkRunner {
public:
  // Subtargets can disable running benchmarks for some instructions by
  // returning an error here.
  class InstructionFilter {
  public:
    virtual ~InstructionFilter();

    virtual Error shouldRun(const LLVMState &State, unsigned Opcode) const {
      return ErrorSuccess();
    }
  };

  virtual ~BenchmarkRunner();

  InstructionBenchmark run(const LLVMState &State, unsigned Opcode,
                           unsigned NumRepetitions,
                           const InstructionFilter &Filter) const;

private:
  virtual const char *getDisplayName() const = 0;

  virtual llvm::Expected<std::vector<MCInst>>
  createCode(const LLVMState &State, unsigned OpcodeIndex,
             unsigned NumRepetitions,
             const JitFunctionContext &Context) const = 0;

  virtual std::vector<BenchmarkMeasure>
  runMeasurements(const LLVMState &State, const JitFunction &Function,
                  unsigned NumRepetitions) const = 0;
};
} // namespace myexegesis

#endif