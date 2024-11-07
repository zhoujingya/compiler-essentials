#include "BenchmarkRunner.h"
#include "BenchmarkResult.h"
#include "InMemoryAssembler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrInfo.h"
#include <string>

namespace myexegesis {
BenchmarkRunner::InstructionFilter::~InstructionFilter() = default;

BenchmarkRunner::~BenchmarkRunner() = default;

InstructionBenchmark
BenchmarkRunner::run(const LLVMState &State, unsigned int Opcode,
                     unsigned int NumRepetitions,
                     const InstructionFilter &Filter) const {
  InstructionBenchmark InstrBenchmark;
  InstrBenchmark.AsmTmpl.Name =
      Twine(getDisplayName())
          .concat(" ")
          .concat(State.getInstrInfo().getName(Opcode))
          .str();
  InstrBenchmark.CpuName = State.getCpuName();
  InstrBenchmark.LLVMTriple = State.getTripleString();
  InstrBenchmark.NumRepetitions = NumRepetitions;

  // Ignore instructions that we cannot run.
  if (State.getInstrInfo().get(Opcode).isPseudo()) {
    InstrBenchmark.Error = "Unsupported opcode: isPseudo";
    return InstrBenchmark;
  }
  if (Error E = Filter.shouldRun(State, Opcode)) {
    InstrBenchmark.Error = toString(std::move(E));
    return InstrBenchmark;
  }

  JitFunctionContext Context(State.createTargetMachine());
  auto ExpectedInstructions =
      createCode(State, Opcode, NumRepetitions, Context);
  if (Error E = ExpectedInstructions.takeError()) {
    InstrBenchmark.Error = toString(std::move(E));
    return InstrBenchmark;
  }

  const std::vector<MCInst> Instructions = *ExpectedInstructions;
  const JitFunction Function(std::move(Context), Instructions);
  const StringRef CodeBytes = Function.getFunctionBytes();

  std::string AsmExcerpt;
  constexpr const int ExcerptSize = 100;
  constexpr const int ExcerptTailSize = 10;
  if (CodeBytes.size() <= ExcerptSize) {
    AsmExcerpt = toHex(CodeBytes);
  } else {
    AsmExcerpt = toHex(CodeBytes.take_front(ExcerptSize - ExcerptTailSize + 3));
    AsmExcerpt += "...";
    AsmExcerpt += toHex(CodeBytes.take_back(ExcerptTailSize));
  }
  outs() << "# Asm excerpt: " << AsmExcerpt << "\n";
  outs().flush(); // In case we crash.

  InstrBenchmark.Measurements =
      runMeasurements(State, Function, NumRepetitions);
  return InstrBenchmark;
}
} // namespace myexegesis