#include "LatencyBenchmarkRunner.h"
#include "BenchmarkResult.h"
#include "InstructionSnippetGenerator.h"
#include "PerfHelper.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <random>

namespace myexegesis {

// FIXME: Handle memory, see PR36905.
static bool isInvalidOperand(const MCOperandInfo &OpInfo) {
  switch (OpInfo.OperandType) {
  default:
    return true;
  case MCOI::OPERAND_IMMEDIATE:
  case MCOI::OPERAND_REGISTER:
    return false;
  }
}

static Error makeError(Twine Msg) {
  return make_error<StringError>(Msg, inconvertibleErrorCode());
}

LatencyBenchmarkRunner::~LatencyBenchmarkRunner() = default;

const char *LatencyBenchmarkRunner::getDisplayName() const { return "latency"; }

Expected<std::vector<MCInst>> LatencyBenchmarkRunner::createCode(
    const LLVMState &State, const unsigned OpcodeIndex,
    const unsigned NumRepetitions, const JitFunctionContext &Context) const {
  std::default_random_engine RandomEngine;
  const auto GetRandomIndex = [&RandomEngine](size_t Size) {
    assert(Size > 0 && "trying to get select a random element of an empty set");
    return std::uniform_int_distribution<>(0, Size - 1)(RandomEngine);
  };

  const auto &InstrInfo = State.getInstrInfo();
  const auto &RegInfo = State.getRegInfo();
  const MCInstrDesc &InstrDesc = InstrInfo.get(OpcodeIndex);
  for (const MCOperandInfo &OpInfo : InstrDesc.operands()) {
    if (isInvalidOperand(OpInfo))
      return makeError("Only registers and immediates are supported");
  }

  const auto Vars = getVariables(RegInfo, InstrDesc, Context.getReservedRegs());
  const std::vector<AssignmentChain> AssignmentChains =
      computeSequentialAssignmentChains(RegInfo, Vars);
  if (AssignmentChains.empty())
    return makeError("Unable to find a dependency chain.");
  const std::vector<MCPhysReg> Regs =
      getRandomAssignment(Vars, AssignmentChains, GetRandomIndex);
  const MCInst Inst = generateMCInst(InstrDesc, Vars, Regs);
  if (!State.canAssemble(Inst))
    return makeError("MCInst does not assemble.");
  return std::vector<MCInst>(NumRepetitions, Inst);
}

std::vector<BenchmarkMeasure>
LatencyBenchmarkRunner::runMeasurements(const LLVMState &State,
                                        const JitFunction &Function,
                                        const unsigned NumRepetitions) const {
  // Cycle measurements include some overhead from the kernel. Repeat the
  // measure several times and take the minimum value.
  constexpr const int NumMeasurements = 30;
  int64_t MinLatency = std::numeric_limits<int64_t>::max();
  // FIXME: Read the perf event from the MCSchedModel (see PR36984).
  const pfm::PerfEvent CyclesPerfEvent("UNHALTED_CORE_CYCLES");
  if (!CyclesPerfEvent.valid())
    report_fatal_error("invalid perf event 'UNHALTED_CORE_CYCLES'");
  for (size_t I = 0; I < NumMeasurements; ++I) {
    pfm::Counter Counter(CyclesPerfEvent);
    Counter.start();
    Function();
    Counter.stop();
    const int64_t Value = Counter.read();
    if (Value < MinLatency)
      MinLatency = Value;
  }
  return {{"latency", static_cast<double>(MinLatency) / NumRepetitions}};
}
} // namespace myexegesis