#include "lib/BenchmarkResult.h"
#include "lib/BenchmarkRunner.h"
#include "lib/LatencyBenchmarkRunner.h"
#include "lib/LlvmState.h"
#include "lib/PerfHelper.h"
#include "lib/UopsBenchmarkRunner.h"
#include "lib/X86.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/PrettyStackTrace.h"
#include <memory>
#include <type_traits>

using namespace llvm;

static cl::opt<unsigned> OpcodeIndex("opcode-index",
                                     cl::desc("opcode to measure, by index"),
                                     cl::init(0));

static cl::opt<std::string> OpcodeName("opcode-name",
                                       cl::desc("opcode to measure, by name"),
                                       cl::init(""));

enum class BenchmarkModeE { Latency, Uops };
static cl::opt<BenchmarkModeE> BenchmarkMode(
    "benchmark-mode", cl::desc("the benchmark mode to run"),
    cl::values(clEnumValN(BenchmarkModeE::Latency, "latency",
                          "Instruction Latency"),
               clEnumValN(BenchmarkModeE::Uops, "uops", "Uop Decomposition")));

static cl::opt<unsigned>
    NumRepetitions("num-repetitions",
                   cl::desc("number of time to repeat the asm snippet"),
                   cl::init(10000));

namespace myexegesis {
void main() {
  if (OpcodeName.empty() == (OpcodeIndex == 0)) {
    report_fatal_error(
        "please provide one and only one of 'opcode-index' or 'opcode-name' ");
  }

  LLVMInitializeX86Target();
  LLVMInitializeX86AsmPrinter();

  // FIXME: Target-specific filter.
  X86Filter Filter;

  const LLVMState State;

  unsigned Opcode = OpcodeIndex;

  if (Opcode == 0) {
    // Resolve opcode name -> opcode.
    for (unsigned I = 0, E = State.getInstrInfo().getNumOpcodes(); I < E; ++I) {
      if (State.getInstrInfo().getName(I) == OpcodeName) {
        Opcode = I;
        break;
      }
    }
    if (Opcode == 0) {
      report_fatal_error(Twine("unknown opcode ").concat(OpcodeName));
    }
  }

  std::unique_ptr<BenchmarkRunner> Runner;
  switch (BenchmarkMode) {
  case BenchmarkModeE::Latency:
    Runner = std::make_unique<LatencyBenchmarkRunner>();
    break;
  case BenchmarkModeE::Uops:
    Runner = std::make_unique<UopsBenchmarkRunner>();
    break;
  }

  Runner->run(State, Opcode, NumRepetitions > 0 ? NumRepetitions : 1, Filter)
      .writeYamlOrDie("-");
}
} // namespace myexegesis

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "");

  if (myexegesis::pfm::pfmInitialize()) {
    errs() << "cannot initialize libpfm\n";
    return EXIT_FAILURE;
  }

  myexegesis::main();

  myexegesis::pfm::pfmTerminate();
  return EXIT_SUCCESS;
}
