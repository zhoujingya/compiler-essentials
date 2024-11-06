#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "lib/X86.h"

#include "lib/PerfHelper.h"

using namespace llvm;

static cl::opt<unsigned>
    OpcodeIndex("opcode-index", cl::desc("opcode to measure, by index"),
                cl::init(0));

static cl::opt<std::string>
    OpcodeName("opcode-name", cl::desc("opcode to measure, by name"),
               cl::init(""));

namespace myexegesis {
void main() {
  if (OpcodeName.empty() == (OpcodeIndex == 0)) {
    report_fatal_error(
      "please provide one and only one of 'opcode-index' or 'opcode-name' ");
  }

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  // FIXME: Target-specific filter.
  // X86Filter Filter;
  
  return;
}
}

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
