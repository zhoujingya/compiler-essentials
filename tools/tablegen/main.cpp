/*
Driver for TableGen.
*/

#include "llvm/TableGen/Main.h"
#include "TinyTablegenEmitter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Record.h"
using namespace llvm;

enum ActionType {
  PrintRecords,
  DumpJSON,
  GenTokens,
};
namespace {
cl::opt<ActionType>
    Action(cl::desc("Action to perform:"),
           cl::values(clEnumValN(PrintRecords, "print-records",
                                 "Print all records to stdout (default)"),
                      clEnumValN(DumpJSON, "dump-json",
                                 "Dump all records as "
                                 "machine-readable JSON"),
                      clEnumValN(GenTokens, "gen-tokens",
                                 "Generate token kinds and keyword "
                                 "filter")));

bool Main(raw_ostream &OS, RecordKeeper &Records) {
  tinytblgen::Emitter Emitter(OS, Records);
  switch (Action) {
  case PrintRecords:
    OS << Records; // No argument, dump all contents
    break;
  case DumpJSON:
    EmitJSON(Records, OS);
    break;
  case GenTokens:
    Emitter.EmitTokensAndKeywordFilter(OS, Records);
    break;
  }

  return false;
}
} // namespace

// Main function
int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], Main);
}
