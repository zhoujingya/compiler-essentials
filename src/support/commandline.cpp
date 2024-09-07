#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

enum Print { hello, world };
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
static cl::opt<enum Print>
    testhello("emit", cl::desc("have to emit hello world"),
              cl::values(clEnumValN(hello, "hello", "print hello"),
                         clEnumValN(world, "world", "print world")));
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "LLVM commandline learning\n");
  llvm::outs() << "InpurFile name: " << inputFilename << "\n";
  if (testhello == Print::hello)
    llvm::outs() << "hello" << "\n";
  else if (testhello == Print::world)
    llvm::outs() << "world" << "\n";
  return 0;
}
