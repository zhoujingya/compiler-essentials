#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;
cl::opt<std::string> inputParsedString("input-string",
                                       cl::desc("string to be parsed"),
                                       cl::init("0"),
                                       cl::value_desc("filename"));

/// get input file content
std::string getInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError())
    llvm::report_fatal_error("Could not open input file: " + filename);
  return fileOrErr.get()->getBuffer().str();
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "smallcc compiler\n");

  outs() << "  .globl main\n";
  outs() << "main:\n";
  printf("  li a0, %d\n", atoi(inputParsedString.c_str()));
  printf("  ret\n");

  // TODO: add
}
