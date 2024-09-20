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
  outs() << ".globl main\n";
  std::string main= "main:\n"
                    "    addi    sp, sp, -16\n "
                    "    sw      ra, 12(sp)\n"
                    "    sw      s0, 8(sp)\n"
                    "    addi    s0, sp, 16\n"
                    "    li      a0, 0\n"
                    "    sw      a0, -16(s0)\n"
                    "    sw      a0, -12(s0)\n"
                    "    lui     a0, %hi(.LC0)\n"
                    "    addi    a0, a0, %lo(.LC0)\n"
                    "    call    printf\n"
                    "    lw      a0, -16(s0)\n"
                    "    lw      ra, 12(sp)\n"
                    "    lw      s0, 8(sp)\n"
                    "    addi    sp, sp, 16\n"
                    "    ret\n";
  outs() << main;
  outs() << ".section	.rodata \n";
	outs() << ".align	3\n";
  printf("%s", ".LC0:\n");
  printf("    .string  \"%d\"\n", atoi(inputParsedString.c_str()));
  // TODO: add
}
