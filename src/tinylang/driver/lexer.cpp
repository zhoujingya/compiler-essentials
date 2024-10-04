/**
 * @file lexer.cpp
 * @author zhoujing
 * @brief tinylang lexer driver
 * @version 0.1
 * @date 2024-10-03
 *
 * @copyright Copyright (c) 2024
 */

#include "Lexer/Lexer.h"
#include <llvm/Support/CommandLine.h>
using namespace tinylang;
using namespace llvm;

static cl::opt<bool> printInputFile("print",
                                    cl::desc("Print input file content"),
                                    cl::init(false),
                                    cl::value_desc("print or not"));
static cl::opt<std::string> inputFile(cl::Positional,
                                    cl::desc("<Input file>"),
                                    cl::init("-"),
                                    cl::value_desc("Input file path"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "tinylang lexer driver\n");
  SourceMgr SrcMgr;
  DiagnosticsEngine Diags(SrcMgr);
  // Tell SrcMgr about this buffer, which is what the
  // parser will pick up.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(inputFile);
  SrcMgr.AddNewSourceBuffer(std::move(*FileOrErr), llvm::SMLoc());
  Lexer lexer(SrcMgr, Diags);
  // print file content
  if(printInputFile)
    llvm::outs() << lexer.getBuffer();
  return 0;
}
