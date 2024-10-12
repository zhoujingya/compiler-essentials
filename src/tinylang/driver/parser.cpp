/**
 * @file parser.cpp
 * @author zhoujing
 * @brief tinylang parser driver
 * @version 0.1
 * @date 2024-10-12
 *
 * @copyright Copyright (c) 2024
 */

#include "Parser/Parser.h"
#include "Lexer/Lexer.h"
#include <llvm/Support/CommandLine.h>
using namespace tinylang;
using namespace llvm;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<Input file>"),
                                      cl::init("-"),
                                      cl::value_desc("Input file path"));

class ParserDriver {
  friend class Parser;

private:
  Parser &parser;
  SourceMgr &SrcMgr;

public:
  ParserDriver(Parser &parser, SourceMgr &SrcMgr)
      : parser(parser), SrcMgr(SrcMgr) {}
  ~ParserDriver() = default;
  void parse() {
    parser.parse();
  };
};

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "tinylang parser driver\n");

  SourceMgr SrcMgr;
    // Tell SrcMgr about this buffer, which is what the
  // parser will pick up.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(inputFile);
  SrcMgr.AddNewSourceBuffer(std::move(*FileOrErr), llvm::SMLoc());
  DiagnosticsEngine Diags(SrcMgr);
  Lexer lexer(SrcMgr, Diags);
  Sema sema(Diags);
  Parser parser(lexer, sema);
  ParserDriver driver(parser, SrcMgr);
  driver.parse();
  return 0;
}
