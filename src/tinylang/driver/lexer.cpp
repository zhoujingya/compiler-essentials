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
static cl::opt<bool> dumpTokens("dump-token",
                                cl::desc("Dump tokens in the buffer"),
                                cl::init(false), cl::value_desc("dump or not"));
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<Input file>"),
                                      cl::init("-"),
                                      cl::value_desc("Input file path"));

class LexerDriver {
private:
  Lexer &lexer;
  SourceMgr &SrcMgr;

public:
  LexerDriver(Lexer &lexer, SourceMgr &SrcMgr) : lexer(lexer), SrcMgr(SrcMgr) {}
  ~LexerDriver() = default;
  void dumpTokens() {
    Token tok;
    while (true) {
      lexer.next(tok);
      if (tok.is(tok::eof))
        break;
      SMLoc Loc = tok.getLocation();
      unsigned LineNo = SrcMgr.getLineAndColumn(Loc).first; // Get line number
      unsigned ColNo = SrcMgr.getLineAndColumn(Loc).second; // Get column number
      StringRef Filename =
          SrcMgr.getMemoryBuffer(SrcMgr.FindBufferContainingLoc(Loc))
              ->getBufferIdentifier();
      llvm::outs() << Filename << ":" << LineNo << ":" << ColNo << ": "
                   << tok.getName() << "\n";
    }
  }
};

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
  LexerDriver driver(lexer, SrcMgr);
  // print file content
  if (printInputFile)
    llvm::outs() << lexer.getBuffer();
  if (dumpTokens)
    driver.dumpTokens();
  return 0;
}
