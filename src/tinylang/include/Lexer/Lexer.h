#ifndef TINYLANG_LEXER_LEXER_H
#define TINYLANG_LEXER_LEXER_H

#include "Basic/Diagnostic.h"
#include "Basic/LLVM.h"
#include "Lexer/Token.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

namespace tinylang {

class KeywordFilter {
  llvm::StringMap<tok::TokenKind> HashTable;

  void addKeyword(StringRef Keyword, tok::TokenKind TokenCode);

public:
  void addKeywords();

  tok::TokenKind getKeyword(StringRef Name,
                            tok::TokenKind DefaultTokenCode = tok::unknown) {
    auto Result = HashTable.find(Name);
    if (Result != HashTable.end())
      return Result->second;
    return DefaultTokenCode;
  }
};

class Lexer {
  SourceMgr &SrcMgr;
  DiagnosticsEngine &Diags;

  const char *CurPtr;
  StringRef CurBuf;

  /// CurBuffer - This is the current buffer index we're
  /// lexing from as managed by the SourceMgr object.
  unsigned CurBuffer = 0;

  KeywordFilter Keywords;

public:
  Lexer(SourceMgr &SrcMgr, DiagnosticsEngine &Diags)
      : SrcMgr(SrcMgr), Diags(Diags) {
    CurBuffer = SrcMgr.getMainFileID();
    CurBuf = SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer();
    CurPtr = CurBuf.begin();
    Keywords.addKeywords();
  }

  DiagnosticsEngine &getDiagnostics() const { return Diags; }

  /// Returns the next token from the input.
  void next(Token &Result);

  /// Gets source code buffer.
  StringRef getBuffer() const { return CurBuf; }


private:
  void identifier(Token &Result);
  // Form a number token from the current position.
  void number(Token &Result);
  // Form a string token from the current position.
  void string(Token &Result);
  // Skip the comment.
  void comment();

  // Get the location of the current position.
  SMLoc getLoc() { return SMLoc::getFromPointer(CurPtr); }

  // Form a token from the current position to TokEnd.
  void formToken(Token &Result, const char *TokEnd, tok::TokenKind Kind);
};
} // namespace tinylang
#endif
