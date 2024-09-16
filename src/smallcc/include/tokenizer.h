#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

namespace smallcc {
typedef enum {
  TK_PUNCT, // Punctuators
  TK_NUM,   // Numeric literals
  TK_EOF,   // End-of-file markers
} TokenKind;

// A link list used to represent all tokens
typedef struct Token {
  TokenKind kind; // Token kind
  Token *next;    // Next token
  int val;        // If kind is TK_NUM, its value
  char *loc;      // Token location
  int len;        // Token length
} Token;

class Tokenizer {
  bool 
};

}; // namespace smallcc

#endif // __TOKENIZER_H__
