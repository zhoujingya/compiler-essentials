#include "Basic/Diagnostic.h"
#include "Basic/TokenKinds.h"
#include <gtest/gtest.h>
using namespace tinylang;

// Demonstrate some basic assertions.
TEST(LexerTest, BasicAssertions) {

  // Expect two strings not to be equal.
  EXPECT_EQ(std::string(getKeywordSpelling(tok::kw_AND)), "AND");
  EXPECT_EQ(std::string(tok::getPunctuatorSpelling(tok::l_paren)), "(");
}
