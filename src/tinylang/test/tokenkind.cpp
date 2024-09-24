#include <gtest/gtest.h>
#include <Basic/TokenKinds.h>
  using namespace tinylang;
// Demonstrate some basic assertions.
TEST(TokenKindTest, BasicAssertions) {

  // Expect two strings not to be equal.
  // EXPECT_EQ(getKeywordSpelling(tok::kw_AND), "and");
  EXPECT_EQ(std::string(tok::getPunctuatorSpelling(tok::l_paren)), "(");
}
