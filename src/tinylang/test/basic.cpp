#include "Basic/Diagnostic.h"
#include "Basic/TokenKinds.h"
#include <gtest/gtest.h>
using namespace tinylang;
// Demonstrate some basic assertions.
TEST(TokenKindTest, BasicAssertions) {

  // Expect two strings not to be equal.
  EXPECT_EQ(std::string(getKeywordSpelling(tok::kw_AND)), "AND");
  EXPECT_EQ(std::string(tok::getPunctuatorSpelling(tok::l_paren)), "(");
}

TEST(DiagnosticTest, BasicAssertions) {
  EXPECT_EQ(
      std::string(DiagnosticsEngine::getDiagnosticText(diag::err_expected)),
      "expected {0} but found {1}");
}
