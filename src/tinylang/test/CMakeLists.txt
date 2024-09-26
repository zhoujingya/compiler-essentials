add_executable(
  tokenkind_test
  tokenkind.cpp
)
add_dependencies(tokenkind_test tinylang)

target_link_libraries(
  tokenkind_test
  GTest::gtest
  GTest::gtest_main
  tinylang
)
gtest_discover_tests(tokenkind_test)
