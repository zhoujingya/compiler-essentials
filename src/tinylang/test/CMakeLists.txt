file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(file ${SOURCE})
  # Get file name
  get_filename_component(TEST_FILE "${file}" NAME)
  add_executable(
    ${TEST_FILE}_bin
    ${file}
  )
  add_dependencies(${TEST_FILE}_bin tinylangBasic)
  target_link_libraries(
  ${TEST_FILE}_bin
  GTest::gtest
  GTest::gtest_main
  tinylangBasic
  tinylangLexer
  )
  gtest_discover_tests(${TEST_FILE}_bin)
endforeach()




