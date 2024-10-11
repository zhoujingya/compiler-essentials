file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(file ${SOURCE})
  # 获取文件名
  get_filename_component(TEST_FILE "${file}" NAME)
  add_tinylang_executable(
    ${TEST_FILE}_bin
    PARTIAL_SOURCES_INTENDED
    ${file}
  )
  add_dependencies(${TEST_FILE}_bin tinylangBasic)
  target_link_libraries(
    ${TEST_FILE}_bin
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    tinylangBasic
    tinylangLexer
  )
  gtest_discover_tests(${TEST_FILE}_bin)
endforeach()




