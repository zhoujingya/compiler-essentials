set(LLVM_LINK_COMPONENTS core support)
add_tinylang_executable(tinylang-lexer PARTIAL_SOURCES_INTENDED lexer.cpp)

target_link_libraries(tinylang-lexer PRIVATE tinylangBasic tinylangLexer)
