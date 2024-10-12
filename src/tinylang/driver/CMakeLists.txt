set(LLVM_LINK_COMPONENTS core support)
add_tinylang_executable(tinylang-lexer PARTIAL_SOURCES_INTENDED lexer.cpp)
add_tinylang_executable(tinylang-parser PARTIAL_SOURCES_INTENDED parser.cpp)

target_link_libraries(tinylang-lexer PRIVATE tinylangBasic tinylangLexer)
target_link_libraries(tinylang-parser PRIVATE tinylangBasic tinylangLexer
                      tinylangParser)
