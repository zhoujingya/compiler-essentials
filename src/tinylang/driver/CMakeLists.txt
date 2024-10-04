add_executable(tinylang-lexer lexer.cpp)
target_link_libraries(tinylang-lexer PRIVATE tinylangBasic tinylangLexer LLVMSupport LLVMCore)
