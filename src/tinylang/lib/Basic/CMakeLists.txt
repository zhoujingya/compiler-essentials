
add_library(tinylang SHARED TokenKinds.cpp)

target_link_libraries(tinylang PRIVATE LLVMCore LLVMSupport)
