

file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(tinylangLexer SHARED ${SOURCE})

target_link_libraries(tinylangLexer PRIVATE LLVMCore LLVMSupport tinylangBasic)
