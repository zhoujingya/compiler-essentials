
file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(tinylangBasic SHARED ${SOURCE})

target_link_libraries(tinylangBasic PRIVATE LLVMCore LLVMSupport)
