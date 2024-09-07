
add_library(HelloWorld SHARED HelloWorld.cpp)
target_link_libraries(HelloWorld PRIVATE ${LLVM_LIBS})

target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
