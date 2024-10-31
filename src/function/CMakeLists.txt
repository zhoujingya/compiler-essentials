
add_library(HelloWorld SHARED HelloWorld.cpp)
target_link_libraries(HelloWorld PRIVATE ${LLVM_LIBS})

add_library(Mem2Reg SHARED Mem2Reg.cpp)
target_link_libraries(Mem2Reg PRIVATE ${LLVM_LIBS})

target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
