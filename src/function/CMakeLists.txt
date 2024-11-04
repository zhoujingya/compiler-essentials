add_library(HelloWorld SHARED HelloWorld.cpp)

add_library(Mem2Reg SHARED Mem2Reg.cpp)

target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
