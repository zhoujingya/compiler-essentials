
add_library(loopOptTutorial SHARED LoopOptTutorial.cpp)
add_library(loopUnroll SHARED LoopUnroll.cpp)


target_link_libraries(loopOptTutorial
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
target_link_libraries(loopUnroll
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
