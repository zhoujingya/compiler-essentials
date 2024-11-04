
add_library(loopOptTutorial SHARED LoopOptTutorial.cpp)

target_link_libraries(loopOptTutorial
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
