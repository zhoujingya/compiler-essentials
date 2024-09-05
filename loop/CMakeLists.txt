link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_library(loopOptTutorial SHARED LoopOptTutorial.cpp)
target_link_libraries(loopOptTutorial PRIVATE ${LLVM_LIBS})

target_link_libraries(loopOptTutorial
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
