link_directories(${LLVM_LIBRARY_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

add_subdirectory(function)
add_subdirectory(jit)
add_subdirectory(loop)
add_subdirectory(support)
add_subdirectory(tablegen)
add_subdirectory(tinylang)

if(ENABLE_SMALL_CC)
  add_subdirectory(smallcc)
endif()
