add_subdirectory(function)
add_subdirectory(jit)
add_subdirectory(loop)
add_subdirectory(support)
add_subdirectory(tinylang)

if(ENABLE_SMALL_CC)
  add_subdirectory(smallcc)
endif()
