link_directories(${LLVM_LIBRARY_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

add_subdir_recursive(${CMAKE_CURRENT_SOURCE_DIR})
