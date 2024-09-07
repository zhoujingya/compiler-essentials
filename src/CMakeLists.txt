link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_subdirectory(loop)
add_subdirectory(support)
