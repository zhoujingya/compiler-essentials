link_directories(${LLVM_LIBRARY_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

add_subdirectory(loop)
add_subdirectory(function)
add_subdirectory(support)
