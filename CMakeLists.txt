cmake_minimum_required(VERSION 3.22)
project(llvm-essential
        VERSION 1.0.0)

find_package(LLVM REQUIRED)

message("LLVM version: " ${LLVM_PACKAGE_VERSION})
message("LLVM include directory: " ${LLVM_INCLUDE_DIRS})
message("LLVM library directory: " ${LLVM_LIBRARY_DIRS})

add_subdirectory(loop)
