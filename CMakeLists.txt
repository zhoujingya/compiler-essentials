cmake_minimum_required(VERSION 3.22)
project(llvm-essential
        VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED)
set(CMAKE_CXX_STANDARD 17)
add_definitions(${LLVM_DEFINITIONS})
include(AddLLVM)
message("LLVM version: " ${LLVM_PACKAGE_VERSION})
message("LLVM include directory: " ${LLVM_INCLUDE_DIRS})
message("LLVM library directory: " ${LLVM_LIBRARY_DIRS})
set(LLVM_DIR ${LLVM_INSTALL_PREFIX})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
add_subdirectory(loop)
add_subdirectory(test)
