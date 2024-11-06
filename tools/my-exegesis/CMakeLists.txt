# cmake_minimum_required(VERSION 3.22)

project(MYEXEGESIS)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# list(APPEND CMAKE_PREFIX_PATH "${NATIVE_X86_LLVM}/lib/cmake/llvm")

# find_package(LLVM REQUIRED CONFIG)

message(STATUS "Configuring MYEXEGESIS...")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_executable(my-exegesis my-exegesis.cpp)

add_subdirectory(lib)

# if(NOT LLVM_ENABLE_RTTI)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
# endif()

target_include_directories(my-exegesis PRIVATE ${LLVM_INCLUDE_DIRS})
message(STATUS "Include Dirs: ${LLVM_INCLUDE_DIRS}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
message(STATUS "Add LLVM Definition: ${LLVM_DEFINITIONS_LIST}")

llvm_map_components_to_libnames(llvm_libs support X86)
target_link_libraries(my-exegesis PRIVATE ${llvm_libs} MyExegesis)
message(STATUS "Add LLVM Libs: ${llvm_libs}")
