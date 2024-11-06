
message(STATUS "Configuring MYEXEGESIS...")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Include Dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
message(STATUS "Add LLVM Definition: ${LLVM_DEFINITIONS_LIST}")
message(STATUS "Add LLVM Libs: ${llvm_libs}")

add_executable(my-exegesis my-exegesis.cpp)

add_subdirectory(lib)
llvm_map_components_to_libnames(llvm_libs support X86)
target_link_libraries(my-exegesis PRIVATE ${llvm_libs} MyExegesis)

