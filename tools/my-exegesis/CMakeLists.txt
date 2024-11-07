
message(STATUS "Configuring MYEXEGESIS...")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
message(STATUS "Add LLVM Definition: ${LLVM_DEFINITIONS_LIST}")

add_executable(my-exegesis my-exegesis.cpp)
llvm_map_components_to_libnames(llvm_libs support X86)
target_link_libraries(my-exegesis PRIVATE ${llvm_libs} MyExegesis)

add_subdirectory(lib)
