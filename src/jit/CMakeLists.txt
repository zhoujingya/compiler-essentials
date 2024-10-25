add_executable(jit jit-main.cpp)
llvm_map_components_to_libnames(llvm_libs Core OrcJIT Support native)
target_link_libraries(jit PRIVATE ${llvm_libs})
