add_llvm_executable(tiny-tblgen main.cpp)
target_link_libraries(tiny-tblgen PRIVATE LLVMCore LLVMSupport LLVMTableGen)
