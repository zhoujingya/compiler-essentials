add_executable(commandline commandline.cpp)

target_link_libraries(commandline PRIVATE LLVMCore LLVMSupport)
