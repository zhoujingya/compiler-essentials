
set(CXX_STANDARD c++17)
set(CMAKE_CXX_FLAGS "-fno-common")

add_executable(smallcc smallcc-main.cpp)

target_link_libraries(smallcc PRIVATE LLVMSupport)
