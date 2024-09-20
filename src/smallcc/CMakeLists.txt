
set(CXX_STANDARD c++17)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -fno-common")

add_executable(smallcc smallcc-main.cpp)

# NOTE: it's bery important to disable 
target_link_libraries(smallcc PRIVATE LLVMCore LLVMSupport)
