cmake_minimum_required(VERSION 3.15)

project(LIBPFM)

add_library(MyExegesis STATIC PerfHelper.cpp LlvmState.cpp)

if(PFM_PATH)
  message(STATUS "Set PFM_PATH: ${PFM_PATH}")
  message(STATUS "Add Compile Definition: -DHAVE_LIBPFM")
  target_compile_definitions(MyExegesis PRIVATE HAVE_LIBPFM)
  target_include_directories(MyExegesis PRIVATE ${PFM_PATH}/include)
  
  list(APPEND CMAKE_PREFIX_PATH "${PFM_PATH}")
  find_library(PFM_LIBRARY
                NAMES pfm
                PATHS ${PFM_PATH}/lib
                )
  message(STATUS "Link PFM LIBRARY: ${PFM_LIBRARY}")
  target_link_libraries(MyExegesis PRIVATE ${PFM_LIBRARY})
endif()
