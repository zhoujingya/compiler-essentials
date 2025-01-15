cmake_minimum_required(VERSION 3.22)
project(CUDA C CXX CUDA)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} SYSTEM)

add_subdirectory(divergence)
