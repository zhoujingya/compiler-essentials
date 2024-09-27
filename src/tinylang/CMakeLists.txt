project(tinylang)

include(HandleLLVMOptions)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TINYLANG_VERSION_STRING "0.1")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Basic/Config.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/Basic/Config.inc)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(lib)
add_subdirectory(test)
