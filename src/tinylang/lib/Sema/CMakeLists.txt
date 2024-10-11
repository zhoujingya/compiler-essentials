

file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(LLVM_LINK_COMPONENTS core support)

add_tinylang_library(tinylangSema
  ${SOURCE}

  LINK_LIBS
  tinylangBasic
)
