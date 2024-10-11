

file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(LLVM_LINK_COMPONENTS Core Support)

add_tinylang_library(tinylangLexer
  ${SOURCE}

  LINK_LIBS
  tinylangBasic
)
