include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_BINARY_DIR})
set(LLVM_TARGET_DEFINITIONS toy/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
add_mlir_library(toy
    toy/AST.cpp
    toy/Dialect.cpp
    toy/MLIRGen.cpp
    toy/ToyCombine.cpp
    toy/toyToAffineLoops.cpp
    toy/ToyToLLVM.cpp
    toy/ShapeInferencePass.cpp

    DEPENDS
    ToyOpsIncGen
    ToyCombineIncGen
    ToyShapeInferenceInterfaceIncGen
)

add_executable(mlir-toy toy/toy.cpp)
target_link_libraries(mlir-toy toy ${dialect_libs} ${extension_libs})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
