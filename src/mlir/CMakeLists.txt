project(mlir-learning VERSION 1.0.0)

# Find mlir
set (MLIR_DIR ${LLVM_INSTALL_PREFIX}/lib/cmake/mlir)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: " ${MLIR_FOUND})
message(STATUS "MLIR_PACKAGE_VERSION: " ${MLIR_VERSION})
message(STATUS "MLIR_INCLUDE_DIRS: " ${MLIR_INCLUDE_DIRS})
message(STATUS "MLIR_LIBRARY_DIRS: " ${MLIR_LIBRARY_DIRS})
# Append function list and include
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM) # LLVM stuff
include(AddMLIR) # MLIR stuff
include(HandleLLVMOptions)

# Set the LLVM header and library paths
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})

link_libraries(
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
)


add_subdirectory(include)

set(MLIR_TARGET "toy" CACHE STRING "Default target")

if(${MLIR_TARGET} STREQUAL "toy")
  include(toy.cmake)
  message(STATUS "Building toy target")
elseif(${MLIR_TARGET} STREQUAL "standalone")
  include(standalone.cmake)
  message(STATUS "Building standalone target")
else()
  message("Not suppoted target: ${CMAKE_TARGET}")
endif()
