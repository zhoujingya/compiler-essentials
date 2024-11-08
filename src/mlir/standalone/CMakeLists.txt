get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../include)
add_subdirectory(standalone)
add_mlir_library(MLIRStandalone
                        StandaloneDialect.cpp
                        StandaloneOps.cpp
                        LowerToLLVM.cpp
                        createSimpleAttr.cpp

        DEPENDS
        StandaloneOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)
