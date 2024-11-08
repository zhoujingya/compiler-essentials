#!/usr/bin/bash

LLVM_TOOL_PATH=$HOME/tools/clang16
export PATH=$LLVM_TOOL_PATH/bin:$PATH


cmake -B build_standalone \
    -S . \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLT_LLVM_INSTALL_DIR=$LLVM_TOOL_PATH \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_TARGET="standalone"


cmake --build build_standalone