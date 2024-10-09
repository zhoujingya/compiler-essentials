## How to build

`cmake -B build -G Ninja`

`cmake --build build`

`./install-riscv.sh` to install `riscv` toolchain

> Build debug version llvm, then export toolchain binary path

## How to test

`lit build/test`


## [small cc](src/smallcc)

base on [chibicc](https://github.com/rui314/chibicc), but use RISC-V ISA
