## How to build

`cmake -B build -G Ninja`
`cmake --build build`

> Build debug version llvm, then export toolchain binary path

## How to test

`lit build/test`


## [small cc](src/smallcc)

base on [chibicc](https://github.com/rui314/chibicc), but use RISC-V ISA
