## How to build

`cmake -B build -G Ninja`

`cmake --build build`

`./install-riscv.sh` to install `riscv` toolchain

> Build debug version llvm, then export toolchain binary path

## [tools](tools)

### build my-exegesis with libpfm

The my-exegesis tool depends on the libpfm. If the path to libpfm is not provided, the compiled binary will not work.

You can install libpfm from [here](https://github.com/wcohen/libpfm4). After installing it, configure your build by specifying the installation path to it.

`cmake -B build -G Ninja -DPFM_PATH=/path/to/pfm/install`

## How to test

`lit build/test`


## [small cc](src/smallcc)

base on [chibicc](https://github.com/rui314/chibicc), but use RISC-V ISA
