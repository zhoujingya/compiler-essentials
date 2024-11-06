// RUN: smallcc -input-string="123" > /tmp/test.s
// RUN: riscv64-unknown-linux-gnu-gcc /tmp/test.s  -static -o /tmp/a.out
// RUN: python3 %utils_path/qemu-test.py /tmp/a.out 123
// REQUIRES: smallcc
