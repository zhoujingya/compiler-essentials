//smallcc -input-string="123" > /tmp/test.s
// riscv64-unknown-linux-gnu-gcc /tmp/test.s  -static -o /tmp/a.out
// qemu-riscv64 /tmp/a.out
// echo $? > /tmp/test.log >& /dev/null || true
// RUN: smallcc -input-string="123" > /tmp/test.s
// RUN: riscv64-unknown-linux-gnu-gcc /tmp/test.s  -static -o /tmp/a.out
// RUN: qemu-riscv64 /tmp/a.out | FileCheck %s

// CHECK: 123
