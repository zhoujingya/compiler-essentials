// RUN: smallcc -input-string="12" >& /tmp/test.s |
// RUN: /tmp/test.s FileCheck %s

// CHECK:  .globl main
// CHECK-NEXT: main:
// CHECK-NEXT: li a0, 12
// CHECK-NEXT: ret
