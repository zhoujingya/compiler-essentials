// RUN: mlir-toy %s -emit=mlir -opt 2>&1 | FileCheck  %s
// RUN: mlir-toy %s -emit=run-jit 2>&1 | FileCheck -check-prefix=RUN-JIT %s
// REQUIRES: tiny-tblgen
//CHECK: module {
//CHECK-NEXT:   toy.func @main() {
//CHECK-NEXT:     %0 = toy.constant dense<1.000000e+00> : tensor<f64>
//CHECK-NEXT:     %1 = toy.constant dense<2.000000e+00> : tensor<f64>
//CHECK-NEXT:     %2 = toy.constant dense<3.000000e+00> : tensor<f64>
//CHECK-NEXT:     %3 = toy.muladd %0 : tensor<f64>, %1 : tensor<f64>, %2 : tensor<f64> -> tensor<f64>
//CHECK-NEXT:     %4 = toy.muladd %0 : tensor<f64>, %1 : tensor<f64>, %3 : tensor<f64> -> tensor<f64>
//CHECK-NEXT:     toy.print %4 : tensor<f64>
//CHECK-NEXT:     toy.return
//CHECK-NEXT:   }
//CHECK-NEXT: }

toy.func @main() {
  %0 = toy.constant dense<1.0> : tensor<f64>
  %1 = toy.constant dense<2.0> : tensor<f64>
  %2 = toy.constant dense<3.0> : tensor<f64>
  %3 = toy.mul %0, %1 : tensor<f64>
  %4 = toy.add %3, %2 : tensor<f64>
  %5 = toy.muladd %0 : tensor<f64>, %1 : tensor<f64>, %4 : tensor<f64> -> tensor<f64>
  toy.print %5 : tensor<f64>
  toy.return
}

// RUN-JIT: 7.000000
