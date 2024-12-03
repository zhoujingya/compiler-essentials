// RUN: mlir-toy %s -emit=mlir  -opt=true 2>&1 | FileCheck  %s

// CHECK: module {
// CHECK-NEXT:   toy.func @muladd() {
// CHECK-NEXT:     %0 = toy.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %1 = toy.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %2 = toy.constant dense<3.000000e+00> : tensor<f64>
// CHECK-NEXT:     %3 = "toy.muladd"(%0, %1, %2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     toy.print %3 : tensor<f64>
// CHECK-NEXT:     toy.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

toy.func @muladd() {
  %0 = toy.constant dense<1.0> : tensor<f64>
  %1 = toy.constant dense<2.0> : tensor<f64>
  %2 = toy.constant dense<3.0> : tensor<f64>
  %3 = toy.add %0, %1 : tensor<f64>
  %4 = toy.mul %3, %2 : tensor<f64>
  toy.print %4 : tensor<f64>
  toy.return
}
