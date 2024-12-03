// RUN: mlir-toy %s --emit=mlir-affine 2>&1 | FileCheck  %s
toy.func @muladd() {
  %0 = toy.constant dense<123.0> : tensor<f64>
  %1 = toy.constant dense<132.0> : tensor<f64>
  %2 = toy.constant dense<33.0> : tensor<f64>
  %3 = toy.add %0, %1 : tensor<f64>
  %4 = toy.mul %2, %3 : tensor<f64>
  toy.print %4 : tensor<f64>
  toy.return
}
