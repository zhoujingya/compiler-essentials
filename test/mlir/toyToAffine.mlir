// RUN: mlir-toy %s --emit=mlir-affine 2>&1 | FileCheck  %s
// RUN: mlir-toy %s -emit=run-jit 2>&1 | FileCheck -check-prefix=RUN-JIT %s
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}

// CHECK: module {
// CHECK-NEXT:   func.func @main() {
// CHECK-NEXT:     %cst = arith.constant 6.000000e+00 : f64
// CHECK-NEXT:     %cst_0 = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:     %cst_1 = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:     %cst_2 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:     %cst_3 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:     %cst_4 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:     %alloc = memref.alloc() : memref<3x2xf64>
// CHECK-NEXT:     %alloc_5 = memref.alloc() : memref<3x2xf64>
// CHECK-NEXT:     %alloc_6 = memref.alloc() : memref<2x3xf64>
// CHECK-NEXT:     affine.store %cst_4, %alloc_6[0, 0] : memref<2x3xf64>
// CHECK-NEXT:     affine.store %cst_3, %alloc_6[0, 1] : memref<2x3xf64>
// CHECK-NEXT:     affine.store %cst_2, %alloc_6[0, 2] : memref<2x3xf64>
// CHECK-NEXT:     affine.store %cst_1, %alloc_6[1, 0] : memref<2x3xf64>
// CHECK-NEXT:     affine.store %cst_0, %alloc_6[1, 1] : memref<2x3xf64>
// CHECK-NEXT:     affine.store %cst, %alloc_6[1, 2] : memref<2x3xf64>
// CHECK-NEXT:     affine.for %arg0 = 0 to 3 {
// CHECK-NEXT:       affine.for %arg1 = 0 to 2 {
// CHECK-NEXT:         %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>
// CHECK-NEXT:         affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg0 = 0 to 3 {
// CHECK-NEXT:       affine.for %arg1 = 0 to 2 {
// CHECK-NEXT:         %0 = affine.load %alloc_5[%arg0, %arg1] : memref<3x2xf64>
// CHECK-NEXT:         %1 = arith.mulf %0, %0 : f64
// CHECK-NEXT:         affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     toy.print %alloc : memref<3x2xf64>
// CHECK-NEXT:     memref.dealloc %alloc_6 : memref<2x3xf64>
// CHECK-NEXT:     memref.dealloc %alloc_5 : memref<3x2xf64>
// CHECK-NEXT:     memref.dealloc %alloc : memref<3x2xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// RUN-JIT: 1.000000 16.000000
// RUN-JIT-NEXT: 4.000000 25.000000
// RUN-JIT-NEXT: 9.000000 36.000000
