#include <iostream>

// 这个算法实现了矩阵乘法。它接受三个矩阵A、B和C，以及它们的维度m、k和n。
// A是一个m行k列的矩阵，B是一个k行n列的矩阵，C是一个m行n列的矩阵。
// 算法通过三个嵌套的for循环来计算矩阵A和矩阵B的乘积，并将结果存储在矩阵C中。
// 外层循环遍历矩阵A的行，中间循环遍历矩阵B的列，内层循环计算矩阵A的行和矩阵B的列的点积。
// A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0] + A[0][3] * B[3][0]
// = C[0][0] A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] * B[2][1] + A[0][3]
// * B[3][1] = C[0][1] A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] * B[2][0]
// + A[1][3] * B[3][0] = C[1][0] 以此类推，计算出C矩阵的所有元素。
// 这个算法的时间复杂度是O(m*k*n)，空间复杂度是O(m*n)。
// 这个算法是矩阵乘法的标准实现，可以用于任何矩阵乘法的实现。

void matMul(float *A, float *B, float *C, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      for (int l = 0; l < n; l++) {
        C[i * n + l] += A[i * k + j] * B[j * n + l];
      }
    }
  }
}

int main() {
  // m = 3, k = 4, n = 3
  float a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};     // 3 * 4
  float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};     // 4 * 3
  float dst[9] = {70, 80, 90, 158, 184, 210, 246, 288, 330}; // 3 * 3
  float *c = new float[9];
  matMul(a, b, c, 3, 4, 3);
  int fail = 0;
  for (int i = 0; i < 9; i++) {
    if (c[i] != dst[i])
      fail++;
  }
  if (fail)
    std::cout << "Matrix multiply error\n";
  else
    std::cout << "Matrix multiply succeed\n";
  delete[] c;
  return 0;
}
