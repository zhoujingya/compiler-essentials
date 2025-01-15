from numba import cuda
import numpy as np


@cuda.jit
def matrix_multiply(A, B, C):
    # 获取当前线程的索引
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # 检查索引是否在矩阵范围内
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        # 计算矩阵乘法的一个元素
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


# 创建示例矩阵
A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5, 6], [7, 8]], dtype=np.float32)
C = np.zeros((2, 2), dtype=np.float32)

# 将数据复制到GPU
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.to_device(C)

# 定义线程块和网格大小
threadsperblock = (2, 2)
blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# 执行核函数
matrix_multiply[blockspergrid, threadsperblock](d_A, d_B, d_C)

# 将结果复制回主机
result = d_C.copy_to_host()

print(result)
