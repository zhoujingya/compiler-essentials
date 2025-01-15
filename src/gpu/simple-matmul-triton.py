import triton
import triton.language as tl
import torch


@triton.jit
def matmul_kernel(
    # 指向矩阵的指针
    a_ptr,
    b_ptr,
    c_ptr,
    # 矩阵维度
    M,
    N,
    K,
):
    # 获取当前线程的位置
    print("--------")
    row = tl.program_id(0)
    col = tl.program_id(1)

    # 检查是否在矩阵范围内
    if row < M and col < N:
        # 初始化累加器
        acc = 0.0

        # 计算一个元素的结果
        for k in range(K):
            # 加载输入数据
            a = tl.load(a_ptr + row * K + k)
            b = tl.load(b_ptr + k * N + col)
            # 累加乘积
            acc += a * b

        # 存储结果
        tl.store(c_ptr + row * N + col, acc)


def matmul(a: torch.Tensor, b: torch.Tensor):
    # 检查输入
    assert a.shape[1] == b.shape[0], "不兼容的矩阵维度"
    M, K = a.shape
    K, N = b.shape

    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 计算网格大小 - 每个线程处理一个输出元素
    grid = lambda meta: (tl.constexpr(M), tl.constexpr(N))

    # 运行内核
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
    )

    return c


if __name__ == "__main__":
    # 创建示例矩阵
    torch.manual_seed(0)
    a = torch.randn(2, 3, device="cuda")
    b = torch.randn(3, 4, device="cuda")

    # 使用Triton计算
    c_triton = matmul(a, b)

    # 使用PyTorch计算
    c_torch = torch.matmul(a, b)
    assert torch.allclose(c_triton, c_torch, atol=1e-3, rtol=1e-3)
    # 验证结果
    print(f"最大误差: {torch.max(torch.abs(c_triton - c_torch))}")
    print("Matrix multiplication completed successfully!")
