from numba import cuda

# 获取当前设备信息
device = cuda.get_current_device()
print("当前CUDA设备名称:", device.name)
print("设备计算能力:", device.compute_capability)
print("最大线程数/块:", device.MAX_THREADS_PER_BLOCK)
print("最大共享内存/块:", device.MAX_SHARED_MEMORY_PER_BLOCK, "字节")
print("时钟频率:", device.CLOCK_RATE / 1000, "MHz")
print("Warp大小:", device.WARP_SIZE)
print(
    "最大网格维度:", device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y, device.MAX_GRID_DIM_Z
)
print(
    "最大块维度:",
    device.MAX_BLOCK_DIM_X,
    device.MAX_BLOCK_DIM_Y,
    device.MAX_BLOCK_DIM_Z,
)

# 获取所有可用设备数量
print("\n可用的CUDA设备数量:", len(cuda.list_devices()))

# 列出所有设备信息
for i, device in enumerate(cuda.list_devices()):
    print(f"\n设备 {i}:")
    print("设备名称:", device.name)
    print("计算能力:", device.compute_capability)
