import torch
import time
import numpy as np
from ultralytics import YOLO

model_path = "/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/runs/detect/train6/weights/best.pt"
model = YOLO(model_path)

input_shape = (1, 6, 640, 640)  # 确认你的模型训练时就是6通道输入
input_data = torch.from_numpy(np.random.rand(*input_shape).astype(np.float32))

def benchmark(model, input_data, device='cpu', iterations=1000):
    input_data = input_data.to(device)
    # ultralytics YOLO 不需要 model.eval()，也不能 .to(device)
    with torch.no_grad():
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_data)
        return time.time() - start_time

# 1. CPU 推理
cpu_iterations = 1000
cpu_time = benchmark(model, input_data, device='cpu', iterations=cpu_iterations)
print(f"CPU 推理总时间: {cpu_time:.4f} 秒, 每次推理平均时间: {cpu_time / cpu_iterations:.4f} 秒")

# 2. GPU 推理（如果有GPU）
if torch.cuda.is_available():
    input_data_gpu = input_data.to('cuda')
    gpu_iterations = 1000
    gpu_time = benchmark(model, input_data_gpu, device='cuda', iterations=gpu_iterations)
    print(f"GPU 推理总时间: {gpu_time:.4f} 秒, 每次推理平均时间: {gpu_time / gpu_iterations:.4f} 秒")
    speedup = cpu_time / gpu_time
    print(f"GPU 加速比: {speedup:.2f} 倍")
else:
    print("未检测到可用的GPU，跳过GPU测试。")