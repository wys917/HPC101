# check_env.py
import torch
import sys

print("--- PyTorch CUDA Environment Interrogation Report ---")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 50)

# 1. 检查PyTorch认为的CUDA版本
print(f"PyTorch was compiled with CUDA version: {torch.version.cuda}")

# 2. 检查PyTorch是否能找到CUDA设备
is_available = torch.cuda.is_available()
print(f"Is CUDA available? : {is_available}")

if not is_available:
    print("\nERROR: PyTorch cannot find a CUDA-enabled GPU. Your PyTorch might be a CPU-only build.")
    sys.exit()

# 3. 如果能找到，查看设备详情
try:
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    current_device_id = torch.cuda.current_device()
    print(f"Current CUDA device ID: {current_device_id}")
    device_name = torch.cuda.get_device_name(current_device_id)
    print(f"Current CUDA device name: {device_name}")
    print("-" * 50)

    # 4. 最关键的测试：尝试在GPU上创建一个张量
    print("ACTION: Attempting to create a tensor on CUDA device 0...")
    x = torch.tensor([6, 6, 6]).to("cuda")
    print("SUCCESS: Tensor created on GPU.")
    print("Tensor value on GPU:", x)
    print("Now, CHECK nvidia-smi. A python process SHOULD be listed.")

except Exception as e:
    print(f"\nERROR: An error occurred during CUDA operations: {e}")

print("\n--- Interrogation Finished ---")