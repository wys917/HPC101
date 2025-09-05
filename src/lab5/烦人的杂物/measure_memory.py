# measure_memory.py (调试版)
import time
import os

# --- 眼线 1: 脚本启动 ---
print(f"[{time.ctime()}] --- Script started ---")

print(f"[{time.ctime()}] Importing torch...")
import torch
print(f"[{time.ctime()}] torch imported successfully.")

print(f"[{time.ctime()}] Importing Qwen3ForCausalLM...")
from modules import Qwen3ForCausalLM
print(f"[{time.ctime()}] Qwen3ForCausalLM imported successfully.")


# --- 配置 ---
device = "cuda"
model_path = "/ocean/model/Qwen3-8B" # 确认这是你的模型路径
sleep_duration_seconds = 300

# --- 打印节点信息 ---
hostname = os.uname().nodename
print(f"[{time.ctime()}] INFO: Code is running on node: {hostname}")
print("="*50)

# --- 眼线 2: 加载模型前 ---
print(f"[{time.ctime()}] ACTION: About to load model to GPU. This may take a while...")

# --- 加载模型 ---
try:
    model = Qwen3ForCausalLM.from_pretrained(
        model_path=model_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    # --- 眼线 3: 加载模型后 ---
    print(f"[{time.ctime()}] SUCCESS: Model loaded to GPU!")
    print(f"ACTION: Please SSH to {hostname} and run 'watch -n 1 nvidia-smi'")
    
    time.sleep(sleep_duration_seconds)

    print(f"\n[{time.ctime()}] INFO: Sleep finished. Exiting.")

except Exception as e:
    print(f"[{time.ctime()}] ERROR: An error occurred: {e}")