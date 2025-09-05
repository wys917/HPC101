# measure_memory_final.py
import torch
import time
from modules import Qwen3ForCausalLM

# --- 配置 ---
MODEL_PATH = "/ocean/model/Qwen3-8B"  # 确认这是你的模型路径
DEVICE = "cuda"
SLEEP_SECONDS = 600  # 暂停600秒（10分钟），足够你操作了

# --- 打印信息 ---
print("="*50)
print("Final Static Memory Measurement Script")
print(f"Model Path: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print("="*50)

try:
    # --- 加载模型 ---
    print(f"INFO: Loading model to '{DEVICE}'... This will take a moment.")
    model = Qwen3ForCausalLM.from_pretrained(
        model_path=MODEL_PATH,
        device=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    print("SUCCESS: Model has been loaded to GPU.")
    
    # --- 等待测量 ---
    print(f"ACTION: The script will now pause for {SLEEP_SECONDS} seconds.")
    print("ACTION: Please log in to the compute node and run 'nvtop' to take your measurement screenshot.")
    
    time.sleep(SLEEP_SECONDS)
    
    print("INFO: Pause finished. Script is exiting.")

except Exception as e:
    print(f"ERROR: An error occurred: {e}")