# final_check.py
import torch
import sys
import time
from modules import Qwen3ForCausalLM

print("--- Final Model Loading Verification ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA available?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("="*50)

# --- 配置 ---
device = "cuda"
model_path = "/ocean/model/Qwen3-8B" # 确认路径

try:
    print(f"[{time.ctime()}] ACTION: Attempting to load model with device='{device}'...")
    
    model = Qwen3ForCausalLM.from_pretrained(
        model_path=model_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    print(f"[{time.ctime()}] SUCCESS: 'from_pretrained' call completed without error.")
    print("-" * 50)
    
    # --- 最关键的检查：直接查询模型参数所在的设备 ---
    
    # 检查最外层的 lm_head
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        final_layer_device = model.lm_head.weight.device
        print(f"CRITICAL CHECK 1: Model's output layer (lm_head) is on device: 【{final_layer_device}】")
    else:
        print("CRITICAL CHECK 1: Could not find model.lm_head.weight.")

    # 检查模型深层的一个参数
    if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
        deep_layer_device = model.model.layers[0].input_layernorm.weight.device
        print(f"CRITICAL CHECK 2: Model's first decoder layer is on device: 【{deep_layer_device}】")
    else:
        print("CRITICAL CHECK 2: Could not find a deep layer parameter to check.")

    print("-" * 50)

    if 'cpu' in str(final_layer_device):
        print("\nCASE CLOSED: The model was loaded to CPU, ignoring the 'device=cuda' argument.")
        print("Your 'from_pretrained' function is bugged.")
    else:
        print("\nVERDICT: The model claims to be on the GPU. This is truly bizarre.")


except Exception as e:
    print(f"[{time.ctime()}] ERROR: An error occurred during model loading: {e}")

print("\n--- Final Check Finished ---")