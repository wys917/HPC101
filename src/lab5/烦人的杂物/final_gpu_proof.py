# final_gpu_proof.py
import torch
import sys
import time
from modules import Qwen3ForCausalLM

print("--- Final GPU Visibility Proof ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA available?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("="*50)

device = "cuda"
model_path = "/ocean/model/Qwen3-8B"

try:
    print(f"[{time.ctime()}] ACTION: Loading model...")
    
    model = Qwen3ForCausalLM.from_pretrained(
        model_path=model_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    print(f"[{time.ctime()}] SUCCESS: Model loaded to device: {model.model.layers[0].input_layernorm.weight.device}")
    print("-" * 50)
    
    print(f"[{time.ctime()}] ACTION: Entering ACTIVE GPU loop to stay visible to nvidia-smi...")
    print("ACTION: Please run 'watch -n 1 nvidia-smi' on the node NOW.")

    # 在GPU上创建一个小张量
    a = torch.randn(100, 100, device="cuda")
    
    # 用一个死循环不停地在GPU上做无用功，而不是sleep
    while True:
        # 执行一个真实的GPU计算任务
        a = torch.matmul(a, a)
        time.sleep(1) # 每秒算一次，别把GPU搞得太累

except Exception as e:
    print(f"[{time.ctime()}] ERROR: An error occurred: {e}")