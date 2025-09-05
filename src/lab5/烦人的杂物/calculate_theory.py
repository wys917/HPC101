# calculate_theory.py

# --- 从 config.py 抄来的参数 ---
vocab_size = 152064
hidden_size = 4096
intermediate_size = 11008
num_hidden_layers = 32
num_attention_heads = 32
num_key_value_heads = 32
head_dim = hidden_size // num_attention_heads

# --- 1. 分步计算参数量 ---

# 词嵌入层
embedding_params = vocab_size * hidden_size

# 单个 Decoder Layer 的参数
# 注意力模块 (Qwen3Attention)
# Q, K, V, O 四个线性投影层
q_proj_params = hidden_size * (num_attention_heads * head_dim)
k_proj_params = hidden_size * (num_key_value_heads * head_dim)
v_proj_params = hidden_size * (num_key_value_heads * head_dim)
o_proj_params = (num_attention_heads * head_dim) * hidden_size
# 注意力模块里的两个 RMSNorm
q_norm_params = head_dim
k_norm_params = head_dim
attention_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params + q_norm_params + k_norm_params

# MLP 模块 (Qwen3MLP)
gate_proj_params = hidden_size * intermediate_size
up_proj_params = hidden_size * intermediate_size
down_proj_params = intermediate_size * hidden_size
mlp_params = gate_proj_params + up_proj_params + down_proj_params

# Decoder Layer 里的两个 RMSNorm
input_layernorm_params = hidden_size
post_attention_layernorm_params = hidden_size

one_decoder_layer_params = attention_params + mlp_params + input_layernorm_params + post_attention_layernorm_params

# 所有 Decoder Layer 的总参数
total_decoder_layers_params = num_hidden_layers * one_decoder_layer_params

# 最终的 RMSNorm 层
final_norm_params = hidden_size

# 输出层 (LM Head)
lm_head_params = hidden_size * vocab_size

# --- 2. 汇总并计算理论显存 ---

# 手动计算的总参数量
manual_total_params = embedding_params + total_decoder_layers_params + final_norm_params + lm_head_params

# 每个参数 bfloat16 占用 2 字节
bytes_per_param = 2
theoretical_memory_bytes = manual_total_params * bytes_per_param
theoretical_memory_gb = theoretical_memory_bytes / (1024**3)

print("--- 理论计算结果 ---")
print(f"手动计算的总参数量: {manual_total_params / 1_000_000:.2f} M")
print(f"理论显存占用: {theoretical_memory_gb:.2f} GB")
print("\n")


# --- 3. 用 PyTorch 自动验证 ---
import torch
from modules import Qwen3ForCausalLM

print("--- PyTorch 自动验证 ---")
model = Qwen3ForCausalLM.from_pretrained(
    model_path="/ocean/model/Qwen3-8B", # 这里改成你的模型路径
    device="cpu", # 先放CPU，不算显存
    torch_dtype=torch.bfloat16,
)
auto_total_params = sum(p.numel() for p in model.parameters())

print(f"PyTorch 自动计算的总参数量: {auto_total_params / 1_000_000:.2f} M")

# 检查手动和自动计算是否一致
if abs(manual_total_params - auto_total_params) < 1000: # 加个容差
    print("✅ 计算结果一致，公式正确！")
else:
    print("❌ 计算结果不一致，检查你的公式！")