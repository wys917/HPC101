#!/usr/bin/env python3
"""
精确计算 Qwen3-8B 模型的理论显存占用
"""

import torch
from modules import Qwen3ForCausalLM

def calculate_theoretical_memory():
    """计算理论显存占用"""
    
    print("=" * 60)
    print("Qwen3-8B 模型理论显存计算")
    print("=" * 60)
    
    # 加载模型到CPU先计算参数
    MODEL_PATH = "/ocean/model/Qwen3-8B"
    
    print("正在加载模型到CPU进行参数统计...")
    model = Qwen3ForCausalLM.from_pretrained(
        model_path=MODEL_PATH,
        device="cpu",
        torch_dtype=torch.float32,  # 先用float32加载到CPU
    )
    
    # 精确计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,} ({total_params/1e9:.3f}B)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e9:.3f}B)")
    
    # 计算不同精度下的理论显存
    print("\n" + "=" * 40)
    print("不同精度下的理论显存占用:")
    print("=" * 40)
    
    # float32: 4 bytes per parameter
    memory_fp32 = total_params * 4 / (1024**3)
    print(f"Float32:   {memory_fp32:.3f} GB")
    
    # float16/bfloat16: 2 bytes per parameter  
    memory_fp16 = total_params * 2 / (1024**3)
    print(f"Float16:   {memory_fp16:.3f} GB")
    print(f"BFloat16:  {memory_fp16:.3f} GB  ← 实际使用")
    
    # int8: 1 byte per parameter
    memory_int8 = total_params * 1 / (1024**3)
    print(f"Int8:      {memory_int8:.3f} GB")
    
    print("\n" + "=" * 40)
    print("推理时额外显存开销估算:")
    print("=" * 40)
    
    # 推理时的额外开销
    # 1. KV Cache (估算)
    seq_len = 200  # 假设序列长度
    batch_size = 1
    num_layers = 32  # Qwen3-8B 有32层
    num_kv_heads = 4  # Qwen3-8B 的 KV heads
    head_dim = 128   # Qwen3-8B 的 head dimension
    
    # KV Cache: 2 (K+V) * batch * seq_len * num_layers * num_kv_heads * head_dim * 2_bytes
    kv_cache_size = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * 2 / (1024**3)
    print(f"KV Cache (seq_len={seq_len}): {kv_cache_size:.3f} GB")
    
    # 2. 前向传播中间激活值 (粗略估算)
    hidden_size = 3584  # Qwen3-8B hidden size
    intermediate_size = 18944  # Qwen3-8B intermediate size
    
    # 估算每层的激活值显存
    activation_per_layer = batch_size * seq_len * max(hidden_size, intermediate_size) * 2 / (1024**3)
    total_activation = activation_per_layer * 2  # 假设同时存储2层的激活值
    print(f"前向激活值 (估算): {total_activation:.3f} GB")
    
    # 3. 梯度和优化器状态 (仅推理时不需要)
    print(f"梯度存储: 0.000 GB (推理时不需要)")
    print(f"优化器状态: 0.000 GB (推理时不需要)")
    
    # 总计
    total_inference_memory = memory_fp16 + kv_cache_size + total_activation
    print(f"\n推理总显存估算: {total_inference_memory:.3f} GB")
    
    print("\n" + "=" * 40)
    print("与实际测量对比:")
    print("=" * 40)
    print(f"理论模型权重: {memory_fp16:.3f} GB")
    print(f"实际测量结果: ~14.88 GB")
    print(f"差异: {abs(memory_fp16 - 14.88):.3f} GB")
    print(f"差异百分比: {abs(memory_fp16 - 14.88)/14.88*100:.2f}%")
    
    if abs(memory_fp16 - 14.88) < 0.5:
        print("✅ 理论计算与实际测量高度吻合!")
    else:
        print("⚠️  理论与实际存在较大差异，可能的原因:")
        print("   - 模型结构定义不完全准确")
        print("   - GPU内存对齐和碎片化")
        print("   - PyTorch框架开销")

if __name__ == "__main__":
    calculate_theoretical_memory()
