#!/usr/bin/env python3
"""
Qwen3模型参数量和显存占用计算脚本

本脚本用于：
1. 理论计算Qwen3模型的参数量
2. 计算理论显存占用
3. 查看实际显存占用
4. 对比分析差异
"""

import torch
import subprocess
import os
from modules import Qwen3Config, Qwen3ForCausalLM


def calculate_theoretical_params():
    """计算理论参数量"""
    config = Qwen3Config()
    
    print("=" * 60)
    print("🧮 Qwen3-8B 模型理论参数量计算")
    print("=" * 60)
    
    # 基础配置
    vocab_size = config.vocab_size           # 152,064
    hidden_size = config.hidden_size         # 4,096
    intermediate_size = config.intermediate_size  # 11,008
    num_layers = config.num_hidden_layers    # 32
    num_heads = config.num_attention_heads   # 32
    num_kv_heads = config.num_key_value_heads # 32
    head_dim = hidden_size // num_heads      # 128
    
    print(f"模型配置:")
    print(f"  词汇表大小: {vocab_size:,}")
    print(f"  隐藏层维度: {hidden_size:,}")
    print(f"  FFN中间维度: {intermediate_size:,}")
    print(f"  层数: {num_layers}")
    print(f"  注意力头数: {num_heads}")
    print(f"  KV头数: {num_kv_heads}")
    print()
    
    # 1. Embedding层参数
    print("📝 1. Embedding层参数计算:")
    embedding_params = vocab_size * hidden_size
    print(f"  Word Embedding: {vocab_size:,} × {hidden_size:,} = {embedding_params:,}")
    
    # 2. 每个Decoder Layer的参数
    print("\n🏗️ 2. 每个Decoder Layer参数计算:")
    
    # 2.1 Self-Attention参数
    print("  2.1 Self-Attention:")
    # QKV投影: hidden_size → (Q_dim + K_dim + V_dim)
    q_dim = num_heads * head_dim      # 32 * 128 = 4096
    k_dim = num_kv_heads * head_dim   # 32 * 128 = 4096 (标准MHA，不是GQA)
    v_dim = num_kv_heads * head_dim   # 32 * 128 = 4096
    qkv_dim = q_dim + k_dim + v_dim   # 4096 + 4096 + 4096 = 12288
    
    qkv_proj_params = hidden_size * qkv_dim
    print(f"    QKV投影: {hidden_size:,} × {qkv_dim:,} = {qkv_proj_params:,}")
    
    # 输出投影: hidden_size → hidden_size
    o_proj_params = hidden_size * hidden_size
    print(f"    输出投影: {hidden_size:,} × {hidden_size:,} = {o_proj_params:,}")
    
    # Q/K归一化层 (RMSNorm)
    qk_norm_params = 2 * num_heads * head_dim  # Q_norm + K_norm
    print(f"    Q/K归一化: 2 × {num_heads} × {head_dim} = {qk_norm_params:,}")
    
    attention_params = qkv_proj_params + o_proj_params + qk_norm_params
    print(f"    Self-Attention总计: {attention_params:,}")
    
    # 2.2 MLP参数
    print("  2.2 MLP (SwiGLU):")
    # Gate投影: hidden_size → intermediate_size
    gate_proj_params = hidden_size * intermediate_size
    print(f"    Gate投影: {hidden_size:,} × {intermediate_size:,} = {gate_proj_params:,}")
    
    # Up投影: hidden_size → intermediate_size  
    up_proj_params = hidden_size * intermediate_size
    print(f"    Up投影: {hidden_size:,} × {intermediate_size:,} = {up_proj_params:,}")
    
    # Down投影: intermediate_size → hidden_size
    down_proj_params = intermediate_size * hidden_size
    print(f"    Down投影: {intermediate_size:,} × {hidden_size:,} = {down_proj_params:,}")
    
    mlp_params = gate_proj_params + up_proj_params + down_proj_params
    print(f"    MLP总计: {mlp_params:,}")
    
    # 2.3 LayerNorm参数
    print("  2.3 LayerNorm:")
    layernorm_params = 2 * hidden_size  # input_layernorm + post_attention_layernorm
    print(f"    LayerNorm × 2: 2 × {hidden_size:,} = {layernorm_params:,}")
    
    # 每层总参数
    layer_params = attention_params + mlp_params + layernorm_params
    print(f"  单层总计: {layer_params:,}")
    
    # 3. 所有层的参数
    print(f"\n🔢 3. 所有层参数计算:")
    total_layer_params = num_layers * layer_params
    print(f"  {num_layers}层总计: {num_layers} × {layer_params:,} = {total_layer_params:,}")
    
    # 4. 最终输出层参数
    print(f"\n📤 4. 最终输出层参数:")
    # Final LayerNorm
    final_norm_params = hidden_size
    print(f"  Final LayerNorm: {final_norm_params:,}")
    
    # LM Head (语言模型头)
    if config.tie_word_embeddings:
        lm_head_params = 0  # 共享embedding权重
        print(f"  LM Head: 共享Embedding权重, 0参数")
    else:
        lm_head_params = hidden_size * vocab_size
        print(f"  LM Head: {hidden_size:,} × {vocab_size:,} = {lm_head_params:,}")
    
    output_params = final_norm_params + lm_head_params
    print(f"  输出层总计: {output_params:,}")
    
    # 5. 总参数量
    print(f"\n🎯 5. 总参数量:")
    total_params = embedding_params + total_layer_params + output_params
    print(f"  Embedding: {embedding_params:,}")
    print(f"  Decoder Layers: {total_layer_params:,}")
    print(f"  输出层: {output_params:,}")
    print(f"  总计: {total_params:,}")
    print(f"  约 {total_params / 1e9:.2f}B 参数")
    
    return total_params


def calculate_theoretical_memory(total_params):
    """计算理论显存占用"""
    print("\n" + "=" * 60)
    print("💾 理论显存占用计算 (bfloat16)")
    print("=" * 60)
    
    # bfloat16: 每个参数2字节
    bytes_per_param = 2
    
    # 模型参数显存
    model_memory_bytes = total_params * bytes_per_param
    model_memory_gb = model_memory_bytes / (1024**3)
    
    print(f"模型参数显存:")
    print(f"  参数量: {total_params:,}")
    print(f"  数据类型: bfloat16 (2 bytes/param)")
    print(f"  参数显存: {total_params:,} × 2 = {model_memory_bytes:,} bytes")
    print(f"  参数显存: {model_memory_gb:.2f} GB")
    
    # 激活显存估算 (推理时)
    # 假设batch_size=1, seq_len=1024
    batch_size = 1
    seq_len = 1024
    hidden_size = 4096
    
    # 主要激活显存来源
    print(f"\n激活显存估算 (batch={batch_size}, seq_len={seq_len}):")
    
    # Hidden states
    hidden_memory = batch_size * seq_len * hidden_size * bytes_per_param * 32  # 32层
    hidden_memory_gb = hidden_memory / (1024**3)
    print(f"  Hidden States: {hidden_memory_gb:.2f} GB")
    
    # Attention weights
    attention_memory = batch_size * 32 * seq_len * seq_len * bytes_per_param * 32  # 32头×32层
    attention_memory_gb = attention_memory / (1024**3)
    print(f"  Attention Weights: {attention_memory_gb:.2f} GB")
    
    # KV Cache (推理时)
    kv_cache_memory = batch_size * seq_len * hidden_size * bytes_per_param * 2 * 32  # K+V×32层
    kv_cache_memory_gb = kv_cache_memory / (1024**3)
    print(f"  KV Cache: {kv_cache_memory_gb:.2f} GB")
    
    total_activation_gb = hidden_memory_gb + attention_memory_gb + kv_cache_memory_gb
    print(f"  激活总计: {total_activation_gb:.2f} GB")
    
    # 总显存需求
    total_memory_gb = model_memory_gb + total_activation_gb
    print(f"\n总显存需求:")
    print(f"  模型参数: {model_memory_gb:.2f} GB")
    print(f"  激活内存: {total_activation_gb:.2f} GB") 
    print(f"  理论总计: {total_memory_gb:.2f} GB")
    
    return model_memory_gb, total_memory_gb


def get_actual_memory_usage():
    """获取实际显存占用"""
    print("\n" + "=" * 60)
    print("📊 实际显存占用查询")
    print("=" * 60)
    
    try:
        # 查看GPU信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_name = parts[0]
                    total_mem = int(parts[1])
                    used_mem = int(parts[2])
                    free_mem = int(parts[3])
                    
                    print(f"GPU {i}: {gpu_name}")
                    print(f"  总显存: {total_mem:,} MB ({total_mem/1024:.2f} GB)")
                    print(f"  已用显存: {used_mem:,} MB ({used_mem/1024:.2f} GB)")
                    print(f"  可用显存: {free_mem:,} MB ({free_mem/1024:.2f} GB)")
                    print(f"  使用率: {used_mem/total_mem*100:.1f}%")
                    
        else:
            print("无法获取GPU信息，可能未安装nvidia-smi或无可用GPU")
            
    except FileNotFoundError:
        print("nvidia-smi命令未找到，可能未安装NVIDIA驱动")
    
    # 使用PyTorch查看显存
    if torch.cuda.is_available():
        print(f"\nPyTorch显存信息:")
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            
            print(f"  GPU {i}:")
            print(f"    总显存: {total_mem:.2f} GB")
            print(f"    已分配: {allocated:.2f} GB") 
            print(f"    已缓存: {cached:.2f} GB")
    else:
        print("PyTorch未检测到可用的CUDA设备")


def verify_with_actual_model():
    """用实际模型验证参数量"""
    print("\n" + "=" * 60)
    print("🔍 实际模型参数量验证")
    print("=" * 60)
    
    try:
        # 创建模型但不加载权重
        config = Qwen3Config()
        model = Qwen3ForCausalLM(config)
        
        # 计算实际参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"实际模型参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  约 {total_params / 1e9:.2f}B 参数")
        
        # 按模块统计
        print(f"\n按模块参数统计:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module_params:,} ({module_params/total_params*100:.1f}%)")
            
        return total_params
        
    except Exception as e:
        print(f"创建模型时出错: {e}")
        return None


def main():
    """主函数"""
    print("Qwen3-8B 模型分析工具")
    print("分析包括：参数量计算、显存占用估算、实际使用情况对比")
    
    # 1. 理论计算
    theoretical_params = calculate_theoretical_params()
    
    # 2. 显存占用计算  
    model_memory_gb, total_memory_gb = calculate_theoretical_memory(theoretical_params)
    
    # 3. 实际参数量验证
    actual_params = verify_with_actual_model()
    
    # 4. 实际显存使用情况
    get_actual_memory_usage()
    
    # 5. 对比分析
    print("\n" + "=" * 60)
    print("📈 对比分析")
    print("=" * 60)
    
    if actual_params:
        diff = abs(theoretical_params - actual_params)
        diff_percent = diff / theoretical_params * 100
        print(f"参数量对比:")
        print(f"  理论计算: {theoretical_params:,}")
        print(f"  实际统计: {actual_params:,}")
        print(f"  差异: {diff:,} ({diff_percent:.2f}%)")
        
        if diff_percent < 1:
            print("  ✅ 理论计算与实际基本一致")
        else:
            print("  ❓ 理论计算与实际存在差异，可能原因:")
            print("     - 某些模块的参数计算遗漏")
            print("     - 配置参数与实际实现不一致")
            print("     - 额外的bias参数或特殊层")


if __name__ == "__main__":
    main()
