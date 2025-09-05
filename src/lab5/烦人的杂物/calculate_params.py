#!/usr/bin/env python3
"""
Qwen3模型参数量和显存占用计算脚本
"""

import torch
import numpy as np
from modules import Qwen3Config, Qwen3ForCausalLM
import os
import gc

def format_number(num):
    """格式化数字为易读形式"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def calculate_theoretical_params():
    """理论计算参数量"""
    print("=" * 60)
    print("🧮 理论参数量计算")
    print("=" * 60)
    
    # 从配置文件获取参数
    config = Qwen3Config()
    
    print(f"模型配置:")
    print(f"  vocab_size: {config.vocab_size:,}")
    print(f"  hidden_size: {config.hidden_size:,}")
    print(f"  intermediate_size: {config.intermediate_size:,}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print()
    
    # 1. Word Embedding层参数
    embedding_params = config.vocab_size * config.hidden_size
    print(f"1. Word Embedding层:")
    print(f"   vocab_size × hidden_size = {config.vocab_size:,} × {config.hidden_size:,}")
    print(f"   参数量: {embedding_params:,} ({format_number(embedding_params)})")
    print()
    
    # 2. 每个Decoder Layer的参数量
    print(f"2. 单个Decoder Layer参数量:")
    
    # 2.1 Self-Attention参数
    # QKV投影: hidden_size → (Q_dim + K_dim + V_dim)
    q_dim = config.num_attention_heads * (config.hidden_size // config.num_attention_heads)  # 4096
    kv_dim = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)  # 1024
    qkv_params = config.hidden_size * (q_dim + kv_dim + kv_dim)  # 4096 * (4096 + 1024 + 1024)
    
    # 输出投影: hidden_size → hidden_size
    o_proj_params = config.hidden_size * config.hidden_size
    
    # Q/K归一化层参数
    qk_norm_params = 2 * config.hidden_size  # q_norm + k_norm
    
    attention_params = qkv_params + o_proj_params + qk_norm_params
    
    print(f"   Self-Attention:")
    print(f"     QKV投影: {config.hidden_size:,} × {q_dim + 2*kv_dim:,} = {qkv_params:,}")
    print(f"     输出投影: {config.hidden_size:,} × {config.hidden_size:,} = {o_proj_params:,}")
    print(f"     Q/K归一化: 2 × {config.hidden_size:,} = {qk_norm_params:,}")
    print(f"     小计: {attention_params:,} ({format_number(attention_params)})")
    print()
    
    # 2.2 MLP (SwiGLU) 参数
    gate_params = config.hidden_size * config.intermediate_size
    up_params = config.hidden_size * config.intermediate_size  
    down_params = config.intermediate_size * config.hidden_size
    
    mlp_params = gate_params + up_params + down_params
    
    print(f"   MLP (SwiGLU):")
    print(f"     Gate投影: {config.hidden_size:,} × {config.intermediate_size:,} = {gate_params:,}")
    print(f"     Up投影: {config.hidden_size:,} × {config.intermediate_size:,} = {up_params:,}")
    print(f"     Down投影: {config.intermediate_size:,} × {config.hidden_size:,} = {down_params:,}")
    print(f"     小计: {mlp_params:,} ({format_number(mlp_params)})")
    print()
    
    # 2.3 LayerNorm参数
    layernorm_params = 2 * config.hidden_size  # input_layernorm + post_attention_layernorm
    
    print(f"   LayerNorm:")
    print(f"     2层 × {config.hidden_size:,} = {layernorm_params:,}")
    print()
    
    # 单层总参数
    single_layer_params = attention_params + mlp_params + layernorm_params
    print(f"   单层总参数: {single_layer_params:,} ({format_number(single_layer_params)})")
    print()
    
    # 3. 所有Decoder Layers参数
    all_layers_params = single_layer_params * config.num_hidden_layers
    print(f"3. 所有{config.num_hidden_layers}层Decoder参数:")
    print(f"   {single_layer_params:,} × {config.num_hidden_layers} = {all_layers_params:,} ({format_number(all_layers_params)})")
    print()
    
    # 4. 最终输出层参数 (Language Model Head)
    lm_head_params = config.hidden_size * config.vocab_size
    print(f"4. Language Model Head:")
    print(f"   hidden_size × vocab_size = {config.hidden_size:,} × {config.vocab_size:,}")
    print(f"   参数量: {lm_head_params:,} ({format_number(lm_head_params)})")
    print()
    
    # 5. Final LayerNorm
    final_norm_params = config.hidden_size
    print(f"5. Final LayerNorm:")
    print(f"   参数量: {final_norm_params:,}")
    print()
    
    # 总参数量
    total_params = embedding_params + all_layers_params + lm_head_params + final_norm_params
    
    print("=" * 60)
    print("📊 参数量汇总:")
    print("=" * 60)
    print(f"Word Embedding:     {embedding_params:,} ({format_number(embedding_params)})")
    print(f"Decoder Layers:     {all_layers_params:,} ({format_number(all_layers_params)})")
    print(f"LM Head:           {lm_head_params:,} ({format_number(lm_head_params)})")
    print(f"Final LayerNorm:   {final_norm_params:,}")
    print("-" * 60)
    print(f"总参数量:          {total_params:,} ({format_number(total_params)})")
    print("=" * 60)
    
    return total_params

def calculate_memory_usage(total_params):
    """计算理论显存占用"""
    print("\n" + "=" * 60)
    print("💾 理论显存占用计算")
    print("=" * 60)
    
    # bfloat16: 每个参数2字节
    bytes_per_param = 2
    
    # 模型参数显存
    model_memory_bytes = total_params * bytes_per_param
    model_memory_gb = model_memory_bytes / (1024**3)
    
    print(f"数据类型: bfloat16 (每参数{bytes_per_param}字节)")
    print(f"模型参数显存: {total_params:,} × {bytes_per_param} = {model_memory_bytes:,} bytes")
    print(f"模型参数显存: {model_memory_gb:.2f} GB")
    print()
    
    # 推理时的额外显存开销估算
    print("推理时额外显存开销估算:")
    
    # 激活值显存 (估算)
    batch_size = 1
    seq_len = 100  # 假设序列长度
    hidden_size = 4096
    
    # 每层的激活值
    activation_per_layer = batch_size * seq_len * hidden_size * bytes_per_param
    total_activation = activation_per_layer * 32  # 32层
    activation_gb = total_activation / (1024**3)
    
    print(f"  激活值显存 (估算): {activation_gb:.2f} GB")
    
    # 注意力矩阵显存
    attention_matrix = batch_size * 32 * seq_len * seq_len * bytes_per_param  # 32个注意力头
    attention_gb = attention_matrix / (1024**3)
    print(f"  注意力矩阵显存: {attention_gb:.2f} GB")
    
    # 总显存估算
    inference_total_gb = model_memory_gb + activation_gb + attention_gb + 1.0  # +1GB buffer
    
    print("\n" + "-" * 40)
    print(f"推理时总显存估算: {inference_total_gb:.2f} GB")
    print("=" * 60)
    
    return model_memory_gb, inference_total_gb

def test_actual_model():
    """测试实际模型参数量和显存占用"""
    print("\n" + "=" * 60)
    print("🔬 实际模型测试")
    print("=" * 60)
    
    try:
        # 获取GPU信息
        if torch.cuda.is_available():
            print(f"GPU设备: {torch.cuda.get_device_name()}")
            print(f"GPU总显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("❌ CUDA不可用，使用CPU进行测试")
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 记录初始显存
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"初始显存占用: {initial_memory:.2f} GB")
        
        # 创建模型配置
        print("\n创建模型...")
        config = Qwen3Config()
        
        # 创建模型 (暂时在CPU上)
        model = Qwen3ForCausalLM(config)
        
        # 计算实际参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"实际总参数量: {total_params:,} ({format_number(total_params)})")
        print(f"可训练参数量: {trainable_params:,} ({format_number(trainable_params)})")
        
        # 详细参数分解
        print("\n详细参数分解:")
        for name, param in model.named_parameters():
            if param.numel() > 1000000:  # 只显示大于1M参数的层
                print(f"  {name}: {param.numel():,} ({format_number(param.numel())})")
        
        if torch.cuda.is_available():
            # 移动到GPU并测试显存
            print("\n移动模型到GPU...")
            model = model.cuda()
            model = model.to(torch.bfloat16)  # 转换为bfloat16
            
            # 记录加载后显存
            after_load_memory = torch.cuda.memory_allocated() / (1024**3)
            model_memory_actual = after_load_memory - initial_memory
            print(f"模型加载后显存: {after_load_memory:.2f} GB")
            print(f"模型实际占用显存: {model_memory_actual:.2f} GB")
            
            # 推理测试
            print("\n进行推理测试...")
            model.eval()
            
            with torch.no_grad():
                # 创建输入
                batch_size = 1
                seq_len = 100
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
                
                # 前向传播
                outputs = model(input_ids)
                
                # 记录推理时显存
                inference_memory = torch.cuda.memory_allocated() / (1024**3)
                inference_overhead = inference_memory - after_load_memory
                
                print(f"推理时总显存: {inference_memory:.2f} GB")
                print(f"推理额外开销: {inference_overhead:.2f} GB")
                
                return total_params, model_memory_actual, inference_memory
        else:
            return total_params, None, None
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    print("🚀 Qwen3模型参数量和显存分析")
    print("=" * 80)
    
    # 1. 理论计算
    theoretical_params = calculate_theoretical_params()
    theoretical_model_memory, theoretical_inference_memory = calculate_memory_usage(theoretical_params)
    
    # 2. 实际测试
    actual_params, actual_model_memory, actual_inference_memory = test_actual_model()
    
    # 3. 对比分析
    print("\n" + "=" * 60)
    print("📊 理论vs实际对比分析")
    print("=" * 60)
    
    if actual_params is not None:
        # 参数量对比
        param_diff = abs(theoretical_params - actual_params)
        param_diff_pct = (param_diff / theoretical_params) * 100
        
        print(f"参数量对比:")
        print(f"  理论值: {theoretical_params:,} ({format_number(theoretical_params)})")
        print(f"  实际值: {actual_params:,} ({format_number(actual_params)})")
        print(f"  差异:   {param_diff:,} ({param_diff_pct:.2f}%)")
        
        if param_diff_pct < 1:
            print(f"  ✅ 参数量基本一致")
        else:
            print(f"  ⚠️ 参数量存在差异")
        
        print()
        
        # 显存对比
        if actual_model_memory is not None:
            memory_diff = abs(theoretical_model_memory - actual_model_memory)
            memory_diff_pct = (memory_diff / theoretical_model_memory) * 100
            
            print(f"模型显存对比:")
            print(f"  理论值: {theoretical_model_memory:.2f} GB")
            print(f"  实际值: {actual_model_memory:.2f} GB")
            print(f"  差异:   {memory_diff:.2f} GB ({memory_diff_pct:.2f}%)")
            
            if memory_diff_pct < 10:
                print(f"  ✅ 显存占用基本一致")
            else:
                print(f"  ⚠️ 显存占用存在差异")
        
        print()
        
        # 推理显存分析
        if actual_inference_memory is not None:
            print(f"推理显存分析:")
            print(f"  理论推理显存: {theoretical_inference_memory:.2f} GB")
            print(f"  实际推理显存: {actual_inference_memory:.2f} GB")
            inference_diff = abs(theoretical_inference_memory - actual_inference_memory)
            print(f"  差异: {inference_diff:.2f} GB")
    
    # 4. 差异原因分析
    print("\n" + "=" * 60)
    print("🔍 差异原因分析")
    print("=" * 60)
    print("可能的差异原因:")
    print("1. 内存对齐: GPU要求内存按特定边界对齐，可能导致额外开销")
    print("2. PyTorch开销: 框架本身的元数据和管理开销")
    print("3. CUDA上下文: GPU驱动和CUDA运行时的基础开销")
    print("4. 缓存开销: PyTorch的内存池和缓存机制")
    print("5. 精度差异: 理论计算可能遗漏某些小的参数组件")
    print("6. 动态内存: 推理过程中临时张量的内存分配")
    print()
    
    print("✅ 分析完成!")

if __name__ == "__main__":
    main()
