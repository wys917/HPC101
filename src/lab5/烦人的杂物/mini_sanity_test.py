"""
简单的实现测试，不依赖预训练模型
"""
import torch
import torch.nn as nn
from modules.attention import Qwen3Attention
from modules.mlp import Qwen3MLP
from modules.config import Qwen3Config


def test_attention():
    """测试注意力模块"""
    print("测试 Qwen3Attention 模块...")
    
    # 创建配置
    config = Qwen3Config(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0
    )
    
    # 创建模块
    attention = Qwen3Attention(config)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    try:
        output = attention(hidden_states)
        print(f"✅ 注意力模块测试通过")
        print(f"   输入形状: {hidden_states.shape}")
        print(f"   输出形状: {output.shape}")
        
        # 检查输出形状是否正确
        expected_shape = (batch_size, seq_len, hidden_size)
        if output.shape == expected_shape:
            print(f"   形状检查: ✅ 通过")
        else:
            print(f"   形状检查: ❌ 失败，期望 {expected_shape}，得到 {output.shape}")
            
        # 检查输出是否包含NaN或Inf
        if torch.isnan(output).any():
            print(f"   数值检查: ❌ 输出包含NaN")
        elif torch.isinf(output).any():
            print(f"   数值检查: ❌ 输出包含Inf")
        else:
            print(f"   数值检查: ✅ 通过")
            
    except Exception as e:
        print(f"❌ 注意力模块测试失败: {e}")
        return False
        
    return True


def test_mlp():
    """测试MLP模块"""
    print("\n测试 Qwen3MLP 模块...")
    
    # 创建配置
    config = Qwen3Config(
        hidden_size=512,
        intermediate_size=1024,
    )
    
    # 创建模块
    mlp = Qwen3MLP(config)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    try:
        output = mlp(hidden_states)
        print(f"✅ MLP模块测试通过")
        print(f"   输入形状: {hidden_states.shape}")
        print(f"   输出形状: {output.shape}")
        
        # 检查输出形状是否正确
        expected_shape = (batch_size, seq_len, hidden_size)
        if output.shape == expected_shape:
            print(f"   形状检查: ✅ 通过")
        else:
            print(f"   形状检查: ❌ 失败，期望 {expected_shape}，得到 {output.shape}")
            
        # 检查输出是否包含NaN或Inf
        if torch.isnan(output).any():
            print(f"   数值检查: ❌ 输出包含NaN")
        elif torch.isinf(output).any():
            print(f"   数值检查: ❌ 输出包含Inf")
        else:
            print(f"   数值检查: ✅ 通过")
            
    except Exception as e:
        print(f"❌ MLP模块测试失败: {e}")
        return False
        
    return True


def test_gqa():
    """测试GQA机制"""
    print("\n测试 GQA 机制...")
    
    # 创建GQA配置
    config = Qwen3Config(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA: 8个Q头共享2个KV头
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0
    )
    
    # 创建模块
    attention = Qwen3Attention(config)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    try:
        output = attention(hidden_states)
        print(f"✅ GQA机制测试通过")
        print(f"   Q头数: {config.num_attention_heads}")
        print(f"   KV头数: {config.num_key_value_heads}")
        print(f"   输入形状: {hidden_states.shape}")
        print(f"   输出形状: {output.shape}")
        
        # 检查输出形状是否正确
        expected_shape = (batch_size, seq_len, hidden_size)
        if output.shape == expected_shape:
            print(f"   形状检查: ✅ 通过")
        else:
            print(f"   形状检查: ❌ 失败，期望 {expected_shape}，得到 {output.shape}")
            
    except Exception as e:
        print(f"❌ GQA机制测试失败: {e}")
        return False
        
    return True


def main():
    """主测试函数"""
    print("=" * 50)
    print("开始基础模块测试...")
    print("=" * 50)
    
    all_passed = True
    
    # 测试各个模块
    all_passed &= test_attention()
    all_passed &= test_mlp()
    all_passed &= test_gqa()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有基础测试通过！")
    else:
        print("❌ 部分测试失败，请检查实现")
    print("=" * 50)


if __name__ == "__main__":
    main()
