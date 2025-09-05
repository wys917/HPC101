# Lab5 TinyLLM 实现报告

## 1. 实验概述

本实验通过从零实现 Qwen3-8B 模型中的核心组件来掌握深度学习的核心概念。我成功实现了：

1. **Self Attention模块** (`modules/attention.py` 中的 `Qwen3Attention` 类)
2. **Feed Forward Network模块** (`modules/mlp.py` 中的 `Qwen3MLP` 类)

## 2. 实现思路

### 2.1 Qwen3Attention 实现思路

按照实验文档 2.8 节的要求，实现了完整的 Self-Attention 层，包含以下核心步骤：

#### 步骤1-3: Q, K, V 的准备
```python
# 1. QKV投影：通过线性层获得Q, K, V
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

# 2. 形状变换：重塑为多头格式 [batch, num_heads, seq_len, head_dim]
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

# 3. Q/K Normalization：对Q和K进行RMSNorm归一化
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
```

#### 步骤4: 应用RoPE位置编码
```python
kv_seq_len = key_states.shape[-2]
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

#### 步骤5: GQA机制实现
```python
# 重复K和V以匹配Q的头数（分组查询注意力）
key_states = self._repeat_kv(key_states, self.num_key_value_groups)
value_states = self._repeat_kv(value_states, self.num_key_value_groups)
```

#### 步骤6: 注意力计算
```python
# 计算注意力分数并应用scale
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

# 应用causal mask确保只能看到之前的token
if attention_mask is None:
    attention_mask = torch.tril(torch.ones(q_len, q_len, device=attn_weights.device, dtype=torch.bool))

if attention_mask is not None:
    attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

# 应用softmax
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

# 计算注意力输出
attn_output = torch.matmul(attn_weights, value_states)
```

#### 步骤7: 输出投影
```python
# 重塑输出并应用输出投影
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
attn_output = self.o_proj(attn_output)
```

### 2.2 Qwen3MLP 实现思路

按照实验文档 2.9 节的要求，实现了SwiGLU激活函数的FFN：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 1. 计算门控投影和上投影
    gate = self.gate_proj(x)  # [batch, seq_len, intermediate_size]
    up = self.up_proj(x)      # [batch, seq_len, intermediate_size]
    
    # 2. 应用SwiGLU激活函数：SwiGLU(x) = SiLU(gate) ⊙ up
    # SiLU(x) = x * sigmoid(x)
    silu_gate = gate * torch.sigmoid(gate)
    
    # 3. 元素级相乘
    intermediate = silu_gate * up
    
    # 4. 下投影回原维度
    output = self.down_proj(intermediate)
    
    return output
```

## 3. 关键技术要点

### 3.1 分组查询注意力 (GQA)
- 在Qwen3-8B中，`num_heads=32`，`num_key_value_heads=8`
- 每4个Query头共享1个Key/Value头，减少计算量和内存占用
- 使用`_repeat_kv`函数将KV头重复到匹配Q头的数量

### 3.2 RoPE位置编码
- 通过旋转位置编码增强模型对序列位置的感知
- 在计算注意力之前对Q和K应用位置编码
- 使用已实现的`apply_rotary_pos_emb`函数

### 3.3 Q/K归一化
- 在计算注意力分数前对Q和K进行RMSNorm归一化
- 提高模型训练稳定性和收敛速度

### 3.4 SwiGLU激活函数
- 结合了SiLU和GLU的优点
- 公式：`SwiGLU(x) = SiLU(gate_proj(x)) ⊙ up_proj(x)`
- 其中`SiLU(x) = x * sigmoid(x)`

### 3.5 因果掩码
- 使用下三角掩码确保模型只能看到当前token之前的token
- 将未来位置设为负无穷，softmax后变为0

## 4. 测试结果

### 4.1 基础模块测试
创建了`mini_sanity_test.py`进行基础功能测试：
- ✅ 注意力模块形状检查通过
- ✅ MLP模块形状检查通过  
- ✅ GQA机制测试通过
- ✅ 数值稳定性检查通过

### 4.2 高级实现测试
创建了`test_implementation.py`进行全面测试：
- ✅ 注意力模块（多配置）- 6/6通过
- ✅ MLP模块（多配置）- 测试通过
- ✅ 完整解码器层 - 测试通过
- ✅ 因果掩码正确性 - 测试通过
- ✅ RoPE位置编码 - 测试通过
- ✅ SwiGLU激活函数 - 测试通过

**最终测试结果：6/6 通过 🎉**

## 5. 关键代码设计亮点

### 5.1 模块化设计
- 每个功能都有清晰的职责划分
- 易于理解和维护的代码结构

### 5.2 形状管理
- 严格的张量形状管理，确保维度正确
- 详细的注释说明每个步骤的张量形状变化

### 5.3 数值稳定性
- 正确实现了softmax和归一化操作
- 避免了数值溢出和下溢问题

### 5.4 GQA优化
- 正确实现了KV头重复机制
- 支持不同的Q头和KV头比例配置

## 6. 思考题分析

### 6.1 Qwen3 Decoder Layer结构
Qwen3 Decoder Layer的主要组件：

```
输入 → RMSNorm → Self-Attention → 残差连接 → RMSNorm → FFN → 残差连接 → 输出
```

**输入输出形状说明：**
- 输入：`[batch_size, seq_len, hidden_size]`
- Self-Attention输出：`[batch_size, seq_len, hidden_size]`
- FFN输出：`[batch_size, seq_len, hidden_size]`
- 最终输出：`[batch_size, seq_len, hidden_size]`

**Residual Connection位置：**
- 在Self-Attention之后
- 在FFN之后

**Normalization位置：**
- 在Self-Attention之前（Pre-Norm）
- 在FFN之前（Pre-Norm）

### 6.2 参数量和显存占用计算

**参数量计算（以Qwen3-8B为例）：**

1. **Embedding层：** `vocab_size × hidden_size = 151936 × 4096 ≈ 622M`

2. **每个Transformer层：**
   - Self-Attention: 
     - Q投影：`4096 × 4096 = 16.8M`
     - K投影：`4096 × 1024 = 4.2M` (GQA: 8个KV头)
     - V投影：`4096 × 1024 = 4.2M`
     - O投影：`4096 × 4096 = 16.8M`
     - Q/K Norm: `2 × 128 × 32 = 8.2K` (忽略不计)
   - FFN:
     - Gate投影：`4096 × 12288 = 50.3M`
     - Up投影：`4096 × 12288 = 50.3M` 
     - Down投影：`12288 × 4096 = 50.3M`
   - LayerNorm: `2 × 4096 = 8.2K` (忽略不计)
   
   **单层总计：** `≈ 193M`

3. **36层总计：** `193M × 36 ≈ 6.95B`

4. **LM Head：** `4096 × 151936 ≈ 622M`

5. **最终归一化：** `4096 ≈ 4K` (忽略不计)

**总参数量：** `622M + 6.95B + 622M ≈ 8.2B`

**理论显存占用（推理）：**
- 模型参数（bfloat16）：`8.2B × 2 bytes = 16.4GB`
- 激活值（取决于序列长度和批次大小）
- KV Cache（取决于序列长度）

实际运行时可能由于框架开销、碎片化等因素导致显存占用更高。

## 7. 实验总结

通过本次实验，我深入理解了：

1. **Transformer架构的核心机制：** 特别是Self-Attention和FFN的工作原理
2. **现代LLM的优化技术：** 如GQA、RoPE、RMSNorm等
3. **PyTorch张量操作：** 形状变换、矩阵运算、掩码操作等
4. **模型实现的工程细节：** 数值稳定性、内存优化等

这次实验让我对大语言模型的内部工作机制有了更深刻的理解，为后续的深度学习研究奠定了坚实的基础。

## 8. 代码文件清单

- ✅ `modules/attention.py` - 实现了完整的Qwen3Attention类
- ✅ `modules/mlp.py` - 实现了完整的Qwen3MLP类  
- ✅ `mini_sanity_test.py` - 基础功能测试
- ✅ `test_implementation.py` - 高级实现测试

所有测试均通过，实现正确且稳定。
