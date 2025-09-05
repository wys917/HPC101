"""
Qwen3 Transformer 层实现
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import Qwen3Config
from .attention import Qwen3Attention, RMSNorm
from .mlp import Qwen3MLP


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Decoder 层
    
    这是Transformer Decoder的一个基本单元，包含：
    1. Self-Attention模块：让token之间相互关注
    2. Feed Forward Network：对每个token进行独立的非线性变换
    3. 两个RMSNorm层：在每个子模块前进行归一化（Pre-Norm结构）
    4. 两个残差连接：保持梯度流动，避免梯度消失
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        device = "cpu" if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        # ============ 组件初始化 ============
        
        # Self-Attention模块：实现token之间的相互关注
        # 支持GQA（分组查询注意力）、RoPE位置编码、Q/K归一化等优化
        self.self_attn = Qwen3Attention(
            config=config, layer_idx=layer_idx, device=device, dtype=dtype
        )

        # Feed Forward Network：对每个token进行独立的非线性变换
        # 使用SwiGLU激活函数，增强模型的表达能力
        self.mlp = Qwen3MLP(config, device=device, dtype=dtype)

        # 层归一化组件（Pre-Norm结构）
        # input_layernorm：Self-Attention前的归一化
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        # post_attention_layernorm：FFN前的归一化
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Qwen3 Decoder Layer 的前向传播过程
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码，用于屏蔽某些位置
        
        Returns:
            hidden_states: 输出隐藏状态 [batch_size, seq_len, hidden_size]
        """
        
        # ============ 第一个子模块：Self-Attention ============
        # 步骤1：保存残差连接的输入（原始输入）
        residual = hidden_states  # 形状: [batch_size, seq_len, hidden_size]
        
        # 步骤2：对输入进行RMS归一化（Pre-Norm结构）
        hidden_states = self.input_layernorm(hidden_states)  # 形状不变: [batch_size, seq_len, hidden_size]
        
        # 步骤3：通过Self-Attention模块处理
        # Self-Attention让每个token关注到序列中的其他token
        hidden_states = self.self_attn(
            hidden_states=hidden_states,      # 输入: [batch_size, seq_len, hidden_size]
            attention_mask=attention_mask,    # 掩码: 控制哪些位置可以被关注
        )                                     # 输出: [batch_size, seq_len, hidden_size]
        
        # 步骤4：第一个残差连接 - 将原始输入加回来
        # 这样可以保持梯度流动，避免梯度消失
        hidden_states = residual + hidden_states  # 形状: [batch_size, seq_len, hidden_size]
        
        # ============ 第二个子模块：Feed Forward Network ============
        # 步骤5：保存残差连接的输入（Self-Attention的输出）
        residual = hidden_states  # 形状: [batch_size, seq_len, hidden_size]
        
        # 步骤6：对输入进行RMS归一化（Pre-Norm结构）
        hidden_states = self.post_attention_layernorm(hidden_states)  # 形状不变: [batch_size, seq_len, hidden_size]
        
        # 步骤7：通过FFN模块处理
        # FFN对每个token独立地进行非线性变换，增强表达能力
        hidden_states = self.mlp(hidden_states)  # 输入/输出: [batch_size, seq_len, hidden_size]
        
        # 步骤8：第二个残差连接 - 将FFN输入加回来
        # 再次保持梯度流动
        hidden_states = residual + hidden_states  # 形状: [batch_size, seq_len, hidden_size]
        
        # 返回最终的隐藏状态
        return hidden_states
