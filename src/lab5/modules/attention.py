"""
Multi-Head Attention Implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import RotaryEmbedding, apply_rotary_pos_emb
from .config import Qwen3Config


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3Attention(nn.Module):
    """Qwen3 Multi-Head Attention Module"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if dtype is None:
            dtype = torch.float32

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Linear projection layers
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Rotary position encoding
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            device=device,
        )

        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Reshape tensor for multi-head attention"""
        return tensor.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 请参照实验文档 2.8 节完成 Self-Attention 模块的实现

        # TODO: 根据 1-3 步完成 Q, K, V 的准备
        query_states = hidden_states  # 这里只是一个占位符，实际代码需要替换
        key_states = hidden_states
        value_states = hidden_states

        # TODO: 应用 RoPE
        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # TODO: 根据第 5 步完成 GQA 机制的实现

        # TODO: 根据第 6 步完成注意力计算

        # TODO: 根据第 7 步完成输出投影

        # 这里的 hidden_states 只是一个占位符，需要替换为实际的注意力输出
        return hidden_states

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
        The hidden_states.repeat(...).view(...) is more optimized when using torch.compile.
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
