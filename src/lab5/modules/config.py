"""
Qwen3 模型配置类

这个文件定义了Qwen3模型的所有超参数和配置。
配置类包含了模型架构的所有关键参数，如层数、隐藏维度、注意力头数等。
"""

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Qwen3Config:
    """
    Qwen3 模型配置类
    
    这个类包含了Qwen3模型的所有超参数，包括：
    - 模型架构参数（层数、维度、头数等）
    - 训练相关参数（学习率、dropout等）
    - 特殊配置（RoPE、归一化等）
    
    默认值对应Qwen3-8B模型的配置。
    """

    # ============ 核心架构参数 ============
    vocab_size: int = 152064              # 词汇表大小
    hidden_size: int = 4096               # 隐藏层维度（嵌入维度）
    intermediate_size: int = 11008        # FFN中间层维度
    num_hidden_layers: int = 32           # Transformer层数
    num_attention_heads: int = 32         # 注意力头数
    num_key_value_heads: int = 32         # Key/Value头数（GQA时可能更少）
    max_position_embeddings: int = 32768  # 最大序列长度

    # ============ 激活函数和归一化 ============
    hidden_act: str = "silu"              # 激活函数类型
    rms_norm_eps: float = 1e-6            # RMSNorm的epsilon值
    rope_theta: float = 1000000.0         # RoPE的theta参数

    # ============ 训练相关参数 ============
    attention_dropout: float = 0.0       # 注意力dropout率

    # ============ 其他参数 ============
    initializer_range: float = 0.02      # 权重初始化范围
    tie_word_embeddings: bool = False     # 是否共享输入输出嵌入
    use_cache: bool = True                # 是否使用KV cache（推理优化）

    # ============ 模型特定参数 ============
    def __post_init__(self):
        """
        初始化后处理，确保参数一致性
        
        这个方法在dataclass实例化后自动调用，用于：
        1. 设置默认的KV头数（如果未指定）
        2. 计算每个注意力头的维度
        3. 验证参数的合理性
        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3Config":
        """
        从字典创建配置对象
        
        这个方法允许从预训练模型的配置字典中创建配置对象，
        常用于加载HuggingFace格式的模型配置。
        
        Args:
            config_dict: 包含配置参数的字典
            
        Returns:
            Qwen3Config实例
        """
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "Qwen3Config":
        """
        从JSON文件加载配置
        
        Args:
            json_path: JSON配置文件的路径
            
        Returns:
            Qwen3Config实例
        """
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        这个方法将配置对象的所有参数转换为字典，
        便于序列化和保存。
        
        Returns:
            包含所有配置参数的字典
        """
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "attention_dropout": self.attention_dropout,
            "initializer_range": self.initializer_range,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_cache": self.use_cache,
        }
