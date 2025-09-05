"""
文本生成工具函数

这个模块包含了自回归文本生成的核心逻辑。
语言模型的推理过程是逐个token生成的：给定前面的token，预测下一个最可能的token。
"""

import time
import torch
import torch.nn.functional as F


def generate_text(
    model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, device="cpu"
):
    """
    使用语言模型进行自回归文本生成
    
    自回归生成的过程：
    1. 将输入文本转换为token序列
    2. 逐步生成：每次预测下一个token
    3. 将新生成的token添加到序列中
    4. 重复2-3步骤，直到达到最大长度或生成结束token
    
    Args:
        model: 语言模型（我们实现的Qwen3ForCausalLM）
        tokenizer: 分词器，用于文本和token之间的转换
        prompt: 输入提示文本
        max_length: 最大生成长度（包含输入）
        temperature: 温度参数，控制生成的随机性
                    - 0: 贪心解码，每次选择概率最高的token
                    - >0: 随机采样，值越大越随机
        top_k: Top-K采样，只从概率最高的k个token中采样
        device: 运行设备
    
    Returns:
        str: 生成的完整文本
    """
    
    print(f"\n输入提示: {prompt}")
    print("=" * 50)

    # ============ 输入预处理 ============
    
    # 将输入文本转换为token序列
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)        # token序列
    attention_mask = inputs["attention_mask"].to(device)  # 注意力掩码

    print(f"输入 token 数量: {input_ids.shape[1]}")

    # ============ 生成准备 ============
    
    # 复制输入序列，作为生成的起点
    generated_ids = input_ids.clone()  # [batch_size, input_length]

    start_time = time.time()

    # ============ 自回归生成循环 ============
    
    with torch.no_grad():  # 推理时不需要计算梯度
        for step in range(max_length):

            # 步骤1：前向传播 - 获取下一个token的概率分布
            results = model(input_ids=generated_ids, attention_mask=attention_mask)
            
            # 提取logits（未归一化的概率分数）
            if hasattr(results, "logits"):
                logits = results.logits  # [batch_size, seq_len, vocab_size]
            else:
                logits = results

            # ============ 下一个token的选择策略 ============
            
            if temperature == 0:
                # 贪心解码：直接选择概率最高的token
                # 这会产生确定性的输出，但可能比较单调
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                # 随机采样：根据概率分布随机选择token
                
                # 获取最后一个位置的logits（下一个token的预测）
                next_token_logits = logits[:, -1, :] / temperature  # 温度缩放
                
                # Top-k 采样：只从概率最高的k个token中选择
                if top_k > 0:
                    # 获取 top-k 个最高的分数和对应的token索引
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    
                    # 创建掩码，将非top-k的位置设为负无穷（softmax后概率为0）
                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # 应用 softmax 获得概率分布，然后随机采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # ============ 更新生成序列 ============
            
            # 将新生成的token添加到序列末尾
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # 更新注意力掩码（新token的位置设为1，表示可以关注）
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )

            # ============ 停止条件检查 ============
            
            # 如果生成了结束token，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.time()

    # ============ 结果处理和输出 ============
    
    # 将token序列解码回文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 输出生成统计信息
    print(f"\n生成完成!")
    print(f"生成时间: {end_time - start_time:.2f} 秒")
    print(f"生成 token 数量: {generated_ids.shape[1] - input_ids.shape[1]}")
    print(
        f"生成速度: {(generated_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.2f} tokens/秒"
    )
    print("=" * 50)
    print(f"完整输出: {generated_text}")

    return generated_text
