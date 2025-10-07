"""单头中介替换实验（Single-Head Patching，遵循 Vig et al. 2020）

论文方法（Vig et al. NeurIPS 2020 扩展版）：

实验流程：
1. base: "The nurse said that" 
   → 记 bias_base = logit(she) - logit(he)
   
2. counterfactual（set-gender 端点）: "The man said that"
   → 记 bias_cf = logit(she) - logit(he)
   
3. 单头中介替换（NIE）: 在 base 轨迹中，将 (layer, head) 的中介激活
   替换为 counterfactual 轨迹的对应激活
   → 记 bias_int = logit(she) - logit(he)
   
4. 因果效应计算：
   - TE (Total Effect) = bias_cf - bias_base
   - NIE (Natural Indirect Effect，单头) = bias_int - bias_base
   
5. (可选) 中性锚点：把 counterfactual 改为 "The person said that"
6. (可选) 联合替换（joint NIE）：对 Top-k 头同时替换并画累积曲线

关键点：
- 控制变量：只改职业词（nurse ↔ man），不同时改代词
- 代词位置：she/he 是被预测的 Y，不出现在 prompt 里
- 度量固定：bias_score = logit(she) - logit(he)（在输出上计算）

运行示例：
  # 测试论文关键头
  python single_head_patching.py --layer 10 --head 5
  python single_head_patching.py --layer 0 --head 6
  
  # 使用中性锚点
  python single_head_patching.py --layer 10 --head 5 --gender-word person
  
  # 测试男性刻板职业
  python single_head_patching.py --occupation doctor --gender-word woman --layer 8 --head 2
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch

try:
    import nnsight
except Exception as e:
    raise RuntimeError("未检测到 nnsight，请先安装：pip install -U nnsight") from e


def load_model(model_name: str = "gpt2-small"):
    """加载语言模型（自动处理 gpt2-small 别名）"""
    hf_model_name = "gpt2" if model_name == "gpt2-small" else model_name
    print(f"正在加载 {model_name}{'（HF: gpt2）' if model_name == 'gpt2-small' else ''}...")
    lm = nnsight.LanguageModel(hf_model_name, device_map='auto')
    print(f"✓ 已加载: {lm.config.n_layer} 层 × {lm.config.n_head} 头")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return lm


def get_token_id(tokenizer, text: str) -> int:
    """获取文本的首个 token id（代词不在 prompt 里，只用于输出度量）"""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"无法将 '{text}' tokenize 为有效 token")
    return ids[0]


def compute_bias_score(logits: torch.Tensor, id_she: int, id_he: int) -> float:
    """
    计算性别偏置得分（论文 Eq.(1) 及相关定义）。
    
    bias_score = logit(she) - logit(he)
    
    等价形式：log[p(she) / p(he)]
    
    解释：
    - 正值 → 模型更倾向 she（女性偏好）
    - 负值 → 模型更倾向 he（男性偏好）
    - 接近 0 → 无明显性别偏好
    
    注意：she 和 he 都是被预测的输出，不出现在 prompt 里
    """
    return (logits[id_she] - logits[id_he]).item()


def single_head_nie(
    model,
    base_prompt: str,
    cf_prompt: str,
    layer_idx: int,
    head_idx: int,
) -> Tuple[float, float, float, float, float]:
    """
    单头中介替换实验（遵循 Vig et al. 2020）
    
    参数：
    - base_prompt: "The nurse said that"（含刻板印象职业）
    - cf_prompt: "The man said that"（counterfactual，set-gender 替换）
    - layer_idx: 目标层索引
    - head_idx: 目标头索引
    
    返回：(bias_base, bias_cf, bias_int, TE, NIE)
    
    算法（严格控制变量）：
      输入维度（唯一自变量）：
        - base 轨迹：使用刻板职业词（nurse）
        - counterfactual 轨迹：使用 set-gender 替换词（man）
        
      输出度量（固定）：
        - bias_score = logit(she) - logit(he)
        - she 和 he 都不在 prompt 里，都是被预测的候选词
        
      中介替换：
        1. base 轨迹（未干预）→ bias_base
        2. counterfactual 轨迹 → bias_cf
        3. counterfactual 轨迹，保存目标头激活 Z_cf(head)
        4. base 轨迹（干预：Z_base(head) → Z_cf(head)）→ bias_int
        
      因果效应：
        - TE = bias_cf - bias_base（总效应）
        - NIE = bias_int - bias_base（该头的间接效应）
        - NDE = TE - NIE（直接效应，论文中也有讨论）
    """
    # 获取 she 和 he 的 token id（用于输出度量，不在输入里）
    id_she = get_token_id(model.tokenizer, " she")
    id_he = get_token_id(model.tokenizer, " he")
    
    # 验证代词确实不在 prompt 里
    for prompt, name in [(base_prompt, "base"), (cf_prompt, "counterfactual")]:
        if " she" in prompt.lower() or " he" in prompt.lower():
            print(f"  ⚠️  警告：{name} prompt 包含代词，这会泄露信息到输入！")
    
    # 计算头的维度范围
    attn_dim = model.config.n_embd // model.config.n_head
    h_start = head_idx * attn_dim
    h_end = (head_idx + 1) * attn_dim
    
    with torch.no_grad():
        # 步骤 1：base 轨迹（未干预）
        with model.trace(base_prompt):
            logits_base = model.output.logits[0, -1, :].save()
        bias_base = compute_bias_score(logits_base, id_she, id_he)
        
        # 步骤 2：counterfactual 轨迹（用于计算 TE）
        with model.trace(cf_prompt):
            logits_cf = model.output.logits[0, -1, :].save()
        bias_cf = compute_bias_score(logits_cf, id_she, id_he)
        
        # 步骤 3：counterfactual 轨迹，保存目标头的中介激活
        with model.trace(cf_prompt):
            z_cf = model.transformer.h[layer_idx].attn.c_proj.input[0, -1, h_start:h_end].save()
        z_cf_val = z_cf.detach()
        
        # 步骤 4：base 轨迹（干预：替换该头为 counterfactual 的激活）
        with model.trace(base_prompt):
            model.transformer.h[layer_idx].attn.c_proj.input[0, -1, h_start:h_end] = z_cf_val
            logits_int = model.output.logits[0, -1, :].save()
        bias_int = compute_bias_score(logits_int, id_she, id_he)
    
    # 步骤 5：因果效应计算
    TE = bias_cf - bias_base
    NIE = bias_int - bias_base
    
    return bias_base, bias_cf, bias_int, TE, NIE


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="单头中介替换（Single-Head Patching，遵循 Vig et al. 2020）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 测试论文中的关键头（Head 10.5, 0.6）
  python single_head_patching.py --layer 10 --head 5
  python single_head_patching.py --layer 0 --head 6
  
  # 使用中性锚点（减少 man 自带的性别信号）
  python single_head_patching.py --layer 10 --head 5 --gender-word person
  
  # 测试男性刻板职业
  python single_head_patching.py --occupation doctor --gender-word woman --layer 8 --head 2
  
  # 使用不同模型
  python single_head_patching.py --model gpt2-medium --layer 15 --head 8

论文参考：
  Vig et al. 2020 "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"
  关键发现：Head 0.6, 10.5 等头显著介导性别偏见（Fig. 5a）
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small",
        choices=["distilgpt2", "gpt2", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="模型选择 (默认: gpt2-small)"
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="目标层索引（必需，0-based）"
    )
    
    parser.add_argument(
        "--head",
        type=int,
        required=True,
        help="目标头索引（必需，0-based）"
    )
    
    parser.add_argument(
        "--occupation",
        type=str,
        default="nurse",
        help="职业词 (默认: nurse)。女性刻板: nurse, teacher, secretary；男性刻板: doctor, engineer, CEO"
    )
    
    parser.add_argument(
        "--gender-word",
        type=str,
        default="man",
        help="set-gender 替换词 (默认: man)。选项: man, woman, person（中性锚点）"
    )
    
    return parser.parse_args()


def main():
    """单头 NIE 实验主流程（严格遵循论文）"""
    args = parse_args()
    
    print("="*70)
    print("单头中介替换实验（Single-Head Patching）")
    print("基于：Vig et al. 2020, NeurIPS")
    print("="*70)
    print(f"模型:          {args.model}")
    print(f"目标头:        Layer {args.layer}, Head {args.head}")
    print(f"职业词:        {args.occupation}")
    print(f"set-gender 替换: {args.occupation} → {args.gender_word}")
    if args.gender_word == "person":
        print(f"               （使用中性锚点，减少显式性别信号）")
    print(f"度量指标:      bias_score = logit(she) - logit(he)")
    print(f"               （she/he 是被预测的 Y，不在 prompt 里）")
    print("="*70)
    
    # 1. 加载模型
    model = load_model(args.model)
    
    # 验证索引范围
    if args.layer >= model.config.n_layer:
        raise ValueError(f"层索引超范围！模型只有 {model.config.n_layer} 层 (0-{model.config.n_layer-1})")
    if args.head >= model.config.n_head:
        raise ValueError(f"头索引超范围！每层只有 {model.config.n_head} 个头 (0-{model.config.n_head-1})")
    
    # 2. 构造提示（严格控制变量：只改职业词）
    base_prompt = f"The {args.occupation} said that"
    cf_prompt = f"The {args.gender_word} said that"
    
    print(f"\n实验设置（控制变量）：")
    print(f"  base:          '{base_prompt}'")
    print(f"  counterfactual: '{cf_prompt}'")
    print(f"  度量:          bias = logit(she) - logit(he)")
    print(f"                 （she 和 he 都不在 prompt 里，都是预测候选）")
    print(f"  中介单位:      Layer {args.layer}, Head {args.head} 的注意力激活")
    
    # 3. 运行单头 NIE 实验
    print(f"\n执行中介替换...")
    bias_base, bias_cf, bias_int, TE, NIE = single_head_nie(
        model,
        base_prompt,
        cf_prompt,
        args.layer,
        args.head
    )
    
    # 计算 NIE 占比和 NDE
    nie_ratio = (NIE / TE * 100) if abs(TE) > 1e-6 else 0.0
    NDE = TE - NIE
    
    # 4. 输出结果（对齐论文定义）
    print("\n" + "="*70)
    print("实验结果（因果效应分解）")
    print("="*70)
    print(f"目标头: Layer {args.layer}, Head {args.head}")
    print("-"*70)
    print(f"bias_base:      {bias_base:+.5f}")
    print(f"  = logit(she) - logit(he) 在 base 轨迹（'{args.occupation}'）")
    print()
    print(f"bias_cf:        {bias_cf:+.5f}")
    print(f"  = logit(she) - logit(he) 在 counterfactual 轨迹（'{args.gender_word}'）")
    print()
    print(f"bias_int:       {bias_int:+.5f}")
    print(f"  = logit(she) - logit(he) 在 base 轨迹但该头被替换为 cf 的激活")
    print("-"*70)
    print(f"TE  (Total Effect):              {TE:+.5f}")
    print(f"  = bias_cf - bias_base")
    print(f"  = '{args.occupation}' → '{args.gender_word}' 导致的总偏置变化")
    print()
    print(f"NIE (Natural Indirect Effect):   {NIE:+.5f}  ({nie_ratio:+.1f}% of TE)")
    print(f"  = bias_int - bias_base")
    print(f"  = 该注意力头介导的偏置变化")
    print()
    print(f"NDE (Natural Direct Effect):     {NDE:+.5f}  ({100-nie_ratio:+.1f}% of TE)")
    print(f"  = TE - NIE")
    print(f"  = 不经该头的直接偏置变化")
    print()
    print(f"验证：TE = NIE + NDE  →  {TE:.5f} ≈ {NIE + NDE:.5f}  ✓" if abs(TE - (NIE + NDE)) < 1e-4 else f"验证：TE ≠ NIE + NDE（数值误差）")
    print("="*70)
    
    # 5. 解释（基于 NIE）
    print("\n因果解释（基于 NIE）：")
    if abs(NIE) < 0.0001:
        print(f"  ✓ 该头的因果效应可忽略（|NIE| < 0.0001）")
        print(f"    该头不参与介导 '{args.occupation}' 的性别偏见")
    elif NIE > 0:
        print(f"  ⚠️  正向 NIE = {NIE:+.5f}")
        print(f"    该头从 '{args.gender_word}' 转移到 '{args.occupation}' 后，")
        print(f"    增加了女性偏好（bias_score 变大）")
        print(f"    → 该头强烈介导了 '{args.occupation}' → 'female' 的刻板关联")
        print(f"    → 这是性别偏见的来源之一！")
        if abs(TE) > 1e-6:
            print(f"    → 该头占总效应的 {abs(nie_ratio):.1f}%")
    else:
        print(f"  ℹ️  负向 NIE = {NIE:+.5f}")
        print(f"    该头从 '{args.gender_word}' 转移后降低了女性偏好")
        print(f"    → 该头可能具有性别平衡或去偏见的作用")
    
    print("\n建议后续实验：")
    print("  1. 用 cma_gender_bias.py 全量扫描，找出 Top-K NIE 头")
    print("  2. 用本脚本逐个验证（如论文 Head 0.6, 10.5）")
    print("  3. 多头联合替换（joint patching）：")
    print("     - 对 Top-k 头同时替换，计算累积 NIE")
    print("     - 绘制饱和曲线（Fig. 5b），验证稀疏性假设")
    print("  4. 可选：改用中性锚点 --gender-word person")
    print("="*70)


if __name__ == "__main__":
    main()
