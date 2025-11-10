"""GPT-2 系列性别偏见分析（Winogender 案例）

pip install transformer_lens plotly

复现论文场景：
- Vig et al. 2020: "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"
- 场景：Winogender 句子中的性别代词消歧
- 任务：找出哪些注意力头编码了职业的性别刻板印象

实验设计（正确版本）：
- base: "The nurse said that"
  → 监控下一个 token 是 "she" 的概率
  
- source: "The man said that"
  → 监控下一个 token 是 "she" 的概率

- 逻辑：如果模型将 "nurse" 刻板关联为女性，p(she|nurse) > p(she|man)
- 通过干预注意力头，找出哪些头负责编码这种性别刻板印象

运行示例：
  # 使用默认模型（GPT-2）
  python cma_gender_bias.py
  
  # 使用 DistilGPT2（更快）
  python cma_gender_bias.py --model distilgpt2
  
  # 使用 GPT-2 Medium（更准确）
  python cma_gender_bias.py --model gpt2-medium
  
  # 自定义实验
  python cma_gender_bias.py --model gpt2 --occupation doctor --gender-word woman

支持的模型：
  - distilgpt2: 82M 参数, 6层×12头 (最快, ~2分钟)
  - gpt2-small: 124M 参数, 12层×12头 (推荐, ~3-5分钟) [别名: gpt2]
  - gpt2-medium: 355M 参数, 24层×16头 (~10-15分钟)
  - gpt2-large: 774M 参数, 36层×20头 (~30-45分钟)
  - gpt2-xl: 1.5B 参数, 48层×25头 (~1-2小时)
"""


from __future__ import annotations

import argparse
from typing import List
import time
import os
import csv
from datetime import datetime

import torch

try:
    from transformer_lens import HookedTransformer
except Exception as e:
    raise RuntimeError("未检测到 transformer_lens，请先安装：pip install transformer_lens") from e

try:
    import plotly.express as px
except Exception:
    px = None


def load_model(model_name: str = "gpt2-small", device: str | None = None):
    """加载 GPT-2 模型（自动处理 gpt2-small 别名）, 使用 TransformerLens，并移动到目标设备"""
    hf_model_name = "gpt2" if model_name == "gpt2-small" else model_name
    print(f"正在加载 {model_name}{'（HF 名称: gpt2）' if model_name == 'gpt2-small' else ''}...")
    lm = HookedTransformer.from_pretrained(hf_model_name, center_writing_weights=False)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = lm.to(device)
    print(f"✓ 模型已加载")
    print(f"  - 设备: {device}")
    print(f"  - CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - 层数: {lm.cfg.n_layers}")
    print(f"  - 注意力头/层: {lm.cfg.n_heads}")
    return lm


def get_token_id(tokenizer, text: str) -> int:
    """获取文本的首个 token id（保持与原实现一致的“首 token”策略）"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"无法将 '{text}' tokenize 为有效 token")
    return ids[0]


def _bias_from_logits(logits_1d: torch.Tensor, id_she: int, id_he: int) -> float:
    return (logits_1d[id_she] - logits_1d[id_he]).item()


def run_gender_bias_cma(
    model: HookedTransformer,
    base_prompt: str,
    cf_prompt: str,
) -> List[List[float]]:
    """
    运行性别偏见的因果中介分析（逐头，中介=pre-Wo 的 z；对应 TL: blocks.L.attn.hook_z）
    """
    # she / he token id（用于输出度量）
    id_she = get_token_id(model.tokenizer, " she")
    id_he = get_token_id(model.tokenizer, " he")

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads

    print(f"\n{'='*70}")
    print(f"逐头中介替换（TransformerLens; hook: blocks.L.attn.hook_z, pre-Wo）")
    print(f"{'='*70}")
    print(f"base:          '{base_prompt}'")
    print(f"counterfactual: '{cf_prompt}'")
    print(f"度量固定:      bias = logit(she) - logit(he)")
    print(f"               （she/he 不在 prompt，都是预测候选）")
    print(f"\n扫描范围: {num_layers} 层 × {num_heads} 头 = {num_layers * num_heads} 次干预")
    print(f"{'='*70}\n")

    # 预先 token 化并移动到模型设备
    device = next(model.parameters()).device
    toks_base = model.to_tokens(base_prompt).to(device)
    toks_cf   = model.to_tokens(cf_prompt).to(device)

    causal_effects: List[List[float]] = []
    start_time = time.time()

    for layer_idx in range(num_layers):
        layer_start = time.time()
        print(f"  [Layer {layer_idx:2d}/{num_layers-1}] ", end="", flush=True)
        per_layer: List[float] = []

        # 先取 cf 的该层 z（所有头），只需前向一次
        with torch.no_grad():
            _, cache_cf = model.run_with_cache(toks_cf, remove_batch_dim=False)
        # TL 的 z 形状: (batch, pos, n_heads, d_head)
        z_cf_all = cache_cf[f"blocks.{layer_idx}.attn.hook_z"]  # [1, seq, H, d_head]
        z_cf_lastpos = z_cf_all[0, -1]  # [H, d_head]

        # 计算 base 的 clean logits（不干预）
        with torch.no_grad():
            logits_base = model(toks_base, return_type="logits")[0]  # (seq, d_vocab)
        bias_clean = _bias_from_logits(logits_base[-1, :], id_she, id_he)

        for head_idx in range(num_heads):
            with torch.no_grad():
                # 取该 head 的 cf 中介激活（pre-Wo）
                z_cf_head = z_cf_lastpos[head_idx].detach().clone()  # (d_head,)

                # 定义一个 hook：在该层的 z 上替换“最后一个位置 & 指定 head”
                def patch_z(value: torch.Tensor, hook):
                    # value shape: (batch, pos, H, d_head)
                    out = value.clone()
                    out[0, -1, head_idx, :] = z_cf_head
                    return out

                # 在 base prompt 上应用干预 → logits_int
                with torch.no_grad():
                    logits_int = model.run_with_hooks(
                        toks_base,
                        fwd_hooks=[(f"blocks.{layer_idx}.attn.hook_z", patch_z)],
                        return_type="logits"
                    )[0]
                bias_int = _bias_from_logits(logits_int[-1, :], id_she, id_he)

                # NIE（该头的间接效应）
                nie = bias_int - bias_clean
                per_layer.append(nie)

        causal_effects.append(per_layer)
        layer_time = time.time() - layer_start
        print(f"✓ ({layer_time:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ 完成！总耗时: {total_time:.1f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"{'='*70}\n")

    return causal_effects


def plot_heatmap(
    causal_effects: List[List[float]],
    title: str,
    num_layers: int,
    num_heads: int
):
    """绘制或打印性别偏见热力图"""
    if px is None:
        print(f"\n[{title}]")
        print("层×头热力图（未安装 plotly，仅打印数值）:\n")
        for li, row in enumerate(causal_effects):
            print(f"  layer {li:2d}: ", ", ".join(f"{v:+.4f}" for v in row))
        return

    fig = px.imshow(
        causal_effects,
        x=list(range(num_heads)),
        y=[f"Layer {i}" for i in range(num_layers)],
        template='simple_white',
        color_continuous_scale='RdBu_r',
        title=title,
        labels={'x': 'Attention Head', 'y': 'Layer', 'color': 'Causal Effect (Δp)'},
        aspect='auto',
    )
    fig.update_layout(
        xaxis_title='Attention Head',
        yaxis_title='Layer',
        yaxis=dict(autorange='reversed'),
        width=1000,
        height=700,
    )

    max_val = max(max(row) for row in causal_effects) if causal_effects else 0.0
    threshold = max_val * 0.5 if max_val != 0 else float("inf")

    annotations = []
    for layer_idx, row in enumerate(causal_effects):
        for head_idx, val in enumerate(row):
            if val > threshold:
                annotations.append(
                    dict(
                        x=head_idx,
                        y=layer_idx,
                        text=f"{val:.3f}",
                        showarrow=False,
                        font=dict(size=9, color='black')
                    )
                )

    fig.update_layout(annotations=annotations)
    fig.show()


def save_heads_to_file(
    causal_effects: List[List[float]],
    te: float,
    nie: float,
    nde: float,
    model_name: str,
    occupation: str,
    gender_word: str,
    output_dir: str = "results"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{occupation}_{gender_word}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    heads_data = []
    for layer_idx in range(len(causal_effects)):
        for head_idx in range(len(causal_effects[layer_idx])):
            nie_value = causal_effects[layer_idx][head_idx]
            heads_data.append({
                'layer': layer_idx,
                'head': head_idx,
                'nie': nie_value,
                'abs_nie': abs(nie_value)
            })

    heads_data.sort(key=lambda x: x['abs_nie'], reverse=True)
    for rank, head_data in enumerate(heads_data, 1):
        head_data['rank'] = rank

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['rank', 'layer', 'head', 'nie', 'abs_nie']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.write(f"# Experiment: {model_name} | {occupation} -> {gender_word}\n")
        f.write(f"# Total Effect (TE): {te:.6f}\n")
        f.write(f"# Natural Indirect Effect (NIE): {nie:.6f}\n")
        f.write(f"# Natural Direct Effect (NDE): {nde:.6f}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Format: rank, layer, head, nie, abs_nie\n")
        f.write("#\n")
        writer.writerows(heads_data)
    return filepath


def analyze_top_heads(
    causal_effects: List[List[float]],
    num_layers: int,
    num_heads: int,
    top_k: int = 5
) -> None:
    all_heads = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            effect = causal_effects[layer_idx][head_idx]
            all_heads.append((layer_idx, head_idx, effect))

    all_heads_sorted = sorted(all_heads, key=lambda x: abs(x[2]), reverse=True)
    positive_heads = sorted([h for h in all_heads if h[2] > 0], key=lambda x: x[2], reverse=True)
    negative_heads = sorted([h for h in all_heads if h[2] < 0], key=lambda x: x[2])

    print("\n" + "="*70)
    print(f"Top-{top_k} 影响最大的注意力头（按绝对值）")
    print("="*70)
    for i, (layer, head, effect) in enumerate(all_heads_sorted[:top_k], 1):
        print(f"  #{i}: Layer {layer:2d}, Head {head:2d}  |  NIE = {effect:+.5f}")

    print("\n" + "="*70)
    print(f"Top-{top_k} 正向影响头（促进偏见）")
    print("="*70)
    if positive_heads:
        for i, (layer, head, effect) in enumerate(positive_heads[:top_k], 1):
            print(f"  #{i}: Layer {layer:2d}, Head {head:2d}  |  NIE = +{effect:.5f}")
        print(f"\n  解释：这些头从 source 转移后增加了代词概率，说明它们编码了职业→性别的刻板关联")
    else:
        print("  无正向影响头")

    print("\n" + "="*70)
    print(f"Top-{top_k} 负向影响头（抑制偏见）")
    print("="*70)
    if negative_heads:
        for i, (layer, head, effect) in enumerate(negative_heads[:top_k], 1):
            print(f"  #{i}: Layer {layer:2d}, Head {head:2d}  |  NIE = {effect:.5f}")
        print(f"\n  解释：这些头从 source 转移后降低了代词概率，可能具有去偏见的作用")
    else:
        print("  无负向影响头")
    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT-2 系列模型性别偏见因果中介分析（TransformerLens 版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python cma_gender_bias.py
  python cma_gender_bias.py --model gpt2-medium
  python cma_gender_bias.py --occupation doctor --gender-word woman
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small",
        choices=["distilgpt2", "gpt2", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="选择模型 (默认: gpt2-small)。注: gpt2 和 gpt2-small 等价"
    )
    parser.add_argument("--occupation", type=str, default="nurse")
    parser.add_argument("--gender-word", type=str, default="man")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备（cuda 或 cpu）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("GPT-2 性别偏见因果中介分析（TransformerLens 版本；pre-Wo 逐头替换）")
    print("="*70)
    print(f"模型:          {args.model}")
    print(f"职业词:        {args.occupation}")
    print(f"set-gender 替换: {args.occupation} → {args.gender_word}")
    print(f"度量固定:      logit(she) - logit(he) 的变化（she/he 不在 prompt）")
    print("="*70)

    model = load_model(args.model, device=args.device)

    print("\n实验场景：职业词的性别偏见检测（Vig et al. 2020）")
    base_prompt = f"The {args.occupation} said that"
    cf_prompt   = f"The {args.gender_word} said that"

    # [步骤 1] 检查基础 bias（不干预）
    print("\n[步骤 1] 检查基础性别偏置...")
    id_she = get_token_id(model.tokenizer, " she")
    id_he  = get_token_id(model.tokenizer, " he")
    print(f"  验证代词 tokens: '{model.tokenizer.decode([id_she])}' vs '{model.tokenizer.decode([id_he])}'")

    with torch.no_grad():
        logits_base = model(model.to_tokens(base_prompt), return_type="logits")[0]  # (seq, d_vocab)
        logits_cf   = model(model.to_tokens(cf_prompt),   return_type="logits")[0]

    bias_base = _bias_from_logits(logits_base[-1, :], id_she, id_he)
    bias_cf   = _bias_from_logits(logits_cf[-1, :],   id_she, id_he)
    TE = bias_cf - bias_base

    print(f"  bias_base ('{args.occupation}'):  {bias_base:+.5f}")
    print(f"  bias_cf   ('{args.gender_word}'): {bias_cf:+.5f}")
    print(f"  TE (Total Effect):                {TE:+.5f}")

    if bias_base > 0.1:
        print(f"  ⚠️  检测到性别偏见！'{args.occupation}' 显著偏向 'she'")
    elif bias_base < -0.1:
        print(f"  ⚠️  检测到性别偏见！'{args.occupation}' 显著偏向 'he'")
    else:
        print(f"  ✓ 无明显性别偏见（bias 接近 0）")

    # [步骤 2] 运行逐头 CMA（pre-Wo 的 z）
    print("\n[步骤 2] 运行因果中介分析，定位负责偏见的注意力头...")
    causal_effects = run_gender_bias_cma(model, base_prompt, cf_prompt)

    # [步骤 3] 计算总效应（保留你的原输出格式）
    print(f"\n[步骤 3] 计算总体效应...")
    # 简化：NIE 总和（各头求和）
    total_nie = sum(sum(row) for row in causal_effects)

    # 重新计算 TE（与上面一致，保持结构）
    with torch.no_grad():
        logits_base = model(model.to_tokens(base_prompt), return_type="logits")[0]  # (seq, d_vocab)
        logits_cf   = model(model.to_tokens(cf_prompt),   return_type="logits")[0]

    bias_base = _bias_from_logits(logits_base[-1, :], id_she, id_he)
    bias_cf   = _bias_from_logits(logits_cf[-1, :],   id_she, id_he)
    te = bias_cf - bias_base
    nie = total_nie
    nde = te - nie

    print(f"  TE (Total Effect):      {te:+.6f}")
    print(f"  NIE (Sum of all heads): {nie:+.6f}")
    print(f"  NDE (Direct Effect):    {nde:+.6f}")
    print(f"  NIE/TE 比例:           {nie/te*100:.1f}%" if te != 0 else "  NIE/TE 比例:           N/A")

    # [步骤 4] 保存 CSV
    print(f"\n[步骤 4] 保存所有头的结果到文件...")
    filepath = save_heads_to_file(
        causal_effects=causal_effects,
        te=te, nie=nie, nde=nde,
        model_name=args.model,
        occupation=args.occupation,
        gender_word=args.gender_word,
        output_dir=args.output_dir
    )
    print(f"  ✓ 已保存到: {filepath}")

    # [步骤 5] Top-K 打印
    print(f"\n[步骤 5] 分析 Top-{args.top_k} 影响最大的注意力头...")
    analyze_top_heads(
        causal_effects,
        num_layers=model.cfg.n_layers,
        num_heads=model.cfg.n_heads,
        top_k=args.top_k
    )

    # [步骤 6] 可视化
    print("\n[步骤 6] 生成热力图...")
    plot_heatmap(
        causal_effects,
        title=f"{args.model.upper()}: Attention Head NIE<br>"
              f"'{args.occupation}' → '{args.gender_word}' (Δbias = Δ[logit(she) - logit(he)])",
        num_layers=model.cfg.n_layers,
        num_heads=model.cfg.n_heads
    )

    # 总结
    print("\n" + "="*70)
    print("结果解读（NIE 热力图）")
    print("="*70)
    print(f"✓ 热力图显示哪些注意力头介导 '{args.occupation}' 的性别偏见")
    print(f"✓ 中介：pre-Wo 的 head 输出 z（TL: blocks.L.attn.hook_z）")
    print(f"✓ 正值（红）：替换后增加女性偏好 → 该头编码职业→性别刻板")
    print(f"✓ 负值（蓝）：替换后降低女性偏好 → 可能具有去偏作用")
    print(f"\n✓ 所有头的结果已保存到: {filepath}")
    print("="*70)


if __name__ == "__main__":
    main()
