"""GPT-2 系列性别偏见分析（Winogender 案例）

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
import json
from datetime import datetime

import torch

try:
    import nnsight
except Exception as e:
    raise RuntimeError("未检测到 nnsight，请先安装：pip install -U nnsight") from e

try:
    import plotly.express as px
except Exception:
    px = None


def load_model(model_name: str = "gpt2-small"):
    """加载 GPT-2 模型（自动处理 gpt2-small 别名）"""
    # 处理别名：gpt2-small → gpt2（HuggingFace 的实际名称）
    hf_model_name = "gpt2" if model_name == "gpt2-small" else model_name
    print(f"正在加载 {model_name}{'（HF 名称: gpt2）' if model_name == 'gpt2-small' else ''}...")
    lm = nnsight.LanguageModel(hf_model_name, device_map='auto')
    
    # 获取模型实际设备
    try:
        device = next(lm.parameters()).device
    except StopIteration:
        device = "meta（延迟加载）"
    
    print(f"✓ 模型已加载")
    print(f"  - 设备: {device}")
    print(f"  - CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - 层数: {lm.config.n_layer}")
    print(f"  - 注意力头/层: {lm.config.n_head}")
    
    return lm


def get_token_id(tokenizer, text: str) -> int:
    """获取文本的首个 token id"""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"无法将 '{text}' tokenize 为有效 token")
    return ids[0]


def run_gender_bias_cma(
    model,
    base_prompt: str,
    cf_prompt: str,
) -> List[List[float]]:
    """
    运行性别偏见的因果中介分析（遵循 Vig et al. 2020）。
    
    参数：
    - base_prompt: 含刻板印象职业的句子（如 "The nurse said that"）
    - cf_prompt: counterfactual 句子（set-gender 替换，如 "The man said that"）
    
    返回：层×头的 NIE 效应矩阵
    
    算法（严格控制变量）：
      输入维度：只改职业词（nurse ↔ man）
      输出度量：固定为 bias = logit(she) - logit(he)
      
      对每个注意力头：
        1) counterfactual 轨迹，保存该头的中介激活 Z_cf
        2) base 轨迹（未干预）→ bias_clean
        3) base 轨迹（干预：Z → Z_cf）→ bias_int
        4) NIE = bias_int - bias_clean
        
    解释：
    - 正值 NIE：该头从 cf 转移后增加了女性偏好，编码了职业→性别刻板
    - 负值 NIE：该头抑制了性别偏好
    
    注意：she 和 he 都不在 prompt 里，它们是被预测的输出 Y
    """
    # 获取 she 和 he 的 token id（用于输出度量）
    id_she = get_token_id(model.tokenizer, " she")
    id_he = get_token_id(model.tokenizer, " he")
    
    # 模型配置
    attn_dim = model.config.n_embd // model.config.n_head
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    
    print(f"\n{'='*70}")
    print(f"逐头中介替换（遵循 Vig et al. 2020）")
    print(f"{'='*70}")
    print(f"base:          '{base_prompt}'")
    print(f"counterfactual: '{cf_prompt}'")
    print(f"度量固定:      bias = logit(she) - logit(he)")
    print(f"               （she/he 不在 prompt，都是预测候选）")
    print(f"\n扫描范围: {num_layers} 层 × {num_heads} 头 = {num_layers * num_heads} 次干预")
    print(f"目标：找出哪些头介导 '{base_prompt.split()[1]}' → 性别偏好 的刻板关联")
    print(f"{'='*70}\n")
    
    causal_effects: List[List[float]] = []
    start_time = time.time()
    
    for layer_idx in range(num_layers):
        layer_start = time.time()
        print(f"  [Layer {layer_idx:2d}/{num_layers-1}] ", end="", flush=True)
        per_layer: List[float] = []
        
        for head_idx in range(num_heads):
            with torch.no_grad():
                h_start = head_idx * attn_dim
                h_end = (head_idx + 1) * attn_dim
                
                # 步骤 1：counterfactual 轨迹，保存该头的中介激活
                with model.trace(cf_prompt):
                    z_cf = model.transformer.h[layer_idx].attn.c_proj.input[0, -1, h_start:h_end].save()
                z_cf_val = z_cf.detach()
                
                # 步骤 2：base 轨迹（未干预），计算 bias_clean
                with model.trace(base_prompt):
                    logits_clean = model.output.logits[0, -1, :].save()
                
                # 步骤 3：base 轨迹（干预：替换该头为 cf 的激活）
                with model.trace(base_prompt):
                    model.transformer.h[layer_idx].attn.c_proj.input[0, -1, h_start:h_end] = z_cf_val
                    logits_int = model.output.logits[0, -1, :].save()
                
                # 计算 bias score（论文度量）
                bias_clean = (logits_clean[id_she] - logits_clean[id_he]).item()
                bias_int = (logits_int[id_she] - logits_int[id_he]).item()
                
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
    
    # 添加注释：标注重要的注意力头
    max_val = max(max(row) for row in causal_effects)
    threshold = max_val * 0.5  # 标注效应 > 50% 最大值的头
    
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
    """
    将所有头的TE和NIE排序结果保存到CSV文件
    
    参数：
    - causal_effects: 层×头的NIE矩阵
    - te, nie, nde: 总体效应
    - model_name, occupation, gender_word: 实验参数
    - output_dir: 输出目录
    
    返回：保存的文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名：模型_职业_性别词_时间戳.csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{occupation}_{gender_word}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # 收集所有头的数据
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
    
    # 按NIE绝对值排序（影响最大的在前）
    heads_data.sort(key=lambda x: x['abs_nie'], reverse=True)
    
    # 添加排名
    for rank, head_data in enumerate(heads_data, 1):
        head_data['rank'] = rank
    
    # 写入CSV文件
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['rank', 'layer', 'head', 'nie', 'abs_nie']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 写入头部信息
        writer.writeheader()
        
        # 写入实验信息（作为注释）
        f.write(f"# Experiment: {model_name} | {occupation} -> {gender_word}\n")
        f.write(f"# Total Effect (TE): {te:.6f}\n")
        f.write(f"# Natural Indirect Effect (NIE): {nie:.6f}\n")
        f.write(f"# Natural Direct Effect (NDE): {nde:.6f}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Format: rank, layer, head, nie, abs_nie\n")
        f.write("#\n")
        
        # 写入数据
        writer.writerows(heads_data)
    
    return filepath


def analyze_top_heads(
    causal_effects: List[List[float]], 
    num_layers: int, 
    num_heads: int, 
    top_k: int = 5,
    model_name: str = "",
    occupation: str = "",
    gender_word: str = ""
) -> str:
    """
    分析并输出影响最大的 Top-K 注意力头，并保存为 JSON 格式。
    
    参数：
    - causal_effects: 层×头的效应矩阵
    - num_layers: 层数
    - num_heads: 每层的头数
    - top_k: 输出前 K 个最大影响的头
    - model_name: 模型名称（用于文件名）
    - occupation: 职业词（用于文件名）
    - gender_word: 性别词（用于文件名）
    
    返回：保存的 JSON 文件路径
    """
    # 将矩阵展平，记录每个头的 (层索引, 头索引, 效应值)
    all_heads = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            effect = causal_effects[layer_idx][head_idx]
            all_heads.append((layer_idx, head_idx, effect))
    
    # 按效应绝对值排序（最大影响）
    all_heads_sorted = sorted(all_heads, key=lambda x: abs(x[2]), reverse=True)
    
    # 也按正值和负值分别排序
    positive_heads = sorted([h for h in all_heads if h[2] > 0], key=lambda x: x[2], reverse=True)
    negative_heads = sorted([h for h in all_heads if h[2] < 0], key=lambda x: x[2])
    
    print("\n" + "="*70)
    print(f"Top-{top_k} 影响最大的注意力头（按绝对值）")
    print("="*70)
    for i, (layer, head, effect) in enumerate(all_heads_sorted[:top_k], 1):
        sign = "+" if effect > 0 else ""
        print(f"  #{i}: Layer {layer:2d}, Head {head:2d}  |  NIE = {sign}{effect:+.5f}")
    
    print("\n" + "="*70)
    print(f"Top-{top_k} 正向影响头（促进偏见）")
    print("="*70)
    if positive_heads:
        for i, (layer, head, effect) in enumerate(positive_heads[:top_k], 1):
            print(f"  #{i}: Layer {layer:2d}, Head {head:2d}  |  NIE = +{effect:.5f}")
        print(f"\n  解释：这些头从 source 转移后增加了代词概率，")
        print(f"        说明它们编码了职业→性别的刻板关联")
    else:
        print("  无正向影响头")
    
    print("\n" + "="*70)
    print(f"Top-{top_k} 负向影响头（抑制偏见）")
    print("="*70)
    if negative_heads:
        for i, (layer, head, effect) in enumerate(negative_heads[:top_k], 1):
            print(f"  #{i}: Layer {layer:2d}, Head {head:2d}  |  NIE = {effect:.5f}")
        print(f"\n  解释：这些头从 source 转移后降低了代词概率，")
        print(f"        可能具有去偏见的作用")
    else:
        print("  无负向影响头")
    
    print("="*70)
    
    # 保存 Top-K 到 JSON 文件
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"topk_mediators_{model_name}_{occupation}_{gender_word}_{timestamp}.json"
    json_filepath = os.path.join(data_dir, json_filename)
    
    # 构建 JSON 数据（兼容 nie_feature_ablation.py 的格式）
    mediators_list = []
    for layer, head, effect in all_heads_sorted[:top_k]:
        mediators_list.append({
            "type": "attn",
            "layer": int(layer),
            "head": int(head),
            "nie": float(effect)
        })
    
    json_data = {
        "experiment": {
            "model": model_name,
            "occupation": occupation,
            "gender_word": gender_word,
            "timestamp": timestamp
        },
        "mediators": mediators_list
    }
    
    # 写入 JSON 文件
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Top-{top_k} mediators 已保存为 JSON: {json_filepath}")
    
    return json_filepath


def save_bias_eval_json(
    base_prompt: str,
    cf_prompt: str,
    occupation: str,
    gender_word: str,
    model_name: str
) -> str:
    """
    保存对应的 bias_eval.json 文件，用于后续的特征消融实验
    
    参数：
    - base_prompt: 基准句子（含职业词）
    - cf_prompt: 对照句子（含性别词）
    - occupation: 职业词
    - gender_word: 性别词
    - model_name: 模型名称
    
    返回：保存的文件路径
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f"bias_eval_{model_name}_{occupation}_{gender_word}_{timestamp}.json"
    eval_filepath = os.path.join(data_dir, eval_filename)
    
    # 推断预期偏见方向（基于常见刻板印象）
    # 注意：这只是启发式推断，实际偏见方向应该由实验测量
    female_stereotyped = ["nurse", "teacher", "secretary", "receptionist", "librarian", "assistant"]
    male_stereotyped = ["doctor", "engineer", "programmer", "CEO", "scientist", "mechanic"]
    
    if occupation.lower() in female_stereotyped:
        expected_bias = "female"
    elif occupation.lower() in male_stereotyped:
        expected_bias = "male"
    else:
        expected_bias = "unknown"
    
    # 构建评估数据
    eval_data = {
        "description": f"性别偏见评估数据 - {model_name} - {occupation} vs {gender_word}",
        "experiment": {
            "model": model_name,
            "occupation": occupation,
            "gender_word": gender_word,
            "timestamp": timestamp
        },
        "examples": [
            {
                "base": base_prompt,
                "counterfactual": cf_prompt,
                "occupation": occupation,
                "gender": "neutral",
                "expected_bias": expected_bias
            }
        ]
    }
    
    # 写入文件
    with open(eval_filepath, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Bias eval 数据已保存为 JSON: {eval_filepath}")
    
    return eval_filepath


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GPT-2 系列模型性别偏见因果中介分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 默认：GPT-2 + nurse/she
  python cma_gender_bias.py
  
  # 使用 DistilGPT2（更快）
  python cma_gender_bias.py --model distilgpt2
  
  # 使用 GPT-2 Medium（更准确）
  python cma_gender_bias.py --model gpt2-medium
  
  # 测试其他职业
  python cma_gender_bias.py --occupation doctor --gender-word woman
  python cma_gender_bias.py --occupation teacher --gender-word man
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small",
        choices=["distilgpt2", "gpt2", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="选择模型 (默认: gpt2-small)。注: gpt2 和 gpt2-small 等价"
    )
    
    parser.add_argument(
        "--occupation",
        type=str,
        default="nurse",
        help="职业词 (默认: nurse)。其他选项: doctor, teacher, engineer, secretary 等"
    )
    
    parser.add_argument(
        "--gender-word",
        type=str,
        default="man",
        help="set-gender 替换词 (默认: man)。选项: man, woman, person（中性锚点）"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="输出影响最大的 Top-K 个注意力头 (默认: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="保存结果文件的目录 (默认: results)"
    )
    
    return parser.parse_args()


def main():
    """
    主流程：复现 Vig et al. 2020 的性别偏见实验
    
    预期结果（参考论文 Figure 5a）：
    - 应该在某些层（如 layer 0, 10 等）的特定头发现显著效应
    - 这些头负责将职业（nurse）与性别刻板印象（female）关联
    """
    # 0. 解析命令行参数
    args = parse_args()
    
    print("="*70)
    print("GPT-2 性别偏见因果中介分析（遵循 Vig et al. 2020）")
    print("="*70)
    print(f"模型:          {args.model}")
    print(f"职业词:        {args.occupation}")
    print(f"set-gender 替换: {args.occupation} → {args.gender_word}")
    print(f"度量固定:      logit(she) - logit(he) 的变化")
    print(f"               （she/he 是被预测的输出，不在 prompt 里）")
    print("="*70)
    
    # 1. 加载模型
    model = load_model(args.model)
    
    # 2. 设置实验（使用命令行参数）
    print("\n实验场景：职业词的性别偏见检测")
    print("基于论文：Vig et al. 2020")
    
    base_prompt = f"The {args.occupation} said that"
    cf_prompt = f"The {args.gender_word} said that"
    
    # 度量固定：she vs he（论文方法，不可更改）
    # 注意：she 和 he 都不在 prompt 里，它们是被预测的输出
    
    # 3. 先检查基础偏置（未干预的 TE）
    print("\n[步骤 1] 检查基础性别偏置...")
    id_she = get_token_id(model.tokenizer, " she")
    id_he = get_token_id(model.tokenizer, " he")
    
    # 验证 token
    print(f"  验证代词 tokens: '{model.tokenizer.decode(id_she)}' vs '{model.tokenizer.decode(id_he)}'")
    print(f"  （代词不在 prompt 里，都是被预测的输出候选）")
    
    with torch.no_grad():
        with model.trace(base_prompt):
            logits_base = model.output.logits[0, -1, :].save()
        
        with model.trace(cf_prompt):
            logits_cf = model.output.logits[0, -1, :].save()
    
    # 计算 bias score（论文度量）
    bias_base = (logits_base[id_she] - logits_base[id_he]).item()
    bias_cf = (logits_cf[id_she] - logits_cf[id_he]).item()
    TE = bias_cf - bias_base
    
    print(f"  bias_base ('{args.occupation}'):  {bias_base:+.5f}")
    print(f"    = logit(she) - logit(he)")
    print(f"  bias_cf ('{args.gender_word}'):    {bias_cf:+.5f}")
    print(f"    = logit(she) - logit(he)")
    print(f"  TE (Total Effect):                {TE:+.5f}")
    print(f"    = bias_cf - bias_base")
    
    if bias_base > 0.1:
        print(f"  ⚠️  检测到性别偏见！'{args.occupation}' 显著偏向 'she'（女性刻板）")
    elif bias_base < -0.1:
        print(f"  ⚠️  检测到性别偏见！'{args.occupation}' 显著偏向 'he'（男性刻板）")
    else:
        print(f"  ✓ 无明显性别偏见（bias 接近 0）")
    
    # 4. 运行因果中介分析
    print("\n[步骤 2] 运行因果中介分析，定位负责偏见的注意力头...")
    causal_effects = run_gender_bias_cma(
        model,
        base_prompt,
        cf_prompt
    )
    
    # 5. 计算总体效应（用于保存）
    print(f"\n[步骤 3] 计算总体效应...")
    
    # 计算所有头的NIE总和（近似总体NIE）
    total_nie = sum(sum(layer_effects) for layer_effects in causal_effects)
    
    # 计算TE和NDE（基于之前的bias计算）
    with torch.no_grad():
        with model.trace(base_prompt):
            logits_base = model.output.logits[0, -1, :].save()
        with model.trace(cf_prompt):
            logits_cf = model.output.logits[0, -1, :].save()
    
    bias_base = (logits_base[id_she] - logits_base[id_he]).item()
    bias_cf = (logits_cf[id_she] - logits_cf[id_he]).item()
    te = bias_cf - bias_base
    nie = total_nie  # 所有头NIE的总和
    nde = te - nie
    
    print(f"  TE (Total Effect):     {te:+.6f}")
    print(f"  NIE (Sum of all heads): {nie:+.6f}")
    print(f"  NDE (Direct Effect):    {nde:+.6f}")
    print(f"  NIE/TE 比例:           {nie/te*100:.1f}%" if te != 0 else "  NIE/TE 比例:           N/A")
    
    # 6. 保存所有头的结果到文件
    print(f"\n[步骤 4] 保存所有头的结果到文件...")
    filepath = save_heads_to_file(
        causal_effects=causal_effects,
        te=te,
        nie=nie,
        nde=nde,
        model_name=args.model,
        occupation=args.occupation,
        gender_word=args.gender_word,
        output_dir=args.output_dir
    )
    print(f"  ✓ 已保存到: {filepath}")
    
    # 7. 分析 Top-K 影响最大的头并保存 JSON
    print(f"\n[步骤 5] 分析 Top-{args.top_k} 影响最大的注意力头...")
    json_filepath = analyze_top_heads(
        causal_effects,
        num_layers=model.config.n_layer,
        num_heads=model.config.n_head,
        top_k=args.top_k,
        model_name=args.model,
        occupation=args.occupation,
        gender_word=args.gender_word
    )
    
    # 7b. 保存对应的 bias_eval.json
    eval_filepath = save_bias_eval_json(
        base_prompt=base_prompt,
        cf_prompt=cf_prompt,
        occupation=args.occupation,
        gender_word=args.gender_word,
        model_name=args.model
    )
    
    # 8. 可视化结果
    print("\n[步骤 6] 生成热力图...")
    plot_heatmap(
        causal_effects,
        title=f"{args.model.upper()}: Attention Head NIE<br>"
              f"'{args.occupation}' → '{args.gender_word}' (Δbias = Δ[logit(she) - logit(he)])",
        num_layers=model.config.n_layer,
        num_heads=model.config.n_head
    )
    
    # 9. 总结
    print("\n" + "="*70)
    print("结果解读（NIE 热力图）")
    print("="*70)
    print(f"✓ 热力图显示哪些注意力头介导 '{args.occupation}' 的性别偏见")
    print(f"✓ 度量：NIE = Δ[logit(she) - logit(he)]（在 base 输入上的变化）")
    print(f"✓ 正值（红色）：该头从 '{args.gender_word}' 转移后增加了女性偏好")
    print(f"  → 说明该头强烈编码了 '{args.occupation}' → 'female' 关联（偏见源头）")
    print(f"✓ 负值（蓝色）：该头降低了女性偏好，可能具有去偏见作用")
    print(f"\n✓ 实验结果已保存（共 3 个文件）:")
    print(f"  [1] CSV（所有头排序）: {filepath}")
    print(f"      格式：rank, layer, head, nie, abs_nie")
    print(f"  [2] JSON（Top-{args.top_k} mediators）: {json_filepath}")
    print(f"      格式：用于 nie_feature_ablation.py 的 --topk_json")
    print(f"  [3] JSON（评估数据）: {eval_filepath}")
    print(f"      格式：用于 nie_feature_ablation.py 的 --eval_data")
    print(f"\n✓ 可直接运行后续实验：")
    print(f"  python experiments/nie_feature_ablation.py \\")
    print(f"    --topk_json {json_filepath} \\")
    print(f"    --eval_data {eval_filepath}")
    print("\n参考论文发现（Vig et al. 2020 Figure 5a）：")
    print("  - Head 0.6, 10.5 等头显著介导职业-性别刻板印象")
    print("  - 多头联合替换（joint patching）可达到饱和效应（Fig. 5b）")
    print("\n其他实验建议：")
    print("  python cma_gender_bias.py --occupation doctor --gender-word woman")
    print("  python cma_gender_bias.py --occupation teacher --gender-word person")
    print("  python cma_gender_bias.py --model gpt2-medium --occupation engineer")
    print("="*70)


if __name__ == "__main__":
    main()

