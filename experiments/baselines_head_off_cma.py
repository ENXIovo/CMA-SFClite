"""
True HeadOff baseline (CMA-based): zero out selected attention heads and evaluate bias/perplexity.

Usage example:
  python -m experiments.baselines_head_off_cma \
      --model gpt2 \
      --ranking_csv results/gpt2-small_topk_heads.csv \
      --top_k 5 \
      --prompt_split test \
      --corpus_path data/WikiText.txt \
      --max_corpus_tokens 4096 \
      --output results/headoff_cma
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .cma_gender_bias import run_gender_bias_cma
from .prompts_winogender import get_prompt_examples


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_ranked_mediators_from_csv(csv_path: str, limit: Optional[int]) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Top-K 结果文件不存在: {csv_path}")

    def _filtered_lines(path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                yield line

    filtered_iter = _filtered_lines(csv_path)
    try:
        header_line = next(filtered_iter)
    except StopIteration:
        raise ValueError(f"Top-K CSV ({csv_path}) 内没有有效内容")

    header_raw = next(csv.reader([header_line]))
    header = [col.strip() for col in header_raw]
    required = {"rank", "layer", "head"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"Top-K CSV 缺少必要列: {', '.join(sorted(missing))}")

    reader = csv.DictReader(filtered_iter, fieldnames=header)
    mediators: List[Dict] = []
    for row in reader:
        try:
            layer = int(row["layer"])
            head = int(row["head"])
        except Exception:
            continue
        nie_val = None
        if "nie" in row and row["nie"] not in (None, "", "nan"):
            try:
                nie_val = float(row["nie"])
            except Exception:
                nie_val = None
        mediators.append({"layer": layer, "head": head, "nie": nie_val})
        if limit is not None and len(mediators) >= limit:
            break
    return mediators


def build_layer_to_heads(mediators: Sequence[Dict]) -> Dict[int, List[int]]:
    layer_to_heads: Dict[int, List[int]] = {}
    for m in mediators:
        layer_to_heads.setdefault(int(m["layer"]), []).append(int(m["head"]))
    return layer_to_heads


def tokenize_corpus(model: HookedTransformer, path: str, max_corpus_tokens: Optional[int]) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = model.to_tokens(text, prepend_bos=False)
    if tokens.numel() <= 1:
        return None
    if max_corpus_tokens is not None and tokens.size(1) > max_corpus_tokens:
        tokens = tokens[:, :max_corpus_tokens]
    return tokens


def compute_corpus_perplexity(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hooks: Optional[List[Tuple[str, callable]]] = None,
    block_size: int = 512,
) -> float:
    if tokens is None or tokens.numel() <= 1:
        return float("nan")

    total_log_prob = 0.0
    total_tokens = 0
    for start in range(0, tokens.size(1) - 1, block_size):
        end = min(start + block_size + 1, tokens.size(1))
        slice_tokens = tokens[:, start:end]
        if hooks:
            logits = model.run_with_hooks(
                slice_tokens,
                fwd_hooks=hooks,
                return_type="logits",
            )
        else:
            logits = model(slice_tokens, return_type="logits")
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
        target = slice_tokens[:, 1:]
        gathered = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        total_log_prob += gathered.sum().item()
        total_tokens += target.numel()

    if total_tokens == 0:
        return float("nan")
    avg_log_prob = total_log_prob / total_tokens
    return float(np.exp(-avg_log_prob))


def make_headoff_hooks(
    layer_to_heads: Dict[int, List[int]],
    alpha: float = 0.0,
    scope: str = "last_token",  # "last_token" or "full_sequence"
) -> List[Tuple[str, callable]]:
    hooks: List[Tuple[str, callable]] = []
    for layer_idx, heads in layer_to_heads.items():
        heads_tensor = torch.tensor(heads)  # capture by value

        def make_layer_hook(layer: int, heads_tensor: torch.Tensor, alpha: float, scope: str):
            def hook_fn(value: torch.Tensor, hook):
                # value shape: [batch, seq, n_heads, d_head] at blocks.{layer}.attn.hook_z
                out = value.clone()
                if scope == "last_token":
                    out[:, -1, heads_tensor, :] = out[:, -1, heads_tensor, :] * alpha
                else:
                    out[:, :, heads_tensor, :] = out[:, :, heads_tensor, :] * alpha
                return out

            return hook_fn

        hooks.append((f"blocks.{layer_idx}.attn.hook_z", make_layer_hook(layer_idx, heads_tensor, alpha, scope)))
    return hooks


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_headoff_cma(
    model_name: str,
    ranking_csv_path: str,
    top_k: int,
    output_path: str,
    prompt_split: str,
    corpus_path: Optional[str],
    max_corpus_tokens: Optional[int],
    device: Optional[str] = None,
    nie_mode: str = "heads",
    edit_scope: str = "last_token",
) -> None:
    os.makedirs(output_path, exist_ok=True)

    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"Top-K Mediators CSV: {ranking_csv_path}")
    print(f"Top-K: {top_k}")
    print(f"评估 prompts split: {prompt_split}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device=device)

    mediators = load_ranked_mediators_from_csv(ranking_csv_path, top_k if top_k > 0 else None)
    if not mediators:
        raise ValueError("从 CSV 中没有读取到任何 (layer, head) 条目。")
    layer_to_heads = build_layer_to_heads(mediators)
    hooks_headoff = make_headoff_hooks(layer_to_heads, alpha=0.0, scope=edit_scope)

    # 评估 prompts（NIE 总量）
    examples = get_prompt_examples(prompt_split)

    # heads-only CMA include map (align baseline 'heads' 模式)
    include_map = build_layer_to_heads(mediators) if nie_mode == "heads" else None

    nie_before_vals: List[float] = []
    nie_after_vals: List[float] = []

    for ex in examples:
        base = ex["base"]
        cf = ex["counterfactual"]
        # NIE before
        if nie_mode == "heads" and include_map:
            eff = run_gender_bias_cma(model, base, cf, verbose=False, include_heads_by_layer=include_map)
        else:
            eff = run_gender_bias_cma(model, base, cf, verbose=False)
        nie_before_vals.append(float(sum(sum(row) for row in eff)))
        # NIE after HeadOff (hooks on full sequence)
        with model.hooks(fwd_hooks=hooks_headoff):
            if nie_mode == "heads" and include_map:
                eff_after = run_gender_bias_cma(model, base, cf, verbose=False, include_heads_by_layer=include_map)
            else:
                eff_after = run_gender_bias_cma(model, base, cf, verbose=False)
        nie_after_vals.append(float(sum(sum(row) for row in eff_after)))

    # Align SFC-lite: first average NIE (with sign), then take absolute
    mean_before = float(np.mean(nie_before_vals)) if nie_before_vals else float("nan")
    mean_after = float(np.mean(nie_after_vals)) if nie_after_vals else float("nan")
    nie_before = abs(mean_before)
    nie_after = abs(mean_after)
    # Δ|NIE| = |after| - |before| (lower is better)
    delta_nie = nie_after - nie_before

    # 评估 PPL（WikiText 等）
    baseline_ppl = float("nan")
    headoff_ppl = float("nan")
    delta_ppl = float("nan")

    corpus_tokens = tokenize_corpus(model, corpus_path, max_corpus_tokens=max_corpus_tokens) if corpus_path else None
    if corpus_tokens is not None:
        baseline_ppl = compute_corpus_perplexity(model, corpus_tokens)
        # Apply the same intervention hooks to corpus to align with SFC-lite baselines
        headoff_ppl = compute_corpus_perplexity(model, corpus_tokens, hooks=hooks_headoff)
        if not (np.isnan(baseline_ppl) or np.isnan(headoff_ppl)):
            delta_ppl = headoff_ppl - baseline_ppl

    # 作图：Δ|NIE| vs ΔPPL
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig1_path = os.path.join(output_path, f"headoff_topk{top_k}_deltaNIE_deltaPPL_{ts}.png")
    plt.figure(figsize=(4, 3))
    plt.scatter([delta_nie], [delta_ppl], c="tab:red", label=f"HeadOff Top-{top_k}")
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.xlabel("Δ|NIE| (after - before)")
    plt.ylabel("ΔPPL (after - before)")
    plt.title("HeadOff (CMA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # 作图：Remaining bias (%) vs ΔPPL
    remain_pct = 100.0 * (nie_after / nie_before) if nie_before > 0 else float("nan")
    fig2_path = os.path.join(output_path, f"headoff_topk{top_k}_remainPct_deltaPPL_{ts}.png")
    plt.figure(figsize=(4, 3))
    plt.scatter([remain_pct], [delta_ppl], c="tab:blue", label=f"HeadOff Top-{top_k}")
    plt.axhline(0, color="gray", lw=0.5)
    plt.xlabel("Remaining bias (% of baseline)")
    plt.ylabel("ΔPPL (after - before)")
    plt.title("HeadOff (CMA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # 文本记录
    txt_path = os.path.join(output_path, f"headoff_topk{top_k}_summary_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Top-K CSV: {ranking_csv_path}\n")
        f.write(f"Top-K: {top_k}\n")
        f.write(f"Prompt split: {prompt_split}\n")
        f.write(f"Corpus path: {corpus_path}\n")
        f.write(f"NIE_before (|mean|): {nie_before:.6f}\n")
        f.write(f"NIE_after  (|mean|): {nie_after:.6f}\n")
        f.write(f"Δ|NIE|: {delta_nie:.6f}\n")
        f.write(f"PPL_before: {baseline_ppl:.6f}\n")
        f.write(f"PPL_after : {headoff_ppl:.6f}\n")
        f.write(f"ΔPPL: {delta_ppl:.6f}\n")
        f.write(f"Remaining bias (%): {remain_pct:.2f}\n")
        f.write(f"Figure1 (Δ|NIE| vs ΔPPL): {fig1_path}\n")
        f.write(f"Figure2 (Remaining% vs ΔPPL): {fig2_path}\n")
    print(f"Saved: {fig1_path}\n      {fig2_path}\n      {txt_path}")


def run_headoff_sweep(
    model_name: str,
    ranking_csv_path: str,
    top_k_list: Sequence[int],
    output_path: str,
    prompt_split: str,
    corpus_path: Optional[str],
    max_corpus_tokens: Optional[int],
    device: Optional[str] = None,
    nie_mode: str = "heads",
    edit_scope: str = "last_token",
) -> None:
    os.makedirs(output_path, exist_ok=True)

    # shared model & corpus tokens
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    corpus_tokens = tokenize_corpus(model, corpus_path, max_corpus_tokens=max_corpus_tokens) if corpus_path else None
    baseline_ppl = compute_corpus_perplexity(model, corpus_tokens) if corpus_tokens is not None else float("nan")

    examples = get_prompt_examples(prompt_split)

    # results rows for CSV (align with baseline fields)
    results: List[Dict] = []

    # for plotting aggregation
    xs_delta_nie: List[float] = []
    ys_delta_ppl: List[float] = []
    labels_delta: List[str] = []

    xs_remain: List[float] = []
    ys_remain_ppl: List[float] = []
    labels_remain: List[str] = []

    # include top-0 (no intervention) if not present
    if 0 not in top_k_list:
        top_k_list = [0] + list(top_k_list)

    for top_k in tqdm(top_k_list, desc="Scenarios"):
        mediators = load_ranked_mediators_from_csv(ranking_csv_path, top_k if top_k > 0 else None)
        layer_to_heads = build_layer_to_heads(mediators)
        include_map = layer_to_heads if (nie_mode == "heads") else None

        # cache NIE_before per K（与 α 无关）
        if nie_mode == "heads" and (include_map is not None) and (sum(len(v) for v in include_map.values()) == 0):
            # Top-K = 0 且 heads-only → 理论上 NIE_before = 0
            nie_before = 0.0
        else:
            nie_before_vals: List[float] = []
            for ex in tqdm(examples, desc="Examples (before)", leave=False):
                base = ex["base"]
                cf = ex["counterfactual"]
                if nie_mode == "heads" and (include_map is not None):
                    eff = run_gender_bias_cma(model, base, cf, verbose=False, include_heads_by_layer=include_map)
                else:
                    eff = run_gender_bias_cma(model, base, cf, verbose=False)
                nie_before_vals.append(float(sum(sum(row) for row in eff)))
            # Align SFC-lite: average then abs
            mean_before = float(np.mean(nie_before_vals)) if nie_before_vals else float("nan")
            nie_before = abs(mean_before) if not np.isnan(mean_before) else float("nan")

        # 固定 Headoff（α=0.0）以对齐“隔壁”
        alpha = 0.0
        hooks_headoff = make_headoff_hooks(layer_to_heads, alpha=float(alpha), scope=edit_scope)

        nie_after_vals: List[float] = []
        ppl_clean_vals: List[float] = []
        ppl_edit_vals: List[float] = []

        if nie_mode == "heads" and (include_map is not None) and (sum(len(v) for v in include_map.values()) == 0):
            # Top-K = 0 且 heads-only → NIE_after 也为 0
            nie_after = 0.0
            delta_nie = 0.0
            remain_pct = 100.0
            # prompt-level ppl 与语料 ppl 均不应变化
            mean_ppl_clean = baseline_ppl
            mean_ppl_edit = baseline_ppl
            delta_prompt_ppl = 0.0
            headoff_ppl = baseline_ppl
            delta_ppl = 0.0
        else:
            for ex in tqdm(examples, desc="Examples (after)", leave=False):
                base = ex["base"]
                cf = ex["counterfactual"]
                with model.hooks(fwd_hooks=hooks_headoff):
                    if nie_mode == "heads" and (include_map is not None):
                        eff_after = run_gender_bias_cma(model, base, cf, verbose=False, include_heads_by_layer=include_map)
                    else:
                        eff_after = run_gender_bias_cma(model, base, cf, verbose=False)
                nie_after_vals.append(float(sum(sum(row) for row in eff_after)))

                # prompt-level ppl for parity (optional; cheap if examples small)
                base_tokens = model.to_tokens(base, prepend_bos=False)
                ppl_base = compute_corpus_perplexity(model, base_tokens)
                with model.hooks(fwd_hooks=hooks_headoff):
                    ppl_edit = compute_corpus_perplexity(model, base_tokens)
                ppl_clean_vals.append(ppl_base)
                ppl_edit_vals.append(ppl_edit)

            mean_after = float(np.mean(nie_after_vals)) if nie_after_vals else float("nan")
            nie_after = abs(mean_after) if not np.isnan(mean_after) else float("nan")
            # Δ|NIE| = |after| - |before|
            delta_nie = nie_after - nie_before
            remain_pct = 100.0 * (nie_after / nie_before) if nie_before > 0 else 100.0

        if corpus_tokens is not None:
            headoff_ppl = compute_corpus_perplexity(model, corpus_tokens, hooks=hooks_headoff)
        else:
            headoff_ppl = float("nan")
        # compute delta_ppl after determining headoff_ppl/baseline_ppl
        delta_ppl = headoff_ppl - baseline_ppl if not (np.isnan(headoff_ppl) or np.isnan(baseline_ppl)) else float("nan")

        mean_ppl_clean = float(np.mean(ppl_clean_vals)) if ppl_clean_vals else float("nan")
        mean_ppl_edit = float(np.mean(ppl_edit_vals)) if ppl_edit_vals else float("nan")
        delta_prompt_ppl = mean_ppl_edit - mean_ppl_clean if (not np.isnan(mean_ppl_edit) and not np.isnan(mean_ppl_clean)) else float("nan")

        # CSV row aligned with baseline（无 alpha）
        aggregate_row = {
            "analysis": "headoff_cma",
            "edit_label": "headoff",
            "feature_source": "cma",
            "feature_count": int(top_k),  # 用 head 数对齐字段
            "sum_abs_edit": float(int(top_k)),  # α=0 → 抑制强度按 head 数计
            "bias_original_mean": nie_before,   # 用作 bias proxy（|NIE|）
            "bias_edited_mean": nie_after,
            "ppl_original_mean": mean_ppl_clean,
            "ppl_edited_mean": mean_ppl_edit,
            "delta_ppl_mean": delta_prompt_ppl,
            "remaining_bias_pct": float(remain_pct / 100.0) if not np.isnan(remain_pct) else float("nan"),
            "delta_nie_mean": float(delta_nie),
            "corpus_ppl_original": baseline_ppl,
            "corpus_ppl_edited": headoff_ppl,
            "delta_corpus_ppl": delta_ppl,
            "mediator_layer": ",".join(map(str, sorted(layer_to_heads.keys()))) if layer_to_heads else "",
            "mediator_type": "heads",
            "mediator_head": ",".join(map(str, sum([[(k, h) for h in v] for k, v in layer_to_heads.items()], []))) if layer_to_heads else "",
            "mediator_nie": None,
            "mediator_category": "topk_heads",
            "row_type": "aggregate",
            "nie_source": "evaluation",
        }
        results.append(aggregate_row)

        xs_delta_nie.append(delta_nie)
        ys_delta_ppl.append(delta_ppl)
        labels_delta.append(f"K={top_k}")

        xs_remain.append(remain_pct)
        ys_remain_ppl.append(delta_ppl)
        labels_remain.append(f"K={top_k}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # write CSV (aligned with baseline)
    csv_path = os.path.join(output_path, f"headoff_cma_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        fieldnames = sorted({key for row in results for key in row.keys()})
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ 结果已写入 {csv_path}")

    # PNG 1: Δ|NIE| vs ΔPPL
    fig1_path = os.path.splitext(csv_path)[0] + "_nie_scatter.png"
    plt.figure(figsize=(5, 4))
    for x, y, lab in zip(xs_delta_nie, ys_delta_ppl, labels_delta):
        plt.scatter([x], [y], label=lab, s=24)
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.xlabel("Δ|NIE| (after - before)")
    plt.ylabel("ΔPPL (after - before)")
    plt.title("HeadOff (CMA)")
    if len(labels_delta) <= 20:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制 NIE-ΔPPL 散点图: {fig1_path}")

    # PNG 2: Remaining% vs ΔPPL
    fig2_path = os.path.splitext(csv_path)[0] + "_bias_pareto.png"
    plt.figure(figsize=(5, 4))
    for x, y, lab in zip(xs_remain, ys_remain_ppl, labels_remain):
        plt.scatter([x], [y], label=lab, s=24)
    plt.axhline(0, color="gray", lw=0.5)
    plt.xlabel("Remaining bias (% of baseline)")
    plt.ylabel("ΔPPL (after - before)")
    plt.title("HeadOff (CMA)")
    if len(labels_remain) <= 20:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制 Bias-Pareto 图: {fig2_path}")

    # HTML (optional, align baseline: two html)
    try:
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame({
            "delta_nie": xs_delta_nie,
            "remaining_nie_pct": xs_remain,
            "delta_ppl": ys_delta_ppl,  # use ΔPPL for both; consistent with baseline selector
            "feature_count": [int(lab.split("=")[1]) for lab in labels_delta],
            "edit_label": labels_delta,
        })
        base_path = os.path.splitext(csv_path)[0]
        html1 = f"{base_path}_nie_scatter.html"
        html2 = f"{base_path}_bias_ppl.html"
        fig1 = px.scatter(df, x="delta_nie", y="delta_ppl", color="feature_count", hover_data=["edit_label"],
                          title="Δ|NIE| vs ΔPPL (headoff)", template="simple_white")
        fig1.write_html(html1)
        print(f"✓ 绘制 NIE-ΔPPL 交互图: {html1}")
        fig2 = px.scatter(df, x="remaining_nie_pct", y="delta_ppl", color="feature_count", hover_data=["edit_label"],
                          title="Bias–PPL (Remaining NIE% vs ΔPPL)", template="simple_white")
        fig2.write_html(html2)
        print(f"✓ 绘制 Bias–PPL 交互图: {html2}")
    except Exception:
        pass


def build_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HeadOff Baseline (CMA-based)")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ranking_csv",
        type=str,
        default="data/gpt2-small_nurse_man_20251110_173059.csv",
        help="CMA 生成的 Top-K 头部排名 CSV（默认与 baseline/sfclite 保持一致）",
    )
    parser.add_argument("--top_k", type=int, default=1, help="HeadOff 的 Top-K 头数（默认与 baseline 对齐为 1）")
    parser.add_argument("--top_k_list", type=int, nargs="+", default=[1, 5, 11], help="批量 K 列表，用于 sweep")
    parser.add_argument("--prompt_split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--corpus_path", type=str, default="data/WikiText.txt", help="用于 PPL 评估的语料")
    parser.add_argument("--max_corpus_tokens", type=int, default=4096, help="语料最大 token 数（防止过大）")
    parser.add_argument("--output", type=str, default="results/headoff_cma", help="输出目录")
    parser.add_argument("--nie_mode", type=str, default="heads", choices=["heads", "full"], help="NIE 评估模式：heads=仅选中头；full=全 CMA")
    parser.add_argument("--edit_scope", type=str, default="last_token", choices=["last_token", "full_sequence"], help="干预范围：last_token 与 CMA 对齐；full_sequence 为旧版全序列置零")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_arg_parser()
    # 若用户希望 sweep：遍历 K（固定 headoff 置零）
    if args.top_k_list:
        run_headoff_sweep(
            model_name=args.model,
            ranking_csv_path=args.ranking_csv,
            top_k_list=args.top_k_list,
            output_path=args.output,
            prompt_split=args.prompt_split,
            corpus_path=args.corpus_path,
            max_corpus_tokens=args.max_corpus_tokens,
            device=args.device,
            nie_mode=args.nie_mode,
            edit_scope=args.edit_scope,
        )
    else:
        run_headoff_cma(
            model_name=args.model,
            ranking_csv_path=args.ranking_csv,
            top_k=args.top_k,
            output_path=args.output,
            prompt_split=args.prompt_split,
            corpus_path=args.corpus_path,
            max_corpus_tokens=args.max_corpus_tokens,
            device=args.device,
            nie_mode=args.nie_mode,
            edit_scope=args.edit_scope,
        )


