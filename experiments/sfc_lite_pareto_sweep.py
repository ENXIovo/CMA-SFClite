"""
Pareto sweep for SFC-lite surgery driven by CMA Top-K heads.

This script sweeps intervention strength (via global budget of gated SAE features)
and measures the Bias–Perplexity Pareto curve as specified in the proposal:
  - Bias axis (X): remaining TE or NIE as % of baseline (lower is better)
  - Performance axis (Y): ΔPerplexity on a text corpus (lower is better)

Output:
- CSV with (budget, total_gated, TE_before/after, NIE_before/after, remaining_% (TE/NIE),
            PPL_before/after, ΔPPL)
- Interactive HTML scatter (x=remaining_% (TE or NIE), y=ΔPPL), with Pareto frontier overlay

Usage (example):
  python pareto_sweep.py \
    --cma_csv results/gpt2-small_nurse_man_20251028_042958.csv \
    --budgets 0,100,300,600,1200,2400,4800 \
    --top_k_heads 5 --topn_per_layer 300 --tau_min 0.12 \
    --ppl_file path/to/text.txt \
    --bias_axis te
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math
from contextlib import nullcontext

from .prompts_winogender import get_prompt_examples
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Reuse core logic from the SFC-lite engine
from .sfc_lite_engine import (
    load_model as load_tl_model,
    read_topk_heads,
    build_surgery,
    pronoun_bias_score,
    run_with_surgery,
    make_layer_gate_hook,
)

# CMA (per-head) to compute NIE matrix
from .cma_gender_bias import run_gender_bias_cma


def parse_args():
    ap = argparse.ArgumentParser("Pareto sweep for SFC-lite (budget vs. bias improvement)")
    ap.add_argument("--cma_csv", type=str, default="results/gpt2-small_nurse_man_20251110_173059.csv",
                    help="Path to CMA CSV with columns: rank/layer/head/nie/abs_nie.")
    ap.add_argument("--sae_release", type=str, default="gpt2-small-attn-out-v5-32k",
                    help="SAE-Lens release for attn_out (post-WO).")
    ap.add_argument("--top_k_heads", type=int, default=1,
                    help="Number of heads to take from CMA CSV.")
    ap.add_argument("--topn_per_layer", type=int, default=32000,
                    help="Top-N features per layer (pre-budget) sorted by attribution mass.")
    ap.add_argument("--tau_min", type=float, default=0.0,
                    help="Loose minimum mass floor.")
    ap.add_argument("--layers_limit", type=int, default=None,
                    help="Optional: restrict to the first N unique layers in CMA selection.")

    ap.add_argument(
        "--budgets",
        type=str,
        default="0:32000:1000",
        help="Budgets list or range 'start:end:step' (interpreted as per-head in head mode); 0 disables cap.",
    )

    ap.add_argument("--prompt_base", type=str, default="The nurse said that",
                    help="Base prompt for bias measurement.")
    ap.add_argument("--prompt_cf", type=str, default="The man said that",
                    help="Counterfactual prompt for bias measurement.")

    ap.add_argument(
        "--ppl_file",
        type=str,
        default="data/WikiText.txt",
        help="Text file path for PPL computation (preferred).",
    )
    ap.add_argument(
        "--prompt_split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Prompt split for averaging TE/NIE across multiple base/cf pairs.",
    )
    ap.add_argument("--ppl_text", type=str, default=None,
                    help="Fallback single string for PPL computation if no file is provided.")

    ap.add_argument(
        "--bias_axis",
        type=str,
        default="nie",
        choices=["te", "nie"],
        help="Which bias metric to show on X axis (remaining % of baseline).",
    )
    ap.add_argument(
        "--nie_mode",
        type=str,
        default="heads",
        choices=["local", "heads", "full"],
        help="NIE mode: 'local' single multi-head patch; 'heads' CMA on selected heads; 'full' CMA all heads.",
    )

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run on.")
    ap.add_argument("--out_dir", type=str, default="results",
                    help="Output directory for CSV/HTML.")
    ap.add_argument("--quiet", action="store_true", default=True, help="Minimal logging; still shows progress bar")
    ap.add_argument("--verbose", action="store_true", help="Verbose diagnostics logging")
    return ap.parse_args()


@torch.no_grad()
def compute_avg_nll(model, text: str, device: str) -> float:
    """
    Average next-token negative log-likelihood over the given text.
    """
    toks = model.to_tokens(text, prepend_bos=True).to(device)
    if toks.shape[-1] < 2:
        return float("nan")
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    with amp_ctx:
        logits = model(toks)
    # Shift for next-token prediction
    logits_shifted = logits[:, :-1, :]
    targets = toks[:, 1:]
    loss = F.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )
    return float(loss.item())


def count_total_gated(plan: Dict[int, Dict]) -> int:
    total = 0
    for _L, cfg in plan.items():
        total += int(cfg.get("n_features_gate", 0))
    return total


def compute_pareto_frontier(points: List[Dict[str, float]], x_key: str, y_key: str) -> List[bool]:
    """
    Mark Pareto-optimal points minimizing both x_key and y_key (lower-left frontier).
    """
    sorted_idx = sorted(range(len(points)), key=lambda i: (points[i][x_key], points[i][y_key]))
    frontier_flags = [False] * len(points)
    best_y = float("inf")
    for i in sorted_idx:
        y = points[i][y_key]
        if y < best_y:
            frontier_flags[i] = True
            best_y = y
    return frontier_flags


def read_text_for_ppl(ppl_file: Optional[str], ppl_text: Optional[str]) -> Optional[str]:
    if ppl_file and os.path.isfile(ppl_file):
        with open(ppl_file, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        return txt
    if ppl_text:
        return ppl_text
    return None


def compute_total_nie(model, base_prompt: str, cf_prompt: str) -> float:
    effects = run_gender_bias_cma(model, base_prompt, cf_prompt, verbose=False)
    return float(sum(sum(row) for row in effects))


@torch.no_grad()
def compute_local_nie(
    model: HookedTransformer,
    base_prompt: str,
    cf_prompt: str,
    include_heads_by_layer: Dict[int, List[int]],
) -> float:
    """Fast local NIE: for selected layers/heads, patch cf z at the last position in one shot per layer.
    Returns scalar NIE = bias_int - bias_clean.
    """
    # tokens
    toks_base = model.to_tokens(base_prompt)
    toks_cf = model.to_tokens(cf_prompt)

    # clean bias
    logits_clean = model(toks_base, return_type="logits")[0]
    id_she = model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
    id_he = model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]
    bias_clean = (logits_clean[-1, id_she] - logits_clean[-1, id_he]).item()

    # cache cf z for required layers
    hook_names = {f"blocks.{L}.attn.hook_z" for L in include_heads_by_layer.keys()}
    if hook_names:
        _, cache_cf = model.run_with_cache(
            toks_cf,
            names_filter=lambda name: name in hook_names,
            remove_batch_dim=False,
            return_type="logits",
        )
    else:
        cache_cf = {}

    # build hooks
    fwd_hooks: List[Tuple[str, callable]] = []
    for L, heads in include_heads_by_layer.items():
        hook_name = f"blocks.{L}.attn.hook_z"
        if hook_name not in cache_cf:
            # layer not cached; skip safely
            continue
        z_cf_all = cache_cf[hook_name]
        z_cf_last = z_cf_all[0, -1]  # [H, d_head]

        def make_patch(layer_idx: int, head_list: List[int], z_last: torch.Tensor):
            def patch(value: torch.Tensor, hook):
                out = value.clone()
                for h in head_list:
                    out[0, -1, h, :] = z_last[h]
                return out
            return patch

        fwd_hooks.append((f"blocks.{L}.attn.hook_z", make_patch(L, heads, z_cf_last)))

    # patched bias
    if fwd_hooks:
        logits_int = model.run_with_hooks(toks_base, fwd_hooks=fwd_hooks, return_type="logits")[0]
    else:
        logits_int = logits_clean
    bias_int = (logits_int[-1, id_she] - logits_int[-1, id_he]).item()

    return bias_int - bias_clean


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def log(msg: str):
        if args.verbose or not args.quiet:
            print(msg)

    log(f"Loading model on {args.device} ...")
    model = load_tl_model(device=args.device)

    log(f"Reading CMA top-K heads from {args.cma_csv} ...")
    topk = read_topk_heads(args.cma_csv, args.top_k_heads)
    if args.verbose:
        print("Top-K heads (layer, head, NIE):")
        for (L, h, nie) in topk:
            print(f"  L{L:02d} H{h:02d}  NIE={nie:+.4f}")

    # Load prompt pairs for averaging
    examples = get_prompt_examples(args.prompt_split)
    pairs = []
    for ex in examples:
        base = ex.get("base", ex.get("clean_prefix", ""))
        cf = ex.get("counterfactual", ex.get("patch_prefix", ""))
        if base and cf:
            pairs.append((base, cf))
    if not pairs:
        # fallback to single pair from args
        pairs = [(args.prompt_base, args.prompt_cf)]

    # Build selected heads map for head-mode NIE
    include_map = {}
    for (L, h, _nie) in topk:
        include_map.setdefault(int(L), []).append(int(h))

    # Baseline metrics averaged over prompt split
    log("\nComputing baseline TE/NIE (no surgery) over prompt split...")
    te_vals = []
    nie_vals = []
    for base, cf in pairs:
        b0_i = pronoun_bias_score(model, base, device=args.device)
        c0_i = pronoun_bias_score(model, cf,   device=args.device)
        te_vals.append(c0_i - b0_i)
        if args.nie_mode == "local":
            nie_vals.append(compute_local_nie(model, base, cf, include_map))
        elif args.nie_mode == "heads":
            effects = run_gender_bias_cma(model, base, cf, verbose=False, include_heads_by_layer=include_map)
            nie_vals.append(float(sum(sum(row) for row in effects)))
        else:
            nie_vals.append(compute_total_nie(model, base, cf))
    TE_0 = float(np.mean(te_vals)) if te_vals else float("nan")
    NIE_0 = float(np.mean(nie_vals)) if nie_vals else float("nan")
    # For logging, show first pair's biases
    bias_base_0 = pronoun_bias_score(model, pairs[0][0], device=args.device)
    bias_cf_0   = pronoun_bias_score(model, pairs[0][1], device=args.device)
    log(f"  prompts={len(pairs)}  TE_mean={TE_0:+.4f}  NIE_mean={NIE_0:+.4f}")

    # PPL baseline on provided text
    ppl_text_all = read_text_for_ppl(args.ppl_file, args.ppl_text)
    ppl_0 = None
    nll_0 = None
    if ppl_text_all:
        log("Computing baseline Perplexity on provided text...")
        nll_0 = compute_avg_nll(model, ppl_text_all, device=args.device)
        ppl_0 = math.exp(nll_0) if not math.isnan(nll_0) else float("nan")
        log(f"  PPL_before={ppl_0:.4f}")
    else:
        log("No PPL text provided; ΔPPL will be NaN.")

    # Parse budgets
    budgets: List[int] = []
    for t in args.budgets.split(","):
        t = t.strip()
        if not t:
            continue
        if ":" in t:
            parts = t.split(":")
            if len(parts) != 3:
                raise ValueError(f"Invalid range budget token: '{t}' (expected start:end:step)")
            start, end, step = map(int, parts)
            if step <= 0:
                raise ValueError("Range step must be > 0")
            # inclusive end
            budgets.extend(list(range(start, end + 1, step)))
        else:
            try:
                budgets.append(int(t))
            except ValueError:
                raise ValueError(f"Invalid budget value: '{t}'")

    results: List[Dict] = []
    # participating heads for per-head cap mapping
    participating_heads = len(topk)
    log("\nSweeping budgets...")
    pbar = tqdm(budgets, desc="Budgets", dynamic_ncols=True)
    for b in pbar:
        if args.verbose:
            print(f"\n  [Budget = {b}] Building surgery plan...")
        plan = build_surgery(
            model=model,
            cma_rows=topk,
            sae_release=args.sae_release,
            topn_per_layer=args.topn_per_layer,
            tau_min=args.tau_min,
            layers_limit=args.layers_limit,
            global_budget=b,
            budget_mode="head",
            device=args.device,
            verbose=args.verbose,
        )
        total_gated = count_total_gated(plan)
        pbar.set_postfix_str(f"cap={b}, gated={total_gated}")
        # per-head cap used this round (handle K=0 safely)
        per_head_cap_used = int(b // participating_heads) if participating_heads > 0 else 0

        # Apply surgery and measure TE (averaged over prompts)
        te_after_vals = []
        inner_pbar = tqdm(pairs, desc="Prompts", leave=False, dynamic_ncols=True)
        for base, cf in inner_pbar:
            b1_i, c1_i = run_with_surgery(model, plan, [base, cf], device=args.device)
            te_after_vals.append(c1_i - b1_i)
        inner_pbar.close()
        TE_1 = float(np.mean(te_after_vals)) if te_after_vals else float("nan")
        # For logging, compute biases on first pair after surgery
        b1_first, c1_first = run_with_surgery(model, plan, [pairs[0][0], pairs[0][1]], device=args.device)
        if args.verbose:
            print(f"    AFTER (mean TE over {len(pairs)} prompts) = {TE_1:+.4f}")

        # NIE under surgery (re-run CMA inside hooks)
        hooks = []
        for L, cfg in plan.items():
            hook_fn = make_layer_gate_hook(cfg["sae"], cfg["mask"], device=args.device)
            hooks.append((cfg["hook_name"], hook_fn))
        with model.hooks(fwd_hooks=hooks):
            nie_after_vals = []
            inner_pbar2 = tqdm(pairs, desc="Prompts", leave=False, dynamic_ncols=True)
            for base, cf in inner_pbar2:
                if args.nie_mode == "local":
                    nie_after_vals.append(compute_local_nie(model, base, cf, include_map))
                elif args.nie_mode == "heads":
                    effects = run_gender_bias_cma(model, base, cf, verbose=False, include_heads_by_layer=include_map)
                    nie_after_vals.append(float(sum(sum(row) for row in effects)))
                else:
                    nie_after_vals.append(compute_total_nie(model, base, cf))
            inner_pbar2.close()
        NIE_1 = float(np.mean(nie_after_vals)) if nie_after_vals else float("nan")
        if args.verbose:
            print(f"    NIE_after={NIE_1:+.4f}")

        # Baseline-aligned metric: Δ|NIE|，以及“剩余 NIE%”用于 Bias–PPL 图
        delta_nie = abs(NIE_1) - abs(NIE_0)
        eps = 1e-9
        remaining_nie_pct = (abs(NIE_1) / max(abs(NIE_0), eps)) * 100.0

        # PPL after (and ΔPPL)
        ppl_1 = None
        delta_ppl = None
        if ppl_text_all:
            with model.hooks(fwd_hooks=hooks):
                nll_1 = compute_avg_nll(model, ppl_text_all, device=args.device)
            ppl_1 = math.exp(nll_1) if not math.isnan(nll_1) else float("nan")
            delta_ppl = None if ppl_0 is None or math.isnan(ppl_0) else (ppl_1 - ppl_0)
            if args.verbose:
                print(f"    PPL_after={ppl_1:.4f}  ΔPPL={delta_ppl:+.4f}")

        results.append({
            "budget": b,
            "total_gated": total_gated,
            "per_head_cap": per_head_cap_used,
            "participating_heads": participating_heads,
            "bias_base_before": bias_base_0,
            "bias_cf_before": bias_cf_0,
            "TE_before": TE_0,
            "NIE_before": NIE_0,
            "bias_base_after": b1_first,
            "bias_cf_after": c1_first,
            "TE_after": TE_1,
            "NIE_after": NIE_1,
            "delta_nie": delta_nie,
            "remaining_nie_pct": remaining_nie_pct,
            "PPL_before": ppl_0,
            "PPL_after": ppl_1,
            "delta_ppl": delta_ppl,
            "topk": args.top_k_heads,
            "prompt_split": args.prompt_split,
            "sae_release": args.sae_release,
            "device": args.device,
        })

    # Mark Pareto frontier (X = Δ|NIE|, Y = ΔPPL)
    x_key = "delta_nie"
    y_key = "delta_ppl"
    pareto_flags = compute_pareto_frontier(results, x_key=x_key, y_key=y_key)
    for rec, flag in zip(results, pareto_flags):
        rec["is_pareto"] = bool(flag)

    # Save outputs using unified naming: pareto_topk{topk}_{ts}_...
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_stem = os.path.join(args.out_dir, f"pareto_topk{args.top_k_heads}_{ts}")
    csv_path = f"{base_stem}.csv"
    pd.DataFrame(results).sort_values(by=[x_key, y_key], ascending=[True, True]).to_csv(csv_path, index=False)
    log(f"\nSaved sweep CSV → {csv_path}")

    # Plot
    if px is None or go is None:
        log("Plotly is not installed; skipping HTML plot. Install plotly to enable plotting.")
        return

    df = pd.DataFrame(results)
    fig = px.scatter(
        df,
        x=x_key,
        y=y_key,
        color="total_gated",
        hover_data=["budget", "TE_before", "TE_after", "NIE_before", "NIE_after", "PPL_before", "PPL_after"],
        title=f"SFC-lite Bias–Perplexity Pareto (X=Δ|NIE|, Y=ΔPPL)",
        template="simple_white",
    )
    # Overlay Pareto frontier line
    df_front = df[df["is_pareto"]].sort_values(by=x_key)
    if not df_front.empty:
        fig.add_trace(go.Scatter(
            x=df_front[x_key],
            y=df_front[y_key],
            mode="lines+markers",
            name="Pareto frontier",
            line=dict(color="black", width=2),
            marker=dict(symbol="diamond", size=8),
        ))
    html_path = f"{base_stem}_nie_scatter.html"
    fig.write_html(html_path)
    log(f"Saved Pareto plot (HTML) → {html_path}")
    # Matplotlib static PNG for Δ|NIE| vs ΔPPL
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[x_key], df[y_key], c=df.get("total_gated", 0), cmap="viridis", alpha=0.8)
        plt.xlabel("Δ|NIE|")
        plt.ylabel("ΔPPL")
        plt.title("Δ|NIE| vs ΔPPL")
        png_path = f"{base_stem}_nie_scatter.png"
        plt.grid(True, alpha=0.3)
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Saved PNG → {png_path}")
    except Exception:
        pass

    # Second plot: Bias–PPL Pareto (remaining NIE% vs ΔPPL)
    if "remaining_nie_pct" in df.columns:
        df2 = df.copy()
        fig2 = px.scatter(
            df2,
            x="remaining_nie_pct",
            y="delta_ppl",
            color="total_gated",
            hover_data=["budget", "NIE_before", "NIE_after", "PPL_before", "PPL_after"],
            title="Bias–PPL Pareto (X=remaining NIE %, Y=ΔPPL)",
            template="simple_white",
        )
        df2_front = df2[df2["is_pareto"]].sort_values(by="remaining_nie_pct")
        if not df2_front.empty:
            fig2.add_trace(go.Scatter(
                x=df2_front["remaining_nie_pct"],
                y=df2_front["delta_ppl"],
                mode="lines+markers",
                name="Pareto frontier",
                line=dict(color="black", width=2),
                marker=dict(symbol="diamond", size=8),
            ))
        html_path2 = f"{base_stem}_bias_pareto.html"
        fig2.write_html(html_path2)
        log(f"Saved Bias–PPL Pareto (HTML) → {html_path2}")
        # Matplotlib static PNG for Bias–PPL
        try:
            plt.figure(figsize=(8, 5))
            plt.scatter(df2["remaining_nie_pct"], df2["delta_ppl"], c=df2.get("total_gated", 0), cmap="viridis", alpha=0.8)
            plt.xlabel("Remaining NIE (%)")
            plt.ylabel("ΔPPL")
            plt.title("Bias–PPL (Remaining NIE% vs ΔPPL)")
            png_path2 = f"{base_stem}_bias_pareto.png"
            plt.grid(True, alpha=0.3)
            plt.savefig(png_path2, dpi=150, bbox_inches="tight")
            plt.close()
            log(f"Saved PNG → {png_path2}")
        except Exception:
            pass


if __name__ == "__main__":
    main()


