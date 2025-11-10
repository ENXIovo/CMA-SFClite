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

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Reuse core logic from sfc_lite_from_cma
from sfc_lite_from_cma import (
    load_model as load_tl_model,
    read_topk_heads,
    build_surgery,
    pronoun_bias_score,
    run_with_surgery,
    make_layer_gate_hook,
)

# CMA (per-head) to compute NIE matrix
from cma_gender_bias import run_gender_bias_cma


def parse_args():
    ap = argparse.ArgumentParser("Pareto sweep for SFC-lite (budget vs. bias improvement)")
    ap.add_argument("--cma_csv", type=str, required=True,
                    help="Path to CMA CSV with columns: rank/layer/head/nie/abs_nie.")
    ap.add_argument("--sae_release", type=str, default="gpt2-small-attn-out-v5-32k",
                    help="SAE-Lens release for attn_out (post-WO).")
    ap.add_argument("--top_k_heads", type=int, default=10,
                    help="Number of heads to take from CMA CSV.")
    ap.add_argument("--topn_per_layer", type=int, default=1000,
                    help="Top-N features per layer (pre-budget) sorted by attribution mass.")
    ap.add_argument("--tau_min", type=float, default=0.05,
                    help="Loose minimum mass floor.")
    ap.add_argument("--layers_limit", type=int, default=None,
                    help="Optional: restrict to the first N unique layers in CMA selection.")

    ap.add_argument("--budgets", type=str, default="0,100,300,600,1200,2400,4800",
                    help="Comma-separated global budgets to sweep; 0 disables cap.")

    ap.add_argument("--prompt_base", type=str, default="The nurse said that",
                    help="Base prompt for bias measurement.")
    ap.add_argument("--prompt_cf", type=str, default="The man said that",
                    help="Counterfactual prompt for bias measurement.")

    ap.add_argument("--ppl_file", type=str, default=None,
                    help="Text file path for PPL computation (preferred).")
    ap.add_argument("--ppl_text", type=str, default=None,
                    help="Fallback single string for PPL computation if no file is provided.")

    ap.add_argument("--bias_axis", type=str, default="te", choices=["te", "nie"],
                    help="Which bias metric to show on X axis (remaining % of baseline).")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run on.")
    ap.add_argument("--out_dir", type=str, default="results",
                    help="Output directory for CSV/HTML.")
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
    effects = run_gender_bias_cma(model, base_prompt, cf_prompt)
    return float(sum(sum(row) for row in effects))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model on {args.device} ...")
    model = load_tl_model(device=args.device)

    print(f"Reading CMA top-K heads from {args.cma_csv} ...")
    topk = read_topk_heads(args.cma_csv, args.top_k_heads)
    print("Top-K heads (layer, head, NIE):")
    for (L, h, nie) in topk:
        print(f"  L{L:02d} H{h:02d}  NIE={nie:+.4f}")

    # Baseline bias metrics (no surgery): TE and NIE
    print("\nComputing baseline TE/NIE (no surgery)...")
    bias_base_0 = pronoun_bias_score(model, args.prompt_base, device=args.device)
    bias_cf_0   = pronoun_bias_score(model, args.prompt_cf,   device=args.device)
    TE_0 = bias_cf_0 - bias_base_0
    NIE_0 = compute_total_nie(model, args.prompt_base, args.prompt_cf)
    print(f"  bias_base={bias_base_0:+.4f}  bias_cf={bias_cf_0:+.4f}  TE={TE_0:+.4f}  NIE={NIE_0:+.4f}")

    # PPL baseline on provided text
    ppl_text_all = read_text_for_ppl(args.ppl_file, args.ppl_text)
    ppl_0 = None
    nll_0 = None
    if ppl_text_all:
        print("Computing baseline Perplexity on provided text...")
        nll_0 = compute_avg_nll(model, ppl_text_all, device=args.device)
        ppl_0 = math.exp(nll_0) if not math.isnan(nll_0) else float("nan")
        print(f"  PPL_before={ppl_0:.4f}")
    else:
        print("No PPL text provided; ΔPPL will be NaN.")

    # Parse budgets
    budgets: List[int] = []
    for t in args.budgets.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            budgets.append(int(t))
        except ValueError:
            raise ValueError(f"Invalid budget value: '{t}'")

    results: List[Dict] = []
    print("\nSweeping budgets...")
    for b in budgets:
        print(f"\n  [Budget = {b}] Building surgery plan...")
        plan = build_surgery(
            model=model,
            cma_rows=topk,
            sae_release=args.sae_release,
            topn_per_layer=args.topn_per_layer,
            tau_min=args.tau_min,
            layers_limit=args.layers_limit,
            global_budget=b,
            device=args.device,
        )
        total_gated = count_total_gated(plan)
        print(f"    total gated features = {total_gated}")

        # Apply surgery and measure TE
        b1, c1 = run_with_surgery(model, plan, [args.prompt_base, args.prompt_cf], device=args.device)
        TE_1 = c1 - b1
        print(f"    AFTER: bias_base={b1:+.4f}  bias_cf={c1:+.4f}  TE={TE_1:+.4f}")

        # NIE under surgery (re-run CMA inside hooks)
        hooks = []
        for L, cfg in plan.items():
            hook_fn = make_layer_gate_hook(cfg["sae"], cfg["mask"], device=args.device)
            hooks.append((cfg["hook_name"], hook_fn))
        with model.hooks(fwd_hooks=hooks):
            NIE_1 = compute_total_nie(model, args.prompt_base, args.prompt_cf)
        print(f"    NIE_after={NIE_1:+.4f}")

        # Remaining bias (% of baseline)
        eps = 1e-9
        remaining_te_pct = (abs(TE_1) / max(abs(TE_0), eps)) * 100.0
        remaining_nie_pct = (abs(NIE_1) / max(abs(NIE_0), eps)) * 100.0

        # PPL after (and ΔPPL)
        ppl_1 = None
        delta_ppl = None
        if ppl_text_all:
            with model.hooks(fwd_hooks=hooks):
                nll_1 = compute_avg_nll(model, ppl_text_all, device=args.device)
            ppl_1 = math.exp(nll_1) if not math.isnan(nll_1) else float("nan")
            delta_ppl = None if ppl_0 is None or math.isnan(ppl_0) else (ppl_1 - ppl_0)
            print(f"    PPL_after={ppl_1:.4f}  ΔPPL={delta_ppl:+.4f}")

        results.append({
            "budget": b,
            "total_gated": total_gated,
            "bias_base_before": bias_base_0,
            "bias_cf_before": bias_cf_0,
            "TE_before": TE_0,
            "NIE_before": NIE_0,
            "bias_base_after": b1,
            "bias_cf_after": c1,
            "TE_after": TE_1,
            "NIE_after": NIE_1,
            "remaining_te_pct": remaining_te_pct,
            "remaining_nie_pct": remaining_nie_pct,
            "PPL_before": ppl_0,
            "PPL_after": ppl_1,
            "delta_ppl": delta_ppl,
        })

    # Mark Pareto frontier
    x_key = "remaining_te_pct" if args.bias_axis == "te" else "remaining_nie_pct"
    y_key = "delta_ppl"
    pareto_flags = compute_pareto_frontier(results, x_key=x_key, y_key=y_key)
    for rec, flag in zip(results, pareto_flags):
        rec["is_pareto"] = bool(flag)

    # Save CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"pareto_{ts}.csv")
    pd.DataFrame(results).sort_values(by=[x_key, y_key], ascending=[True, True]).to_csv(csv_path, index=False)
    print(f"\nSaved sweep CSV → {csv_path}")

    # Plot
    if px is None or go is None:
        print("Plotly is not installed; skipping HTML plot. Install plotly to enable plotting.")
        return

    df = pd.DataFrame(results)
    fig = px.scatter(
        df,
        x=x_key,
        y=y_key,
        color="total_gated",
        hover_data=["budget", "TE_before", "TE_after", "NIE_before", "NIE_after", "PPL_before", "PPL_after"],
        title=f"SFC-lite Bias–Perplexity Pareto (X={x_key}, Y={y_key})",
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
    html_path = os.path.join(args.out_dir, f"pareto_{ts}.html")
    fig.write_html(html_path)
    print(f"Saved Pareto plot (HTML) → {html_path}")


if __name__ == "__main__":
    main()


