"""
Minimal SFC-lite surgery driven by CMA top-K heads (attn_out / post-WO).
- Loads CMA CSV (layer, head, nie, rank)
- Picks layers/heads
- Loads attn_out SAE per layer (gpt2-small-attn-out-v5-32k)
- Attributes SAE features to heads via WO row-space projection
- Gates features strongly attributed to selected heads
- Evaluates a tiny bias metric before/after (for sanity)

Usage:
  pip install -U torch transformer_lens sae_lens pandas
  python sfc_lite_from_cma.py \
      --cma_csv results/gpt2-small_nurse_man_20251028_042958.csv \
      --top_k_heads 5 \
      --topn_per_layer 300 \
      --tau_min 0.12 \
      --global_budget 3000
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

from transformer_lens import HookedTransformer
from sae_lens import SAE


# =========================================================
# 1) CLI
# =========================================================
def get_args():
    ap = argparse.ArgumentParser("SFC-lite gate by CMA top-K heads (attn_out, post-WO)")

    # --- inputs / outputs
    ap.add_argument("--cma_csv", type=str, required=True,
                    help="Path to CMA CSV (needs columns: layer, head, nie, rank or abs_nie).")

    # --- CMA selection
    ap.add_argument("--top_k_heads", type=int, default=10,
                    help="Take top-K heads from CMA (by 'rank' if present else by |NIE|).")
    ap.add_argument("--layers_limit", type=int, default=None,
                    help="Optional: only keep the first N unique layers from the selected heads.")

    # --- SAE source (attn_out / post-WO)
    ap.add_argument("--sae_release", type=str, default="gpt2-small-attn-out-v5-32k",
                    help="SAE-Lens release name (not HF repo path).")

    # --- feature selection (hybrid: Top-N per layer + a loose floor)
    ap.add_argument("--topn_per_layer", type=int, default=300,
                    help="How many features per layer to gate (sorted by head-attribution mass).")
    ap.add_argument("--tau_min", type=float, default=0.12,
                    help="Loose minimum mass floor; features below this are dropped even if in Top-N.")
    ap.add_argument("--global_budget", type=int, default=3000,
                    help="Global cap across all layers after per-layer picks (0 or negative to disable).")

    # --- device / run
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry_run", action="store_true",
                    help="Build the plan and print stats, but do not run a forward pass.")

    # --- tiny bias sanity prompts (replace with real eval later)
    ap.add_argument("--prompt_base", type=str, default="The nurse said that")
    ap.add_argument("--prompt_cf",   type=str, default="The man said that")

    return ap.parse_args()


# =========================================================
# 2) Model
# =========================================================
def load_model(device: str = "cpu") -> HookedTransformer:
    """
    Load GPT-2 small into TransformerLens, eval mode.
    """
    model = HookedTransformer.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    return model


# =========================================================
# 3) CMA heads
# =========================================================
def read_topk_heads(cma_csv: str, top_k: int) -> List[Tuple[int, int, float]]:
    """
    Read CMA CSV and return top-K (layer, head, nie) tuples.

    Priority:
      - If 'rank' exists, sort ascending by rank (1 = best)
      - Else sort by 'abs_nie' (desc) if present, otherwise by 'nie' (desc)
    """
    df = pd.read_csv(cma_csv, comment="#")
    if "rank" in df.columns:
        df = df.sort_values(by="rank", ascending=True)
    elif "abs_nie" in df.columns:
        df = df.sort_values(by="abs_nie", ascending=False)
    elif "nie" in df.columns:
        df = df.sort_values(by="nie", ascending=False)
    else:
        raise ValueError("CSV must contain either 'rank' or 'abs_nie' or 'nie'.")

    sel = df.head(top_k)[["layer", "head", "nie"]].values.tolist()
    return [(int(L), int(h), float(nie)) for (L, h, nie) in sel]


# =========================================================
# 4) Per-head row-space projectors from W_O
# =========================================================
def build_head_rowspace_projectors(model: HookedTransformer, L: int, device: str) -> List[torch.Tensor]:
    """
    Build row-space projectors P_h for each head at layer L.

    In TransformerLens, W_O has shape [n_heads, d_head, d_model].
    Each head's output v_h (1 x d_head) maps to model space via v_h @ W_O[h] -> (1 x d_model).
    The reachable set is the ROW-SPACE of W_O[h]. Orthogonal projector:
        P_h = W_h^T (W_h W_h^T)^(-1) W_h
      where W_h is [d_head, d_model].
    """
    W_O = model.blocks[L].attn.W_O  # [n_heads, d_head, d_model]
    n_heads, d_head, d_model = W_O.shape
    projectors = []
    eye = torch.eye(d_head, device=device)

    for h in range(n_heads):
        W_h = W_O[h].to(device)                    # [d_head, d_model]
        gram = W_h @ W_h.T                         # [d_head, d_head]
        gram_inv = torch.inverse(gram + 1e-6 * eye)
        P_h = W_h.T @ gram_inv @ W_h               # [d_model, d_model]
        projectors.append(P_h)

    return projectors


# =========================================================
# 5) SAE loader (attn_out / post-WO)
# =========================================================
def load_sae_for_layer(release: str, layer: int, model: HookedTransformer, device: str):
    """
    SAE-Lens new API: SAE.from_pretrained(release, sae_id) -> SAE
    Use attn_out / post-WO hook id.
    """
    sae_id = f"blocks.{layer}.hook_attn_out"
    sae = SAE.from_pretrained(release, sae_id)
    sae.to(model.W_U.device if hasattr(model, "W_U") else device)
    sae.eval()
    return sae


# =========================================================
# 6) Feature→Head attribution via projectors
# =========================================================
def feature_head_attribution(sae, projectors: List[torch.Tensor], device: str) -> torch.Tensor:
    """
    Compute attribution mass per (feature, head):
      A[i, h] = || P_h d_i ||_2

    Where d_i is the decoder vector of feature i in model space.
    SAE-lens stores decoder as W_dec with either shape [d_model, d_sae] or [d_sae, d_model].
    We normalize per-feature to get a distribution over heads (mass sums to 1.0 per feature).
    """
    W_dec = sae.W_dec.detach().to(device)
    # Ensure columns are features: D = [d_model, d_sae]
    if W_dec.shape[0] <= W_dec.shape[1]:
        D = W_dec
    else:
        D = W_dec.T

    d_model, d_sae = D.shape
    A = torch.zeros((d_sae, len(projectors)), device=device)

    for h, P_h in enumerate(projectors):
        # Project all decoder vectors onto head h's row space
        proj = P_h @ D  # [d_model, d_sae]
        # L2 per feature (column)
        A[:, h] = torch.linalg.vector_norm(proj, dim=0)

    mass = A / (A.sum(dim=1, keepdim=True) + 1e-9)  # normalize per feature
    return mass  # [n_features, n_heads]


# =========================================================
# 7) Hybrid selection: Top-N per layer + loose floor tau_min
# =========================================================
def pick_features_hybrid(
    mass: torch.Tensor,
    heads_L: List[int],
    topn_per_layer: int = 300,
    tau_min: float = 0.12,
) -> np.ndarray:
    """
    mass: [n_features, n_heads]  (row-normalized attribution per feature)
    heads_L: head indices (this layer) picked from CMA

    Steps:
      1) sel_mass = sum mass over the selected heads
      2) take Top-N features by sel_mass
      3) drop those whose sel_mass < tau_min (loose safety floor)
    """
    if len(heads_L) == 0:
        return np.zeros(mass.shape[0], dtype=bool)

    sel_mass = mass[:, heads_L].sum(dim=1)  # [n_features]
    topn = int(min(topn_per_layer, sel_mass.shape[0]))

    if topn <= 0:
        return np.zeros(mass.shape[0], dtype=bool)

    idx = torch.topk(sel_mass, k=topn, largest=True).indices
    mask = torch.zeros_like(sel_mass, dtype=torch.bool)
    mask[idx] = True

    if tau_min is not None and tau_min > 0:
        mask = mask & (sel_mass >= tau_min)

    return mask.detach().cpu().numpy()

# ========= SAE encode/decode 兼容封装 =========
def sae_encode_compat(sae, x):
    """
    兼容两种返回形式：
      - 新版：sae.encode(x) -> s
      - 旧版：sae.encode(x) -> (s, cache/info)
    统一返回：(s, cache)；新版则 cache=None
    """
    out = sae.encode(x)
    if isinstance(out, tuple):
        s, cache = out
    else:
        s, cache = out, None
    return s, cache

def sae_decode_compat(sae, s, cache=None):
    """
    兼容两种调用形式：
      - 新版：sae.decode(s)
      - 旧版：sae.decode(s, cache/info)
    自动尝试合适的签名。
    """
    try:
        if cache is not None:
            return sae.decode(s, cache)
        else:
            return sae.decode(s)
    except TypeError:
        # 有些实现即便有 cache 也只接受 s
        return sae.decode(s)

# =========================================================
# 8) Hook builder (gating masked SAE features)
# =========================================================
def make_layer_gate_hook(sae, feature_mask: np.ndarray, device: str):
    """
    Forward hook for blocks.L.hook_attn_out:
      - Encode activations into SAE latent s
      - Zero out selected features
      - Decode back to edited activations
    兼容 SAE-Lens 新旧 encode/decode API。
    """
    feat_idx_np = np.where(feature_mask)[0]
    if len(feat_idx_np) == 0:
        # No-op hook for this layer
        def no_op_hook(y, hook):
            return y
        return no_op_hook

    feat_idx = torch.tensor(feat_idx_np, device=device, dtype=torch.long)

    def gate_hook(y, hook):
        # y: [batch, pos, d_model]
        b, t, d = y.shape
        y2 = y.reshape(-1, d).to(device)            # [B*T, d_model]
        y2 = y2.float()                              # 避免 dtype 不匹配

        with torch.no_grad():
            # 兼容获取 s 与 cache/info
            s, cache = sae_encode_compat(sae, y2)   # s: [B*T, d_sae]
            if feat_idx.numel() > 0:
                s[:, feat_idx] = 0.0                # gate 选中特征
            y_new = sae_decode_compat(sae, s, cache)  # [B*T, d_model]
            y_new = y_new.reshape(b, t, d)

        return y_new

    return gate_hook


# =========================================================
# 9) Build surgery plan
# =========================================================
def build_surgery(
    model: HookedTransformer,
    cma_rows: List[Tuple[int, int, float]],
    sae_release: str,
    topn_per_layer: int,
    tau_min: float,
    layers_limit: int,
    global_budget: int,
    device: str,
) -> Dict[int, Dict]:
    """
    Returns: plan dict
      layer -> {
        "heads": List[int],
        "sae": SAE,
        "mask": np.ndarray(bool) of shape [n_features],
        "hook_name": str,
        "n_features_total": int,
        "n_features_gate": int
      }
    """
    # Group heads by layer
    by_layer: Dict[int, List[int]] = {}
    for L, h, _nie in cma_rows:
        by_layer.setdefault(L, []).append(h)

    # Optional: restrict to first N unique layers
    layers = sorted(by_layer.keys())
    if layers_limit is not None:
        layers = layers[:layers_limit]

    # Per-layer selection
    raw_masks: Dict[int, np.ndarray] = {}
    layer_info: Dict[int, Dict] = {}
    for L in layers:
        heads_L = sorted(set(by_layer[L]))

        # (1) Build projectors
        projs = build_head_rowspace_projectors(model, L, device)

        # (2) Load SAE for attn_out (post-WO)
        sae = load_sae_for_layer(sae_release, L, model, device)

        # (3) Attribute features to heads
        mass = feature_head_attribution(sae, projs, device)

        # (3.1) Diagnostics: print a few quantiles to guide tau_min/topN
        sel_mass = mass[:, heads_L].sum(dim=1)
        qs = torch.tensor([0.5, 0.8, 0.9, 0.95, 0.99], device=sel_mass.device)
        quant = torch.quantile(sel_mass, qs)
        print(
            f"Layer {L} attribution mass over heads {heads_L} :: "
            f"median={quant[0]:.3f} p80={quant[1]:.3f} p90={quant[2]:.3f} "
            f"p95={quant[3]:.3f} p99={quant[4]:.3f}"
        )

        # (4) Hybrid pick: Top-N per layer + tau_min
        mask = pick_features_hybrid(
            mass, heads_L, topn_per_layer=topn_per_layer, tau_min=tau_min
        )

        raw_masks[L] = mask
        layer_info[L] = {
            "heads": heads_L,
            "sae": sae,
            "hook_name": f"blocks.{L}.hook_attn_out",
            "n_features_total": mass.shape[0],
        }

    # (5) Global budget (optional): downsample per layer to meet total cap
    if global_budget and global_budget > 0:
        # Flatten all picked indices
        all_idx = []
        for L, m in raw_masks.items():
            idx = np.where(m)[0]
            all_idx.extend([(L, int(i)) for i in idx])

        over = len(all_idx) - global_budget
        if over > 0:
            # Simple fair cut: cap each layer to per_layer_cap
            per_layer_cap = max(1, global_budget // max(1, len(raw_masks)))
            for L in raw_masks.keys():
                idx = np.where(raw_masks[L])[0]
                if len(idx) > per_layer_cap:
                    # Keep the first per_layer_cap (they are already the largest by our Top-N)
                    keep = np.sort(idx)[:per_layer_cap]
                    new_mask = np.zeros_like(raw_masks[L], dtype=bool)
                    new_mask[keep] = True
                    raw_masks[L] = new_mask

    # (6) Assemble final plan
    plan: Dict[int, Dict] = {}
    for L in layers:
        mask = raw_masks[L]
        info = layer_info[L]
        plan[L] = {
            "heads": info["heads"],
            "sae": info["sae"],
            "mask": mask,
            "hook_name": info["hook_name"],
            "n_features_total": int(info["n_features_total"]),
            "n_features_gate": int(mask.sum()),
        }
    return plan


# =========================================================
# 10) Tiny bias metric (demo only)
# =========================================================
@torch.no_grad()
def pronoun_bias_score(model: HookedTransformer, prompt: str, device: str) -> float:
    """
    bias = logit(" she") - logit(" he") at next token.
    (Demo only; replace with your real evaluation.)
    """
    toks = model.to_tokens(prompt, prepend_bos=True).to(device)
    logits = model(toks)
    next_logits = logits[0, -1]  # [vocab]
    id_she = model.to_single_token(" she")
    id_he = model.to_single_token(" he")
    return (next_logits[id_she] - next_logits[id_he]).item()


def run_with_surgery(model, plan: Dict[int, Dict], text_list: List[str], device: str):
    """
    Apply per-layer gating hooks and get bias scores for each text in text_list.
    """
    hooks = []
    for L, cfg in plan.items():
        hook_name = cfg["hook_name"]
        hook_fn = make_layer_gate_hook(cfg["sae"], cfg["mask"], device=device)
        hooks.append((hook_name, hook_fn))

    outs = []
    with model.hooks(fwd_hooks=hooks):
        for s in text_list:
            outs.append(pronoun_bias_score(model, s, device=device))
    return outs


# =========================================================
# 11) Main
# =========================================================
def main():
    args = get_args()

    print(f"Loading model on {args.device} ...")
    model = load_model(device=args.device)

    print(f"Reading CMA top-K heads from {args.cma_csv} ...")
    topk = read_topk_heads(args.cma_csv, args.top_k_heads)
    print("Top-K heads (layer, head, NIE):")
    for (L, h, nie) in topk:
        print(f"  L{L:02d} H{h:02d}  NIE={nie:+.4f}")

    print(f"\nBuilding SFC-lite plan using SAE release: {args.sae_release}")
    plan = build_surgery(
        model=model,
        cma_rows=topk,
        sae_release=args.sae_release,
        topn_per_layer=args.topn_per_layer,
        tau_min=args.tau_min,
        layers_limit=args.layers_limit,
        global_budget=args.global_budget,
        device=args.device,
    )

    for L, cfg in plan.items():
        ratio = cfg['n_features_gate'] / max(1, cfg['n_features_total'])
        print(f"Layer {L}: heads={cfg['heads']}, gate_features={cfg['n_features_gate']}/{cfg['n_features_total']} "
              f"({ratio:.1%})")

    if args.dry_run:
        print("\n[DRY RUN] Surgery plan built; no forward pass executed.")
        return

    # Quick demo evaluation
    base, cf = args.prompt_base, args.prompt_cf
    print("\nQuick bias sanity check (Δ = after - before):")
    b0 = pronoun_bias_score(model, base, device=args.device)
    c0 = pronoun_bias_score(model, cf,   device=args.device)
    print(f"  BEFORE  bias_base={b0:+.4f}   bias_cf={c0:+.4f}")

    b1, = run_with_surgery(model, plan, [base], device=args.device)
    c1, = run_with_surgery(model, plan, [cf],   device=args.device)
    print(f"  AFTER   bias_base={b1:+.4f}   bias_cf={c1:+.4f}")
    print(f"  Δbase={b1-b0:+.4f}  Δcf={c1-c0:+.4f}")

    # Save plan for reproducibility
    dump = {
        "sae_release": args.sae_release,
        "top_k_heads": args.top_k_heads,
        "topn_per_layer": args.topn_per_layer,
        "tau_min": args.tau_min,
        "global_budget": args.global_budget,
        "layers": {
            str(L): {
                "heads": cfg["heads"],
                "n_features_gate": cfg["n_features_gate"],
                "n_features_total": cfg["n_features_total"],
            } for L, cfg in plan.items()
        }
    }
    with open("sfc_surgery_plan.json", "w") as f:
        json.dump(dump, f, indent=2)
    print("\nSaved plan → sfc_surgery_plan.json")


if __name__ == "__main__":
    main()
