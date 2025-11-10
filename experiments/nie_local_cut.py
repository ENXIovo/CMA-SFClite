"""
NIE Local Cut Experiment
------------------------
Runs feature-level hard ablations (alpha = 0) on top CMA mediators using TransformerLens.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from tqdm import tqdm

try:
    from prompt_sets import get_prompt_examples
except ModuleNotFoundError:
    from experiments.prompt_sets import get_prompt_examples


# =========================================================================
# Configuration
# =========================================================================

SAE_MODE = "attn"  # 修改为 "attn" 即可切换到注意力 SAE

if SAE_MODE == "attn":
    SAE_RELEASE = "gpt2-small-attn-out-v5-128k"
    SAE_HOOK_TEMPLATE = "blocks.{layer}.hook_attn_out"
else:
    SAE_RELEASE = "gpt2-small-res-jb"
    SAE_HOOK_TEMPLATE = "blocks.{layer}.hook_resid_pre"


# =========================================================================
# Helper structures
# =========================================================================


@dataclass(frozen=True)
class Submodule:
    name: str
    hook_name: str
    kind: str
    layer: int

    def __hash__(self) -> int:
        return hash((self.name, self.hook_name))


@dataclass
class NodeMask:
    act: Optional[torch.Tensor] = None
    resc: Optional[torch.Tensor] = None

    def complemented(self) -> "NodeMask":
        act = None if self.act is None else (~self.act)
        resc = None if self.resc is None else (~self.resc)
        return NodeMask(act=act, resc=resc)


@dataclass
class PatchState:
    act: torch.Tensor
    res: Optional[torch.Tensor] = None

    def clone(self) -> "PatchState":
        return PatchState(
            act=self.act.clone(),
            res=None if self.res is None else self.res.clone(),
        )


@dataclass
class FeatureEffect:
    scores: torch.Tensor
    delta: torch.Tensor
    grad: torch.Tensor
    total_effect: torch.Tensor


DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


# =========================================================================
# SAE loading utilities
# =========================================================================


def load_layer_sae(layer_idx: int, device: str) -> SAE:
    sae_id = SAE_HOOK_TEMPLATE.format(layer=layer_idx)
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    return sae


def load_saes_and_submodules(
    model: HookedTransformer,
    device: str,
) -> Tuple[DictionaryStash, Dict[Submodule, SAE]]:
    attns: List[Submodule] = []
    resids: List[Submodule] = []
    dictionaries: Dict[Submodule, SAE] = {}

    for layer_idx in range(model.cfg.n_layers):
        sae = load_layer_sae(layer_idx, device=device)
        hook_name = sae.cfg.metadata.get("hook_name", SAE_HOOK_TEMPLATE.format(layer=layer_idx))
        submodule = Submodule(
            name=f"{SAE_MODE}_{layer_idx}",
            hook_name=hook_name,
            kind=SAE_MODE,
            layer=layer_idx,
        )
        if SAE_MODE == "attn":
            attns.append(submodule)
        else:
            resids.append(submodule)
        dictionaries[submodule] = sae

    stash = DictionaryStash(embed=None, attns=attns, mlps=[], resids=resids)
    return stash, dictionaries


def find_mediator_submodule(mediator: Dict, stash: DictionaryStash) -> Tuple[Optional[Submodule], str]:
    layer_idx = mediator["layer"]
    mediator_type = mediator.get("type", "attn")
    if mediator_type == "attn" and layer_idx < len(stash.attns):
        return stash.attns[layer_idx], "attn"
    if mediator_type == "resid" and layer_idx < len(stash.resids):
        return stash.resids[layer_idx], "resid"
    if layer_idx < len(stash.resids):
        return stash.resids[layer_idx], mediator_type
    return None, mediator_type


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
        if not row:
            continue
        row = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
        if row.get("rank") in (None, ""):
            continue
        mediator = {
            "rank": int(row["rank"]),
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "nie": float(row["nie"]) if row.get("nie") not in (None, "") else None,
            "abs_nie": float(row["abs_nie"]) if row.get("abs_nie") not in (None, "") else None,
            "type": "attn",
        }
        mediators.append(mediator)

    if limit is not None and limit > 0:
        mediators = mediators[:limit]
    return mediators


# =========================================================================
# TransformerLens helpers
# =========================================================================


def tokenize_prompt(model: HookedTransformer, prompt: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=False)
    return tokens.to(model.cfg.device)


def _ensure_same_shape(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


def _expand_to(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
    expanded = tensor
    while expanded.dim() < target_ndim:
        expanded = expanded.unsqueeze(0)
    return expanded


def build_feature_edit_hook(
    dictionary: SAE,
    node_mask: NodeMask,
    patch_state: PatchState,
    handle_errors: str,
) -> callable:
    feature_dim = int(dictionary.cfg.d_sae)
    act_mask = None if node_mask.act is None else node_mask.act.to(dtype=torch.bool).reshape(-1)
    if act_mask is None:
        act_mask = torch.zeros(feature_dim, dtype=torch.bool)
    elif act_mask.numel() != feature_dim:
        raise ValueError("Feature mask size mismatch.")
    replace_mask = (~act_mask).to(dtype=torch.bool)

    resc_mask = None
    if node_mask.resc is not None:
        resc_mask = node_mask.resc.to(dtype=torch.bool).reshape(-1)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        features = dictionary.encode(value)
        recon = dictionary.decode(features)
        residual = value - recon

        patch_act = patch_state.act.to(value.device, value.dtype)
        _ensure_same_shape("feature", features, patch_act)

        mask_expanded = _expand_to(replace_mask.to(value.device), features.dim())
        features = torch.where(mask_expanded, patch_act, features)

        if patch_state.res is not None:
            patch_res = patch_state.res.to(value.device, value.dtype)
            _ensure_same_shape("residual", residual, patch_res)
            if resc_mask is not None:
                if resc_mask.numel() == 1:
                    if not resc_mask.item():
                        residual = patch_res
                else:
                    res_replace = _expand_to((~resc_mask.to(value.device)), residual.dim())
                    residual = torch.where(res_replace, patch_res, residual)

        return dictionary.decode(features) + residual

    return hook_fn


def make_scaling_hook(
    dictionary: SAE,
    feature_indices: Optional[List[int]],
    alpha: float,
):
    device = next(dictionary.parameters()).device
    if feature_indices:
        feature_tensor = torch.tensor(feature_indices, device=device, dtype=torch.long)
    else:
        feature_tensor = None

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        features = dictionary.encode(value)
        recon = dictionary.decode(features)
        residual = value - recon
        if feature_tensor is None:
            features = features * alpha
        else:
            features[..., feature_tensor] = features[..., feature_tensor] * alpha
        return dictionary.decode(features) + residual

    return hook_fn


def tokenize_corpus(
    model: HookedTransformer,
    path: Optional[str],
    max_corpus_tokens: Optional[int] = None,
) -> Optional[torch.Tensor]:
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
    tokens: Optional[torch.Tensor],
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
            logits = model.run_with_hooks(slice_tokens, fwd_hooks=hooks, return_type="logits")
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


def run_with_ablations(
    model: HookedTransformer,
    clean_prompt: str,
    patch_prompt: Optional[str],
    submodules: Sequence[Submodule],
    dictionaries: Dict[Submodule, SAE],
    nodes: Dict[Submodule, NodeMask],
    complement: bool = False,
    ablation_fn=None,
    handle_errors: str = "default",
) -> torch.Tensor:
    if not submodules:
        raise ValueError("No submodules provided for ablation.")

    if patch_prompt is None:
        patch_prompt = clean_prompt

    clean_tokens = tokenize_prompt(model, clean_prompt)
    patch_tokens = tokenize_prompt(model, patch_prompt)

    hook_names = {submodule.hook_name for submodule in submodules}

    with torch.no_grad():
        _, patch_cache = model.run_with_cache(
            patch_tokens,
            names_filter=lambda name: name in hook_names,
            remove_batch_dim=False,
            return_type="logits",
        )

    patch_states: Dict[Submodule, PatchState] = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        activation = patch_cache[submodule.hook_name]
        features = dictionary.encode(activation)
        recon = dictionary.decode(features)
        residual = activation - recon
        state = PatchState(act=features, res=residual)
        if ablation_fn is not None:
            state = ablation_fn(state)
        patch_states[submodule] = state

    hooks = []
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        node_mask = nodes[submodule]
        if complement:
            node_mask = node_mask.complemented()
        hook_fn = build_feature_edit_hook(
            dictionary=dictionary,
            node_mask=node_mask,
            patch_state=patch_states[submodule],
            handle_errors=handle_errors,
        )
        hooks.append((submodule.hook_name, hook_fn))

    with torch.no_grad():
        logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=hooks,
            return_type="logits",
        )
    return logits


# =========================================================================
# Metric helpers
# =========================================================================


def compute_bias_and_ppl(model: HookedTransformer, logits: torch.Tensor, tokens: torch.Tensor) -> Tuple[float, float]:
    id_she = model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
    id_he = model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]
    bias = (logits[0, -1, id_she] - logits[0, -1, id_he]).item()
    ppl = _perplexity_from_logits(logits[0], tokens)
    return bias, ppl


def _perplexity_from_logits(logits_val: torch.Tensor, tokens: torch.Tensor) -> float:
    if len(tokens) < 2:
        return float("inf")
    log_probs = torch.nn.functional.log_softmax(logits_val[:-1], dim=-1)
    target_log_probs = log_probs[range(len(tokens) - 1), tokens[1:]]
    return torch.exp(-target_log_probs.mean()).item()


def determine_dictionary_size(
    model: HookedTransformer,
    submodule: Submodule,
    dictionary: SAE,
    prompt: str,
) -> int:
    return int(dictionary.cfg.d_sae)


def apply_alpha_gate(
    model: HookedTransformer,
    submodule: Submodule,
    dictionary: SAE,
    base_prompt: str,
    cf_prompt: str,
    feature_indices: Optional[List[int]],
    alpha: float,
) -> Tuple[float, float, float, float, float]:
    tokens_base = tokenize_prompt(model, base_prompt)[0]

    dict_size = int(dictionary.cfg.d_sae)
    mask_keep = torch.ones(dict_size, dtype=torch.bool)
    if feature_indices is None:
        mask_keep[:] = False
    else:
        mask_keep[feature_indices] = False

    def make_nodes() -> Dict[Submodule, NodeMask]:
        return {
            submodule: NodeMask(
                act=mask_keep.clone(),
                resc=torch.ones(1, dtype=torch.bool),
            )
        }

    def ablation_fn(state: PatchState) -> PatchState:
        act = state.act.clone()
        target_mask = (~mask_keep).to(torch.bool)
        if feature_indices is None:
            act = act * alpha
        else:
            act[..., target_mask] = act[..., target_mask] * alpha
        res = state.res.clone() if state.res is not None else None
        return PatchState(act=act, res=res)

    with torch.no_grad():
        logits_clean = model(tokenize_prompt(model, base_prompt))

    logits_gated = run_with_ablations(
        model=model,
        clean_prompt=base_prompt,
        patch_prompt=base_prompt,
        submodules=[submodule],
        dictionaries={submodule: dictionary},
        nodes=make_nodes(),
        ablation_fn=ablation_fn,
        handle_errors="keep",
    )

    logits_cf = run_with_ablations(
        model=model,
        clean_prompt=base_prompt,
        patch_prompt=cf_prompt,
        submodules=[submodule],
        dictionaries={submodule: dictionary},
        nodes=make_nodes(),
        ablation_fn=ablation_fn,
        handle_errors="keep",
    )

    bias_clean, ppl_clean = compute_bias_and_ppl(model, logits_clean, tokens_base)
    bias_gated, ppl_gated = compute_bias_and_ppl(model, logits_gated, tokens_base)
    bias_cf_replaced, _ = compute_bias_and_ppl(model, logits_cf, tokens_base)
    # 与 baselines 对齐：NIE = bias_cf_replaced - bias_clean
    nie = bias_cf_replaced - bias_clean
    return bias_clean, bias_gated, ppl_clean, ppl_gated, nie


# =========================================================================
# Feature scoring utilities
# =========================================================================


def compute_feature_effect(
    model: HookedTransformer,
    submodule: Submodule,
    dictionary: SAE,
    base_prompt: str,
    cf_prompt: str,
    metric_fn,
) -> FeatureEffect:
    tokens_base = tokenize_prompt(model, base_prompt)
    tokens_cf = tokenize_prompt(model, cf_prompt)

    captured: Dict[str, torch.Tensor] = {}

    def grad_hook(value: torch.Tensor, hook) -> torch.Tensor:
        value = value.requires_grad_(True)
        features = dictionary.encode(value)
        recon = dictionary.decode(features)
        residual = (value - recon).detach()
        features.retain_grad()
        captured["features"] = features
        captured["residual"] = residual
        return recon + residual

    model.zero_grad(set_to_none=True)
    logits_clean = model.run_with_hooks(
        tokens_base,
        fwd_hooks=[(submodule.hook_name, grad_hook)],
        return_type="logits",
    )
    metric_clean = metric_fn(logits_clean)
    metric_clean.sum().backward()

    features_clean = captured["features"].detach()
    grad = captured["features"].grad.detach()

    with torch.no_grad():
        logits_cf, cache_cf = model.run_with_cache(
            tokens_cf,
            names_filter=lambda name: name == submodule.hook_name,
            remove_batch_dim=False,
            return_type="logits",
        )

    activation_cf = cache_cf[submodule.hook_name].to(features_clean.device, features_clean.dtype)
    features_cf = dictionary.encode(activation_cf)
    recon_cf = dictionary.decode(features_cf)
    residual_cf = activation_cf - recon_cf

    delta = features_cf - features_clean
    effect_tensor = delta * grad
    scores = effect_tensor.abs().mean(dim=tuple(range(effect_tensor.ndim - 1)))

    total_effect = metric_fn(logits_cf).detach() - metric_clean.detach()
    return FeatureEffect(
        scores=scores.detach(),
        delta=delta.detach(),
        grad=grad.detach(),
        total_effect=total_effect,
    )


# =========================================================================
# Local cut logic
# =========================================================================


def run_local_cut(
    model_name: str,
    ranking_csv_path: str,
    output_path: str,
    prompt_split: str,
    topk: int,
    control_count: int,
    num_features: int,
    seed: int,
    device: str,
    corpus_path: Optional[str],
    max_corpus_tokens: Optional[int],
) -> str:
    print("=" * 70)
    print("NIE Local Cut")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"Top-K CSV: {ranking_csv_path}")
    print(f"Prompt split: {prompt_split}")
    print(f"SAE 模式: {SAE_MODE}")
    print("=" * 70)

    dtype = torch.float32
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()

    stash, dictionaries = load_saes_and_submodules(model, device=device)

    total_needed = topk + control_count
    mediators = load_ranked_mediators_from_csv(ranking_csv_path, total_needed if total_needed > 0 else None)
    entries = [(med, "topk") for med in mediators[:topk]]
    entries.extend((med, "control") for med in mediators[topk: topk + control_count])

    examples = get_prompt_examples(prompt_split)
    if not examples:
        raise ValueError(f"Prompt split '{prompt_split}' 没有可用样本")

    rng = np.random.default_rng(seed)
    corpus_tokens = tokenize_corpus(model, corpus_path, max_corpus_tokens=max_corpus_tokens)
    baseline_corpus_ppl = compute_corpus_perplexity(model, corpus_tokens) if corpus_tokens is not None else float("nan")

    results: List[Dict] = []
    metric_fn = lambda logits: logits[:, -1, :]

    for idx, (mediator, category) in enumerate(tqdm(entries, desc="Mediators")):
        submodule, mediator_type = find_mediator_submodule(mediator, stash)
        if submodule is None:
            print(f"  ⚠ Mediator {idx} 无法匹配 hook")
            continue
        dictionary = dictionaries[submodule]

        valid_examples = []
        for example_idx, example in enumerate(examples):
            base_prompt = example.get("base", example.get("clean_prefix", ""))
            cf_prompt = example.get("counterfactual", example.get("patch_prefix", ""))
            if not base_prompt or not cf_prompt:
                continue
            bias_base, _, ppl_base, _, nie_base = apply_alpha_gate(
                model,
                submodule,
                dictionary,
                base_prompt,
                cf_prompt,
                feature_indices=None,
                alpha=1.0,
            )
            valid_examples.append({
                "example_id": example_idx,
                "base_prompt": base_prompt,
                "cf_prompt": cf_prompt,
                "bias": bias_base,
                "ppl": ppl_base,
                "nie": nie_base,
            })

        if not valid_examples:
            continue

        mean_bias_clean = float(np.mean([ex["bias"] for ex in valid_examples]))
        mean_ppl_clean = float(np.mean([ex["ppl"] for ex in valid_examples]))

        anchor = valid_examples[0]
        effect = compute_feature_effect(
            model=model,
            submodule=submodule,
            dictionary=dictionary,
            base_prompt=anchor["base_prompt"],
            cf_prompt=anchor["cf_prompt"],
            metric_fn=metric_fn,
        )
        scores = effect.scores
        dict_size = scores.shape[-1]
        top_k = min(num_features, dict_size)
        if top_k > 0:
            top_scores, top_indices = torch.topk(scores, k=top_k)
            top_features = list(zip(top_indices.tolist(), top_scores.tolist()))
        else:
            top_features = []
        random_features = []
        if dict_size > 0:
            rand_indices = rng.choice(dict_size, size=min(num_features, dict_size), replace=False).tolist()
            random_features = [(idx_val, 0.0) for idx_val in rand_indices]

        feature_groups = []
        if top_features:
            feature_groups.append(("cma", top_features))
        if random_features:
            feature_groups.append(("random", random_features))

        for source, feature_list in feature_groups:
            prefix: List[int] = []
            for feat_idx, feat_score in feature_list:
                prefix.append(int(feat_idx))
                bias_edit_vals = []
                ppl_edit_vals = []
                delta_nie_vals = []

                for entry in valid_examples:
                    _, bias_edit, _, ppl_edit, nie_after = apply_alpha_gate(
                        model,
                        submodule,
                        dictionary,
                        entry["base_prompt"],
                        entry["cf_prompt"],
                        feature_indices=prefix,
                        alpha=0.0,
                    )
                    bias_edit_vals.append(bias_edit)
                    ppl_edit_vals.append(ppl_edit)
                    # 与 baselines 对齐：Δ|NIE| = |NIE_after| - |NIE_before|
                    delta_nie_vals.append(abs(nie_after) - abs(entry["nie"]))

                if not bias_edit_vals:
                    continue

                mean_bias_edit = float(np.mean(bias_edit_vals))
                mean_ppl_edit = float(np.mean(ppl_edit_vals))
                mean_delta_nie = float(np.mean(delta_nie_vals))
                remaining_pct = float(abs(mean_bias_edit) / max(abs(mean_bias_clean), 1e-9))

                gated_corpus_ppl = float("nan")
                delta_corpus_ppl = float("nan")
                if corpus_tokens is not None:
                    hook_fn = make_scaling_hook(dictionary, prefix, 0.0)
                    gated_corpus_ppl = compute_corpus_perplexity(
                        model,
                        corpus_tokens,
                        hooks=[(submodule.hook_name, hook_fn)],
                    )
                    if not np.isnan(baseline_corpus_ppl):
                        delta_corpus_ppl = gated_corpus_ppl - baseline_corpus_ppl

                results.append({
                    "analysis": "local_cut",
                    "row_type": "aggregate",
                    "feature_idx": int(feat_idx),
                    "feature_set": json.dumps(prefix),
                    "prefix_size": len(prefix),
                    "feature_score": float(feat_score),
                    "feature_source": source,
                    "alpha": 0.0,
                    "feature_count": len(prefix),
                    "sum_abs_edit": len(prefix),
                    "bias_original_mean": mean_bias_clean,
                    "bias_edited_mean": mean_bias_edit,
                    "ppl_original_mean": mean_ppl_clean,
                    "ppl_edited_mean": mean_ppl_edit,
                    "delta_ppl_mean": mean_ppl_edit - mean_ppl_clean,
                    "remaining_bias_pct": remaining_pct,
                    "delta_nie_mean": mean_delta_nie,
                    "corpus_ppl_original": baseline_corpus_ppl,
                    "corpus_ppl_edited": gated_corpus_ppl,
                    "delta_corpus_ppl": delta_corpus_ppl,
                    "mediator_layer": mediator.get("layer"),
                    "mediator_type": mediator_type,
                    "mediator_head": mediator.get("head"),
                    "mediator_nie": mediator.get("nie"),
                    "mediator_category": category,
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ 结果已写入 {output_path}")
    plot_local_cut_results(results, output_path)
    pareto_data = build_pareto_frontiers(results)
    if pareto_data:
        pareto_path = os.path.splitext(output_path)[0] + "_pareto.json"
        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(pareto_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 保存 Pareto 数据: {pareto_path}")
        plot_pareto_curves(pareto_data, output_path, "Local Cut")
    return output_path


def plot_local_cut_results(rows: List[Dict], csv_path: str) -> None:
    agg_rows = [r for r in rows if r.get("row_type") == "aggregate"]
    if not agg_rows:
        return
    remaining = [r.get("remaining_bias_pct", 1.0) * 100 for r in agg_rows]
    delta_vals = []
    for r in agg_rows:
        val = r.get("delta_corpus_ppl")
        if np.isnan(val) or val is None:
            val = r.get("delta_ppl_mean", float("nan"))
        delta_vals.append(val)
    sources = [r.get("feature_source", "cma") for r in agg_rows]

    colors = {"cma": "tab:blue", "random": "tab:orange", "control": "tab:green"}

    plt.figure(figsize=(8, 5))
    for x, y, src in zip(remaining, delta_vals, sources):
        plt.scatter(x, y, c=colors.get(src, "gray"), alpha=0.7, label=src)
    handles = []
    for src, color in colors.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=src, markerfacecolor=color, markersize=8))
    plt.legend(handles, [h.get_label() for h in handles])
    plt.xlabel("剩余偏差（% 基线）")
    plt.ylabel("ΔPPL（语料或均值）")
    plt.title("Local Cut Bias–Perplexity")
    plt.grid(True, alpha=0.3)
    plot_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制 Local Cut 图: {plot_path}")


def build_pareto_frontiers(rows: List[Dict]) -> Dict[str, List[Dict]]:
    frontiers: Dict[str, List[Dict]] = {}
    aggregate_rows = [r for r in rows if r.get("row_type") == "aggregate"]
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in aggregate_rows:
        key = (
            row.get("mediator_type"),
            row.get("mediator_layer"),
            row.get("mediator_head"),
            row.get("mediator_category"),
        )
        groups[key].append(row)

    for key, group in groups.items():
        if not group:
            continue
        group_sorted = sorted(group, key=lambda r: r.get("prefix_size", 0))
        pareto: List[Dict] = []
        best_metric = float("inf")
        for row in group_sorted:
            metric = row.get("delta_corpus_ppl")
            if metric is None or np.isnan(metric):
                metric = row.get("delta_ppl_mean", float("inf"))
            if metric <= best_metric + 1e-9:
                pareto.append({
                    "step": row.get("prefix_size"),
                    "feature_idx": row.get("feature_idx"),
                    "feature_set": row.get("feature_set"),
                    "alpha": row.get("alpha", 0.0),
                    "remaining_bias_pct": row.get("remaining_bias_pct"),
                    "delta_ppl": metric,
                    "sum_abs_edit": row.get("sum_abs_edit"),
                    "feature_source": row.get("feature_source"),
                })
                best_metric = min(best_metric, metric)
        if pareto:
            key_str = f"{key[0] or 'unknown'}_layer{key[1]}_head{key[2]}_{key[3]}"
            frontiers[key_str] = pareto

    return frontiers


def plot_pareto_curves(pareto_data: Dict[str, List[Dict]], csv_path: str, title: str) -> None:
    if not pareto_data:
        return
    base_path = os.path.splitext(csv_path)[0]
    for label, points in pareto_data.items():
        if not points:
            continue
        sorted_pts = sorted(points, key=lambda p: p.get("remaining_bias_pct", 1.0))
        xs = [p.get("remaining_bias_pct", 0.0) * 100 for p in sorted_pts]
        ys = [p.get("delta_ppl", 0.0) for p in sorted_pts]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o", linestyle="-", label="Pareto")
        plt.xlabel("剩余偏差（% 基线 NIE）")
        plt.ylabel("累计 ΔPPL")
        plt.title(f"{title}: {label}")
        plt.grid(True, alpha=0.3)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        plot_path = f"{base_path}_{safe_label}_pareto.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ 绘制 Pareto 曲线: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIE Local Cut")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ranking_csv", type=str, default="results/gpt2-small_nurse_man_20251109_173606.csv")
    parser.add_argument("--output", type=str, default="results/nie_local_cut.csv")
    parser.add_argument("--prompt_split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--num_features", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus_path", type=str, default="data/WikiText‑2.txt", help="用于 ΔPPL 的语料文件")
    parser.add_argument("--max_corpus_tokens", type=int, default=4096, help="语料最大 token 数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_local_cut(
        model_name=args.model,
        ranking_csv_path=args.ranking_csv,
        output_path=args.output,
        prompt_split=args.prompt_split,
        topk=args.topk,
        control_count=args.controls,
        num_features=args.num_features,
        seed=args.seed,
        device=args.device,
        corpus_path=args.corpus_path,
        max_corpus_tokens=args.max_corpus_tokens,
    )


if __name__ == "__main__":
    main()
