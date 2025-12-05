"""
Local Cut & Gate via SAE feature ranking.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .cma_gender_bias import run_gender_bias_cma
from .prompts_winogender import get_prompt_examples


# ============================================================================
# Configuration (aligned with baselines_head_off_random)
# ============================================================================

SAE_MODE = "attn"

if SAE_MODE == "attn":
    SAE_RELEASE = "gpt2-small-attn-out-v5-32k"
    SAE_HOOK_TEMPLATE = "blocks.{layer}.hook_attn_out"
else:
    SAE_RELEASE = "gpt2-small-res-jb"
    SAE_HOOK_TEMPLATE = "blocks.{layer}.hook_resid_pre"


# ============================================================================
# Helper data structures（保持与 baselines_head_off_random 一致）
# ============================================================================


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


DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


# ============================================================================
# SAE loading utilities（拷贝自 baselines_head_off_random）
# ============================================================================


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


# ============================================================================
# TransformerLens helpers
# ============================================================================


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

    # 语义对齐：
    # - feature_indices is None → 作用于“全部特征”
    # - feature_indices == []   → 不作用于任何特征（no-op）
    # - 否则仅作用于给定索引
    if feature_indices is None:
        mode = "all"
        feature_tensor = None
    elif len(feature_indices) == 0:
        mode = "none"
        feature_tensor = None
    else:
        mode = "some"
        feature_tensor = torch.tensor(feature_indices, device=device, dtype=torch.long)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        features = dictionary.encode(value)
        recon = dictionary.decode(features)
        residual = value - recon
        if mode == "all":
            features = features * alpha
        elif mode == "some":
            features[..., feature_tensor] = features[..., feature_tensor] * alpha
        else:
            # mode == "none" → 不改动
            pass
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


# ============================================================================
# Metric helpers
# ============================================================================


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


# ============================================================================
# Multi-layer SAE gating helpers
# ============================================================================


def _build_nodes_for_layers(
    submodules: Sequence[Submodule],
    dictionaries: Dict[Submodule, SAE],
    layer_to_indices: Dict[int, Optional[List[int]]],
) -> Dict[Submodule, NodeMask]:
    nodes: Dict[Submodule, NodeMask] = {}
    for sub in submodules:
        dictionary = dictionaries[sub]
        dict_size = int(dictionary.cfg.d_sae)
        mask_keep = torch.ones(dict_size, dtype=torch.bool)
        feat_indices = layer_to_indices.get(sub.layer, None)
        if feat_indices is None:
            mask_keep[:] = False
        else:
            mask_keep[feat_indices] = False
        nodes[sub] = NodeMask(
            act=mask_keep.clone(),
            resc=torch.ones(1, dtype=torch.bool),
        )
    return nodes


def _multi_layer_hooks(
    model: HookedTransformer,
    patch_tokens: torch.Tensor,
    submodules: Sequence[Submodule],
    dictionaries: Dict[Submodule, SAE],
    nodes: Dict[Submodule, NodeMask],
    alpha: float,
) -> List[Tuple[str, callable]]:
    hook_names = {sub.hook_name for sub in submodules}
    with torch.no_grad():
        _, patch_cache = model.run_with_cache(
            patch_tokens,
            names_filter=lambda name: name in hook_names,
            remove_batch_dim=False,
            return_type="logits",
        )
    hooks: List[Tuple[str, callable]] = []
    for sub in submodules:
        dictionary = dictionaries[sub]
        activation = patch_cache[sub.hook_name]
        features = dictionary.encode(activation)
        recon = dictionary.decode(features)
        residual = activation - recon
        node = nodes[sub]
        target_mask = (~node.act).to(torch.bool).to(features.device)
        feat_scaled = features.clone()
        if target_mask.numel() == feat_scaled.numel():
            feat_scaled = feat_scaled * alpha
        else:
            while target_mask.dim() < feat_scaled.dim():
                target_mask = target_mask.unsqueeze(0)
            feat_scaled = torch.where(target_mask, feat_scaled * alpha, feat_scaled)
        state = PatchState(act=feat_scaled, res=residual)
        hook_fn = build_feature_edit_hook(
            dictionary=dictionary,
            node_mask=node,
            patch_state=state,
            handle_errors="keep",
        )
        hooks.append((sub.hook_name, hook_fn))
    return hooks


def apply_multi_layer_alpha_gate(
    model: HookedTransformer,
    submodules: Sequence[Submodule],
    dictionaries: Dict[Submodule, SAE],
    base_prompt: str,
    cf_prompt: str,
    layer_to_indices: Dict[int, Optional[List[int]]],
    alpha: float,
) -> Tuple[float, float, float, float, float]:
    clean_tokens = tokenize_prompt(model, base_prompt)
    patch_base_tokens = tokenize_prompt(model, base_prompt)
    patch_cf_tokens = tokenize_prompt(model, cf_prompt)

    nodes = _build_nodes_for_layers(submodules, dictionaries, layer_to_indices)

    with torch.no_grad():
        logits_clean = model(clean_tokens, return_type="logits")

    hooks_base = _multi_layer_hooks(model, patch_base_tokens, submodules, dictionaries, nodes, alpha)
    with torch.no_grad():
        logits_gated = model.run_with_hooks(clean_tokens, fwd_hooks=hooks_base, return_type="logits")

    hooks_cf = _multi_layer_hooks(model, patch_cf_tokens, submodules, dictionaries, nodes, alpha)
    with torch.no_grad():
        logits_cf = model.run_with_hooks(clean_tokens, fwd_hooks=hooks_cf, return_type="logits")

    tokens_base_1d = clean_tokens[0]
    id_she = model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
    id_he = model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]
    bias_clean = (logits_clean[0, -1, id_she] - logits_clean[0, -1, id_he]).item()
    bias_gated = (logits_gated[0, -1, id_she] - logits_gated[0, -1, id_he]).item()
    bias_cf_replaced = (logits_cf[0, -1, id_she] - logits_cf[0, -1, id_he]).item()
    ppl_clean = _perplexity_from_logits(logits_clean[0], tokens_base_1d)
    ppl_gated = _perplexity_from_logits(logits_gated[0], tokens_base_1d)
    nie = bias_cf_replaced - bias_clean
    return bias_clean, bias_gated, ppl_clean, ppl_gated, nie


@dataclass
class FeatureEffect:
    scores: torch.Tensor
    delta: torch.Tensor
    grad: torch.Tensor
    total_effect: torch.Tensor


def build_bias_metric_fn(model: HookedTransformer):
    id_she = model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
    id_he = model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, id_she] - logits[:, -1, id_he]

    return metric_fn


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
    reduce_dims = tuple(range(effect_tensor.ndim - 1))
    scores = effect_tensor.abs().mean(dim=reduce_dims)

    total_effect = metric_fn(logits_cf).detach() - metric_clean.detach()
    return FeatureEffect(
        scores=scores.detach(),
        delta=delta.detach(),
        grad=grad.detach(),
        total_effect=total_effect,
    )


def rank_features_for_layers(
    model: HookedTransformer,
    submodules: Sequence[Submodule],
    dictionaries: Dict[Submodule, SAE],
    base_prompt: str,
    cf_prompt: str,
    metric_fn,
) -> Dict[int, List[int]]:
    layer_to_rank: Dict[int, List[int]] = {}
    for sub in submodules:
        effect = compute_feature_effect(
            model=model,
            submodule=sub,
            dictionary=dictionaries[sub],
            base_prompt=base_prompt,
            cf_prompt=cf_prompt,
            metric_fn=metric_fn,
        )
        scores = effect.scores.reshape(-1)
        order = torch.argsort(scores, descending=True)
        layer_to_rank[sub.layer] = order.tolist()
    return layer_to_rank


def format_layer_indices(layer_to_indices: Dict[int, List[int]]) -> str:
    parts = []
    for layer in sorted(layer_to_indices):
        idxs = layer_to_indices[layer]
        if not idxs:
            parts.append(f"{layer}:[]")
        else:
            preview = ",".join(map(str, idxs[:10]))
            suffix = "..." if len(idxs) > 10 else ""
            parts.append(f"{layer}:[{preview}{suffix}]")
    return ";".join(parts)


def run_local_cut_gate(
    model_name: str,
    ranking_csv_path: str,
    output_path: str,
    prompt_split: str,
    topk: int,
    feature_counts: List[int],
    alphas: List[float],
    nie_mode: str,
    seed: int,
    device: str,
    corpus_path: Optional[str],
    max_corpus_tokens: Optional[int],
) -> str:
    print("=" * 70)
    print("Local Cut & Gate (SAE feature ranking)")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"Top-K CSV: {ranking_csv_path}")
    print(f"Prompt split: {prompt_split}")
    print(f"Top-K: {topk}")
    if feature_counts:
        print(f"Feature counts: {len(feature_counts)} 档, 范围 [{min(feature_counts)}, {max(feature_counts)}]")
    if alphas:
        print(f"α: {len(alphas)} 档, 范围 [{min(alphas):.3f}, {max(alphas):.3f}]")
    print(f"NIE 模式: {nie_mode}")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    mediators = load_ranked_mediators_from_csv(ranking_csv_path, topk if topk > 0 else None)
    topk_mediators = mediators[:topk]
    if not topk_mediators:
        raise ValueError("Top-K mediator 列表为空")

    include_map: Dict[int, List[int]] = {}
    for m in topk_mediators:
        include_map.setdefault(int(m["layer"]), []).append(int(m.get("head")))

    target_layers = sorted({int(m["layer"]) for m in topk_mediators})
    submodules: List[Submodule] = []
    for sub in dictionaries.keys():
        if sub.layer in target_layers:
            submodules.append(sub)
    if not submodules:
        raise ValueError("目标层没有可用的 SAE 子模块")

    examples = get_prompt_examples(prompt_split)
    if not examples:
        raise ValueError(f"Prompt split '{prompt_split}' 没有可用样本")

    metric_fn = build_bias_metric_fn(model)

    unique_feature_counts = sorted({max(0, int(fc)) for fc in feature_counts})
    unique_alphas = [float(alpha) for alpha in alphas]
    scenarios = [(fc, alpha) for fc in unique_feature_counts for alpha in unique_alphas]
    stats: Dict[Tuple[int, float], Dict[str, List[float]]] = {
        (fc, alpha): {
            "bias_clean": [],
            "bias_edit": [],
            "ppl_clean": [],
            "ppl_edit": [],
            "delta_nie": [],
            "examples": [],
            "feature_sets": [],
        }
        for fc, alpha in scenarios
    }

    corpus_tokens = tokenize_corpus(model, corpus_path, max_corpus_tokens=max_corpus_tokens)
    baseline_corpus_ppl = compute_corpus_perplexity(model, corpus_tokens) if corpus_tokens is not None else float("nan")

    for ex_idx, example in enumerate(tqdm(examples, desc="Examples")):
        base_prompt = example.get("base", example.get("clean_prefix", ""))
        cf_prompt = example.get("counterfactual", example.get("patch_prefix", ""))
        if not base_prompt or not cf_prompt:
            continue

        layer_rankings = rank_features_for_layers(
            model=model,
            submodules=submodules,
            dictionaries=dictionaries,
            base_prompt=base_prompt,
            cf_prompt=cf_prompt,
            metric_fn=metric_fn,
        )

        fc_to_indices: Dict[int, Dict[int, List[int]]] = {}
        for fc in unique_feature_counts:
            per_layer: Dict[int, List[int]] = {}
            for layer, ranking in layer_rankings.items():
                if fc <= 0:
                    per_layer[layer] = []
                else:
                    per_layer[layer] = ranking[: min(fc, len(ranking))]
            fc_to_indices[fc] = per_layer

        fc_to_nie_before: Dict[int, float] = {}
        for fc, layer_to_indices in fc_to_indices.items():
            _, _, _, _, nie_local = apply_multi_layer_alpha_gate(
                model=model,
                submodules=submodules,
                dictionaries=dictionaries,
                base_prompt=base_prompt,
                cf_prompt=cf_prompt,
                layer_to_indices=layer_to_indices,
                alpha=1.0,
            )
            if nie_mode == "local":
                fc_to_nie_before[fc] = nie_local
            elif nie_mode == "heads":
                effects = run_gender_bias_cma(
                    model,
                    base_prompt,
                    cf_prompt,
                    verbose=False,
                    include_heads_by_layer=include_map,
                )
                fc_to_nie_before[fc] = float(sum(sum(row) for row in effects))
            else:
                effects = run_gender_bias_cma(model, base_prompt, cf_prompt, verbose=False)
                fc_to_nie_before[fc] = float(sum(sum(row) for row in effects))

        for fc, alpha in scenarios:
            layer_to_indices = fc_to_indices[fc]
            feature_label = format_layer_indices(layer_to_indices)
            bias_clean, bias_edit, ppl_clean, ppl_edit, nie_local = apply_multi_layer_alpha_gate(
                model=model,
                submodules=submodules,
                dictionaries=dictionaries,
                base_prompt=base_prompt,
                cf_prompt=cf_prompt,
                layer_to_indices=layer_to_indices,
                alpha=alpha,
            )

            if nie_mode == "local":
                nie_after = nie_local
            else:
                hooks_eval = []
                for sub in submodules:
                    feats = layer_to_indices.get(sub.layer, [])
                    hook_fn = make_scaling_hook(dictionaries[sub], feats, alpha)
                    hooks_eval.append((sub.hook_name, hook_fn))
                with model.hooks(fwd_hooks=hooks_eval):
                    if nie_mode == "heads":
                        eff = run_gender_bias_cma(
                            model,
                            base_prompt,
                            cf_prompt,
                            verbose=False,
                            include_heads_by_layer=include_map,
                        )
                    else:
                        eff = run_gender_bias_cma(model, base_prompt, cf_prompt, verbose=False)
                nie_after = float(sum(sum(row) for row in eff))

            entry = stats[(fc, alpha)]
            entry["bias_clean"].append(float(bias_clean))
            entry["bias_edit"].append(float(bias_edit))
            entry["ppl_clean"].append(float(ppl_clean))
            entry["ppl_edit"].append(float(ppl_edit))
            entry["delta_nie"].append(float(abs(nie_after) - abs(fc_to_nie_before[fc])))
            entry["examples"].append(ex_idx)
            entry["feature_sets"].append(feature_label)

    results: List[Dict] = []
    for fc, alpha in scenarios:
        entry = stats[(fc, alpha)]
        if not entry["bias_clean"]:
            continue
        mean_bias_clean = float(np.mean(entry["bias_clean"]))
        mean_bias_edit = float(np.mean(entry["bias_edit"]))
        mean_ppl_clean = float(np.mean(entry["ppl_clean"]))
        mean_ppl_edit = float(np.mean(entry["ppl_edit"]))
        mean_delta_nie = float(np.mean(entry["delta_nie"])) if entry["delta_nie"] else float("nan")
        remaining_pct = (
            float(abs(mean_bias_edit) / (abs(mean_bias_clean) + 1e-9))
            if abs(mean_bias_clean) > 1e-9
            else float("nan")
        )
        delta_prompt_ppl = (
            mean_ppl_edit - mean_ppl_clean
            if (not np.isnan(mean_ppl_edit) and not np.isnan(mean_ppl_clean))
            else float("nan")
        )
        sum_abs_edit = abs(1.0 - alpha) * float(fc * len(submodules))

        # 语料级 ΔPPL（与 baselines_head_off_random 对齐：使用最后一次样本的 layer_to_indices 作为代表）
        gated_corpus_ppl = float("nan")
        delta_corpus_ppl = float("nan")
        if corpus_tokens is not None and "fc_to_indices" in locals():
            layer_to_indices_eval = fc_to_indices.get(fc, {})
            hooks_eval: List[Tuple[str, callable]] = []
            for sub in submodules:
                dictionary = dictionaries[sub]
                feats = layer_to_indices_eval.get(sub.layer, [])
                hook_fn = make_scaling_hook(dictionary, feats, alpha)
                hooks_eval.append((sub.hook_name, hook_fn))
            gated_corpus_ppl = compute_corpus_perplexity(model, corpus_tokens, hooks=hooks_eval)
            if not np.isnan(baseline_corpus_ppl):
                delta_corpus_ppl = gated_corpus_ppl - baseline_corpus_ppl

        aggregate_row = {
            "analysis": "local_feature_gate",
            "row_type": "aggregate",
            "edit_label": f"bias_rank_fc{fc}_alpha{alpha}",
            "feature_source": "bias_ranked",
            "feature_selection": "local_top_by_bias",
            "feature_count": fc,
            "alpha": alpha,
            "sum_abs_edit": sum_abs_edit,
            "bias_original_mean": mean_bias_clean,
            "bias_edited_mean": mean_bias_edit,
            "ppl_original_mean": mean_ppl_clean,
            "ppl_edited_mean": mean_ppl_edit,
            "delta_ppl_mean": delta_prompt_ppl,
            "remaining_bias_pct": remaining_pct,
            "delta_nie_mean": mean_delta_nie,
            "corpus_ppl_original": baseline_corpus_ppl,
            "corpus_ppl_edited": gated_corpus_ppl,
            "delta_corpus_ppl": delta_corpus_ppl,
            "mediator_layer": ",".join(map(str, target_layers)),
            "mediator_type": "sae_layers",
            "mediator_head": None,
            "mediator_nie": None,
            "mediator_category": "topk_layers",
            "nie_source": nie_mode,
            "example_count": len(entry["bias_clean"]),
            "top_features_snapshot": entry["feature_sets"][:3],
        }
        results.append(aggregate_row)

    if not results:
        raise RuntimeError("没有生成任何有效结果")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ 结果已写入 {output_path}")
    plot_baseline_results(results, output_path)
    plot_nie_vs_ppl(results, output_path)
    plot_alpha_pareto_lines(results, output_path)
    return output_path


def _resolve_feature_counts(
    num_features: int,
    num_features_range: Optional[List[int]],
) -> Tuple[List[int], Optional[str]]:
    if num_features_range and len(num_features_range) == 3:
        start, end, step = [int(x) for x in num_features_range]
        step = 1 if step <= 0 else step
        lo, hi = (start, end) if start <= end else (end, start)
        lo = max(0, lo)
        nf_list = list(range(lo, hi + 1, step))
        if not nf_list:
            nf_list = [int(num_features)]
        msg = f"num_features_range → {lo}..{hi} step {step} → {len(nf_list)} 档"
        return nf_list, msg
    return [int(num_features)], None


def _resolve_alphas(
    alphas: List[float],
    alpha_range: Optional[List[float]],
) -> Tuple[List[float], Optional[str]]:
    if alpha_range and len(alpha_range) == 3:
        start, end, step = [float(x) for x in alpha_range]
        if step == 0.0:
            step = -0.25 if end < start else 0.25
        vals: List[float] = []
        cur = start
        # 支持 start > end 的递减
        if step > 0:
            while cur <= end + 1e-9:
                vals.append(cur)
                cur += step
        else:
            while cur >= end - 1e-9:
                vals.append(cur)
                cur += step
        msg = f"alpha_range → {start}..{end} step {step} → {len(vals)} 档"
        return vals, msg
    # 无 range 时直接使用传入列表
    return [float(a) for a in alphas], None


FEATURE_SOURCE_MARKERS = {
    "cma": "o",
    "random": "^",
    "bias_ranked": "s",
}


def _select_delta_ppl(row: Dict) -> float:
    val = row.get("delta_corpus_ppl")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        val = row.get("delta_ppl_mean")
    return val


def build_bias_delta_pareto(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    sorted_points = sorted(points, key=lambda item: item[0])
    frontier: List[Tuple[float, float]] = []
    best_delta = float("inf")
    for x_val, y_val in sorted_points:
        if y_val < best_delta - 1e-9:
            frontier.append((x_val, y_val))
            best_delta = y_val
    return frontier


def plot_baseline_results(rows: List[Dict], csv_path: str) -> None:
    from matplotlib.colors import Normalize

    scatter_points = []
    for row in rows:
        bias_ratio = row.get("remaining_bias_pct")
        delta = _select_delta_ppl(row)
        if bias_ratio is None or delta is None:
            continue
        if isinstance(bias_ratio, float) and np.isnan(bias_ratio):
            continue
        if isinstance(delta, float) and np.isnan(delta):
            continue
        scatter_points.append((bias_ratio * 100.0, delta, row))

    if not scatter_points:
        return

    nfs = [point[2].get("feature_count", 0.0) for point in scatter_points]
    nf_min, nf_max = (min(nfs), max(nfs)) if nfs else (0.0, 0.0)
    norm = Normalize(vmin=nf_min, vmax=nf_max) if nf_max - nf_min > 1e-6 else None
    cmap = plt.get_cmap("plasma")

    plt.figure(figsize=(8, 5))
    scatter_handles = []
    feature_sources = sorted({point[2].get("feature_source", "unknown") for point in scatter_points})
    for source in feature_sources:
        subset = [point for point in scatter_points if point[2].get("feature_source", "unknown") == source]
        if not subset:
            continue
        xs = [point[0] for point in subset]
        ys = [point[1] for point in subset]
        colors = [point[2].get("feature_count", 0.0) for point in subset]
        marker = FEATURE_SOURCE_MARKERS.get(source, "o")
        # 点大小用 α 编码（越大 α，点越大）
        sizes = [
            30.0 + 70.0 * float(point[2].get("alpha", 0.0))
            for point in subset
        ]
        scatter = plt.scatter(
            xs,
            ys,
            c=colors,
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=sizes,
            alpha=0.85,
            edgecolors="none",
        )
        scatter_handles.append(scatter)

    if scatter_handles:
        cbar = plt.colorbar(scatter_handles[0])
        cbar.set_label("num_features")

    pareto_points = build_bias_delta_pareto([(point[0], point[1]) for point in scatter_points])
    if pareto_points:
        px, py = zip(*pareto_points)
        plt.plot(px, py, color="black", linewidth=2)

    plt.xlabel("Remaining bias (% baseline)")
    plt.ylabel("ΔPPL")
    plt.title("SAE Multi-layer Random Cut: Bias-Perplexity Pareto")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels)
    plt.grid(True, alpha=0.3)
    plot_path = os.path.splitext(csv_path)[0] + "_bias_pareto.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制 Bias-Pareto 图: {plot_path}")

    # Plotly HTML 交互图（可选）
    try:
        import pandas as pd
        import plotly.express as px

        aggregate_rows = [r for r in rows if r.get("row_type") == "aggregate"]
        if not aggregate_rows:
            return

        base_path = os.path.splitext(csv_path)[0]

        def _y(row: Dict) -> float:
            val = row.get("delta_corpus_ppl")
            if val is None or (isinstance(val, float) and (val != val)):
                val = row.get("delta_ppl_mean")
            return val

        df = pd.DataFrame(
            {
                "delta_nie": [r.get("delta_nie_mean") for r in aggregate_rows],
                "remaining_nie_pct": [
                    r.get("remaining_bias_pct", float("nan")) * 100 for r in aggregate_rows
                ],
                "delta_ppl": [_y(r) for r in aggregate_rows],
                "feature_count": [r.get("feature_count") for r in aggregate_rows],
                "alpha": [r.get("alpha") for r in aggregate_rows],
                "edit_label": [r.get("edit_label") for r in aggregate_rows],
            }
        )

        fig1 = px.scatter(
            df,
            x="delta_nie",
            y="delta_ppl",
            color="feature_count",
            size="alpha",
            hover_data=["edit_label", "alpha"],
            title="Δ|NIE| vs ΔPPL (local cut/gate)",
            template="simple_white",
        )
        html1 = f"{base_path}_nie_scatter.html"
        fig1.write_html(html1)
        print(f"✓ 绘制 NIE-ΔPPL 交互图: {html1}")

        fig2 = px.scatter(
            df,
            x="remaining_nie_pct",
            y="delta_ppl",
            color="feature_count",
            size="alpha",
            hover_data=["edit_label", "alpha"],
            title="Bias–PPL (Remaining NIE% vs ΔPPL, local cut/gate)",
            template="simple_white",
        )
        html2 = f"{base_path}_bias_ppl.html"
        fig2.write_html(html2)
        print(f"✓ 绘制 Bias–PPL 交互图: {html2}")
    except Exception:
        pass


def plot_nie_vs_ppl(rows: List[Dict], csv_path: str) -> None:
    from matplotlib.colors import Normalize

    scatter_points = []
    for row in rows:
        nie_delta = row.get("delta_nie_mean")
        delta = _select_delta_ppl(row)
        if nie_delta is None or delta is None:
            continue
        if isinstance(nie_delta, float) and np.isnan(nie_delta):
            continue
        if isinstance(delta, float) and np.isnan(delta):
            continue
        scatter_points.append((nie_delta, delta, row))

    if not scatter_points:
        return

    nf_vals = [point[2].get("feature_count", 0.0) for point in scatter_points]
    nf_min, nf_max = (min(nf_vals), max(nf_vals)) if nf_vals else (0.0, 1.0)
    nf_norm = Normalize(vmin=nf_min, vmax=nf_max if nf_max != nf_min else nf_min + 1)
    cmap = plt.get_cmap("plasma")

    plt.figure(figsize=(8, 5))
    color_ref = None
    feature_sources = sorted({point[2].get("feature_source", "unknown") for point in scatter_points})
    for source in feature_sources:
        subset = [point for point in scatter_points if point[2].get("feature_source", "unknown") == source]
        if not subset:
            continue
        xs = [point[0] for point in subset]
        ys = [point[1] for point in subset]
        colors = [point[2].get("feature_count", 0.0) for point in subset]
        marker = FEATURE_SOURCE_MARKERS.get(source, "o")
        sizes = [
            30.0 + 70.0 * float(point[2].get("alpha", 0.0))
            for point in subset
        ]
        scatter = plt.scatter(
            xs,
            ys,
            c=colors,
            cmap=cmap,
            norm=nf_norm,
            marker=marker,
            s=sizes,
            alpha=0.85,
            edgecolors="none",
        )
        color_ref = color_ref or scatter

    if color_ref:
        cbar = plt.colorbar(color_ref)
        cbar.set_label("num_features")

    plt.xlabel("Δ|NIE|")
    plt.ylabel("ΔPPL")
    plt.title("SAE Multi-Layer Random Cut: Δ|NIE| vs ΔPPL")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels)
    plt.grid(True, alpha=0.3)
    plot_path = os.path.splitext(csv_path)[0] + "_nie_scatter.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制 NIE-ΔPPL 散点图: {plot_path}")


def plot_alpha_pareto_lines(rows: List[Dict], csv_path: str) -> None:
    agg_rows = [r for r in rows if r.get("row_type") == "aggregate"]
    if not agg_rows:
        return
    groups: Dict[float, List[Dict]] = {}
    for row in agg_rows:
        alpha = row.get("alpha")
        if alpha is None:
            continue
        if isinstance(alpha, float) and np.isnan(alpha):
            continue
        groups.setdefault(float(alpha), []).append(row)
    if not groups:
        return

    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    alphas_sorted = sorted(groups.keys())
    denom = max(1, len(alphas_sorted) - 1)
    for idx, alpha in enumerate(alphas_sorted):
        group = groups[alpha]
        # 将 remaining_bias_pct < 1 的点用于画线；=1 的点只做散点，不连线，避免图右侧杂乱
        pts_line = [
            g
            for g in group
            if g.get("remaining_bias_pct") is not None
            and not (isinstance(g.get("remaining_bias_pct"), float) and np.isnan(g.get("remaining_bias_pct")))
            and g.get("remaining_bias_pct") < 0.999
        ]
        pts_tail = [
            g
            for g in group
            if g.get("remaining_bias_pct") is not None
            and not (isinstance(g.get("remaining_bias_pct"), float) and np.isnan(g.get("remaining_bias_pct")))
            and g.get("remaining_bias_pct") >= 0.999
        ]
        color = cmap(idx / denom)

        if pts_line:
            sorted_group = sorted(pts_line, key=lambda g: g.get("remaining_bias_pct", float("inf")))
            xs = [g.get("remaining_bias_pct", float("nan")) * 100 for g in sorted_group]
            ys = [_select_delta_ppl(g) for g in sorted_group]
            plt.plot(
                xs,
                ys,
                marker="o",
                label=f"α={alpha}",
                color=color,
            )
        if pts_tail:
            xs_tail = [g.get("remaining_bias_pct", float("nan")) * 100 for g in pts_tail]
            ys_tail = [_select_delta_ppl(g) for g in pts_tail]
            plt.scatter(
                xs_tail,
                ys_tail,
                color=color,
                s=30,
                alpha=0.7,
                edgecolors="none",
            )
    plt.xlabel("Remaining bias (% baseline)")
    plt.ylabel("ΔPPL")
    plt.title("Alpha-wise Pareto (Remaining Bias vs ΔPPL)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = os.path.splitext(csv_path)[0] + "_alpha_pareto.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制 Alpha Pareto 图: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local SAE feature cut/gate (bias-ranked)")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ranking_csv", type=str, default="results/gpt2-small_nurse_man_20251110_181041.csv")
    parser.add_argument("--output", type=str, default="results/nie_local_cut_gate.csv")
    parser.add_argument("--prompt_split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_features", type=int, default=1000)
    parser.add_argument("--num_features_range", type=int, nargs=3, default=[0, 32000, 1000], help="start end step（含端点）生成多档 feature 数（唯一多档来源）")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5])
    parser.add_argument("--alpha_range", type=float, nargs=3, default=None, help="可选：start end step，为 α 生成多档值；若提供则覆盖 --alphas")
    parser.add_argument("--nie_mode", type=str, default="local", choices=["local", "heads", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus_path", type=str, default="data/WikiText.txt")
    parser.add_argument("--max_corpus_tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_counts, nf_log = _resolve_feature_counts(
        num_features=args.num_features,
        num_features_range=args.num_features_range,
    )
    if nf_log:
        print(nf_log)
    alphas, alpha_log = _resolve_alphas(
        alphas=args.alphas,
        alpha_range=args.alpha_range,
    )
    if alpha_log:
        print(alpha_log)
    run_local_cut_gate(
        model_name=args.model,
        ranking_csv_path=args.ranking_csv,
        output_path=args.output,
        prompt_split=args.prompt_split,
        topk=args.topk,
        feature_counts=feature_counts,
        alphas=alphas,
        nie_mode=args.nie_mode,
        seed=args.seed,
        device=args.device,
        corpus_path=args.corpus_path,
        max_corpus_tokens=args.max_corpus_tokens,
    )


if __name__ == "__main__":
    main()

