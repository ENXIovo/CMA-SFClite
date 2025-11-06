"""
NIE Local Gate Experiment
-------------------------
Applies soft gates (alpha ∈ [1.0, ..., 0.0]) on top CMA mediators.
Helper code copied locally for clarity. """

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from nnsight import LanguageModel
except ImportError as exc:
    raise SystemExit("请先安装 nnsight：pip install nnsight") from exc


SAE_MODE = "attn"  # 改成 "attn" 即可加载注意力 SAE
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESID_SAE_TEMPLATE = os.path.join(
    PROJECT_ROOT,
    "models",
    "GPT2-Small-SAEs",
    "final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt",
)

ATTN_SAE_LOCAL_TEMPLATE = os.path.join(
    PROJECT_ROOT,
    "models",
    "GPT2-Small-OAI-v5-32k-attn-out-SAEs",
    "v5_32k_layer_{layer}",
    "sae_weights.safetensors",
)


class SparseAct:
    def __init__(self, act: torch.Tensor, res: Optional[torch.Tensor] = None, resc: Optional[torch.Tensor] = None):
        self.act = act
        self.res = res
        self.resc = resc

    def _map(self, fn, other=None) -> "SparseAct":
        kwargs = {}
        if isinstance(other, SparseAct):
            for attr in ("act", "res", "resc"):
                x = getattr(self, attr)
                y = getattr(other, attr)
                if x is not None and y is not None:
                    kwargs[attr] = fn(x, y)
        else:
            for attr in ("act", "res", "resc"):
                x = getattr(self, attr)
                if x is not None:
                    kwargs[attr] = fn(x, other)
        return SparseAct(**kwargs)

    def save(self) -> "SparseAct":
        return self._map(lambda x, _: x.save())

    @property
    def value(self) -> "SparseAct":
        kwargs = {}
        for attr in ("act", "res", "resc"):
            x = getattr(self, attr)
            if x is not None:
                kwargs[attr] = x.value
        return SparseAct(**kwargs)

    @property
    def grad(self) -> "SparseAct":
        kwargs = {}
        for attr in ("act", "res", "resc"):
            x = getattr(self, attr)
            grad = getattr(x, "grad", None)
            if grad is not None:
                kwargs[attr] = grad
        return SparseAct(**kwargs)

    def detach(self) -> "SparseAct":
        return self._map(lambda x, _: x.detach())

    def __sub__(self, other: "SparseAct") -> "SparseAct":
        return self._map(lambda x, y: x - y, other)

    def __add__(self, other: "SparseAct") -> "SparseAct":
        return self._map(lambda x, y: x + y, other)

    def __matmul__(self, other: "SparseAct") -> "SparseAct":
        kwargs = {}
        for attr in ("act", "res", "resc"):
            x = getattr(self, attr)
            y = getattr(other, attr)
            if x is not None and y is not None:
                kwargs[attr] = (x * y).sum(dim=-1, keepdim=True)
        return SparseAct(**kwargs)


@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: object
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def get_activation(self) -> torch.Tensor:
        out = self.submodule.input if self.use_input else self.submodule.output
        return out[0] if self.is_tuple else out

    def set_activation(self, x: torch.Tensor) -> None:
        target = self.submodule.input if self.use_input else self.submodule.output
        if self.is_tuple:
            target = target[0]
        target[:] = x


DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


def load_gpt2_sae_from_file(path: str, dtype: torch.dtype, device: torch.device) -> Optional[AutoEncoder]:
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                W_enc = f.get_tensor("W_enc").to(dtype=dtype, device=device)
                W_dec = f.get_tensor("W_dec").to(dtype=dtype, device=device)
                b_enc = f.get_tensor("b_enc").to(dtype=dtype, device=device)
        else:
            data = torch.load(path, map_location=device, weights_only=False)
            state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
            W_enc = state_dict["W_enc"].to(dtype=dtype, device=device)
            W_dec = state_dict["W_dec"].to(dtype=dtype, device=device)
            b_enc = state_dict.get("b_enc")
            if b_enc is not None:
                b_enc = b_enc.to(dtype=dtype, device=device)
    except Exception as exc:
        print(f"  ⚠ SAE 加载异常: {exc}")
        return None

    activation_dim, dict_size = W_enc.shape
    sae = AutoEncoder(activation_dim, dict_size).to(dtype=dtype, device=device)
    sae.encoder.weight.data = W_enc.T.clone()
    sae.decoder.weight.data = W_dec.T.clone()
    if b_enc is not None:
        sae.encoder.bias.data = b_enc.clone()
    return sae


def load_layer_sae(layer_idx: int, dtype: torch.dtype, device: torch.device) -> Optional[AutoEncoder]:
    cfgs = {
        "resid": {"source": "local", "template": RESID_SAE_TEMPLATE},
        "attn": {"source": "local", "template": ATTN_SAE_LOCAL_TEMPLATE},
    }
    cfg = cfgs[SAE_MODE]
    try:
        if cfg["source"] == "local":
            path = cfg["template"].format(layer=layer_idx)
            sae = load_gpt2_sae_from_file(path, dtype=dtype, device=device)
            if sae is None:
                print(f"  ⚠ 未找到 {path}，使用 Identity SAE")
            return sae
    except Exception as exc:
        print(f"  ⚠ SAE 加载异常: {exc}")
        return None


def load_saes_and_submodules(model: LanguageModel, dtype: torch.dtype, device: str) -> Tuple[DictionaryStash, Dict[Submodule, AutoEncoder]]:
    attns: List[Submodule] = []
    mlps: List[Submodule] = []
    resids: List[Submodule] = []
    dictionaries: Dict[Submodule, AutoEncoder] = {}

    for layer_idx, block in enumerate(model.transformer.h):
        sae = load_layer_sae(layer_idx, dtype=dtype, device=torch.device(device))
        if SAE_MODE == "attn":
            attn_mod = Submodule(name=f"attn_{layer_idx}", submodule=block.attn, is_tuple=True)
            attns.append(attn_mod)
            dictionaries[attn_mod] = sae if sae is not None else IdentityDict(768)

            resid_mod = Submodule(name=f"resid_{layer_idx}", submodule=block, is_tuple=True)
            resids.append(resid_mod)
            dictionaries[resid_mod] = IdentityDict(768)
        else:
            resid_mod = Submodule(name=f"resid_{layer_idx}", submodule=block, is_tuple=True)
            resids.append(resid_mod)
            dictionaries[resid_mod] = sae if sae is not None else IdentityDict(768)

            attn_alias = Submodule(name=f"attn_{layer_idx}", submodule=block, is_tuple=True)
            attns.append(attn_alias)
            dictionaries[attn_alias] = dictionaries[resid_mod]

        mlp_alias = Submodule(name=f"mlp_{layer_idx}", submodule=block, is_tuple=True)
        mlps.append(mlp_alias)
        dictionaries[mlp_alias] = dictionaries[attns[-1]]

    return DictionaryStash(embed=None, attns=attns, mlps=mlps, resids=resids), dictionaries


def find_mediator_submodule(mediator: Dict, stash: DictionaryStash) -> Tuple[Optional[Submodule], str]:
    layer_idx = mediator["layer"]
    mediator_type = mediator.get("type", "attn")
    if mediator_type == "attn" and layer_idx < len(stash.attns):
        return stash.attns[layer_idx], "attn"
    if layer_idx < len(stash.resids):
        return stash.resids[layer_idx], "resid"
    return None, mediator_type


def run_with_ablations(clean, patch, model, submodules, dictionaries, nodes, metric_fn, ablation_fn):
    if patch is None:
        patch = clean
    patch_states = {}
    with model.trace(patch), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k: ablation_fn(v.value) for k, v in patch_states.items()}

    with model.trace(clean), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            res = x - dictionary.decode(f)

            mask = ~submod_nodes.act
            f[..., mask] = patch_states[submodule].act[..., mask]
            res.copy_(patch_states[submodule].res)
            submodule.set_activation(dictionary.decode(f) + res)

        metric = metric_fn(model).save()
    return metric.value


EffectOut = namedtuple("EffectOut", ["effects", "deltas", "grads", "total_effect"])


def reduce_feature_scores(tensor: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(tensor.ndim - 1))
    return tensor.abs().mean(dim=dims)


def patching_effect(clean, patch, model, submodules, dictionaries, metric_fn):
    hidden_states_clean = {}
    grads = {}
    with model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            x_hat, f = dictionary(x, output_features=True)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
            grads[submodule] = hidden_states_clean[submodule].grad.save()
            residual.grad = torch.zeros_like(residual)
            submodule.set_activation(x_hat + residual)
            x.grad = (x_hat + residual).grad
        metric_clean = metric_fn(model).save()
        metric_clean.sum().backward()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}
    grads = {k: v.value for k, v in grads.items()}

    hidden_states_patch = {}
    with torch.no_grad(), model.trace(patch):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            x_hat, f = dictionary(x, output_features=True)
            residual = x - x_hat
            hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
        metric_patch = metric_fn(model).save()
    total_effect = (metric_patch.value - metric_clean.value).detach()
    hidden_states_patch = {k: v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state = hidden_states_patch[submodule]
        clean_state = hidden_states_clean[submodule]
        grad = grads[submodule]
        delta = patch_state - clean_state.detach()
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
    return EffectOut(effects, deltas, grads, total_effect)


def _perplexity_from_logits(logits_val: torch.Tensor, tokens: torch.Tensor) -> float:
    if len(tokens) < 2:
        return float("inf")
    log_probs = torch.nn.functional.log_softmax(logits_val[:-1], dim=-1)
    target_log_probs = log_probs[range(len(tokens) - 1), tokens[1:]]
    return torch.exp(-target_log_probs.mean()).item()


def apply_alpha_gate(
    model: LanguageModel,
    submodule: Submodule,
    dictionary: AutoEncoder,
    base_prompt: str,
    cf_prompt: str,
    feature_indices: Optional[List[int]],
    alpha: float,
) -> Tuple[float, float, float, float, float]:
    tokens_base = model.tokenizer(base_prompt, return_tensors="pt")["input_ids"][0]

    if hasattr(dictionary, "dict_size"):
        dict_size = dictionary.dict_size
    elif hasattr(dictionary, "encoder") and hasattr(dictionary.encoder, "weight"):
        dict_size = dictionary.encoder.weight.shape[0]
    else:
        with torch.no_grad():
            with model.trace(base_prompt):
                sample = submodule.get_activation().detach()
        dict_size = sample.shape[-1]

    mask_keep = torch.ones(dict_size, dtype=torch.bool)
    if feature_indices is None:
        mask_keep[:] = False
    else:
        mask_keep[feature_indices] = False

    def make_nodes() -> Dict[Submodule, SparseAct]:
        return {submodule: SparseAct(act=mask_keep.clone(), resc=torch.ones(1, dtype=torch.bool))}

    def ablation_fn(sparse_act: SparseAct) -> SparseAct:
        act = sparse_act.act.clone()
        res = sparse_act.res.clone() if sparse_act.res is not None else None
        resc = sparse_act.resc.clone() if sparse_act.resc is not None else None
        target_mask = (~mask_keep).to(torch.bool)
        if feature_indices is None:
            act = act * alpha
        else:
            act[..., target_mask] = act[..., target_mask] * alpha
        return SparseAct(act=act, res=res, resc=resc)

    with model.trace(base_prompt), torch.no_grad():
        logits_clean = model.output.logits.save()
    logits_clean = logits_clean.value

    logits_gated = run_with_ablations(
        clean=base_prompt,
        patch=base_prompt,
        model=model,
        submodules=[submodule],
        dictionaries={submodule: dictionary},
        nodes=make_nodes(),
        metric_fn=lambda m: m.output.logits,
        ablation_fn=ablation_fn,
    )
    logits_cf = run_with_ablations(
        clean=base_prompt,
        patch=cf_prompt,
        model=model,
        submodules=[submodule],
        dictionaries={submodule: dictionary},
        nodes=make_nodes(),
        metric_fn=lambda m: m.output.logits,
        ablation_fn=ablation_fn,
    )

    id_she = model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
    id_he = model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]
    bias_clean = (logits_clean[0, -1, id_she] - logits_clean[0, -1, id_he]).item()
    bias_gated = (logits_gated[0, -1, id_she] - logits_gated[0, -1, id_he]).item()
    ppl_clean = _perplexity_from_logits(logits_clean[0], tokens_base)
    ppl_gated = _perplexity_from_logits(logits_gated[0], tokens_base)
    bias_cf_replaced = (logits_cf[0, -1, id_she] - logits_cf[0, -1, id_he]).item()
    nie = bias_cf_replaced - bias_clean
    return bias_clean, bias_gated, ppl_clean, ppl_gated, nie


def run_local_gate(
    model_name: str,
    mediator_path: str,
    eval_path: str,
    output_path: str,
    topk: int,
    control_count: int,
    num_features: int,
    alphas: List[float],
    seed: int,
    device: str,
) -> str:
    print("=" * 70)
    print("NIE Local Gate")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"Top-K 文件: {mediator_path}")
    print(f"评估数据: {eval_path}")
    print(f"SAE 模式: {SAE_MODE}")
    print("α 列表:", alphas)
    print("=" * 70)

    model = LanguageModel(model_name, device_map=device, torch_dtype=torch.float32, dispatch=True)
    stash, dictionaries = load_saes_and_submodules(model, dtype=torch.float32, device=device)

    with open(mediator_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mediators = data if isinstance(data, list) else data.get("mediators", [])
    mediators = mediators[: topk + control_count]
    entries = [(med, "topk") for med in mediators[:topk]]
    entries.extend((med, "control") for med in mediators[topk: topk + control_count])

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    examples = eval_data.get("examples", eval_data if isinstance(eval_data, list) else [])
    if not examples:
        raise ValueError("评估数据为空")
    example = examples[0]
    base_prompt = example.get("base", example.get("clean_prefix", ""))
    cf_prompt = example.get("counterfactual", example.get("patch_prefix", ""))
    if not base_prompt or not cf_prompt:
        raise ValueError("评估样本缺少 base/counterfactual 字段")

    rng = np.random.default_rng(seed)
    results: List[Dict] = []

    for idx, (mediator, category) in enumerate(tqdm(entries, desc="Mediators")):
        submodule, mediator_type = find_mediator_submodule(mediator, stash)
        if submodule is None:
            print(f"  ⚠ Mediator {idx} 无法匹配 submodule")
            continue
        dictionary = dictionaries[submodule]

        effect = patching_effect(
            clean=base_prompt,
            patch=cf_prompt,
            model=model,
            submodules=[submodule],
            dictionaries={submodule: dictionary},
            metric_fn=lambda m: m.output.logits[:, -1, :],
        )
        scores = reduce_feature_scores(effect.effects[submodule].act.detach())
        top_tensor, top_idx = torch.topk(scores, k=min(num_features, scores.shape[-1]))
        top_indices = top_idx.tolist()
        top_scores = top_tensor.tolist()

        base_orig, base_orig_gated, ppl_orig, ppl_orig_gated, nie_orig = apply_alpha_gate(
            model,
            submodule,
            dictionary,
            base_prompt,
            cf_prompt,
            feature_indices=None,
            alpha=1.0,
        )

        for feat_idx, feat_score in zip(top_indices, top_scores):
            if isinstance(feat_idx, (list, tuple)):
                if not feat_idx:
                    continue
                feat_idx = feat_idx[0]
            if isinstance(feat_score, (list, tuple)):
                feat_score = feat_score[0] if feat_score else 0.0
            for alpha in alphas:
                _, bias_after, _, ppl_after, nie_val = apply_alpha_gate(
                    model,
                    submodule,
                    dictionary,
                    base_prompt,
                    cf_prompt,
                    feature_indices=[int(feat_idx)],
                    alpha=alpha,
                )
                results.append({
                    "analysis": "local_gate",
                    "edit_label": "feature_gate",
                    "feature_idx": int(feat_idx),
                    "feature_score": float(feat_score),
                    "alpha": float(alpha),
                    "nie_original": float(nie_orig),
                    "nie_ablated": float(nie_val),
                    "delta_nie": float(nie_val - nie_orig),
                    "ppl_original": float(ppl_orig),
                    "ppl_ablated": float(ppl_after),
                    "delta_ppl": float(ppl_after - ppl_orig),
                    "mediator_layer": mediator["layer"],
                    "mediator_type": mediator_type,
                    "mediator_head": mediator.get("head"),
                    "mediator_nie": mediator.get("nie"),
                    "mediator_category": category,
                    "example_id": 0,
                    "example_base": base_prompt[:100],
                    "bias_original": float(base_orig),
                    "bias_edited": float(bias_after),
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ 结果已写入 {output_path}")
    plot_local_gate_results(results, output_path)

    pareto_data = build_pareto_frontiers(results)
    if pareto_data:
        pareto_path = os.path.splitext(output_path)[0] + "_pareto.json"
        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(pareto_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 保存 Pareto 数据: {pareto_path}")
        plot_pareto_curves(pareto_data, output_path, "Local Gate")
    return output_path


def plot_local_gate_results(rows: List[Dict], csv_path: str) -> None:
    if not rows:
        return
    abs_nie = [abs(r["delta_nie"]) for r in rows]
    delta_ppl = [r["delta_ppl"] for r in rows]
    alphas = [r.get("alpha", 1.0) for r in rows]

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(abs_nie, delta_ppl, c=alphas, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="alpha")
    plt.xlabel("|ΔNIE|")
    plt.ylabel("ΔPPL")
    plt.title("Local Gate (Soft Scaling)")
    plt.grid(True, alpha=0.3)
    plot_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制 Local Gate 图: {plot_path}")


def _select_best_gate_rows(rows: List[Dict]) -> List[Dict]:
    baseline_rows: List[Dict] = []
    best_by_feature: Dict[Tuple, Dict] = {}
    for row in rows:
        alpha = float(row.get("alpha", 1.0))
        if math.isclose(alpha, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            baseline_rows.append(row)
            continue
        key = (
            row.get("mediator_type"),
            row.get("mediator_layer"),
            row.get("mediator_head"),
            row.get("mediator_category"),
            row.get("feature_idx"),
        )
        current = best_by_feature.get(key)
        if current is None:
            best_by_feature[key] = row
            continue
        delta_nie = float(row.get("delta_nie", 0.0))
        curr_delta_nie = float(current.get("delta_nie", 0.0))
        if delta_nie < curr_delta_nie - 1e-6:
            best_by_feature[key] = row
        elif math.isclose(delta_nie, curr_delta_nie, rel_tol=1e-4, abs_tol=1e-6):
            if float(row.get("delta_ppl", 0.0)) < float(current.get("delta_ppl", 0.0)):
                best_by_feature[key] = row
    return baseline_rows + list(best_by_feature.values())


def build_pareto_frontiers(rows: List[Dict]) -> Dict[str, List[Dict]]:
    frontiers: Dict[str, List[Dict]] = {}
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    filtered_rows = _select_best_gate_rows(rows)

    for row in filtered_rows:
        key = (
            row.get("mediator_type"),
            row.get("mediator_layer"),
            row.get("mediator_head"),
            row.get("mediator_category"),
        )
        grouped[key].append(row)

    for key, group in grouped.items():
        if not group:
            continue
        baseline_nie = float(np.mean([g.get("nie_original", 0.0) for g in group]))
        denom = abs(baseline_nie) if abs(baseline_nie) > 1e-9 else 1e-9

        points: List[Dict] = [
            {
                "step": 0,
                "feature_idx": None,
                "alpha": 1.0,
                "remaining_bias": float(baseline_nie),
                "remaining_bias_pct": 1.0,
                "delta_ppl": 0.0,
                "delta_nie": 0.0,
            }
        ]

        cumulative_delta_ppl = 0.0
        cumulative_delta_nie = 0.0
        current_nie = baseline_nie

        candidates = [g for g in group if abs(g.get("delta_nie", 0.0)) > 1e-12 or abs(g.get("delta_ppl", 0.0)) > 1e-12]
        candidates.sort(key=lambda g: (abs(g.get("nie_ablated", g.get("nie_original", 0.0))), abs(g.get("delta_ppl", 0.0))))

        for idx, row in enumerate(candidates, start=1):
            delta_nie = float(row.get("delta_nie", 0.0))
            delta_ppl = float(row.get("delta_ppl", 0.0))
            cumulative_delta_nie += delta_nie
            cumulative_delta_ppl += delta_ppl
            current_nie += delta_nie
            remaining_pct = abs(current_nie) / denom
            points.append({
                "step": idx,
                "feature_idx": row.get("feature_idx"),
                "alpha": row.get("alpha"),
                "remaining_bias": float(current_nie),
                "remaining_bias_pct": float(remaining_pct),
                "delta_ppl": float(cumulative_delta_ppl),
                "delta_nie": float(cumulative_delta_nie),
                "last_delta_nie": delta_nie,
                "last_delta_ppl": delta_ppl,
            })

        pareto: List[Dict] = []
        best_delta = float("inf")
        for point in sorted(points, key=lambda p: p["remaining_bias_pct"]):
            if point["delta_ppl"] <= best_delta + 1e-9:
                pareto.append(point)
                best_delta = min(best_delta, point["delta_ppl"])

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
    parser = argparse.ArgumentParser(description="NIE Local Gate")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--topk_json", type=str, default="/Users/qjzheng/Desktop/CMA-SFC-integration/experiments/data/topk_mediators_gpt2-small_doctor_woman_20251026_150239.json")
    parser.add_argument("--eval_data", type=str, default="/Users/qjzheng/Desktop/CMA-SFC-integration/experiments/data/bias_eval_gpt2-small_doctor_woman_20251026_150239.json")
    parser.add_argument("--output", type=str, default="results/nie_local_gate.csv")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--controls", type=int, default=0)
    parser.add_argument("--num_features", type=int, default=5)
    parser.add_argument("--alphas", type=str, default="1.0,0.75,0.5,0.25,0.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    run_local_gate(
        model_name=args.model,
        mediator_path=args.topk_json,
        eval_path=args.eval_data,
        output_path=args.output,
        topk=args.topk,
        control_count=args.controls,
        num_features=args.num_features,
        alphas=alphas,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
