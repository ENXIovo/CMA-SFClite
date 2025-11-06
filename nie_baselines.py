"""
NIE Baselines Experiment
------------------------
Implements head-off and random feature ablation baselines using CMA-selected mediators.

This script is self-contained: helper classes/functions copied from sfc_utils modules
for easier inspection. Switch between residual- and attention-trained SAEs by editing
the SAE_MODE constant defined below.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict, namedtuple
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


# ============================================================================
# Configuration
# ============================================================================

# Set to "resid" (default) or "attn". Change this single line to switch SAE type.
SAE_MODE = "attn"  

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local residual SAE files (pre-downloaded in this repo).
RESID_SAE_TEMPLATE = os.path.join(
    PROJECT_ROOT,
    "models",
    "GPT2-Small-SAEs",
    "final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt",
)

# Attention SAE repo/template (adjust filename template to match downloaded repo layout).
ATTN_SAE_LOCAL_TEMPLATE = os.path.join(
    PROJECT_ROOT,
    "models",
    "GPT2-Small-OAI-v5-32k-attn-out-SAEs",
    "v5_32k_layer_{layer}",
    "sae_weights.safetensors",
)


# ============================================================================
# Helper data structures copied from sfc_utils.{activation_utils,loading_utils}
# ============================================================================


class SparseAct:
    """Sparse activation container used by SFC utilities."""

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

    def __mul__(self, other) -> "SparseAct":
        return self._map(lambda x, y: x * y, other)

    __rmul__ = __mul__

    def __add__(self, other) -> "SparseAct":
        return self._map(lambda x, y: x + y, other)

    def __sub__(self, other) -> "SparseAct":
        return self._map(lambda x, y: x - y, other)

    def __neg__(self) -> "SparseAct":
        return self._map(lambda x, _: -x)

    def clone(self) -> "SparseAct":
        kwargs = {}
        for attr in ("act", "res", "resc"):
            value = getattr(self, attr)
            if value is not None:
                kwargs[attr] = value.clone()
        return SparseAct(**kwargs)

    def save(self) -> "SparseAct":
        return self._map(lambda x, _: x.save())

    @property
    def value(self) -> "SparseAct":
        kwargs = {}
        for attr in ("act", "res", "resc"):
            value = getattr(self, attr)
            if value is not None:
                kwargs[attr] = value.value
        return SparseAct(**kwargs)


@dataclass(frozen=True)
class Submodule:
    """Hook handle describing a model submodule."""

    name: str
    submodule: object
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def get_activation(self) -> torch.Tensor:
        if self.use_input:
            out = self.submodule.input
        else:
            out = self.submodule.output
        tensor = out[0] if self.is_tuple else out
        return tensor

    def set_activation(self, x: torch.Tensor) -> None:
        if self.use_input:
            target = self.submodule.input
        else:
            target = self.submodule.output
        if self.is_tuple:
            target = target[0]
        target[:] = x


DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


# ============================================================================
# Helper functions copied/adapted from sfc_utils
# ============================================================================


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


def ensure_torch_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"Expected torch.Tensor, got {type(x)}")


def load_layer_sae(layer_idx: int, dtype: torch.dtype, device: torch.device) -> Optional[AutoEncoder]:
    cfgs = {
        "resid": {
            "source": "local",
            "template": RESID_SAE_TEMPLATE,
            "dict_size": 24576,
        },
        "attn": {
            "source": "local",
            "template": ATTN_SAE_LOCAL_TEMPLATE,
            "dict_size": 32768,
        },
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


def load_saes_and_submodules(
    model: LanguageModel,
    separate_by_type: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Tuple[DictionaryStash, Dict[Submodule, AutoEncoder]]:
    attns: List[Submodule] = []
    mlps: List[Submodule] = []
    resids: List[Submodule] = []
    dictionaries: Dict[Submodule, AutoEncoder] = {}

    for layer_idx, block in enumerate(model.transformer.h):
        sae = load_layer_sae(layer_idx, dtype=dtype, device=torch.device(device))
        if SAE_MODE == "attn":
            attn_mod = Submodule(
                name=f"attn_{layer_idx}",
                submodule=block.attn,
                is_tuple=True,
            )
            attns.append(attn_mod)
            dictionaries[attn_mod] = sae if sae is not None else IdentityDict(768)

            resid_mod = Submodule(
                name=f"resid_{layer_idx}",
                submodule=block,
                is_tuple=True,
            )
            resids.append(resid_mod)
            dictionaries[resid_mod] = IdentityDict(768)
        else:
            resid_mod = Submodule(
                name=f"resid_{layer_idx}",
                submodule=block,
                is_tuple=True,
            )
            resids.append(resid_mod)
            dictionaries[resid_mod] = sae if sae is not None else IdentityDict(768)

            attn_alias = Submodule(
                name=f"attn_{layer_idx}",
                submodule=block,
                is_tuple=True,
            )
            attns.append(attn_alias)
            dictionaries[attn_alias] = dictionaries[resid_mod]

        mlp_alias = Submodule(
            name=f"mlp_{layer_idx}",
            submodule=block,
            is_tuple=True,
        )
        mlps.append(mlp_alias)
        dictionaries[mlp_alias] = dictionaries[attns[-1]]

    stash = DictionaryStash(embed=None, attns=attns, mlps=mlps, resids=resids)
    return stash, dictionaries


def find_mediator_submodule(
    mediator: Dict,
    stash: DictionaryStash,
) -> Tuple[Optional[Submodule], str]:
    layer_idx = mediator["layer"]
    mediator_type = mediator.get("type", "attn")

    if mediator_type == "attn" and layer_idx < len(stash.attns):
        return stash.attns[layer_idx], "attn"
    if mediator_type == "resid" and layer_idx < len(stash.resids):
        return stash.resids[layer_idx], "resid"
    if layer_idx < len(stash.resids):
        return stash.resids[layer_idx], "resid"
    return None, mediator_type


def run_with_ablations(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    nodes,
    metric_fn,
    metric_kwargs=dict(),
    complement=False,
    ablation_fn=lambda x: x.mean(dim=0).expand_as(x),
    handle_errors="default",
):
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

            if complement:
                submod_nodes = ~submod_nodes
            submod_nodes.resc = torch.ones_like(res, dtype=torch.bool)
            if handle_errors == "remove":
                submod_nodes.resc = torch.zeros_like(submod_nodes.resc)
            if handle_errors == "keep":
                submod_nodes.resc = torch.ones_like(submod_nodes.resc)

            mask = ~submod_nodes.act
            f[..., mask] = patch_states[submodule].act[..., mask]
            res[..., ~submod_nodes.resc] = patch_states[submodule].res[..., ~submod_nodes.resc]

            submodule.set_activation(dictionary.decode(f) + res)

        metric = metric_fn(model, **metric_kwargs).save()
    return metric.value


# ============================================================================
# Ablation utilities
# ============================================================================


def compute_bias_and_ppl(model: LanguageModel, logits: torch.Tensor, tokens: torch.Tensor) -> Tuple[float, float]:
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


def apply_alpha_gate(
    model: LanguageModel,
    mediator: Dict,
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
                sample_act = submodule.get_activation().detach()
        dict_size = sample_act.shape[-1]

    mask_keep = torch.ones(dict_size, dtype=torch.bool)
    if feature_indices is None:
        mask_keep[:] = False
    else:
        mask_keep[feature_indices] = False

    def make_nodes() -> Dict[Submodule, SparseAct]:
        return {
            submodule: SparseAct(
                act=mask_keep.clone(),
                resc=torch.ones(1, dtype=torch.bool),
            )
        }

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
        complement=False,
        handle_errors="keep",
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
        complement=False,
        handle_errors="keep",
    )

    bias_clean, ppl_clean = compute_bias_and_ppl(model, logits_clean, tokens_base)
    bias_gated, ppl_gated = compute_bias_and_ppl(model, logits_gated, tokens_base)
    bias_cf_replaced, _ = compute_bias_and_ppl(model, logits_cf, tokens_base)
    nie = bias_cf_replaced - bias_clean
    return bias_clean, bias_gated, ppl_clean, ppl_gated, nie


# ============================================================================
# Baseline experiment logic
# ============================================================================


def run_baselines(
    model_name: str,
    mediator_path: str,
    eval_data_path: str,
    output_path: str,
    topk: int,
    control_count: int,
    num_features: int,
    seed: int,
    device: str,
) -> str:
    if SAE_MODE not in {"resid", "attn"}:
        raise ValueError("SAE_MODE must be 'resid' or 'attn'")

    print("=" * 70)
    print("NIE Baselines")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"Top-K Mediators 文件: {mediator_path}")
    print(f"评估数据: {eval_data_path}")
    print(f"SAE 模式: {SAE_MODE}")
    print(f"Top-K: {topk}, 控制数量: {control_count}")
    print("=" * 70)

    dtype = torch.float32
    model = LanguageModel(model_name, device_map=device, torch_dtype=dtype, dispatch=True)

    stash, dictionaries = load_saes_and_submodules(model, separate_by_type=True, dtype=dtype, device=device)

    with open(mediator_path, "r", encoding="utf-8") as f:
        mediator_data = json.load(f)
    mediators = mediator_data if isinstance(mediator_data, list) else mediator_data.get("mediators", [])

    total_needed = topk + control_count
    mediators = mediators[:total_needed] if total_needed > 0 else mediators
    mediators_main = mediators[:topk]
    mediators_control = mediators[topk:topk + control_count]

    mediator_entries = [(mediator, "topk") for mediator in mediators_main]
    mediator_entries.extend((mediator, "control") for mediator in mediators_control)

    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    examples = eval_data.get("examples", eval_data if isinstance(eval_data, list) else [])
    if not examples:
        raise ValueError("评估数据为空")

    rng = np.random.default_rng(seed)
    results: List[Dict] = []

    for idx, (mediator, category) in enumerate(tqdm(mediator_entries, desc="Mediators")):
        submodule, mediator_type = find_mediator_submodule(mediator, stash)
        if submodule is None:
            print(f"  ⚠ Mediator {idx} 无法找到对应 submodule，跳过")
            continue
        dictionary = dictionaries[submodule]

        for example_idx, example in enumerate(examples):
            base_prompt = example.get("base", example.get("clean_prefix", ""))
            cf_prompt = example.get("counterfactual", example.get("patch_prefix", ""))
            if not base_prompt or not cf_prompt:
                continue

            bias_base_orig, bias_base_gated, ppl_base_orig, ppl_base_gated, nie_orig = apply_alpha_gate(
                model,
                mediator,
                submodule,
                dictionary,
                base_prompt,
                cf_prompt,
                feature_indices=None,
                alpha=1.0,
            )

            _, bias_head_after, _, ppl_head_after, nie_head = apply_alpha_gate(
                model,
                mediator,
                submodule,
                dictionary,
                base_prompt,
                cf_prompt,
                feature_indices=None,
                alpha=0.0,
            )
            results.append({
                "analysis": "baseline",
                "edit_label": "head_off",
                "feature_source": "global",
                "feature_idx": -1,
                "alpha": 0.0,
                "nie_original": float(nie_orig),
                "nie_ablated": float(nie_head),
                "delta_nie": float(nie_head - nie_orig),
                "ppl_original": float(ppl_base_orig),
                "ppl_ablated": float(ppl_head_after),
                "delta_ppl": float(ppl_head_after - ppl_base_orig),
                "mediator_layer": mediator["layer"],
                "mediator_type": mediator_type,
                "mediator_head": mediator.get("head"),
                "mediator_nie": mediator.get("nie"),
                "mediator_category": category,
                "example_id": example_idx,
                "example_base": base_prompt[:100],
                "bias_original": float(bias_base_orig),
                "bias_edited": float(bias_head_after),
            })

            if hasattr(dictionary, "dict_size"):
                dict_size = dictionary.dict_size
            elif hasattr(dictionary, "encoder") and hasattr(dictionary.encoder, "weight"):
                dict_size = dictionary.encoder.weight.shape[0]
            else:
                with torch.no_grad():
                    with model.trace(base_prompt):
                        sample_act = submodule.get_activation().detach()
                dict_size = sample_act.shape[-1]

            random_features = rng.choice(dict_size, size=min(num_features, dict_size), replace=False)
            _, bias_rand_after, _, ppl_rand_after, nie_rand = apply_alpha_gate(
                model,
                mediator,
                submodule,
                dictionary,
                base_prompt,
                cf_prompt,
                feature_indices=random_features.tolist(),
                alpha=0.0,
            )
            results.append({
                "analysis": "baseline",
                "edit_label": "random_cut",
                "feature_source": "random",
                "feature_idx": -2,
                "alpha": 0.0,
                "nie_original": float(nie_orig),
                "nie_ablated": float(nie_rand),
                "delta_nie": float(nie_rand - nie_orig),
                "ppl_original": float(ppl_base_orig),
                "ppl_ablated": float(ppl_rand_after),
                "delta_ppl": float(ppl_rand_after - ppl_base_orig),
                "mediator_layer": mediator["layer"],
                "mediator_type": mediator_type,
                "mediator_head": mediator.get("head"),
                "mediator_nie": mediator.get("nie"),
                "mediator_category": category,
                "example_id": example_idx,
                "example_base": base_prompt[:100],
                "bias_original": float(bias_base_orig),
                "bias_edited": float(bias_rand_after),
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ 结果已写入 {output_path}")
    plot_baseline_results(results, output_path)
    return output_path


def plot_baseline_results(rows: List[Dict], csv_path: str) -> None:
    if not rows:
        return
    abs_nie = [abs(r["delta_nie"]) for r in rows]
    delta_ppl = [r["delta_ppl"] for r in rows]
    labels = [r["edit_label"] for r in rows]
    colors = {
        "head_off": "tab:red",
        "random_cut": "tab:blue",
    }

    plt.figure(figsize=(8, 5))
    for x, y, lbl in zip(abs_nie, delta_ppl, labels):
        plt.scatter(x, y, c=colors.get(lbl, "gray"), label=lbl, alpha=0.6)
    plt.xlabel("|ΔNIE|")
    plt.ylabel("ΔPPL")
    plt.title("NIE Baselines")
    handles, unique_labels = [], []
    for lbl, color in colors.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=lbl, markerfacecolor=color, markersize=8))
        unique_labels.append(lbl)
    plt.legend(handles, unique_labels)
    plt.grid(True, alpha=0.3)
    plot_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制基线图: {plot_path}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIE Baselines (Head-off, Random Cut)")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--topk_json", type=str, default="/Users/qjzheng/Desktop/CMA-SFClite/data/topk_mediators_gpt2-small_nurse_man_20251106_145026.json")
    parser.add_argument("--eval_data", type=str, default="/Users/qjzheng/Desktop/CMA-SFClite/data/bias_eval_gpt2-small_nurse_man_20251106_145026.json")
    parser.add_argument("--output", type=str, default="results/nie_baselines.csv")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--controls", type=int, default=0)
    parser.add_argument("--num_features", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baselines(
        model_name=args.model,
        mediator_path=args.topk_json,
        eval_data_path=args.eval_data,
        output_path=args.output,
        topk=args.topk,
        control_count=args.controls,
        num_features=args.num_features,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
