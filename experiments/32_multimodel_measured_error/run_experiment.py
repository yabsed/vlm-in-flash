#!/usr/bin/env python3
"""Experiment 32: multi-model measured-error and laptop-I/O evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import time
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_31 = PROJECT_ROOT / "experiments" / "31_wave_balanced_dp" / "run_experiment.py"
LOCAL_PROFILE = (
    PROJECT_ROOT / "experiments" / "26_frontier_adaptive_trim"
    / "results_laptop" / "laptop_sn850x_profile.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP31 = _load_module("experiment_31_for_32", EXPERIMENT_31)
EXP29 = EXP31.EXP29
EXP22 = EXP31.EXP22
EXP13 = EXP31.EXP13
BASE = EXP31.BASE

METHODS = ("paper", "tile8_ceil", *EXP31.WAVE_METHODS)
METHOD_LABELS = {
    "paper": "Paper",
    "tile8_ceil": "Cell-8",
    **{method: EXP31.METHOD_LABELS[method] for method in EXP31.WAVE_METHODS},
}
METHOD_COLORS = {
    "paper": "#D97706", "tile8_ceil": "#0F766E",
    "wave_k1": "#64748B", "wave_k6": "#2563EB",
    "wave_k12": "#7C3AED", "wave_k18": "#DB2777", "wave_k24": "#DC2626",
}
E2E_POLICIES = (
    "paper", "tile8_ceil", "wave_k6", "wave_dispatch",
    "min_total_dispatch", "error_guard_dispatch",
)
E2E_LABELS = {
    **METHOD_LABELS,
    "wave_dispatch": "Wave calibration dispatch",
    "min_total_dispatch": "All-method min-total dispatch",
    "error_guard_dispatch": "Measured-error guard dispatch",
}

MODEL_SPECS = {
    "qwen05": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5-0.5B-Instruct",
    },
    "smol360": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "label": "SmolLM2-360M-Instruct",
    },
    "tiny11": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "label": "TinyLlama-1.1B-Chat-v1.0",
    },
}

PROMPTS = (
    {
        "name": "cal_short_factual", "split": "calibration", "situation": "short",
        "text": "In one sentence, explain why contiguous storage reads are fast.",
    },
    {
        "name": "cal_medium_reasoning", "split": "calibration", "situation": "medium",
        "text": (
            "A system can read either eight scattered 32 KiB blocks or one contiguous "
            "256 KiB block. Discuss queueing, fixed request overhead, parallelism, and "
            "when the scattered option could still win. Give a compact engineering answer."
        ),
    },
    {
        "name": "cal_long_context", "split": "calibration", "situation": "long",
        "text": (
            "You are reviewing a neural-network inference pipeline. The selector runs on "
            "the CPU after receiving activation magnitudes from the GPU. Selected weight "
            "rows are stored on NVMe, contiguous rows merge into one request, each request "
            "has fixed overhead, and six reader threads receive tasks round-robin. The GPU "
            "then receives the selected rows and computes a projection. Explain how mask "
            "geometry, row size, request splitting at 768 KiB, thread imbalance, selector "
            "runtime, and retained signal jointly affect total latency. Separate assumptions "
            "from measurements and propose two falsifiable tests."
        ),
    },
    {
        "name": "test_short_korean", "split": "holdout", "situation": "short",
        "text": "연속된 플래시 읽기가 빠른 이유를 한 문장으로 설명해 줘.",
    },
    {
        "name": "test_medium_code", "split": "holdout", "situation": "medium",
        "text": (
            "Write a Python function that receives a Boolean mask, converts consecutive "
            "true values into half-open intervals, splits intervals into pieces no larger "
            "than a byte limit, and greedily assigns those pieces to six workers. Explain "
            "the time complexity and one edge case."
        ),
    },
    {
        "name": "test_long_structured", "split": "holdout", "situation": "long",
        "text": (
            "Consider three pruning policies evaluated at 25%, 50%, and 75% retained rows. "
            "Policy A has the fastest selector but loses some activation importance. Policy B "
            "uses six balanced intervals and retains more importance, while Policy C greedily "
            "optimizes chunks using a latency table. Measurements include selector latency, "
            "O_DIRECT read time, GPU upload time, projection relative L2 error, cosine error, "
            "next-token KL divergence, and top-1 agreement. Build a careful decision rule that "
            "does not assume activation importance is a perfect proxy for output error. Explain "
            "how calibration and held-out prompts prevent choosing a policy on the test data, "
            "and identify what conclusion would remain invalid without task-level evaluation."
        ),
    },
)

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--layer-samples", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--selector-repetitions", type=int, default=3)
    parser.add_argument("--io-repetitions", type=int, default=3)
    parser.add_argument("--io-warmup", type=int, default=1)
    parser.add_argument("--io-threads", type=int, default=6)
    parser.add_argument("--io-max-read-kib", type=int, default=768)
    parser.add_argument("--io-blob", type=Path)
    parser.add_argument("--profile", type=Path, default=LOCAL_PROFILE)
    parser.add_argument("--saturation-kib", type=float, default=240.0)
    parser.add_argument("--error-guard-slack", type=float, default=0.01)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and not args.analyze_only:
        raise SystemExit("Experiment 32 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if not args.budgets or any(not 0.0 < value < 1.0 for value in args.budgets):
        raise SystemExit("--budgets must lie strictly inside (0, 1)")
    if args.layer_samples < 1 or args.selector_repetitions < 1 or args.io_repetitions < 1:
        raise SystemExit("sample and repetition counts must be positive")
    if args.prompt_limit < 0 or args.max_input_tokens < 2:
        raise SystemExit("invalid prompt limit or token limit")


def cached_snapshot(repo_id: str) -> Path:
    root = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + repo_id.replace("/", "--")
    )
    ref = root / "refs" / "main"
    if not ref.is_file():
        raise SystemExit(f"model is not cached locally: {repo_id}")
    snapshot = root / "snapshots" / ref.read_text().strip()
    required = ("config.json", "model.safetensors", "tokenizer_config.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise SystemExit(f"incomplete cache for {repo_id}: missing {missing}")
    return snapshot


def tokenize(tokenizer, text: str, max_tokens: int) -> dict[str, torch.Tensor]:
    rendered = text
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer(
        rendered, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    return {name: tensor.to("cuda") for name, tensor in encoded.items()}


def layer_samples(num_layers: int, count: int) -> list[int]:
    if count >= num_layers:
        return list(range(num_layers))
    return sorted(set(int(round(x)) for x in np.linspace(0, num_layers - 1, count)))


def projection_name(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1].removesuffix("_proj")


class Selector:
    def __init__(self, lookup_table, saturation_kib: float):
        self.lookup_table = lookup_table
        self.saturation_kib = float(saturation_kib)
        self.contexts: dict[tuple[int, int], dict] = {}
        self.known_shapes = {item["shape"]: dict(item) for item in EXP31.SHAPES}

    def context(self, n: int, d: int) -> dict:
        key = (int(n), int(d))
        if key in self.contexts:
            return self.contexts[key]
        shape_name = f"{n}x{d}"
        spec = self.known_shapes.get(shape_name, {
            "shape": shape_name, "n": n, "d": d,
            "start_kib": 8.0, "jump_kib": 8.0,
        })
        row_kib = 2.0 * d / 1024.0
        params = EXP22.make_params(spec, self.saturation_kib)
        model = EXP13.fit_continuous_two_line(
            self.lookup_table, row_kib, self.saturation_kib
        )
        context = {
            "shape": shape_name, "row_kib": row_kib, "params": params,
            "model": model, "paper_parameter_source": (
                "table2" if shape_name in self.known_shapes else "generic-8KiB"
            ),
        }
        self.contexts[key] = context
        return context

    def select(self, method: str, importance: torch.Tensor, row_budget: int,
               d: int) -> tuple[torch.Tensor, dict]:
        n = int(importance.numel())
        context = self.context(n, d)
        normalized = importance / importance.sum()
        if method == "paper":
            mask, metadata = EXP22.paper_select(
                normalized, int(row_budget), context["row_kib"],
                self.lookup_table, context["params"], "native",
            )
            return mask.to(device=importance.device, dtype=torch.bool), {
                **metadata, "fallback_used": False, "error": "",
            }
        values = normalized.detach().to("cpu").numpy().astype(np.float64)
        mask_np, metadata = EXP31.safe_select(
            method, values, int(row_budget), context["model"], self.lookup_table
        )
        mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(
            importance.device
        )
        return mask, metadata


def benchmark_selector(selector: Selector, method: str, importance: torch.Tensor,
                       row_budget: int, d: int, repetitions: int):
    selector.select(method, importance, row_budget, d)
    samples = []
    masks = []
    metadata = {}
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        mask, metadata = selector.select(method, importance, row_budget, d)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        masks.append(mask.detach().to("cpu").numpy().astype(bool, copy=True))
    deterministic = all(np.array_equal(masks[0], candidate) for candidate in masks[1:])
    return masks[0], metadata, samples, deterministic


def measure_io(native_reader, blob: Path, mask_np: np.ndarray, row_bytes: int,
               args: argparse.Namespace):
    mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to("cuda")
    recorded = []
    for repetition in range(args.io_warmup + args.io_repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output, io_us, upload_us, direct = native_reader.read_rows(
            str(blob), mask, int(row_bytes), "cuda", args.io_threads,
            args.io_max_read_kib * 1024, True,
        )
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        del output
        if repetition >= args.io_warmup:
            recorded.append((float(io_us) / 1000.0, float(upload_us) / 1000.0,
                             wall_ms, bool(direct)))
    return recorded


def output_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().float()
    cand = candidate.detach().float()
    diff = cand - ref
    ref_l2 = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    cand_l2 = torch.linalg.vector_norm(cand).clamp_min(1e-12)
    dot = torch.sum(ref * cand)
    return {
        "relative_l2_error": float(torch.linalg.vector_norm(diff) / ref_l2),
        "relative_mae": float(diff.abs().mean() / ref.abs().mean().clamp_min(1e-12)),
        "cosine_error": float(1.0 - dot / (ref_l2 * cand_l2)),
        "output_norm_ratio": float(cand_l2 / ref_l2),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_projection_hook(state: dict, model_key: str, model_label: str,
                         module_name: str, selector: Selector, native_reader,
                         args: argparse.Namespace, local_rows: list[dict],
                         timing_rows: list[dict], io_rows: list[dict]):
    layer_match = LAYER_RE.search(module_name)
    layer_index = int(layer_match.group(1)) if layer_match else -1
    proj = projection_name(module_name)

    def hook(module, inputs, output):
        x = inputs[0].detach()
        reference = output.detach()
        n, d = int(module.in_features), int(module.out_features)
        importance = x.abs().reshape(-1, n).mean(dim=0).float()
        importance_total = float(importance.sum())
        case_context = selector.context(n, d)
        for budget_index, fraction in enumerate(args.budgets):
            row_budget = max(1, min(n - 1, int(round(n * fraction))))
            for method in METHODS:
                case_id = f"{model_key}-{len(local_rows):06d}"
                mask_np, metadata, selector_samples, deterministic = benchmark_selector(
                    selector, method, importance, row_budget, d,
                    args.selector_repetitions,
                )
                selected = int(mask_np.sum())
                mask = torch.from_numpy(mask_np).to(x.device)
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                sparse = F.linear(
                    x.index_select(-1, indices),
                    module.weight.index_select(1, indices), module.bias,
                )
                errors = output_error(reference, sparse)
                retained = float(importance[mask].sum()) / importance_total
                io_samples = measure_io(
                    native_reader, args.io_blob, mask_np, d * 2, args
                )
                selector_median = float(np.median(selector_samples))
                wall_values = [sample[2] for sample in io_samples]
                read_median = float(np.median(wall_values))
                starts, _ = EXP31._run_bounds(mask_np)
                local_rows.append({
                    "case_id": case_id, "model": model_key,
                    "model_label": model_label, "prompt": state["prompt"],
                    "split": state["split"], "situation": state["situation"],
                    "num_tokens": state["num_tokens"], "module": module_name,
                    "layer": layer_index, "projection": proj,
                    "shape": f"{n}x{d}", "n": n, "d": d,
                    "row_size_kib": 2.0 * d / 1024.0,
                    "paper_parameter_source": case_context["paper_parameter_source"],
                    "budget_index": budget_index, "budget_fraction": fraction,
                    "row_budget": row_budget, "method": method,
                    "method_label": METHOD_LABELS[method], "selected_rows": selected,
                    "row_match": selected == row_budget,
                    "importance_retention": retained,
                    **errors,
                    "chunks": int(len(starts)),
                    "selector_median_ms": selector_median,
                    "selector_p95_ms": float(np.quantile(selector_samples, 0.95)),
                    "read_wall_median_ms": read_median,
                    "read_wall_p95_ms": float(np.quantile(wall_values, 0.95)),
                    "actual_total_ms": selector_median + read_median,
                    "direct_rate": float(np.mean([sample[3] for sample in io_samples])),
                    "deterministic": deterministic,
                    "fallback_used": bool(metadata.get("fallback_used", False)),
                    "error": metadata.get("error", ""),
                })
                for repetition, elapsed in enumerate(selector_samples):
                    timing_rows.append({
                        "case_id": case_id, "repetition": repetition,
                        "runtime_ms": elapsed,
                    })
                for repetition, (io_ms, upload_ms, wall_ms, direct) in enumerate(io_samples):
                    io_rows.append({
                        "case_id": case_id, "repetition": repetition,
                        "io_ms": io_ms, "upload_ms": upload_ms,
                        "read_wall_ms": wall_ms, "direct": direct,
                    })
                del sparse, mask, indices
        print(
            f"  measured {model_key} {state['prompt']} {module_name}", flush=True
        )

    return hook


def build_dispatch(local: pd.DataFrame, slack: float) -> tuple[pd.DataFrame, dict]:
    calibration = local[local.split == "calibration"]
    group_keys = ["model", "shape", "budget_index"]
    rows = []
    maps = {"wave_dispatch": {}, "min_total_dispatch": {}, "error_guard_dispatch": {}}
    for key, group in calibration.groupby(group_keys, sort=True):
        means = group.groupby("method").agg(
            actual_total_ms=("actual_total_ms", "mean"),
            relative_l2_error=("relative_l2_error", "mean"),
            importance_retention=("importance_retention", "mean"),
        )
        wave = means.loc[list(EXP31.WAVE_METHODS)]
        wave_choice = (wave.importance_retention / wave.actual_total_ms).idxmax()
        total_choice = means.actual_total_ms.idxmin()
        paper_error = float(means.loc["paper", "relative_l2_error"])
        allowed = means[
            means.relative_l2_error <= paper_error * (1.0 + slack) + 1e-9
        ]
        guard_choice = allowed.actual_total_ms.idxmin()
        serial_key = "|".join(map(str, key))
        choices = {
            "wave_dispatch": wave_choice,
            "min_total_dispatch": total_choice,
            "error_guard_dispatch": guard_choice,
        }
        for policy, method in choices.items():
            maps[policy][serial_key] = method
            rows.append({
                "model": key[0], "shape": key[1], "budget_index": int(key[2]),
                "policy": policy, "selected_method": method,
                "calibration_cases": int(len(group[group.method == method])),
                "calibration_total_ms": float(means.loc[method, "actual_total_ms"]),
                "calibration_relative_l2_error": float(
                    means.loc[method, "relative_l2_error"]
                ),
                "calibration_importance_retention": float(
                    means.loc[method, "importance_retention"]
                ),
                "paper_calibration_error": paper_error,
            })
    return pd.DataFrame(rows), maps


class DynamicPolicy:
    def __init__(self, selector: Selector, selection_type, model_key: str,
                 dispatch_maps: dict):
        self.selector = selector
        self.selection_type = selection_type
        self.model_key = model_key
        self.dispatch_maps = dispatch_maps
        self.policy_name = "paper"
        self.budget_index = 0
        self.calls: list[dict] = []

    def configure(self, policy_name: str, budget_index: int) -> None:
        self.policy_name = policy_name
        self.budget_index = int(budget_index)
        self.calls = []

    def resolve(self, n: int, d: int) -> str:
        if self.policy_name in METHODS:
            return self.policy_name
        key = f"{self.model_key}|{n}x{d}|{self.budget_index}"
        return self.dispatch_maps[self.policy_name][key]

    def __call__(self, importance: torch.Tensor, num_load_rows: int,
                 row_size_kib: float):
        d = int(round(row_size_kib * 1024.0 / 2.0))
        method = self.resolve(int(importance.numel()), d)
        mask, metadata = self.selector.select(method, importance, num_load_rows, d)
        self.calls.append({
            "method": method, "rows": int(mask.sum()),
            "fallback": bool(metadata.get("fallback_used", False)),
        })
        return self.selection_type(
            mask, float(importance[mask].sum()), None
        )


def logits_error(reference: torch.Tensor, candidate: torch.Tensor,
                 input_ids: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().float()
    cand = candidate.detach().float()
    basic = output_error(ref, cand)
    ref_logp = F.log_softmax(ref, dim=-1)
    cand_logp = F.log_softmax(cand, dim=-1)
    kl = torch.sum(ref_logp.exp() * (ref_logp - cand_logp), dim=-1).mean()
    agreement = (ref.argmax(dim=-1) == cand.argmax(dim=-1)).float().mean()
    if input_ids.shape[1] > 1:
        labels = input_ids[:, 1:].reshape(-1)
        dense_nll = F.cross_entropy(ref[:, :-1].reshape(-1, ref.shape[-1]), labels)
        sparse_nll = F.cross_entropy(cand[:, :-1].reshape(-1, cand.shape[-1]), labels)
    else:
        dense_nll = torch.tensor(float("nan"), device=ref.device)
        sparse_nll = torch.tensor(float("nan"), device=ref.device)
    return {
        "logit_relative_l2_error": basic["relative_l2_error"],
        "logit_cosine_error": basic["cosine_error"],
        "dense_to_sparse_kl": float(kl),
        "top1_agreement": float(agreement),
        "dense_nll": float(dense_nll),
        "sparse_nll": float(sparse_nll),
        "nll_delta": float(sparse_nll - dense_nll),
    }


def run_end_to_end(model, tokenizer, model_key: str, model_label: str,
                   selector: Selector, dispatch_maps: dict, prompts: list[dict],
                   args: argparse.Namespace, vlmflash) -> list[dict]:
    policy = DynamicPolicy(selector, vlmflash.Selection, model_key, dispatch_maps)
    handle = vlmflash.attach(
        model, policy=policy, include=vlmflash.DEFAULT_INCLUDE, sparsity=0.5
    )
    rows = []
    try:
        for prompt in prompts:
            if prompt["split"] != "holdout":
                continue
            inputs = tokenize(tokenizer, prompt["text"], args.max_input_tokens)
            with torch.inference_mode():
                dense = model(**inputs, use_cache=False).logits.detach()
            for budget_index, fraction in enumerate(args.budgets):
                for name in handle.names:
                    model.get_submodule(name).nc_sparsity = 1.0 - float(fraction)
                for policy_name in E2E_POLICIES:
                    policy.configure(policy_name, budget_index)
                    with torch.inference_mode(), vlmflash.enabled():
                        sparse = model(**inputs, use_cache=False).logits.detach()
                    metrics = logits_error(dense, sparse, inputs["input_ids"])
                    choice_counts = pd.Series(
                        [call["method"] for call in policy.calls]
                    ).value_counts().to_dict()
                    rows.append({
                        "model": model_key, "model_label": model_label,
                        "prompt": prompt["name"], "split": prompt["split"],
                        "situation": prompt["situation"],
                        "num_tokens": int(inputs["input_ids"].numel()),
                        "budget_index": budget_index, "budget_fraction": fraction,
                        "policy": policy_name, "policy_label": E2E_LABELS[policy_name],
                        **metrics, "projection_calls": len(policy.calls),
                        "fallback_calls": sum(call["fallback"] for call in policy.calls),
                        "resolved_method_counts": json.dumps(choice_counts, sort_keys=True),
                    })
                    print(
                        f"  e2e {model_key} {prompt['name']} budget={fraction:g} "
                        f"policy={policy_name}", flush=True,
                    )
                    del sparse
            del dense, inputs
    finally:
        handle.detach()
    return rows


def run_model(model_key: str, selector: Selector, native_reader,
              prompts: list[dict], args: argparse.Namespace, vlmflash):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    spec = MODEL_SPECS[model_key]
    snapshot = cached_snapshot(spec["repo"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float16, local_files_only=True
    ).to("cuda").eval()
    num_layers = int(model.config.num_hidden_layers)
    sampled_layers = layer_samples(num_layers, args.layer_samples)
    targets = []
    for name, module in model.named_modules():
        match = LAYER_RE.search(name)
        if (
            match and int(match.group(1)) in sampled_layers
            and re.search(vlmflash.DEFAULT_INCLUDE, name)
            and type(module) is torch.nn.Linear
        ):
            targets.append((name, module))
    expected = len(sampled_layers) * 7
    if len(targets) != expected:
        raise RuntimeError(
            f"{model_key}: expected {expected} sampled projections, got {len(targets)}"
        )
    local_rows: list[dict] = []
    timing_rows: list[dict] = []
    io_rows: list[dict] = []
    state = {"prompt": "", "split": "", "situation": "", "num_tokens": 0}
    hooks = [
        module.register_forward_hook(make_projection_hook(
            state, model_key, spec["label"], name, selector, native_reader,
            args, local_rows, timing_rows, io_rows,
        ))
        for name, module in targets
    ]
    try:
        with torch.inference_mode():
            for prompt in prompts:
                inputs = tokenize(tokenizer, prompt["text"], args.max_input_tokens)
                state.update({
                    "prompt": prompt["name"], "split": prompt["split"],
                    "situation": prompt["situation"],
                    "num_tokens": int(inputs["input_ids"].numel()),
                })
                output = model(**inputs, use_cache=False)
                torch.cuda.synchronize()
                del output, inputs
                print(
                    f"captured {model_key} {prompt['name']} with "
                    f"{state['num_tokens']} tokens", flush=True,
                )
    finally:
        for hook in hooks:
            hook.remove()
    local = pd.DataFrame(local_rows)
    dispatch_frame, dispatch_maps = build_dispatch(local, args.error_guard_slack)
    e2e_rows = []
    if not args.skip_end_to_end and any(p["split"] == "holdout" for p in prompts):
        e2e_rows = run_end_to_end(
            model, tokenizer, model_key, spec["label"], selector,
            dispatch_maps, prompts, args, vlmflash,
        )
    model_metadata = {
        "model": model_key, "label": spec["label"], "repo": spec["repo"],
        "snapshot": snapshot.name, "model_type": model.config.model_type,
        "layers": num_layers, "sampled_layers": sampled_layers,
        "sampled_projection_count": len(targets),
        "shapes": sorted({f"{m.in_features}x{m.out_features}" for _, m in targets}),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "dtype": str(next(model.parameters()).dtype),
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return local_rows, timing_rows, io_rows, dispatch_frame, dispatch_maps, e2e_rows, model_metadata


def aggregate_and_report(args: argparse.Namespace) -> None:
    output = args.output_dir
    local = pd.read_csv(output / "projection_cases.csv")
    dispatch = pd.read_csv(output / "dispatch.csv")
    e2e_path = output / "end_to_end.csv"
    e2e = pd.read_csv(e2e_path) if e2e_path.is_file() else pd.DataFrame()
    holdout = local[local.split == "holdout"]
    evaluation = holdout if len(holdout) else local
    summary = evaluation.groupby(["model", "budget_index", "method"], as_index=False).agg(
        cases=("case_id", "size"),
        actual_total_ms_mean=("actual_total_ms", "mean"),
        actual_total_ms_median=("actual_total_ms", "median"),
        relative_l2_error_mean=("relative_l2_error", "mean"),
        relative_l2_error_median=("relative_l2_error", "median"),
        cosine_error_mean=("cosine_error", "mean"),
        importance_retention_mean=("importance_retention", "mean"),
        row_match_rate=("row_match", "mean"),
        direct_rate=("direct_rate", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    summary["method_label"] = summary.method.map(METHOD_LABELS)
    summary.to_csv(output / "projection_summary.csv", index=False)
    overall = evaluation.groupby("method", as_index=False).agg(
        cases=("case_id", "size"), actual_total_ms_mean=("actual_total_ms", "mean"),
        relative_l2_error_mean=("relative_l2_error", "mean"),
        relative_l2_error_median=("relative_l2_error", "median"),
        cosine_error_mean=("cosine_error", "mean"),
        importance_retention_mean=("importance_retention", "mean"),
        direct_rate=("direct_rate", "mean"), fallback_rate=("fallback_used", "mean"),
    )
    overall["method_label"] = overall.method.map(METHOD_LABELS)
    overall.to_csv(output / "overall_summary.csv", index=False)
    within_rows = []
    paired_keys = ["model", "prompt", "module", "budget_index"]
    for key, group in evaluation.groupby(paired_keys, sort=False):
        within_rows.append({
            **dict(zip(paired_keys, key)),
            "importance_vs_relative_l2_spearman": float(
                group.importance_retention.corr(
                    group.relative_l2_error, method="spearman"
                )
            ),
            "importance_vs_cosine_error_spearman": float(
                group.importance_retention.corr(group.cosine_error, method="spearman")
            ),
        })
    within = pd.DataFrame(within_rows)
    within.to_csv(output / "within_case_correlations.csv", index=False)
    correlations = []
    for model_name, group in evaluation.groupby("model"):
        model_within = within[within.model == model_name]
        correlations.append({
            "model": model_name,
            "importance_vs_relative_l2_spearman": float(
                group.importance_retention.corr(group.relative_l2_error, method="spearman")
            ),
            "importance_vs_cosine_error_spearman": float(
                group.importance_retention.corr(group.cosine_error, method="spearman")
            ),
            "within_case_relative_l2_spearman_mean": float(
                model_within.importance_vs_relative_l2_spearman.mean()
            ),
            "within_case_relative_l2_spearman_median": float(
                model_within.importance_vs_relative_l2_spearman.median()
            ),
            "cases": int(len(group)),
        })
    correlations.append({
        "model": "all",
        "importance_vs_relative_l2_spearman": float(
            evaluation.importance_retention.corr(
                evaluation.relative_l2_error, method="spearman"
            )
        ),
        "importance_vs_cosine_error_spearman": float(
            evaluation.importance_retention.corr(
                evaluation.cosine_error, method="spearman"
            )
        ),
        "within_case_relative_l2_spearman_mean": float(
            within.importance_vs_relative_l2_spearman.mean()
        ),
        "within_case_relative_l2_spearman_median": float(
            within.importance_vs_relative_l2_spearman.median()
        ),
        "cases": int(len(evaluation)),
    })
    pd.DataFrame(correlations).to_csv(output / "correlations.csv", index=False)
    if len(e2e):
        e2e_summary = e2e.groupby(["model", "budget_index", "policy"], as_index=False).agg(
            cases=("prompt", "size"), dense_to_sparse_kl_mean=("dense_to_sparse_kl", "mean"),
            top1_agreement_mean=("top1_agreement", "mean"),
            nll_delta_mean=("nll_delta", "mean"),
            logit_relative_l2_error_mean=("logit_relative_l2_error", "mean"),
            fallback_calls=("fallback_calls", "sum"),
        )
        e2e_summary["policy_label"] = e2e_summary.policy.map(E2E_LABELS)
        e2e_summary.to_csv(output / "end_to_end_summary.csv", index=False)
    else:
        e2e_summary = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in overall.itertuples(index=False):
        ax.scatter(row.actual_total_ms_mean, row.relative_l2_error_mean, s=75,
                   color=METHOD_COLORS[row.method], label=row.method_label)
        ax.annotate(row.method_label, (row.actual_total_ms_mean, row.relative_l2_error_mean),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Measured selector + O_DIRECT read (ms)")
    ax.set_ylabel("Projection relative L2 error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "speed_error.png", dpi=180)
    fig.savefig(output / "speed_error.pdf")
    plt.close(fig)

    if len(e2e_summary):
        plotted = e2e_summary.groupby("policy", as_index=False).agg(
            kl=("dense_to_sparse_kl_mean", "mean"),
            agreement=("top1_agreement_mean", "mean"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        x = np.arange(len(plotted))
        axes[0].bar(x, plotted.kl)
        axes[0].set_ylabel("Dense→sparse logit KL")
        axes[1].bar(x, plotted.agreement)
        axes[1].set_ylabel("Next-token top-1 agreement")
        for ax in axes:
            ax.set_xticks(x, [E2E_LABELS[p] for p in plotted.policy], rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "end_to_end_error.png", dpi=180)
        fig.savefig(output / "end_to_end_error.pdf")
        plt.close(fig)

    metadata = json.loads((output / "metadata.json").read_text())
    result = {
        "format": "experiment-32-multimodel-measured-error-v1",
        "models": metadata["models"], "prompts": metadata["prompts"],
        "projection_cases": int(len(local)), "holdout_projection_cases": int(len(holdout)),
        "summary_split": "holdout" if len(holdout) else "all-available",
        "selector_samples": int(len(pd.read_csv(output / "selector_samples.csv"))),
        "io_samples": int(len(pd.read_csv(output / "io_samples.csv"))),
        "all_direct": bool((local.direct_rate == 1.0).all()),
        "all_deterministic": bool(local.deterministic.all()),
        "fallback_cases": int(local.fallback_used.sum()),
        "correlations": correlations,
    }
    if len(e2e):
        result["end_to_end_cases"] = int(len(e2e))
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(
        args, overall, summary, dispatch, e2e_summary, correlations,
        metadata, evaluation,
    )


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_report(args, overall, summary, dispatch, e2e_summary, correlations,
                 metadata, evaluation) -> None:
    report_path = args.report_output or HERE / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    indexed = overall.set_index("method").reindex(METHODS)
    fastest = indexed.actual_total_ms_mean.idxmin()
    lowest_error = indexed.relative_l2_error_mean.idxmin()
    correlation = next(item for item in correlations if item["model"] == "all")
    lines = [
        "# Experiment 32 보고서: 다중 모델 실제 오차", "",
        f"`{metadata['gpu']}`와 `{metadata['flash']}`에서 세 실제 모델, calibration/holdout "
        "prompt, 세 row budget을 측정했다. Importance 외에 실제 projection 출력 오차와 "
        "전체 forward logit 오차를 직접 계산했다.", "",
        "## Holdout projection 결과", "",
        "| 방법 | actual total | relative L2 | cosine error | importance retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in indexed.iterrows():
        lines.append(
            f"| {METHOD_LABELS[method]} | {row.actual_total_ms_mean:.3f} ms "
            f"| {row.relative_l2_error_mean:.4f} | {row.cosine_error_mean:.4f} "
            f"| {100*row.importance_retention_mean:.2f}% |"
        )
    paired_keys = ["model", "prompt", "module", "budget_index"]
    time_pivot = evaluation.pivot(index=paired_keys, columns="method", values="actual_total_ms")
    error_pivot = evaluation.pivot(
        index=paired_keys, columns="method", values="relative_l2_error"
    )
    paper_total = float(indexed.loc["paper", "actual_total_ms_mean"])
    cell_total = float(indexed.loc["tile8_ceil", "actual_total_ms_mean"])
    paper_error = float(indexed.loc["paper", "relative_l2_error_mean"])
    cell_error = float(indexed.loc["tile8_ceil", "relative_l2_error_mean"])
    cell_saving_pct = 100.0 * (paper_total - cell_total) / paper_total
    cell_error_pct = 100.0 * (cell_error / paper_error - 1.0)
    lines.extend([
        "", f"최저 actual total은 **{METHOD_LABELS[fastest]}**, 최저 projection relative "
        f"L2 오차는 **{METHOD_LABELS[lowest_error]}**였다.", "",
        f"같은 projection×prompt×budget 안에서 방법만 바꾼 importance retention과 "
        f"relative L2 error의 Spearman 상관은 평균 "
        f"`{correlation['within_case_relative_l2_spearman_mean']:.3f}`, 중앙값 "
        f"`{correlation['within_case_relative_l2_spearman_median']:.3f}`이다. 완전한 "
        "대리변수라면 -1에 가까워야 한다.", "",
        "## 모델별 Cell-8 대 Paper", "",
        "| 모델 | Paper total | Cell-8 total | 시간 절감 | Paper rel-L2 | Cell-8 rel-L2 | 오차 변화 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    model_means = evaluation.groupby(["model", "method"]).agg(
        total=("actual_total_ms", "mean"), error=("relative_l2_error", "mean")
    )
    for model_name in evaluation.model.drop_duplicates():
        paper_row = model_means.loc[(model_name, "paper")]
        cell_row = model_means.loc[(model_name, "tile8_ceil")]
        lines.append(
            f"| {model_name} | {paper_row.total:.3f} ms | {cell_row.total:.3f} ms "
            f"| {100*(paper_row.total-cell_row.total)/paper_row.total:.2f}% "
            f"| {paper_row.error:.4f} | {cell_row.error:.4f} "
            f"| {100*(cell_row.error/paper_row.error-1):+.2f}% |"
        )
    lines.extend([
        "", "## Calibration dispatch", "",
        "Dispatcher는 calibration prompt에서만 model×shape×budget별 방법을 고르고 "
        "holdout에는 고정한다. Error guard는 calibration Paper relative L2의 1% 이내인 "
        "후보 중 actual total이 가장 작은 방법이다.", "",
        "| 정책 | Paper | Cell-8 | Wave K | 총 선택 수 |",
        "|---|---:|---:|---:|---:|",
    ])
    for policy, group in dispatch.groupby("policy", sort=False):
        counts = group.selected_method.value_counts()
        lines.append(
            f"| {E2E_LABELS[policy]} | {int(counts.get('paper', 0))} "
            f"| {int(counts.get('tile8_ceil', 0))} "
            f"| {int(sum(counts.get(method, 0) for method in EXP31.WAVE_METHODS))} "
            f"| {len(group)} |"
        )
    if len(e2e_summary):
        aggregate = e2e_summary.groupby("policy", as_index=False).agg(
            kl=("dense_to_sparse_kl_mean", "mean"),
            top1=("top1_agreement_mean", "mean"),
            nll=("nll_delta_mean", "mean"),
            rel_l2=("logit_relative_l2_error_mean", "mean"),
        ).set_index("policy").reindex(E2E_POLICIES)
        lines.extend([
            "", "## Holdout end-to-end logit 오차", "",
            "| 정책 | logit relative L2 | dense→sparse KL | top-1 agreement | NLL delta |",
            "|---|---:|---:|---:|---:|",
        ])
        for policy, row in aggregate.iterrows():
            lines.append(
                f"| {E2E_LABELS[policy]} | {row.rel_l2:.4f} | {row.kl:.4f} "
                f"| {100*row.top1:.2f}% | {row.nll:+.4f} |"
            )
        lines.extend([
            "", "### Budget별 Paper·Cell-8·Wave K=6", "",
            "| budget | 정책 | KL | top-1 | NLL delta |",
            "|---:|---|---:|---:|---:|",
        ])
        budget_view = e2e_summary[e2e_summary.policy.isin(
            ["paper", "tile8_ceil", "wave_k6"]
        )].groupby(["budget_index", "policy"], as_index=False).agg(
            kl=("dense_to_sparse_kl_mean", "mean"),
            top1=("top1_agreement_mean", "mean"),
            nll=("nll_delta_mean", "mean"),
        )
        for budget_index in sorted(budget_view.budget_index.unique()):
            for policy in ("paper", "tile8_ceil", "wave_k6"):
                row = budget_view[
                    (budget_view.budget_index == budget_index)
                    & (budget_view.policy == policy)
                ].iloc[0]
                lines.append(
                    f"| {100*metadata['budgets'][int(budget_index)]:.0f}% "
                    f"| {E2E_LABELS[policy]} | {row.kl:.4f} "
                    f"| {100*row.top1:.2f}% | {row.nll:+.4f} |"
                )
        lines.extend([
            "", "### 모델별 주요 end-to-end 결과", "",
            "| 모델 | 정책 | KL | top-1 | NLL delta |",
            "|---|---|---:|---:|---:|",
        ])
        selected_policies = ("paper", "tile8_ceil", "wave_k6", "error_guard_dispatch")
        for model_name in e2e_summary.model.drop_duplicates():
            for policy in selected_policies:
                group = e2e_summary[
                    (e2e_summary.model == model_name) & (e2e_summary.policy == policy)
                ]
                if not len(group):
                    continue
                lines.append(
                    f"| {model_name} | {E2E_LABELS[policy]} "
                    f"| {group.dense_to_sparse_kl_mean.mean():.4f} "
                    f"| {100*group.top1_agreement_mean.mean():.2f}% "
                    f"| {group.nll_delta_mean.mean():+.4f} |"
                )
        paper_kl = float(aggregate.loc["paper", "kl"])
        guard_kl = float(aggregate.loc["error_guard_dispatch", "kl"])
        lines.extend([
            "", "## 판정", "",
            f"순수 actual_total에서는 Cell-8이 가장 좋다. Paper 대비 평균 "
            f"`{cell_saving_pct:.2f}%` 빠르고 paired case의 "
            f"`{100*(time_pivot.tile8_ceil < time_pivot.paper).mean():.1f}%`에서 이겼다. "
            f"대신 projection relative L2는 평균 `{cell_error_pct:+.2f}%` 악화됐고, "
            f"오차까지 개선한 case는 `{100*(error_pivot.tile8_ceil < error_pivot.paper).mean():.1f}%`뿐이다.", "",
            f"Calibration projection-error guard도 end-to-end 보장은 못 했다. 평균 logit "
            f"KL은 Paper `{paper_kl:.3f}`에서 guard `{guard_kl:.3f}`로 "
            f"`{100*(guard_kl/paper_kl-1):+.2f}%` 증가했다. 따라서 importance뿐 아니라 "
            "projection 오차조차 누적된 모델 오차의 완전한 대리변수가 아니다.", "",
        ])
    lines.extend([
        "", "## 측정 범위", "",
        f"- 모델: {', '.join(item['label'] for item in metadata['models'])}.",
        "- 각 모델의 초·중·후반 layer에서 q/k/v/o/gate/up/down projection을 측정했다.",
        "- Projection 오차는 원래 dense activation과 실제 checkpoint weight에서 계산했다.",
        "- End-to-end 오차는 모든 decoder projection을 실제로 sparsify한 forward의 logits다.",
        "- actual total은 selector와 동일 mask의 native O_DIRECT+GPU upload 합이며, "
        "전체 LLM forward latency는 아니다.",
        "- Paper는 원 알고리즘의 whole-window 제약 때문에 holdout case 일부에서 "
        "명목 row budget을 소폭 underfill할 수 있으며, 후보들은 exact-R이다.",
        "- End-to-end 평가는 모든 text-prefill decoder projection을 동시에 sparsify한 "
        "stress scenario이며 논문의 visual-token 적용 조건과 동일하지 않다.",
        "- Logit/NLL은 측정 가능한 모델 오차지만 downstream task 정확도 자체는 아니다.",
        "", "![Speed-error frontier](results_laptop/speed_error.png)", "",
        "![End-to-end error](results_laptop/end_to_end_error.png)", "",
    ])
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        aggregate_and_report(args)
        return
    prompts = list(PROMPTS[:args.prompt_limit] if args.prompt_limit else PROMPTS)
    lookup_table = BASE.LatencyTable.load(args.profile)
    selector = Selector(lookup_table, args.saturation_kib)
    vlmflash = EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")
    max_required = 0
    for key in args.models:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(cached_snapshot(MODEL_SPECS[key]["repo"]),
                                            local_files_only=True)
        max_required = max(
            max_required,
            int(config.intermediate_size) * int(config.hidden_size) * 2,
        )
    if args.io_blob.stat().st_size < max_required:
        raise SystemExit(
            f"I/O blob has {args.io_blob.stat().st_size} bytes; need {max_required}"
        )

    all_local, all_timing, all_io, all_dispatch, all_e2e, model_metadata = [], [], [], [], [], []
    dispatch_maps_by_model = {}
    for model_key in args.models:
        print(f"loading and measuring {MODEL_SPECS[model_key]['label']}", flush=True)
        result = run_model(model_key, selector, native_reader, prompts, args, vlmflash)
        local_rows, timing_rows, io_rows, dispatch, dispatch_maps, e2e_rows, metadata = result
        all_local.extend(local_rows)
        all_timing.extend(timing_rows)
        all_io.extend(io_rows)
        all_dispatch.append(dispatch)
        all_e2e.extend(e2e_rows)
        model_metadata.append(metadata)
        dispatch_maps_by_model[model_key] = dispatch_maps
        pd.DataFrame(all_local).to_csv(args.output_dir / "projection_cases.partial.csv", index=False)

    pd.DataFrame(all_local).to_csv(args.output_dir / "projection_cases.csv", index=False)
    pd.DataFrame(all_timing).to_csv(args.output_dir / "selector_samples.csv", index=False)
    pd.DataFrame(all_io).to_csv(args.output_dir / "io_samples.csv", index=False)
    pd.concat(all_dispatch, ignore_index=True).to_csv(args.output_dir / "dispatch.csv", index=False)
    if all_e2e:
        pd.DataFrame(all_e2e).to_csv(args.output_dir / "end_to_end.csv", index=False)
    partial = args.output_dir / "projection_cases.partial.csv"
    if partial.exists():
        partial.unlink()
    profile_meta = lookup_table.meta
    metadata = {
        "format": "experiment-32-multimodel-measured-error-v1",
        "models": model_metadata,
        "prompts": [
            {k: v for k, v in prompt.items() if k != "text"} | {
                "sha256": sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "budgets": args.budgets, "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions, "io_warmup": args.io_warmup,
        "io_threads": args.io_threads, "io_max_read_kib": args.io_max_read_kib,
        "error_guard_slack": args.error_guard_slack,
        "profile": str(args.profile), "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "flash": profile_meta.get("flash", "unknown"),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
        "dispatch_maps": dispatch_maps_by_model,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    aggregate_and_report(args)


if __name__ == "__main__":
    main()
