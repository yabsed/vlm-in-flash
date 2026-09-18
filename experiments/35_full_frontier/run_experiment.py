#!/usr/bin/env python3
"""Experiment 35: full Paper/Cell-8 quality-latency frontiers."""

from __future__ import annotations

import argparse
import gc
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


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXP34_PATH = (
    PROJECT_ROOT / "experiments" / "34_quality_constrained_variable_r"
    / "run_experiment.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP34 = _load_module("experiment_34_for_35", EXP34_PATH)
EXP32 = EXP34.EXP32
EXP31 = EXP34.EXP31
MODEL_SPECS = EXP34.MODEL_SPECS
PROMPTS = EXP34.PROMPTS
LAYER_RE = EXP34.LAYER_RE
LOCAL_PROFILE = EXP34.LOCAL_PROFILE

METHODS = ("paper", "cell8")
METHOD_LABELS = {"paper": "Paper", "cell8": "Cell-8"}
METHOD_COLORS = {"paper": "#D97706", "cell8": "#0F766E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument(
        "--budgets", type=float, nargs="+",
        default=[x / 100.0 for x in range(10, 100, 5)],
    )
    parser.add_argument(
        "--quality-targets", type=float, nargs="+",
        default=[
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90, 0.925, 0.95,
        ],
    )
    parser.add_argument("--calibration-error-ceiling", type=float, default=0.42)
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--layer-samples", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--selector-repetitions", type=int, default=2)
    parser.add_argument("--io-repetitions", type=int, default=2)
    parser.add_argument("--io-warmup", type=int, default=1)
    parser.add_argument("--gemm-repetitions", type=int, default=5)
    parser.add_argument("--gemm-warmup", type=int, default=2)
    parser.add_argument("--io-threads", type=int, default=6)
    parser.add_argument("--io-max-read-kib", type=int, default=768)
    parser.add_argument("--io-blob", type=Path)
    parser.add_argument("--profile", type=Path, default=LOCAL_PROFILE)
    parser.add_argument("--saturation-kib", type=float, default=240.0)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.analyze_only and not torch.cuda.is_available():
        raise SystemExit("Experiment 35 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if args.budgets != sorted(set(args.budgets)) or any(
        not 0.0 < value < 1.0 for value in args.budgets
    ):
        raise SystemExit("--budgets must be unique, increasing, and inside (0, 1)")
    if args.quality_targets != sorted(set(args.quality_targets)) or any(
        not 0.0 < value < 1.0 for value in args.quality_targets
    ):
        raise SystemExit(
            "--quality-targets must be unique, increasing, and inside (0, 1)"
        )
    counts = (
        args.layer_samples, args.selector_repetitions, args.io_repetitions,
        args.gemm_repetitions,
    )
    if any(value < 1 for value in counts):
        raise SystemExit("sample and repetition counts must be positive")


def benchmark_selector(selector, method: str, scores: torch.Tensor,
                       row_budget: int, d: int, repetitions: int):
    internal = "paper" if method == "paper" else "tile8_ceil"
    samples, masks = [], []
    metadata = {}
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        mask, metadata = selector.select(internal, scores, row_budget, d)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        masks.append(mask.detach().cpu().numpy().astype(bool, copy=True))
    deterministic = all(np.array_equal(masks[0], item) for item in masks[1:])
    return masks[0], metadata, samples, deterministic


def make_projection_hook(
    state: dict, model_key: str, model_label: str, module_name: str, module,
    selector, native_reader, args: argparse.Namespace, candidate_rows: list[dict],
    selector_rows: list[dict], io_rows: list[dict], gemm_rows: list[dict],
    feature_rows: list[dict],
):
    layer_match = LAYER_RE.search(module_name)
    layer_index = int(layer_match.group(1)) if layer_match else -1
    projection = EXP32.projection_name(module_name)
    n, d = int(module.in_features), int(module.out_features)
    context = selector.context(n, d)

    def hook(_module, inputs, output):
        x = inputs[0].detach()
        reference = output.detach()
        scores = x.float().reshape(-1, n).abs().mean(dim=0)
        case_key = f"{model_key}|{state['prompt']}|{module_name}"
        required, predictor_samples = EXP34.benchmark_unstructured_requirements(
            scores, args.quality_targets, args.selector_repetitions
        )
        for target, required_rows in zip(args.quality_targets, required):
            feature_rows.append({
                "case_key": case_key, "model": model_key,
                "prompt": state["prompt"], "split": state["split"],
                "module": module_name, "shape": f"{n}x{d}",
                "target": target, "unstructured_required_rows": required_rows,
                "unstructured_required_fraction": required_rows / n,
                "predictor_median_ms": float(np.median(predictor_samples)),
                "predictor_p95_ms": float(np.quantile(predictor_samples, 0.95)),
            })

        measured_masks: dict[str, dict] = {}

        def get_measurement(mask_np: np.ndarray):
            digest = EXP34.mask_digest(mask_np)
            if digest not in measured_masks:
                measured = EXP34.measure_mask(
                    x, reference, module, mask_np, native_reader, args
                )
                measured_masks[digest] = measured
                io_rows.append({
                    "case_key": case_key, "mask_sha256": digest,
                    "read_wall_median_ms": measured["read_wall_median_ms"],
                    "read_wall_p95_ms": measured["read_wall_p95_ms"],
                    "io_median_ms": measured["io_median_ms"],
                    "upload_median_ms": measured["upload_median_ms"],
                    "direct_rate": measured["direct_rate"],
                })
                gemm_rows.append({
                    "case_key": case_key, "mask_sha256": digest,
                    "gather_median_ms": measured["gather_median_ms"],
                    "gather_p95_ms": measured["gather_p95_ms"],
                    "gemm_median_ms": measured["gemm_median_ms"],
                    "gemm_p95_ms": measured["gemm_p95_ms"],
                })
            return digest, measured_masks[digest]

        warm_budget = max(1, min(n - 1, int(round(n * args.budgets[0]))))
        selector.select("paper", scores, warm_budget, d)
        selector.select("tile8_ceil", scores, warm_budget, d)
        for budget_index, fraction in enumerate(args.budgets):
            row_budget = max(1, min(n - 1, int(round(n * fraction))))
            for method in METHODS:
                mask_np, metadata, samples, deterministic = benchmark_selector(
                    selector, method, scores, row_budget, d,
                    args.selector_repetitions,
                )
                digest, measured = get_measurement(mask_np)
                selected = int(mask_np.sum())
                selector_median = float(np.median(samples))
                candidate_rows.append({
                    "case_key": case_key, "model": model_key,
                    "model_label": model_label, "prompt": state["prompt"],
                    "split": state["split"], "situation": state["situation"],
                    "num_tokens": state["num_tokens"], "module": module_name,
                    "layer": layer_index, "projection": projection,
                    "shape": f"{n}x{d}", "n": n, "d": d,
                    "row_size_kib": 2.0 * d / 1024.0,
                    "paper_parameter_source": context["paper_parameter_source"],
                    "method": method, "method_label": METHOD_LABELS[method],
                    "budget_index": budget_index, "budget_fraction": fraction,
                    "row_budget": row_budget, "selected_rows": selected,
                    "selected_fraction": selected / n,
                    "row_match": selected == row_budget, "mask_sha256": digest,
                    "importance_retention": EXP34.retention(scores, mask_np),
                    "selector_median_ms": selector_median,
                    "selector_p95_ms": float(np.quantile(samples, 0.95)),
                    "actual_total_ms": selector_median + measured["nonselector_total_ms"],
                    "deterministic": deterministic,
                    "fallback_used": bool(metadata.get("fallback_used", False)),
                    "error": metadata.get("error", ""), **measured,
                })
                for repetition, elapsed in enumerate(samples):
                    selector_rows.append({
                        "case_key": case_key, "method": method,
                        "budget_index": budget_index, "repetition": repetition,
                        "runtime_ms": elapsed,
                    })
        print(
            f"  measured {model_key} {state['prompt']} {module_name}: "
            f"{2*len(args.budgets)} candidates, {len(measured_masks)} masks",
            flush=True,
        )

    return hook


def calibration_costs(candidates: pd.DataFrame) -> pd.DataFrame:
    calibration = candidates[
        (candidates.split == "calibration") & (candidates.method == "cell8")
    ]
    return calibration.groupby(
        ["model", "shape", "budget_index"], as_index=False
    ).agg(
        calibrated_nonselector_ms=("nonselector_total_ms", "mean"),
        calibrated_selector_ms=("selector_median_ms", "mean"),
    )


def calibrate_margins(candidates: pd.DataFrame, features: pd.DataFrame):
    cell = candidates[candidates.method == "cell8"]
    rows = []
    for feature in features[features.split == "calibration"].itertuples(index=False):
        case = cell[cell.case_key == feature.case_key].sort_values("budget_fraction")
        allowed = case[
            case.importance_retention + 1e-12 >= feature.target
        ]
        if len(allowed):
            structured = float(allowed.iloc[0].budget_fraction)
            feasible = True
        else:
            structured = float(case.iloc[-1].budget_fraction)
            feasible = False
        rows.append({
            "model": feature.model, "shape": feature.shape,
            "target": EXP34.target_key(feature.target),
            "case_key": feature.case_key,
            "required_margin": max(
                0.0, structured - feature.unstructured_required_fraction
            ),
            "grid_feasible": feasible,
        })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["model", "shape", "target"], as_index=False
    ).agg(
        predictor_margin=("required_margin", "max"),
        calibration_cases=("case_key", "size"),
        calibration_grid_feasible_rate=("grid_feasible", "mean"),
    )
    mapping = {
        (row.model, row.shape, EXP34.target_key(row.target)):
        float(row.predictor_margin)
        for row in summary.itertuples(index=False)
    }
    return summary, mapping


def build_adaptive_decisions(candidates, features, costs, targets, margins):
    cell = candidates[candidates.method == "cell8"]
    feature_lookup = features.set_index(["case_key", "target"]).to_dict("index")
    cost_lookup = costs.set_index(
        ["model", "shape", "budget_index"]
    ).to_dict("index")
    rows = []
    for case_key, case in cell.groupby("case_key", sort=False):
        first = case.iloc[0]
        for target in targets:
            feature = feature_lookup[(case_key, target)]
            margin = margins[(
                first.model, first["shape"], EXP34.target_key(target)
            )]
            predicted = min(
                1.0, feature["unstructured_required_fraction"] + margin
            )
            ordered = []
            for item in case.itertuples(index=False):
                if item.budget_fraction + 1e-12 < predicted:
                    continue
                cost = cost_lookup[(item.model, item.shape, item.budget_index)]
                ordered.append((cost["calibrated_nonselector_ms"], item))
            if not ordered:
                item = max(case.itertuples(index=False), key=lambda row: row.budget_fraction)
                cost = cost_lookup[(item.model, item.shape, item.budget_index)]
                ordered = [(cost["calibrated_nonselector_ms"], item)]
            ordered.sort(key=lambda pair: (pair[0], pair[1].budget_fraction))
            attempted, chosen = [], None
            for _, item in ordered:
                attempted.append(item)
                if item.importance_retention + 1e-12 >= target:
                    chosen = item
                    break
            feasible = chosen is not None
            if chosen is None:
                chosen = max(ordered, key=lambda pair: pair[1].importance_retention)[1]
            predictor_ms = float(feature["predictor_median_ms"])
            selector_ms = float(sum(item.selector_median_ms for item in attempted))
            rows.append({
                "case_key": case_key, "model": first.model,
                "prompt": first.prompt, "split": first.split,
                "module": first.module, "shape": first["shape"],
                "target": target, "policy": f"adaptive_t{target:g}",
                "selected_fraction": chosen.selected_fraction,
                "importance_retention": chosen.importance_retention,
                "relative_l2_error": chosen.relative_l2_error,
                "cosine_error": chosen.cosine_error,
                "predictor_ms": predictor_ms, "selector_mask_ms": selector_ms,
                "selector_total_ms": predictor_ms + selector_ms,
                "search_attempts": len(attempted), "feasible": feasible,
                "nonselector_total_ms": chosen.nonselector_total_ms,
                "actual_total_ms": (
                    predictor_ms + selector_ms + chosen.nonselector_total_ms
                ),
            })
    return pd.DataFrame(rows)


def calibrate_targets(decisions: pd.DataFrame, ceiling: float):
    calibration = decisions[decisions.split == "calibration"]
    summary = calibration.groupby(
        ["model", "shape", "target"], as_index=False
    ).agg(
        relative_l2_error=("relative_l2_error", "mean"),
        actual_total_ms=("actual_total_ms", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        feasible_rate=("feasible", "mean"),
    )
    rows, mapping = [], {}
    for key, group in summary.groupby(["model", "shape"], sort=True):
        allowed = group[
            (group.relative_l2_error <= ceiling) & (group.feasible_rate == 1.0)
        ]
        if len(allowed):
            chosen = allowed.sort_values(["actual_total_ms", "target"]).iloc[0]
            met = True
        else:
            chosen = group.sort_values(
                ["relative_l2_error", "actual_total_ms"]
            ).iloc[0]
            met = False
        mapping[key] = float(chosen.target)
        rows.append({
            "model": key[0], "shape": key[1],
            "selected_target": float(chosen.target),
            "calibration_relative_l2_error": float(chosen.relative_l2_error),
            "calibration_actual_total_ms": float(chosen.actual_total_ms),
            "calibration_selected_fraction": float(chosen.selected_fraction),
            "error_ceiling": ceiling, "ceiling_met": met,
        })
    return pd.DataFrame(rows), mapping


class EndToEndPolicy:
    def __init__(self, selector, selection_type, model_key, shape, d, budgets,
                 cost_order, target_map, margin_map, collector):
        self.selector = selector
        self.selection_type = selection_type
        self.model_key = model_key
        self.shape = shape
        self.d = d
        self.budgets = budgets
        self.cost_order = cost_order
        self.target_map = target_map
        self.margin_map = margin_map
        self.collector = collector
        self.mode = "paper"
        self.budget_index = 0
        self.target = 0.5

    def configure(self, mode, budget_index=0, target=0.5):
        self.mode = mode
        self.budget_index = int(budget_index)
        self.target = float(target)

    def __call__(self, importance, _num_load_rows, _row_size_kib):
        n = int(importance.numel())
        attempts, feasible = 1, True
        if self.mode in METHODS:
            fraction = self.budgets[self.budget_index]
            rows = max(1, min(n - 1, int(round(n * fraction))))
            internal = "paper" if self.mode == "paper" else "tile8_ceil"
            mask, metadata = self.selector.select(
                internal, importance, rows, self.d
            )
            target = float("nan")
        else:
            target = (
                self.target_map[(self.model_key, self.shape)]
                if self.mode == "adaptive_calibrated" else self.target
            )
            ordered = torch.sort(importance, descending=True).values
            cumulative = torch.cumsum(ordered, dim=0)
            required = int(torch.searchsorted(
                cumulative, target * cumulative[-1], right=False
            )) + 1
            predicted = min(
                1.0,
                required / n + self.margin_map[(
                    self.model_key, self.shape, EXP34.target_key(target)
                )],
            )
            eligible = [
                pair for pair in self.cost_order[(self.model_key, self.shape)]
                if pair[1] + 1e-12 >= predicted
            ]
            if not eligible:
                eligible = [max(
                    self.cost_order[(self.model_key, self.shape)],
                    key=lambda pair: pair[1],
                )]
            mask, metadata, feasible = None, {}, False
            attempts = 0
            best = None
            for budget_index, fraction in eligible:
                rows = max(1, min(n - 1, int(round(n * fraction))))
                candidate, candidate_meta = self.selector.select(
                    "tile8_ceil", importance, rows, self.d
                )
                attempts += 1
                retained = float(
                    importance[candidate].sum()
                    / importance.sum().clamp_min(1e-20)
                )
                if best is None or retained > best[0]:
                    best = (retained, candidate, candidate_meta)
                if retained + 1e-12 >= target:
                    mask, metadata, feasible = candidate, candidate_meta, True
                    break
            if mask is None:
                assert best is not None
                _, mask, metadata = best
        self.collector.append({
            "selected_fraction": float(mask.sum()) / n,
            "search_attempts": attempts, "feasible": feasible,
            "fallback": bool(metadata.get("fallback_used", False)),
        })
        return self.selection_type(mask, float(importance[mask].sum()), None)


def run_end_to_end(model, tokenizer, model_key, selector, costs, target_map,
                   margin_map, prompts, args, vlmflash):
    cost_order = {}
    for key, group in costs.groupby(["model", "shape"], sort=False):
        ordered = group.sort_values(["calibrated_nonselector_ms", "budget_index"])
        cost_order[key] = [
            (int(row.budget_index), args.budgets[int(row.budget_index)])
            for row in ordered.itertuples(index=False)
        ]
    dummy = lambda importance, num_load_rows, row_size_kib: vlmflash.Selection(
        torch.ones_like(importance, dtype=torch.bool), float(importance.sum()), None
    )
    handle = vlmflash.attach(
        model, policy=dummy, include=vlmflash.DEFAULT_INCLUDE, sparsity=0.5
    )
    collector, policies = [], []
    for name in handle.names:
        module = model.get_submodule(name)
        policy = EndToEndPolicy(
            selector, vlmflash.Selection, model_key,
            f"{module.in_features}x{module.out_features}", module.out_features,
            args.budgets, cost_order, target_map, margin_map, collector,
        )
        module.nc_policy = policy
        policies.append(policy)
    configurations = []
    for method in METHODS:
        for budget_index, fraction in enumerate(args.budgets):
            configurations.append((
                f"{method}_r{100*fraction:g}", method, budget_index, float("nan")
            ))
    for target in args.quality_targets:
        configurations.append((
            f"adaptive_t{target:g}", "adaptive", 0, target
        ))
    configurations.append((
        "adaptive_calibrated", "adaptive_calibrated", 0, 0.0
    ))
    rows = []
    try:
        for prompt in prompts:
            if prompt["split"] != "holdout":
                continue
            inputs = EXP32.tokenize(tokenizer, prompt["text"], args.max_input_tokens)
            with torch.inference_mode():
                dense = model(**inputs, use_cache=False).logits.detach()
            for policy_name, mode, budget_index, target in configurations:
                collector.clear()
                for policy in policies:
                    policy.configure(mode, budget_index, target)
                with torch.inference_mode(), vlmflash.enabled():
                    sparse = model(**inputs, use_cache=False).logits.detach()
                metrics = EXP32.logits_error(dense, sparse, inputs["input_ids"])
                rows.append({
                    "model": model_key, "prompt": prompt["name"],
                    "situation": prompt["situation"], "policy": policy_name,
                    "mode": mode, "budget_index": budget_index,
                    "budget_fraction": (
                        args.budgets[budget_index] if mode in METHODS else float("nan")
                    ),
                    "target": target, **metrics,
                    "projection_calls": len(collector),
                    "selected_fraction_mean": float(np.mean([
                        call["selected_fraction"] for call in collector
                    ])),
                    "search_attempts_mean": float(np.mean([
                        call["search_attempts"] for call in collector
                    ])),
                    "feasible_rate": float(np.mean([
                        call["feasible"] for call in collector
                    ])),
                    "fallback_calls": int(sum(call["fallback"] for call in collector)),
                })
            print(f"  e2e {model_key} {prompt['name']} complete", flush=True)
            del dense, inputs
    finally:
        handle.detach()
    return rows


def run_model(model_key, selector, native_reader, prompts, args, vlmflash):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    spec = MODEL_SPECS[model_key]
    snapshot = EXP32.cached_snapshot(spec["repo"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float16, local_files_only=True
    ).to("cuda").eval()
    sampled_layers = EXP32.layer_samples(
        int(model.config.num_hidden_layers), args.layer_samples
    )
    targets = []
    for name, module in model.named_modules():
        match = LAYER_RE.search(name)
        if (
            match and int(match.group(1)) in sampled_layers
            and re.search(vlmflash.DEFAULT_INCLUDE, name)
            and type(module) is torch.nn.Linear
        ):
            targets.append((name, module))
    if len(targets) != len(sampled_layers) * 7:
        raise RuntimeError(f"{model_key}: sampled projection count mismatch")
    candidate_rows, selector_rows, io_rows, gemm_rows, feature_rows = (
        [], [], [], [], []
    )
    state = {"prompt": "", "split": "", "situation": "", "num_tokens": 0}
    hooks = [
        module.register_forward_hook(make_projection_hook(
            state, model_key, spec["label"], name, module, selector,
            native_reader, args, candidate_rows, selector_rows, io_rows,
            gemm_rows, feature_rows,
        ))
        for name, module in targets
    ]
    try:
        with torch.inference_mode():
            for prompt in prompts:
                inputs = EXP32.tokenize(tokenizer, prompt["text"], args.max_input_tokens)
                state.update({
                    "prompt": prompt["name"], "split": prompt["split"],
                    "situation": prompt["situation"],
                    "num_tokens": int(inputs["input_ids"].numel()),
                })
                output = model(**inputs, use_cache=False)
                torch.cuda.synchronize()
                del output, inputs
                print(f"captured {model_key} {prompt['name']}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    metadata = {
        "model": model_key, "label": spec["label"], "repo": spec["repo"],
        "snapshot": snapshot.name, "model_type": model.config.model_type,
        "layers": int(model.config.num_hidden_layers),
        "sampled_layers": sampled_layers, "sampled_projection_count": len(targets),
        "shapes": sorted({f"{m.in_features}x{m.out_features}" for _, m in targets}),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "dtype": str(next(model.parameters()).dtype),
    }
    return (
        model, tokenizer, candidate_rows, selector_rows, io_rows, gemm_rows,
        feature_rows, metadata,
    )


def best_at_error(curve: pd.DataFrame, ceiling: float):
    allowed = curve[curve.relative_l2_error <= ceiling]
    if not len(allowed):
        return None
    return allowed.sort_values("actual_total_ms").iloc[0]


def interpolate_latency_at_error(curve: pd.DataFrame, error: float) -> float:
    """Piecewise-linear latency on a measured error/latency curve.

    This is a diagnostic for gaps between the 5%-spaced measurements, not a
    replacement for a measured point.
    """
    points = (
        curve[["relative_l2_error", "actual_total_ms"]]
        .dropna()
        .groupby("relative_l2_error", as_index=False)
        .actual_total_ms.min()
        .sort_values("relative_l2_error")
    )
    errors = points.relative_l2_error.to_numpy(dtype=float)
    latencies = points.actual_total_ms.to_numpy(dtype=float)
    if not len(errors) or error < errors[0] or error > errors[-1]:
        return float("nan")
    return float(np.interp(error, errors, latencies))


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    features = pd.read_csv(output / "score_features.csv")
    costs = calibration_costs(candidates)
    costs.to_csv(output / "calibration_costs.csv", index=False)
    margin_frame, margin_map = calibrate_margins(candidates, features)
    margin_frame.to_csv(output / "predictor_margins.csv", index=False)
    adaptive = build_adaptive_decisions(
        candidates, features, costs, args.quality_targets, margin_map
    )
    adaptive.to_csv(output / "adaptive_decisions.csv", index=False)
    calibration, target_map = calibrate_targets(
        adaptive, args.calibration_error_ceiling
    )
    calibration.to_csv(output / "calibrated_targets.csv", index=False)

    holdout = candidates[candidates.split == "holdout"]
    fixed = holdout.groupby(
        ["method", "budget_index", "budget_fraction"], as_index=False
    ).agg(
        cases=("case_key", "size"),
        actual_total_ms=("actual_total_ms", "mean"),
        selector_ms=("selector_median_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        importance_retention=("importance_retention", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        row_match_rate=("row_match", "mean"),
    )
    fixed.to_csv(output / "fixed_frontiers.csv", index=False)
    holdout_adaptive = adaptive[adaptive.split == "holdout"]
    adaptive_sweep = holdout_adaptive.groupby("target", as_index=False).agg(
        cases=("case_key", "size"),
        actual_total_ms=("actual_total_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        importance_retention=("importance_retention", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        selector_total_ms=("selector_total_ms", "mean"),
        search_attempts=("search_attempts", "mean"),
        feasible_rate=("feasible", "mean"),
    )
    adaptive_sweep.to_csv(output / "adaptive_frontier.csv", index=False)
    calibrated_rows = []
    for row in holdout_adaptive.itertuples(index=False):
        target = target_map[(row.model, row.shape)]
        if math.isclose(row.target, target, abs_tol=1e-12, rel_tol=0.0):
            calibrated_rows.append(row._asdict())
    calibrated_holdout = pd.DataFrame(calibrated_rows)
    calibrated_summary = calibrated_holdout.agg({
        "actual_total_ms": "mean", "relative_l2_error": "mean",
        "selected_fraction": "mean", "feasible": "mean",
    }).to_frame().T
    calibrated_summary.to_csv(output / "adaptive_calibrated_summary.csv", index=False)

    error_levels = [0.50, 0.45, 0.42, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
    comparison_rows = []
    curves = {
        "Paper fixed-R envelope": fixed[fixed.method == "paper"],
        "Cell-8 fixed-R envelope": fixed[fixed.method == "cell8"],
        "Adaptive Cell-8 envelope": adaptive_sweep,
    }
    for ceiling in error_levels:
        for label, curve in curves.items():
            best = best_at_error(curve, ceiling)
            comparison_rows.append({
                "error_ceiling": ceiling, "strategy": label,
                "feasible": best is not None,
                "actual_total_ms": (
                    float(best.actual_total_ms) if best is not None else float("nan")
                ),
                "achieved_error": (
                    float(best.relative_l2_error) if best is not None else float("nan")
                ),
                "selected_fraction": (
                    float(best.selected_fraction) if best is not None else float("nan")
                ),
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "quality_constrained_comparison.csv", index=False)

    # Test whether any deployable adaptive point escapes the fixed Cell-8
    # envelope at equal or better measured projection error.
    cell_curve = fixed[fixed.method == "cell8"]
    strategy_rows = []
    for row in adaptive_sweep.itertuples(index=False):
        fixed_best = best_at_error(cell_curve, row.relative_l2_error)
        interpolated_ms = interpolate_latency_at_error(
            cell_curve, row.relative_l2_error
        )
        strategy_rows.append({
            "target": row.target,
            "adaptive_total_ms": row.actual_total_ms,
            "adaptive_error": row.relative_l2_error,
            "fixed_cell_total_ms": (
                float(fixed_best.actual_total_ms) if fixed_best is not None else float("nan")
            ),
            "fixed_cell_error": (
                float(fixed_best.relative_l2_error) if fixed_best is not None else float("nan")
            ),
            "adaptive_frontier_win": bool(
                fixed_best is not None
                and row.actual_total_ms < fixed_best.actual_total_ms
            ),
            "interpolated_cell_total_ms": interpolated_ms,
            "adaptive_minus_interpolated_ms": (
                row.actual_total_ms - interpolated_ms
            ),
            "adaptive_interpolated_win": bool(
                not math.isnan(interpolated_ms)
                and row.actual_total_ms < interpolated_ms
            ),
        })
    strategy_test = pd.DataFrame(strategy_rows)
    strategy_test.to_csv(output / "adaptive_vs_fixed_cell.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    for method in METHODS:
        curve = fixed[fixed.method == method].sort_values("budget_fraction")
        ax.plot(
            curve.actual_total_ms, curve.relative_l2_error,
            marker="s" if method == "paper" else "o", linewidth=2.0,
            markersize=5, color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} fixed-R sweep",
        )
        for row in curve.itertuples(index=False):
            pct = int(round(100 * row.budget_fraction))
            if pct % 10 == 0 or pct == 95:
                ax.annotate(
                    f"R={pct}%", (row.actual_total_ms, row.relative_l2_error),
                    xytext=(4, 4), textcoords="offset points", fontsize=7,
                    color=METHOD_COLORS[method],
                )
    curve = adaptive_sweep.sort_values("target")
    ax.plot(
        curve.actual_total_ms, curve.relative_l2_error,
        marker="D", linestyle="--", linewidth=1.8, markersize=5,
        color="#2563EB", label="Adaptive Cell-8 (quality threshold)",
    )
    for row in curve.itertuples(index=False):
        pct = 100 * row.target
        if pct in (50, 60, 70, 80, 90, 95):
            ax.annotate(
                f"τ={pct:g}%", (row.actual_total_ms, row.relative_l2_error),
                xytext=(4, -11), textcoords="offset points", fontsize=7,
                color="#2563EB",
            )
    calibrated_point = calibrated_summary.iloc[0]
    ax.scatter(
        calibrated_point.actual_total_ms,
        calibrated_point.relative_l2_error,
        marker="*", s=190, color="#DC2626", edgecolor="white", linewidth=0.7,
        zorder=5, label="Adaptive calibration choice",
    )
    ax.set_xlabel("Measured projection-path latency (ms)")
    ax.set_ylabel("Projection relative L2 error (lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "quality_latency_frontier.png", dpi=200)
    fig.savefig(output / "quality_latency_frontier.pdf")
    plt.close(fig)

    e2e_path = output / "end_to_end.csv"
    e2e_strategy_test = pd.DataFrame()
    if e2e_path.is_file():
        e2e = pd.read_csv(e2e_path)
        e2e_summary = e2e.groupby(
            ["policy", "mode", "budget_index", "budget_fraction", "target"],
            dropna=False, as_index=False,
        ).agg(
            cases=("prompt", "size"),
            dense_to_sparse_kl=("dense_to_sparse_kl", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            nll_delta=("nll_delta", "mean"),
            logit_relative_l2_error=("logit_relative_l2_error", "mean"),
            selected_fraction=("selected_fraction_mean", "mean"),
            feasible_rate=("feasible_rate", "mean"),
        )
        latency_lookup = {
            (row.method, int(row.budget_index)): row.actual_total_ms
            for row in fixed.itertuples(index=False)
        }
        adaptive_lookup = {
            EXP34.target_key(row.target): row.actual_total_ms
            for row in adaptive_sweep.itertuples(index=False)
        }
        projection_latency = []
        for row in e2e_summary.itertuples(index=False):
            if row.mode in METHODS:
                value = latency_lookup[(row.mode, int(row.budget_index))]
            elif row.mode == "adaptive":
                value = adaptive_lookup[EXP34.target_key(row.target)]
            else:
                value = float(calibrated_point.actual_total_ms)
            projection_latency.append(value)
        e2e_summary["projection_path_ms"] = projection_latency
        e2e_summary.to_csv(output / "end_to_end_summary.csv", index=False)

        fixed_cell_e2e = e2e_summary[e2e_summary["mode"] == "cell8"]
        adaptive_e2e = e2e_summary[
            e2e_summary["mode"].isin(["adaptive", "adaptive_calibrated"])
        ]
        e2e_strategy_rows = []
        for row in adaptive_e2e.itertuples(index=False):
            allowed = fixed_cell_e2e[
                fixed_cell_e2e.dense_to_sparse_kl <= row.dense_to_sparse_kl
            ].sort_values("projection_path_ms")
            fixed_best = allowed.iloc[0] if len(allowed) else None
            e2e_strategy_rows.append({
                "policy": row.policy,
                "mode": row.mode,
                "target": row.target,
                "adaptive_projection_path_ms": row.projection_path_ms,
                "adaptive_dense_to_sparse_kl": row.dense_to_sparse_kl,
                "fixed_cell_projection_path_ms": (
                    float(fixed_best.projection_path_ms)
                    if fixed_best is not None else float("nan")
                ),
                "fixed_cell_dense_to_sparse_kl": (
                    float(fixed_best.dense_to_sparse_kl)
                    if fixed_best is not None else float("nan")
                ),
                "adaptive_e2e_frontier_win": bool(
                    fixed_best is not None
                    and row.projection_path_ms < fixed_best.projection_path_ms
                ),
            })
        e2e_strategy_test = pd.DataFrame(e2e_strategy_rows)
        e2e_strategy_test.to_csv(
            output / "adaptive_vs_fixed_cell_e2e.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(10.5, 6.6))
        for method in METHODS:
            group = e2e_summary[e2e_summary["mode"] == method].sort_values(
                "budget_fraction"
            )
            ax.plot(
                group.projection_path_ms, group.dense_to_sparse_kl,
                marker="s" if method == "paper" else "o", linewidth=2,
                markersize=5, color=METHOD_COLORS[method],
                label=f"{METHOD_LABELS[method]} fixed-R sweep",
            )
        group = e2e_summary[
            e2e_summary["mode"] == "adaptive"
        ].sort_values("target")
        ax.plot(
            group.projection_path_ms, group.dense_to_sparse_kl,
            marker="D", linestyle="--", color="#2563EB",
            label="Adaptive Cell-8",
        )
        ax.set_xlabel("Measured mean projection-path latency (ms)")
        ax.set_ylabel("Dense→sparse logit KL (lower is better)")
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(output / "end_to_end_frontier.png", dpi=200)
        fig.savefig(output / "end_to_end_frontier.pdf")
        plt.close(fig)
    else:
        e2e_summary = pd.DataFrame()

    metadata = json.loads((output / "metadata.json").read_text())
    summary = {
        "format": "experiment-35-full-frontier-v1",
        "candidate_cases": int(len(candidates)),
        "adaptive_decisions": int(len(adaptive)),
        "all_direct": bool((candidates.direct_rate == 1.0).all()),
        "all_deterministic": bool(candidates.deterministic.all()),
        "fallback_cases": int(candidates.fallback_used.sum()),
        "paper_row_match_rate": float(
            candidates[candidates.method == "paper"].row_match.mean()
        ),
        "cell_row_match_rate": float(
            candidates[candidates.method == "cell8"].row_match.mean()
        ),
        "adaptive_frontier_wins": int(strategy_test.adaptive_frontier_win.sum()),
        "adaptive_interpolated_wins": int(
            strategy_test.adaptive_interpolated_win.sum()
        ),
        "adaptive_frontier_points": int(len(strategy_test)),
        "adaptive_e2e_frontier_wins": int(
            e2e_strategy_test[
                e2e_strategy_test["mode"] == "adaptive"
            ].adaptive_e2e_frontier_win.sum()
        ) if len(e2e_strategy_test) else 0,
        "e2e_cases": int(pd.read_csv(e2e_path).shape[0]) if e2e_path.is_file() else 0,
        "gpu": metadata["gpu"], "flash": metadata["flash"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        args, fixed, adaptive_sweep, calibrated_summary, calibration,
        comparison, strategy_test, e2e_summary, e2e_strategy_test, summary,
    )


def write_report(args, fixed, adaptive, calibrated, calibration, comparison,
                 strategy_test, e2e, e2e_strategy_test, summary):
    path = args.report_output or HERE / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    calibrated_row = calibrated.iloc[0]
    lines = [
        "# Experiment 35 보고서: R을 제거한 full frontier", "",
        "Paper와 Cell-8을 모두 `R=10%..95%`에서 측정하고, R은 curve를 만드는 "
        "숨은 매개변수로만 사용했다. 최종 비교축은 실제 latency와 실제 error다. "
        "Adaptive 정책은 calibration prompt만으로 threshold와 structured margin을 "
        "고정한 뒤 holdout에서 평가했다.", "",
        "## Quality-constrained value", "",
        "각 error ceiling에서 이를 만족하는 최소 latency를 고른 결과다. 따라서 아래 "
        "표에는 특정 R을 먼저 고정하는 가정이 없다.", "",
        "| error ceiling | Paper | Cell-8 | Adaptive |",
        "|---:|---:|---:|---:|",
    ]
    pivot = comparison.pivot(
        index="error_ceiling", columns="strategy", values="actual_total_ms"
    )
    for ceiling, row in pivot.sort_index(ascending=False).iterrows():
        def value(name):
            item = row.get(name, float("nan"))
            return "—" if pd.isna(item) else f"{item:.3f} ms"
        lines.append(
            f"| {ceiling:.2f} | {value('Paper fixed-R envelope')} "
            f"| {value('Cell-8 fixed-R envelope')} "
            f"| {value('Adaptive Cell-8 envelope')} |"
        )
    lines.extend([
        "", "## Fixed-R 측정 곡선", "",
        "| R | Paper time | Paper error | Cell-8 time | Cell-8 error |",
        "|---:|---:|---:|---:|---:|",
    ])
    paper = fixed[fixed.method == "paper"].set_index("budget_index")
    cell = fixed[fixed.method == "cell8"].set_index("budget_index")
    for index in paper.index:
        p, c = paper.loc[index], cell.loc[index]
        lines.append(
            f"| {100*p.budget_fraction:.0f}% | {p.actual_total_ms:.3f} ms "
            f"| {p.relative_l2_error:.4f} | {c.actual_total_ms:.3f} ms "
            f"| {c.relative_l2_error:.4f} |"
        )
    lines.extend([
        "", "## Adaptive sweep", "",
        "| quality threshold | time | error | selected R | attempts | feasible |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in adaptive.itertuples(index=False):
        lines.append(
            f"| τ={100*row.target:g}% | {row.actual_total_ms:.3f} ms "
            f"| {row.relative_l2_error:.4f} | {100*row.selected_fraction:.1f}% "
            f"| {row.search_attempts:.2f} | {100*row.feasible_rate:.1f}% |"
        )
    lines.extend([
        "", "## Calibration 선택", "",
        f"Shape별 calibration target을 적용한 holdout 평균은 "
        f"`{calibrated_row.actual_total_ms:.3f} ms`, error "
        f"`{calibrated_row.relative_l2_error:.4f}`, 선택 R "
        f"`{100*calibrated_row.selected_fraction:.1f}%`다.", "",
        "| 모델 | shape | target | calibration error | R | ceiling 충족 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in calibration.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.shape} | {100*row.selected_target:g}% "
            f"| {row.calibration_relative_l2_error:.4f} "
            f"| {100*row.calibration_selected_fraction:.1f}% "
            f"| {'yes' if row.ceiling_met else 'no'} |"
        )
    wins = int(strategy_test.adaptive_frontier_win.sum())
    interpolated_wins = int(strategy_test.adaptive_interpolated_win.sum())
    lines.extend([
        "", "## 판정", "",
        f"Adaptive threshold 점 `{len(strategy_test)}`개 중 같은 수준 이하의 error를 "
        f"내는 5%-간격 fixed Cell-8 실측점보다 빨랐던 점은 **`{wins}`개**다. "
        f"측정점 사이를 선형 보간한 진단에서는 **`{interpolated_wins}`개**다. "
        "보간은 실측값이 아니지만, coarse grid의 빈틈을 전략의 승리로 오인하는지 "
        "검사한다.", "",
    ])
    measured_winners = strategy_test[strategy_test.adaptive_frontier_win]
    if len(measured_winners):
        row = measured_winners.iloc[0]
        lines.extend([
            f"유일한 실측-grid 승리는 `τ={100*row.target:g}%`에서 "
            f"`{row.fixed_cell_total_ms-row.adaptive_total_ms:.3f} ms`였지만, 같은 error의 "
            f"보간 Cell-8은 `{row.interpolated_cell_total_ms:.3f} ms`로 adaptive "
            f"`{row.adaptive_total_ms:.3f} ms`보다 빠르다. 따라서 현재 증거는 adaptive "
            "R 전략이 fixed Cell-8 frontier 자체를 개선한다는 주장을 지지하지 않는다. "
            "Paper 대비 개선과 Cell-8 대비 전략 개선은 구분해야 한다.", "",
        ])
    if len(e2e):
        threshold_e2e = e2e_strategy_test[e2e_strategy_test["mode"] == "adaptive"]
        calibrated_e2e = e2e_strategy_test[
            e2e_strategy_test["mode"] == "adaptive_calibrated"
        ].iloc[0]
        lines.extend([
            f"End-to-end sparse forward는 `{summary['e2e_cases']}`개를 실행했다. "
            f"실제 dense→sparse logit KL 기준으로 adaptive threshold가 fixed Cell-8 "
            f"실측 frontier를 이긴 점은 `{int(threshold_e2e.adaptive_e2e_frontier_win.sum())}`/"
            f"`{len(threshold_e2e)}`개다. Calibration 선택도 "
            f"`{calibrated_e2e.adaptive_projection_path_ms:.3f} ms`, KL "
            f"`{calibrated_e2e.adaptive_dense_to_sparse_kl:.3f}`이고, 이를 지배하는 "
            f"fixed Cell-8 실측점은 `{calibrated_e2e.fixed_cell_projection_path_ms:.3f} ms`, "
            f"KL `{calibrated_e2e.fixed_cell_dense_to_sparse_kl:.3f}`이다. 상세 결과는 "
            "`end_to_end_summary.csv`, `adaptive_vs_fixed_cell_e2e.csv`, "
            "`end_to_end_frontier.pdf`에 있다.", "",
        ])
    lines.extend([
        "## 측정 범위", "",
        f"- 후보 case: `{summary['candidate_cases']}`, adaptive decision: "
        f"`{summary['adaptive_decisions']}`.",
        f"- 모든 native read는 O_DIRECT였고 fallback은 "
        f"`{summary['fallback_cases']}`건이다.",
        "- latency는 projection 하나의 selector + O_DIRECT/upload + activation gather + "
        "compact GEMM이며 전체 LLM wall-clock은 아니다.",
        "- Error 축은 시각적 quality 방향을 맞추기 위해 뒤집었다. 따라서 작은 error가 "
        "그래프 위쪽에 있다.",
        "- 평균 curve는 per-case tail guarantee를 증명하지 않는다.", "",
        "![Full quality-latency frontier](results_laptop/quality_latency_frontier.png)", "",
        "![End-to-end frontier](results_laptop/end_to_end_frontier.png)", "",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return
    prompts = list(PROMPTS[:args.prompt_limit] if args.prompt_limit else PROMPTS)
    lookup = EXP32.BASE.LatencyTable.load(args.profile)
    selector = EXP32.Selector(lookup, args.saturation_kib)
    vlmflash = EXP32.EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")
    from transformers import AutoConfig
    max_required = 0
    for key in args.models:
        config = AutoConfig.from_pretrained(
            EXP32.cached_snapshot(MODEL_SPECS[key]["repo"]), local_files_only=True
        )
        max_required = max(
            max_required, int(config.intermediate_size) * int(config.hidden_size) * 2
        )
    if args.io_blob.stat().st_size < max_required:
        raise SystemExit(f"I/O blob needs at least {max_required} bytes")

    all_candidates, all_selectors, all_io, all_gemm = [], [], [], []
    all_features, all_e2e, models = [], [], []
    for model_key in args.models:
        print(f"loading and measuring {MODEL_SPECS[model_key]['label']}", flush=True)
        result = run_model(
            model_key, selector, native_reader, prompts, args, vlmflash
        )
        (
            model, tokenizer, candidates, selectors, io_rows, gemm_rows,
            features, metadata,
        ) = result
        all_candidates.extend(candidates)
        all_selectors.extend(selectors)
        all_io.extend(io_rows)
        all_gemm.extend(gemm_rows)
        all_features.extend(features)
        models.append(metadata)
        pd.DataFrame(all_candidates).to_csv(
            args.output_dir / "candidates.partial.csv", index=False
        )
        if not args.skip_end_to_end and any(p["split"] == "holdout" for p in prompts):
            model_candidates = pd.DataFrame(candidates)
            model_features = pd.DataFrame(features)
            model_costs = calibration_costs(model_candidates)
            _, margin_map = calibrate_margins(model_candidates, model_features)
            decisions = build_adaptive_decisions(
                model_candidates, model_features, model_costs,
                args.quality_targets, margin_map,
            )
            _, target_map = calibrate_targets(
                decisions, args.calibration_error_ceiling
            )
            all_e2e.extend(run_end_to_end(
                model, tokenizer, model_key, selector, model_costs,
                target_map, margin_map, prompts, args, vlmflash,
            ))
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(all_candidates).to_csv(args.output_dir / "candidates.csv", index=False)
    pd.DataFrame(all_selectors).to_csv(
        args.output_dir / "selector_samples.csv", index=False
    )
    pd.DataFrame(all_io).to_csv(args.output_dir / "io_aggregates.csv", index=False)
    pd.DataFrame(all_gemm).to_csv(args.output_dir / "gemm_aggregates.csv", index=False)
    pd.DataFrame(all_features).to_csv(
        args.output_dir / "score_features.csv", index=False
    )
    if all_e2e:
        pd.DataFrame(all_e2e).to_csv(args.output_dir / "end_to_end.csv", index=False)
    partial = args.output_dir / "candidates.partial.csv"
    if partial.exists():
        partial.unlink()
    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-35-full-frontier-v1",
        "models": models,
        "prompts": [
            {k: v for k, v in prompt.items() if k != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "budgets": args.budgets, "quality_targets": args.quality_targets,
        "calibration_error_ceiling": args.calibration_error_ceiling,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions, "io_warmup": args.io_warmup,
        "gemm_repetitions": args.gemm_repetitions,
        "gemm_warmup": args.gemm_warmup,
        "io_threads": args.io_threads, "io_max_read_kib": args.io_max_read_kib,
        "profile": str(args.profile), "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "flash": profile_meta.get("flash", "unknown"),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    analyze(args)


if __name__ == "__main__":
    main()
