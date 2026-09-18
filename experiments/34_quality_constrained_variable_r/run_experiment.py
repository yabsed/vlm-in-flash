#!/usr/bin/env python3
"""Experiment 34: quality-constrained variable-R selection on this laptop."""

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
EXP32_PATH = (
    PROJECT_ROOT / "experiments" / "32_multimodel_measured_error"
    / "run_experiment.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP32 = _load_module("experiment_32_for_34", EXP32_PATH)
EXP31 = EXP32.EXP31
MODEL_SPECS = EXP32.MODEL_SPECS
PROMPTS = EXP32.PROMPTS
LAYER_RE = EXP32.LAYER_RE
LOCAL_PROFILE = EXP32.LOCAL_PROFILE

FAMILIES = ("raw", "bound", "diag")
FAMILY_LABELS = {
    "raw": "Activation L1",
    "bound": "Weight-aware bound",
    "diag": "Weight-aware diagonal",
}
COLORS = {"raw": "#2563EB", "bound": "#0F766E", "diag": "#7C3AED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument(
        "--budgets", type=float, nargs="+",
        default=[x / 100.0 for x in range(10, 100, 5)],
        help="Dense candidate retained-row ratios.",
    )
    parser.add_argument(
        "--quality-targets", type=float, nargs="+",
        default=[
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90, 0.925, 0.95, 0.975,
        ],
    )
    parser.add_argument("--baseline-budget", type=float, default=0.50)
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
        raise SystemExit("Experiment 34 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if not args.budgets or any(not 0.0 < x < 1.0 for x in args.budgets):
        raise SystemExit("--budgets must lie strictly inside (0, 1)")
    if args.budgets != sorted(set(args.budgets)):
        raise SystemExit("--budgets must be unique and increasing")
    if not args.quality_targets or any(not 0.0 < x < 1.0 for x in args.quality_targets):
        raise SystemExit("--quality-targets must lie strictly inside (0, 1)")
    if args.quality_targets != sorted(set(args.quality_targets)):
        raise SystemExit("--quality-targets must be unique and increasing")
    if not 0.0 < args.baseline_budget < 1.0:
        raise SystemExit("--baseline-budget must lie strictly inside (0, 1)")
    if args.baseline_budget not in args.budgets:
        raise SystemExit("--baseline-budget must be included in --budgets")
    positive = (
        args.layer_samples, args.selector_repetitions, args.io_repetitions,
        args.gemm_repetitions,
    )
    if any(x < 1 for x in positive) or args.io_warmup < 0 or args.gemm_warmup < 0:
        raise SystemExit("sample and repetition counts are invalid")


def score_vectors(x: torch.Tensor, weight_norm: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = x.detach().float().reshape(-1, x.shape[-1])
    raw = flat.abs().mean(dim=0)
    activation_l2 = torch.linalg.vector_norm(flat, dim=0)
    bound = activation_l2 * weight_norm
    diag = bound.square()
    return {"raw": raw, "bound": bound, "diag": diag}


def retention(scores: torch.Tensor, mask_np: np.ndarray) -> float:
    mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(scores.device)
    return float(scores[mask].sum() / scores.sum().clamp_min(1e-20))


def benchmark_unstructured_requirements(
    scores: torch.Tensor, targets: list[float], repetitions: int,
) -> tuple[list[int], list[float]]:
    """Time a single score sort and derive the unstructured lower bound on R."""
    rows = []
    samples = []
    target_tensor = torch.tensor(targets, device=scores.device, dtype=scores.dtype)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        ordered = torch.sort(scores, descending=True).values
        cumulative = torch.cumsum(ordered, dim=0)
        thresholds = target_tensor * cumulative[-1]
        indices = torch.searchsorted(cumulative, thresholds, right=False)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        rows = [min(int(scores.numel()), int(index) + 1) for index in indices]
    return rows, samples


def benchmark_tile_candidate(
    selector, scores: torch.Tensor, row_budget: int, d: int, repetitions: int,
):
    samples = []
    masks = []
    metadata = {}
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        mask, metadata = selector.select("tile8_ceil", scores, row_budget, d)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        masks.append(mask.detach().cpu().numpy().astype(bool, copy=True))
    deterministic = all(np.array_equal(masks[0], other) for other in masks[1:])
    return masks[0], metadata, samples, deterministic


def benchmark_compact_projection(
    x: torch.Tensor, module, mask_np: np.ndarray, warmup: int, repetitions: int,
) -> dict[str, float]:
    mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(x.device)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    weight = module.weight.index_select(1, indices).contiguous()
    gather_values = []
    gemm_values = []
    output = None
    for repetition in range(warmup + repetitions):
        gather_start = torch.cuda.Event(enable_timing=True)
        gather_end = torch.cuda.Event(enable_timing=True)
        gemm_start = torch.cuda.Event(enable_timing=True)
        gemm_end = torch.cuda.Event(enable_timing=True)
        gather_start.record()
        x_selected = x.index_select(-1, indices)
        gather_end.record()
        gemm_start.record()
        output = F.linear(x_selected, weight, module.bias)
        gemm_end.record()
        torch.cuda.synchronize()
        if repetition >= warmup:
            gather_values.append(float(gather_start.elapsed_time(gather_end)))
            gemm_values.append(float(gemm_start.elapsed_time(gemm_end)))
    assert output is not None
    metrics = EXP32.output_error(module(x).detach(), output.detach())
    del output, x_selected, weight, indices, mask
    return {
        **metrics,
        "gather_median_ms": float(np.median(gather_values)),
        "gather_p95_ms": float(np.quantile(gather_values, 0.95)),
        "gemm_median_ms": float(np.median(gemm_values)),
        "gemm_p95_ms": float(np.quantile(gemm_values, 0.95)),
    }


def measure_mask(
    x: torch.Tensor, reference: torch.Tensor, module, mask_np: np.ndarray,
    native_reader, args: argparse.Namespace,
) -> dict[str, float]:
    mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(x.device)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    weight = module.weight.index_select(1, indices).contiguous()
    with torch.inference_mode():
        sparse = F.linear(x.index_select(-1, indices), weight, module.bias)
    errors = EXP32.output_error(reference, sparse)
    del sparse, weight, indices, mask

    # Warm and time x compaction and the compact GEMM separately.  Weight
    # compaction is intentionally excluded: the native reader returns compact
    # selected rows, so charging index_select(weight) would double count it.
    mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(x.device)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    compact_weight = module.weight.index_select(1, indices).contiguous()
    gather_values, gemm_values = [], []
    for repetition in range(args.gemm_warmup + args.gemm_repetitions):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e2, e3 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        x_selected = x.index_select(-1, indices)
        e1.record()
        e2.record()
        y = F.linear(x_selected, compact_weight, module.bias)
        e3.record()
        torch.cuda.synchronize()
        if repetition >= args.gemm_warmup:
            gather_values.append(float(e0.elapsed_time(e1)))
            gemm_values.append(float(e2.elapsed_time(e3)))
        del x_selected, y
    io_samples = EXP32.measure_io(
        native_reader, args.io_blob, mask_np, module.out_features * 2, args
    )
    starts, _ = EXP31._run_bounds(mask_np)
    del compact_weight, indices, mask
    read_values = [sample[2] for sample in io_samples]
    result = {
        **errors,
        "chunks": int(len(starts)),
        "gather_median_ms": float(np.median(gather_values)),
        "gather_p95_ms": float(np.quantile(gather_values, 0.95)),
        "gemm_median_ms": float(np.median(gemm_values)),
        "gemm_p95_ms": float(np.quantile(gemm_values, 0.95)),
        "io_median_ms": float(np.median([sample[0] for sample in io_samples])),
        "upload_median_ms": float(np.median([sample[1] for sample in io_samples])),
        "read_wall_median_ms": float(np.median(read_values)),
        "read_wall_p95_ms": float(np.quantile(read_values, 0.95)),
        "direct_rate": float(np.mean([sample[3] for sample in io_samples])),
    }
    result["nonselector_total_ms"] = (
        result["read_wall_median_ms"] + result["gather_median_ms"]
        + result["gemm_median_ms"]
    )
    return result


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


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
    weight_norm = torch.linalg.vector_norm(module.weight.detach().float(), dim=0)
    context = selector.context(n, d)

    def hook(_module, inputs, output):
        x = inputs[0].detach()
        reference = output.detach()
        scores = score_vectors(x, weight_norm)
        case_key = f"{model_key}|{state['prompt']}|{module_name}"
        measured_masks: dict[str, dict] = {}

        for family in FAMILIES:
            required, predictor_samples = benchmark_unstructured_requirements(
                scores[family], args.quality_targets, args.selector_repetitions
            )
            for target, required_rows in zip(args.quality_targets, required):
                feature_rows.append({
                    "case_key": case_key, "model": model_key,
                    "prompt": state["prompt"], "split": state["split"],
                    "module": module_name, "shape": f"{n}x{d}",
                    "family": family, "target": target,
                    "unstructured_required_rows": required_rows,
                    "unstructured_required_fraction": required_rows / n,
                    "predictor_median_ms": float(np.median(predictor_samples)),
                    "predictor_p95_ms": float(np.quantile(predictor_samples, 0.95)),
                })

        def get_measurement(mask_np: np.ndarray) -> tuple[str, dict]:
            digest = mask_digest(mask_np)
            if digest not in measured_masks:
                measured_masks[digest] = measure_mask(
                    x, reference, module, mask_np, native_reader, args
                )
                metrics = measured_masks[digest]
                # Native/GEMM raw repetitions are reduced inside measure_mask;
                # retain exactly one explicit aggregate record per unique mask.
                io_rows.append({
                    "case_key": case_key, "mask_sha256": digest,
                    "read_wall_median_ms": metrics["read_wall_median_ms"],
                    "read_wall_p95_ms": metrics["read_wall_p95_ms"],
                    "io_median_ms": metrics["io_median_ms"],
                    "upload_median_ms": metrics["upload_median_ms"],
                    "direct_rate": metrics["direct_rate"],
                })
                gemm_rows.append({
                    "case_key": case_key, "mask_sha256": digest,
                    "gather_median_ms": metrics["gather_median_ms"],
                    "gather_p95_ms": metrics["gather_p95_ms"],
                    "gemm_median_ms": metrics["gemm_median_ms"],
                    "gemm_p95_ms": metrics["gemm_p95_ms"],
                })
            return digest, measured_masks[digest]

        # Compile/warm the tile path outside recorded candidate timings.
        warm_budget = max(1, min(n - 1, int(round(n * args.budgets[0]))))
        for family in FAMILIES:
            selector.select("tile8_ceil", scores[family], warm_budget, d)

        for family in FAMILIES:
            for budget_index, fraction in enumerate(args.budgets):
                row_budget = max(1, min(n - 1, int(round(n * fraction))))
                mask_np, metadata, samples, deterministic = benchmark_tile_candidate(
                    selector, scores[family], row_budget, d,
                    args.selector_repetitions,
                )
                digest, measured = get_measurement(mask_np)
                selected = int(mask_np.sum())
                selector_median = float(np.median(samples))
                row = {
                    "case_key": case_key, "model": model_key,
                    "model_label": model_label, "prompt": state["prompt"],
                    "split": state["split"], "situation": state["situation"],
                    "num_tokens": state["num_tokens"], "module": module_name,
                    "layer": layer_index, "projection": projection,
                    "shape": f"{n}x{d}", "n": n, "d": d,
                    "row_size_kib": 2.0 * d / 1024.0,
                    "paper_parameter_source": context["paper_parameter_source"],
                    "family": family, "family_label": FAMILY_LABELS[family],
                    "budget_index": budget_index, "budget_fraction": fraction,
                    "row_budget": row_budget, "selected_rows": selected,
                    "row_match": selected == row_budget, "mask_sha256": digest,
                    "raw_retention": retention(scores["raw"], mask_np),
                    "bound_retention": retention(scores["bound"], mask_np),
                    "diag_retention": retention(scores["diag"], mask_np),
                    "selector_median_ms": selector_median,
                    "selector_p95_ms": float(np.quantile(samples, 0.95)),
                    "single_total_ms": selector_median + measured["nonselector_total_ms"],
                    "deterministic": deterministic,
                    "fallback_used": bool(metadata.get("fallback_used", False)),
                    "error": metadata.get("error", ""),
                    **measured,
                }
                candidate_rows.append(row)
                for repetition, elapsed in enumerate(samples):
                    selector_rows.append({
                        "case_key": case_key, "family": family,
                        "budget_index": budget_index, "repetition": repetition,
                        "runtime_ms": elapsed,
                    })
        paper_budget = max(1, min(n - 1, int(round(n * args.baseline_budget))))
        mask_np, metadata, samples, deterministic = EXP32.benchmark_selector(
            selector, "paper", scores["raw"], paper_budget, d,
            args.selector_repetitions,
        )
        digest, measured = get_measurement(mask_np)
        selected = int(mask_np.sum())
        selector_median = float(np.median(samples))
        candidate_rows.append({
            "case_key": case_key, "model": model_key, "model_label": model_label,
            "prompt": state["prompt"], "split": state["split"],
            "situation": state["situation"], "num_tokens": state["num_tokens"],
            "module": module_name, "layer": layer_index,
            "projection": projection, "shape": f"{n}x{d}", "n": n, "d": d,
            "row_size_kib": 2.0 * d / 1024.0,
            "paper_parameter_source": context["paper_parameter_source"],
            "family": "paper", "family_label": "Paper fixed 50%",
            "budget_index": -1, "budget_fraction": args.baseline_budget,
            "row_budget": paper_budget, "selected_rows": selected,
            "row_match": selected == paper_budget, "mask_sha256": digest,
            "raw_retention": retention(scores["raw"], mask_np),
            "bound_retention": retention(scores["bound"], mask_np),
            "diag_retention": retention(scores["diag"], mask_np),
            "selector_median_ms": selector_median,
            "selector_p95_ms": float(np.quantile(samples, 0.95)),
            "single_total_ms": selector_median + measured["nonselector_total_ms"],
            "deterministic": deterministic,
            "fallback_used": bool(metadata.get("fallback_used", False)),
            "error": metadata.get("error", ""), **measured,
        })
        for repetition, elapsed in enumerate(samples):
            selector_rows.append({
                "case_key": case_key, "family": "paper", "budget_index": -1,
                "repetition": repetition, "runtime_ms": elapsed,
            })
        print(
            f"  measured {model_key} {state['prompt']} {module_name}: "
            f"{3 * len(args.budgets) + 1} candidates, "
            f"{len(measured_masks)} unique masks", flush=True,
        )

    return hook


def calibration_cost_table(candidates: pd.DataFrame) -> pd.DataFrame:
    calibration = candidates[
        (candidates.split == "calibration") & candidates.family.isin(FAMILIES)
    ]
    return calibration.groupby(
        ["model", "shape", "family", "budget_index"], as_index=False
    ).agg(
        calibrated_nonselector_ms=("nonselector_total_ms", "mean"),
        calibrated_selector_ms=("selector_median_ms", "mean"),
        calibration_candidates=("case_key", "size"),
    )


def target_key(value: float) -> float:
    return round(float(value), 6)


def calibrate_predictor_margins(
    candidates: pd.DataFrame, features: pd.DataFrame, targets: list[float],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, float], float]]:
    """Calibrate a safe structured-overhead margin using calibration only."""
    rows = []
    tile = candidates[candidates.family.isin(FAMILIES)]
    calibration_features = features[features.split == "calibration"]
    for feature in calibration_features.itertuples(index=False):
        case = tile[
            (tile.case_key == feature.case_key) & (tile.family == feature.family)
        ].sort_values("budget_fraction")
        retention_name = f"{feature.family}_retention"
        allowed = case[case[retention_name] + 1e-12 >= feature.target]
        if len(allowed):
            structured = float(allowed.iloc[0].budget_fraction)
            feasible = True
        else:
            structured = float(case.iloc[-1].budget_fraction)
            feasible = False
        rows.append({
            "model": feature.model, "shape": feature.shape,
            "family": feature.family, "target": target_key(feature.target),
            "case_key": feature.case_key,
            "unstructured_fraction": feature.unstructured_required_fraction,
            "minimum_structured_fraction": structured,
            "required_margin": max(
                0.0, structured - feature.unstructured_required_fraction
            ),
            "grid_feasible": feasible,
        })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["model", "shape", "family", "target"], as_index=False
    ).agg(
        predictor_margin=("required_margin", "max"),
        calibration_cases=("case_key", "size"),
        calibration_grid_feasible_rate=("grid_feasible", "mean"),
        unstructured_fraction_mean=("unstructured_fraction", "mean"),
        minimum_structured_fraction_mean=("minimum_structured_fraction", "mean"),
    )
    mapping = {
        (row.model, row.shape, row.family, target_key(row.target)):
        float(row.predictor_margin)
        for row in summary.itertuples(index=False)
    }
    return summary, mapping


def build_threshold_decisions(
    candidates: pd.DataFrame, costs: pd.DataFrame, targets: list[float],
    features: pd.DataFrame,
    margins: dict[tuple[str, str, str, float], float],
) -> pd.DataFrame:
    lookup = costs.set_index(
        ["model", "shape", "family", "budget_index"]
    ).to_dict("index")
    rows = []
    tile = candidates[candidates.family.isin(FAMILIES)]
    feature_lookup = features.set_index(
        ["case_key", "family", "target"]
    ).to_dict("index")
    for case_key, case in tile.groupby("case_key", sort=False):
        first = case.iloc[0]
        for family in FAMILIES:
            family_rows = case[case.family == family].copy()
            for target in targets:
                feature = feature_lookup[(case_key, family, target)]
                margin = margins[(
                    first.model, first["shape"], family, target_key(target)
                )]
                predicted_fraction = min(
                    1.0,
                    float(feature["unstructured_required_fraction"]) + margin,
                )
                ordered = []
                for item in family_rows.itertuples(index=False):
                    if item.budget_fraction + 1e-12 < predicted_fraction:
                        continue
                    cost = lookup[(item.model, item.shape, family, item.budget_index)]
                    ordered.append((cost["calibrated_nonselector_ms"], item))
                if not ordered:
                    item = family_rows.sort_values("budget_fraction").iloc[-1]
                    item = next(
                        row for row in family_rows.itertuples(index=False)
                        if row.budget_index == item.budget_index
                    )
                    cost = lookup[(item.model, item.shape, family, item.budget_index)]
                    ordered.append((cost["calibrated_nonselector_ms"], item))
                ordered.sort(key=lambda pair: (pair[0], pair[1].budget_fraction))
                attempted = []
                chosen = None
                retention_name = f"{family}_retention"
                for _, item in ordered:
                    attempted.append(item)
                    if float(getattr(item, retention_name)) + 1e-12 >= target:
                        chosen = item
                        break
                feasible = chosen is not None
                if chosen is None:
                    chosen = max(
                        (item for _, item in ordered),
                        key=lambda item: (getattr(item, retention_name), -item.single_total_ms),
                    )
                predictor_ms = float(feature["predictor_median_ms"])
                selector_mask_ms = float(sum(
                    item.selector_median_ms for item in attempted
                ))
                selector_search = predictor_ms + selector_mask_ms
                rows.append({
                    "case_key": case_key, "model": first.model,
                    "prompt": first.prompt, "split": first.split,
                    "situation": first.situation, "module": first.module,
                    "layer": int(first.layer), "projection": first.projection,
                    "shape": first["shape"], "family": family,
                    "family_label": FAMILY_LABELS[family], "target": target,
                    "policy": f"{family}_t{target:g}",
                    "budget_index": int(chosen.budget_index),
                    "selected_fraction": float(chosen.selected_rows / chosen.n),
                    "selected_rows": int(chosen.selected_rows),
                    "quality_retention": float(getattr(chosen, retention_name)),
                    "raw_retention": float(chosen.raw_retention),
                    "bound_retention": float(chosen.bound_retention),
                    "diag_retention": float(chosen.diag_retention),
                    "relative_l2_error": float(chosen.relative_l2_error),
                    "cosine_error": float(chosen.cosine_error),
                    "selector_search_ms": selector_search,
                    "predictor_ms": predictor_ms,
                    "selector_mask_ms": selector_mask_ms,
                    "search_attempts": len(attempted), "feasible": feasible,
                    "unstructured_required_fraction": float(
                        feature["unstructured_required_fraction"]
                    ),
                    "predictor_margin": margin,
                    "predicted_fraction": predicted_fraction,
                    "read_wall_median_ms": float(chosen.read_wall_median_ms),
                    "gather_median_ms": float(chosen.gather_median_ms),
                    "gemm_median_ms": float(chosen.gemm_median_ms),
                    "nonselector_total_ms": float(chosen.nonselector_total_ms),
                    "actual_total_ms": selector_search + float(chosen.nonselector_total_ms),
                })
    return pd.DataFrame(rows)


def calibrate_targets(
    decisions: pd.DataFrame, error_ceiling: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], float]]:
    calibration = decisions[decisions.split == "calibration"]
    summary = calibration.groupby(
        ["model", "shape", "family", "target"], as_index=False
    ).agg(
        relative_l2_error=("relative_l2_error", "mean"),
        actual_total_ms=("actual_total_ms", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        feasible_rate=("feasible", "mean"),
    )
    rows = []
    target_map = {}
    for key, group in summary.groupby(["model", "shape", "family"], sort=True):
        allowed = group[
            (group.relative_l2_error <= error_ceiling) & (group.feasible_rate == 1.0)
        ]
        if len(allowed):
            chosen = allowed.sort_values(["actual_total_ms", "target"]).iloc[0]
            met = True
        else:
            chosen = group.sort_values(
                ["relative_l2_error", "actual_total_ms"], ascending=[True, True]
            ).iloc[0]
            met = False
        target_map[key] = float(chosen.target)
        rows.append({
            "model": key[0], "shape": key[1], "family": key[2],
            "selected_target": float(chosen.target),
            "calibration_relative_l2_error": float(chosen.relative_l2_error),
            "calibration_actual_total_ms": float(chosen.actual_total_ms),
            "calibration_selected_fraction": float(chosen.selected_fraction),
            "error_ceiling": error_ceiling, "ceiling_met": met,
        })
    return pd.DataFrame(rows), target_map


def calibrated_policy_rows(
    decisions: pd.DataFrame, target_map: dict[tuple[str, str, str], float],
) -> pd.DataFrame:
    selected = []
    for row in decisions.itertuples(index=False):
        target = target_map[(row.model, row.shape, row.family)]
        if math.isclose(row.target, target, rel_tol=0.0, abs_tol=1e-12):
            item = row._asdict()
            item["policy"] = f"{row.family}_calibrated"
            item["calibrated_target"] = target
            selected.append(item)
    return pd.DataFrame(selected)


def fixed_policy_rows(candidates: pd.DataFrame, baseline: float) -> pd.DataFrame:
    rows = []
    paper = candidates[candidates.family == "paper"]
    cell = candidates[
        (candidates.family == "raw")
        & np.isclose(candidates.budget_fraction, baseline)
    ]
    for policy, frame in (("paper_fixed", paper), ("cell_fixed", cell)):
        for item in frame.itertuples(index=False):
            rows.append({
                "case_key": item.case_key, "model": item.model,
                "prompt": item.prompt, "split": item.split,
                "situation": item.situation, "module": item.module,
                "layer": int(item.layer), "projection": item.projection,
                "shape": item.shape, "family": item.family,
                "family_label": "Paper" if policy == "paper_fixed" else "Cell-8",
                "target": float("nan"), "policy": policy,
                "budget_index": int(item.budget_index),
                "selected_fraction": float(item.selected_rows / item.n),
                "selected_rows": int(item.selected_rows),
                "quality_retention": float(item.raw_retention),
                "raw_retention": float(item.raw_retention),
                "bound_retention": float(item.bound_retention),
                "diag_retention": float(item.diag_retention),
                "relative_l2_error": float(item.relative_l2_error),
                "cosine_error": float(item.cosine_error),
                "selector_search_ms": float(item.selector_median_ms),
                "search_attempts": 1, "feasible": True,
                "read_wall_median_ms": float(item.read_wall_median_ms),
                "gather_median_ms": float(item.gather_median_ms),
                "gemm_median_ms": float(item.gemm_median_ms),
                "nonselector_total_ms": float(item.nonselector_total_ms),
                "actual_total_ms": float(item.single_total_ms),
                "calibrated_target": float("nan"),
            })
    return pd.DataFrame(rows)


def error_oracle_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_key, case in candidates.groupby("case_key", sort=False):
        paper = case[case.family == "paper"].iloc[0]
        pool = case[case.family.isin(FAMILIES)].copy()
        allowed = pool[pool.relative_l2_error <= paper.relative_l2_error + 1e-12]
        if not len(allowed):
            chosen = pool.sort_values(["relative_l2_error", "single_total_ms"]).iloc[0]
            feasible = False
        else:
            chosen = allowed.sort_values("single_total_ms").iloc[0]
            feasible = True
        rows.append({
            "case_key": case_key, "model": paper.model, "prompt": paper.prompt,
            "split": paper.split, "module": paper.module, "shape": paper["shape"],
            "paper_total_ms": paper.single_total_ms,
            "paper_relative_l2_error": paper.relative_l2_error,
            "oracle_family": chosen.family,
            "oracle_budget_fraction": chosen.selected_rows / chosen.n,
            "oracle_total_ms": chosen.single_total_ms,
            "oracle_relative_l2_error": chosen.relative_l2_error,
            "feasible": feasible,
        })
    return pd.DataFrame(rows)


class ModuleVariablePolicy:
    """Per-projection policy used only for held-out end-to-end quality runs."""

    def __init__(self, selector, selection_type, model_key: str, shape: str,
                 d: int, weight_norm: torch.Tensor, cost_order: dict,
                 target_map: dict, margin_map: dict, collector: list[dict]):
        self.selector = selector
        self.selection_type = selection_type
        self.model_key = model_key
        self.shape = shape
        self.d = d
        self.weight_norm = weight_norm
        self.cost_order = cost_order
        self.target_map = target_map
        self.margin_map = margin_map
        self.collector = collector
        self.mode = "paper_fixed"
        self.family = "raw"
        self.target = 0.9
        self.current_scores: dict[str, torch.Tensor] | None = None

    def prehook(self, _module, inputs):
        self.current_scores = score_vectors(inputs[0], self.weight_norm)

    def configure(self, mode: str, family: str = "raw", target: float = 0.9):
        self.mode, self.family, self.target = mode, family, float(target)

    def __call__(self, importance: torch.Tensor, _num_load_rows: int,
                 _row_size_kib: float):
        n = int(importance.numel())
        if self.mode == "paper_fixed":
            row_budget = max(1, min(n - 1, int(round(n * 0.50))))
            mask, metadata = self.selector.select(
                "paper", importance, row_budget, self.d
            )
            attempts, feasible, target = 1, True, float("nan")
            family = "paper"
        elif self.mode == "cell_fixed":
            row_budget = max(1, min(n - 1, int(round(n * 0.50))))
            mask, metadata = self.selector.select(
                "tile8_ceil", importance, row_budget, self.d
            )
            attempts, feasible, target = 1, True, float("nan")
            family = "raw"
        else:
            assert self.current_scores is not None
            family = self.family
            scores = self.current_scores[family]
            if self.mode == "calibrated":
                target = self.target_map[(self.model_key, self.shape, family)]
            else:
                target = self.target
            ordered_scores = torch.sort(scores, descending=True).values
            cumulative = torch.cumsum(ordered_scores, dim=0)
            required_rows = int(torch.searchsorted(
                cumulative, target * cumulative[-1], right=False
            )) + 1
            unstructured_fraction = required_rows / n
            margin = self.margin_map[(
                self.model_key, self.shape, family, target_key(target)
            )]
            predicted_fraction = min(1.0, unstructured_fraction + margin)
            mask, metadata, feasible = None, {}, False
            attempts = 0
            best = None
            candidates = [
                pair for pair in self.cost_order[
                    (self.model_key, self.shape, family)
                ]
                if pair[1] + 1e-12 >= predicted_fraction
            ]
            if not candidates:
                candidates = [max(
                    self.cost_order[(self.model_key, self.shape, family)],
                    key=lambda pair: pair[1],
                )]
            for budget_index, fraction in candidates:
                row_budget = max(1, min(n - 1, int(round(n * fraction))))
                candidate, candidate_meta = self.selector.select(
                    "tile8_ceil", scores, row_budget, self.d
                )
                attempts += 1
                candidate_retention = float(
                    scores[candidate].sum() / scores.sum().clamp_min(1e-20)
                )
                if best is None or candidate_retention > best[0]:
                    best = (candidate_retention, candidate, candidate_meta)
                if candidate_retention + 1e-12 >= target:
                    mask, metadata, feasible = candidate, candidate_meta, True
                    break
            if mask is None:
                assert best is not None
                _, mask, metadata = best
        raw_retention = float(
            importance[mask].sum() / importance.sum().clamp_min(1e-20)
        )
        self.collector.append({
            "family": family, "target": target, "selected_rows": int(mask.sum()),
            "num_rows": n, "selected_fraction": float(mask.sum()) / n,
            "raw_retention": raw_retention, "search_attempts": attempts,
            "feasible": feasible,
            "fallback": bool(metadata.get("fallback_used", False)),
        })
        return self.selection_type(mask, float(importance[mask].sum()), None)


def run_end_to_end(
    model, tokenizer, model_key: str, selector, costs: pd.DataFrame,
    target_map: dict, margin_map: dict, prompts: list[dict],
    args: argparse.Namespace, vlmflash,
) -> list[dict]:
    cost_order = {}
    for key, group in costs.groupby(["model", "shape", "family"], sort=False):
        ordered = group.sort_values(
            ["calibrated_nonselector_ms", "budget_index"]
        )
        cost_order[key] = [
            (int(row.budget_index), args.budgets[int(row.budget_index)])
            for row in ordered.itertuples(index=False)
        ]
    dummy = lambda importance, num_load_rows, row_size_kib: vlmflash.Selection(
        torch.ones_like(importance, dtype=torch.bool), float(importance.sum()), None
    )
    handle = vlmflash.attach(
        model, policy=dummy, include=vlmflash.DEFAULT_INCLUDE, sparsity=0.50
    )
    collector: list[dict] = []
    policies = []
    hooks = []
    for name in handle.names:
        module = model.get_submodule(name)
        weight_norm = torch.linalg.vector_norm(module.weight.detach().float(), dim=0)
        policy = ModuleVariablePolicy(
            selector, vlmflash.Selection, model_key,
            f"{module.in_features}x{module.out_features}", module.out_features,
            weight_norm, cost_order, target_map, margin_map, collector,
        )
        module.nc_policy = policy
        hooks.append(module.register_forward_pre_hook(policy.prehook))
        policies.append(policy)
    configurations = [
        ("paper_fixed", "paper_fixed", "raw", float("nan")),
        ("cell_fixed", "cell_fixed", "raw", float("nan")),
    ]
    for family in FAMILIES:
        configurations.append((f"{family}_calibrated", "calibrated", family, 0.0))
        configurations.append((f"{family}_t0.9", "threshold", family, 0.90))
    rows = []
    try:
        for prompt in prompts:
            if prompt["split"] != "holdout":
                continue
            inputs = EXP32.tokenize(tokenizer, prompt["text"], args.max_input_tokens)
            with torch.inference_mode():
                dense = model(**inputs, use_cache=False).logits.detach()
            for policy_name, mode, family, target in configurations:
                collector.clear()
                for policy in policies:
                    policy.configure(mode, family, target)
                with torch.inference_mode(), vlmflash.enabled():
                    sparse = model(**inputs, use_cache=False).logits.detach()
                metrics = EXP32.logits_error(dense, sparse, inputs["input_ids"])
                rows.append({
                    "model": model_key, "prompt": prompt["name"],
                    "situation": prompt["situation"], "policy": policy_name,
                    "family": family, "target": target,
                    **metrics, "projection_calls": len(collector),
                    "selected_fraction_mean": float(np.mean([
                        call["selected_fraction"] for call in collector
                    ])),
                    "selected_fraction_min": float(np.min([
                        call["selected_fraction"] for call in collector
                    ])),
                    "selected_fraction_max": float(np.max([
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
                print(
                    f"  e2e {model_key} {prompt['name']} {policy_name}: "
                    f"R={rows[-1]['selected_fraction_mean']:.3f}", flush=True,
                )
                del sparse
            del dense, inputs
    finally:
        for hook in hooks:
            hook.remove()
        handle.detach()
    return rows


def run_model(model_key: str, selector, native_reader, prompts, args, vlmflash):
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
        raise RuntimeError(f"{model_key}: expected {len(sampled_layers)*7} projections")
    candidate_rows, selector_rows, io_rows, gemm_rows, feature_rows = (
        [], [], [], [], []
    )
    state = {"prompt": "", "split": "", "situation": "", "num_tokens": 0}
    hooks = [
        module.register_forward_hook(make_projection_hook(
            state, model_key, spec["label"], name, module, selector,
            native_reader, args, candidate_rows, selector_rows, io_rows, gemm_rows,
            feature_rows,
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


def aggregate(args: argparse.Namespace, run_e2e: bool = False) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    features = pd.read_csv(output / "score_features.csv")
    costs = calibration_cost_table(candidates)
    costs.to_csv(output / "calibration_costs.csv", index=False)
    margin_frame, margin_map = calibrate_predictor_margins(
        candidates, features, args.quality_targets
    )
    margin_frame.to_csv(output / "predictor_margins.csv", index=False)
    decisions = build_threshold_decisions(
        candidates, costs, args.quality_targets, features, margin_map
    )
    decisions.to_csv(output / "threshold_decisions.csv", index=False)
    calibration, target_map = calibrate_targets(
        decisions, args.calibration_error_ceiling
    )
    calibration.to_csv(output / "calibrated_targets.csv", index=False)
    calibrated = calibrated_policy_rows(decisions, target_map)
    fixed = fixed_policy_rows(candidates, args.baseline_budget)
    policies = pd.concat([fixed, calibrated], ignore_index=True)
    policies.to_csv(output / "policy_projection_cases.csv", index=False)
    oracle = error_oracle_rows(candidates)
    oracle.to_csv(output / "error_oracle.csv", index=False)
    holdout = policies[policies.split == "holdout"]
    summary = holdout.groupby("policy", as_index=False).agg(
        cases=("case_key", "size"), actual_total_ms=("actual_total_ms", "mean"),
        selector_search_ms=("selector_search_ms", "mean"),
        read_wall_ms=("read_wall_median_ms", "mean"),
        gather_ms=("gather_median_ms", "mean"), gemm_ms=("gemm_median_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        search_attempts=("search_attempts", "mean"), feasible_rate=("feasible", "mean"),
    )
    summary.to_csv(output / "policy_summary.csv", index=False)
    per_model = holdout.groupby(["model", "policy"], as_index=False).agg(
        actual_total_ms=("actual_total_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        selected_fraction=("selected_fraction", "mean"),
    )
    per_model.to_csv(output / "policy_by_model.csv", index=False)
    time_pivot = holdout.pivot(
        index="case_key", columns="policy", values="actual_total_ms"
    )
    error_pivot = holdout.pivot(
        index="case_key", columns="policy", values="relative_l2_error"
    )
    paired_rows = []
    for policy in ("cell_fixed", *[f"{family}_calibrated" for family in FAMILIES]):
        paired_rows.append({
            "policy": policy,
            "time_win_rate": float(
                (time_pivot[policy] < time_pivot.paper_fixed).mean()
            ),
            "error_win_rate": float(
                (error_pivot[policy] < error_pivot.paper_fixed).mean()
            ),
            "joint_win_rate": float((
                (time_pivot[policy] < time_pivot.paper_fixed)
                & (error_pivot[policy] <= error_pivot.paper_fixed)
            ).mean()),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "paired_summary.csv", index=False)
    target_summary = decisions[decisions.split == "holdout"].groupby(
        ["family", "target"], as_index=False
    ).agg(
        actual_total_ms=("actual_total_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        search_attempts=("search_attempts", "mean"),
        feasible_rate=("feasible", "mean"),
    )
    target_summary.to_csv(output / "target_sweep_summary.csv", index=False)
    holdout_oracle = oracle[oracle.split == "holdout"]
    oracle_summary = {
        "paper_total_ms": float(holdout_oracle.paper_total_ms.mean()),
        "oracle_total_ms": float(holdout_oracle.oracle_total_ms.mean()),
        "saving_vs_paper_pct": float(100 * (
            1 - holdout_oracle.oracle_total_ms.mean()
            / holdout_oracle.paper_total_ms.mean()
        )),
        "strict_win_rate": float(
            (holdout_oracle.oracle_total_ms < holdout_oracle.paper_total_ms).mean()
        ),
        "feasible_rate": float(holdout_oracle.feasible.mean()),
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for family in FAMILIES:
        group = target_summary[target_summary.family == family].sort_values("target")
        ax.plot(group.actual_total_ms, group.relative_l2_error, marker="o",
                label=FAMILY_LABELS[family], color=COLORS[family])
        for row in group.itertuples(index=False):
            ax.annotate(f"{100*row.target:g}%", (row.actual_total_ms,
                        row.relative_l2_error), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    fixed_summary = summary[summary.policy.isin(["paper_fixed", "cell_fixed"])]
    for row in fixed_summary.itertuples(index=False):
        label = "Paper 50%" if row.policy == "paper_fixed" else "Cell-8 50%"
        ax.scatter(row.actual_total_ms, row.relative_l2_error, marker="*", s=150,
                   label=label)
    ax.set_xlabel("Selector search + O_DIRECT/upload + gather + compact GEMM (ms)")
    ax.set_ylabel("Projection relative L2 error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "quality_latency_frontier.png", dpi=180)
    fig.savefig(output / "quality_latency_frontier.pdf")
    plt.close(fig)

    e2e_path = output / "end_to_end.csv"
    if e2e_path.is_file():
        e2e = pd.read_csv(e2e_path)
        e2e_summary = e2e.groupby("policy", as_index=False).agg(
            cases=("prompt", "size"), dense_to_sparse_kl=("dense_to_sparse_kl", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            nll_delta=("nll_delta", "mean"),
            logit_relative_l2_error=("logit_relative_l2_error", "mean"),
            selected_fraction=("selected_fraction_mean", "mean"),
            feasible_rate=("feasible_rate", "mean"),
        )
        e2e_summary.to_csv(output / "end_to_end_summary.csv", index=False)
        e2e_by_model = e2e.groupby(["model", "policy"], as_index=False).agg(
            dense_to_sparse_kl=("dense_to_sparse_kl", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            nll_delta=("nll_delta", "mean"),
            selected_fraction=("selected_fraction_mean", "mean"),
        )
        e2e_by_model.to_csv(output / "end_to_end_by_model.csv", index=False)
    else:
        e2e_summary = pd.DataFrame()
        e2e_by_model = pd.DataFrame()
    write_report(
        args, summary, per_model, paired, target_summary, calibration,
        oracle_summary, e2e_summary, e2e_by_model,
    )
    result = {
        "format": "experiment-34-quality-constrained-variable-r-v1",
        "candidate_cases": int(len(candidates)),
        "threshold_decisions": int(len(decisions)),
        "holdout_policy_cases": int(len(holdout)),
        "all_direct": bool((candidates.direct_rate == 1.0).all()),
        "all_deterministic": bool(candidates.deterministic.all()),
        "fallback_cases": int(candidates.fallback_used.sum()),
        "oracle": oracle_summary,
        "e2e_cases": int(pd.read_csv(e2e_path).shape[0]) if e2e_path.is_file() else 0,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


def write_report(args, summary, per_model, paired, sweep, calibration, oracle,
                 e2e, e2e_by_model) -> None:
    report = args.report_output or HERE / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    indexed = summary.set_index("policy")
    lines = [
        "# Experiment 34 보고서: 품질 하한 기반 가변 R", "",
        "고정된 row ratio를 목적함수로 사용하지 않고, calibration에서만 정한 품질 "
        "하한을 만족하는 후보 중 이 노트북의 실제 비용이 가장 작은 구조를 선택했다. "
        "Holdout의 Paper importance나 실제 오차는 정책 입력으로 사용하지 않았다.", "",
        "## Holdout projection 정책", "",
        "| 정책 | actual total | selector search | 선택 R | relative L2 | feasible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["paper_fixed", "cell_fixed", *[f"{f}_calibrated" for f in FAMILIES]]
    labels = {
        "paper_fixed": "Paper fixed 50%", "cell_fixed": "Cell-8 fixed 50%",
        "raw_calibrated": "Variable-R activation",
        "bound_calibrated": "Variable-R weight-bound",
        "diag_calibrated": "Variable-R weight-diagonal",
    }
    for policy in order:
        if policy not in indexed.index:
            continue
        row = indexed.loc[policy]
        lines.append(
            f"| {labels[policy]} | {row.actual_total_ms:.3f} ms "
            f"| {row.selector_search_ms:.3f} ms | {100*row.selected_fraction:.1f}% "
            f"| {row.relative_l2_error:.4f} | {100*row.feasible_rate:.1f}% |"
        )
    lines.extend([
        "", "### 모델별 calibrated 정책", "",
        "| 모델 | 정책 | actual total | 선택 R | relative L2 |",
        "|---|---|---:|---:|---:|",
    ])
    for row in per_model[
        per_model.policy.isin(["paper_fixed", *[f"{f}_calibrated" for f in FAMILIES]])
    ].itertuples(index=False):
        lines.append(
            f"| {row.model} | {labels.get(row.policy, row.policy)} "
            f"| {row.actual_total_ms:.3f} ms | {100*row.selected_fraction:.1f}% "
            f"| {row.relative_l2_error:.4f} |"
        )
    lines.extend([
        "", "### Paper와 paired 비교", "",
        "| 정책 | latency win | local-error win | 둘 다 win |",
        "|---|---:|---:|---:|",
    ])
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {labels.get(row.policy, row.policy)} "
            f"| {100*row.time_win_rate:.1f}% | {100*row.error_win_rate:.1f}% "
            f"| {100*row.joint_win_rate:.1f}% |"
        )
    lines.extend([
        "", f"Calibration의 projection relative-L2 ceiling은 "
        f"`{args.calibration_error_ceiling:.3f}`이다. 아래 target은 Paper의 온라인 "
        "importance가 아니라 각 score의 보존율 하한이다.", "",
        "## 품질 하한 sweep", "",
        "| score | 하한 | actual total | 선택 R | relative L2 | 탐색 수 | feasible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sweep.sort_values(["family", "target"]).itertuples(index=False):
        lines.append(
            f"| {FAMILY_LABELS[row.family]} | {100*row.target:g}% "
            f"| {row.actual_total_ms:.3f} ms | {100*row.selected_fraction:.1f}% "
            f"| {row.relative_l2_error:.4f} | {row.search_attempts:.2f} "
            f"| {100*row.feasible_rate:.1f}% |"
        )
    lines.extend([
        "", "## Calibration이 선택한 하한", "",
        "| 모델 | shape | score | target | calibration error | R | ceiling 충족 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in calibration.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.shape} | {FAMILY_LABELS[row.family]} "
            f"| {100*row.selected_target:g}% | {row.calibration_relative_l2_error:.4f} "
            f"| {100*row.calibration_selected_fraction:.1f}% "
            f"| {'yes' if row.ceiling_met else 'no'} |"
        )
    lines.extend([
        "", "## 사후 error oracle (정책 아님)", "",
        f"Holdout의 실제 projection error를 사후에 볼 수 있다고 가정하면 Paper 50%와 "
        f"같거나 작은 오차에서 평균 `{oracle['saving_vs_paper_pct']:.2f}%`를 절약했다. "
        f"strict latency win rate는 `{100*oracle['strict_win_rate']:.1f}%`, feasible rate는 "
        f"`{100*oracle['feasible_rate']:.1f}%`다. 이는 가능한 headroom의 상한이며 배포 "
        "가능한 결과로 해석하면 안 된다.", "",
    ])
    if len(e2e):
        lines.extend([
            "## Holdout end-to-end 오차", "",
            "| 정책 | 선택 R | logit rel-L2 | KL | top-1 | NLL delta | feasible |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in e2e.itertuples(index=False):
            lines.append(
                f"| {labels.get(row.policy, row.policy)} | {100*row.selected_fraction:.1f}% "
                f"| {row.logit_relative_l2_error:.4f} | {row.dense_to_sparse_kl:.4f} "
                f"| {100*row.top1_agreement:.2f}% | {row.nll_delta:+.4f} "
                f"| {100*row.feasible_rate:.1f}% |"
            )
        lines.extend([
            "", "### 모델별 Paper 대 calibrated activation", "",
            "| 모델 | 정책 | 선택 R | KL | top-1 | NLL delta |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in e2e_by_model[
            e2e_by_model.policy.isin(["paper_fixed", "raw_calibrated"])
        ].itertuples(index=False):
            lines.append(
                f"| {row.model} | {labels.get(row.policy, row.policy)} "
                f"| {100*row.selected_fraction:.1f}% | {row.dense_to_sparse_kl:.4f} "
                f"| {100*row.top1_agreement:.2f}% | {row.nll_delta:+.4f} |"
            )
    paper = indexed.loc["paper_fixed"]
    candidates = indexed.drop(index=[p for p in ("paper_fixed", "cell_fixed") if p in indexed.index])
    fastest = candidates.actual_total_ms.idxmin()
    fast = indexed.loc[fastest]
    raw = indexed.loc["raw_calibrated"]
    raw_time_saving = 100 * (1 - raw.actual_total_ms / paper.actual_total_ms)
    raw_error_change = 100 * (
        raw.relative_l2_error / paper.relative_l2_error - 1
    )
    if len(e2e):
        e2e_indexed = e2e.set_index("policy")
        raw_kl_change = 100 * (
            e2e_indexed.loc["raw_calibrated", "dense_to_sparse_kl"]
            / e2e_indexed.loc["paper_fixed", "dense_to_sparse_kl"] - 1
        )
        raw_nll_change = 100 * (
            e2e_indexed.loc["raw_calibrated", "nll_delta"]
            / e2e_indexed.loc["paper_fixed", "nll_delta"] - 1
        )
    else:
        raw_kl_change = raw_nll_change = float("nan")
    raw_pair = paired.set_index("policy").loc["raw_calibrated"]
    lines.extend([
        "", "## 판정", "",
        f"Calibration 기반 가변 정책 중 최저 시간은 **{labels.get(fastest, fastest)}**로 "
        f"`{fast.actual_total_ms:.3f} ms`이다. Paper fixed 50%의 "
        f"`{paper.actual_total_ms:.3f} ms` 대비 "
        f"`{100*(1-fast.actual_total_ms/paper.actual_total_ms):.2f}%` 빨랐고, "
        f"projection relative-L2는 `{paper.relative_l2_error:.4f}`에서 "
        f"`{fast.relative_l2_error:.4f}`로 변했다.", "",
        f"End-to-end까지 함께 보면 activation 기반 가변 정책이 핵심 결과다. Paper보다 "
        f"`{raw_time_saving:.2f}%` 빠르면서 local error는 `{raw_error_change:.2f}%`, "
        f"logit KL은 `{raw_kl_change:.2f}%`, NLL delta는 `{raw_nll_change:.2f}%` "
        "변했다. 음수인 오차 변화는 개선을 뜻한다. 반면 weight-bound와 diagonal은 "
        "local error를 줄였지만 평균 end-to-end KL을 악화시켰다.", "",
        f"따라서 **고정 R 대신 측정 가능한 품질 하한에서 R을 결정한다**는 통찰은 "
        "aggregate 결과에서 지지된다. 그러나 activation 정책이 개별 projection에서 "
        f"latency와 local error를 동시에 이긴 비율은 `{100*raw_pair.joint_win_rate:.1f}%`이고, "
        "모델별 end-to-end 결과도 일관되지 않다. 아직 per-case 또는 모델 보편적 "
        "우월성을 주장할 수는 없다.", "",
        f"사후 error oracle의 Paper 대비 절약은 `{oracle['saving_vs_paper_pct']:.2f}%`로 "
        "deployable 정책보다 크다. 이는 더 좋은 threshold calibration과 global layer "
        "allocation에 남은 여지가 있음을 보여준다.", "",
        "## 측정 범위와 제한", "",
        "- 세 checkpoint의 초·중·후반 layer와 q/k/v/o/gate/up/down projection을 "
        "calibration 3 prompts, holdout 3 prompts에서 측정했다.",
        "- actual total은 projection 하나의 selector, O_DIRECT+upload, activation gather, "
        "compact GEMM 합이다. 전체 LLM wall-clock latency가 아니다.",
        "- 모든 정책에서 activation score 자체를 만드는 reduction은 selector 밖으로 "
        "두었다. Paper와 activation 정책에는 같은 관례지만, weight-aware score의 추가 "
        "L2/norm 연산도 제외되어 그 두 정책의 latency는 다소 낙관적이다.",
        f"- `{args.calibration_error_ceiling:.3f}`은 projection-level 설계 허용치이며 "
        "task accuracy 보장이 아니다.",
        "- End-to-end 표는 resident checkpoint weight로 모든 decoder projection을 "
        "동시에 sparsify한 quality stress test다. 논문의 visual-token 조건과 같지 않다.",
        "- O_DIRECT는 동일 mask의 byte geometry를 실제 NVMe/GPU 경로에서 측정했고, "
        "projection 오차는 실제 checkpoint weight로 계산했다.", "",
        "이 비교에는 selector 탐색, native O_DIRECT+GPU upload, activation gather, 실제 "
        "compact GEMM이 모두 포함된다. End-to-end 표는 실제 sparse forward의 품질 "
        "측정이며 resident-weight 구현 특성상 전체 LLM wall-clock speedup 표는 아니다.", "",
        "![Quality-latency frontier](results_laptop/quality_latency_frontier.png)", "",
    ])
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        aggregate(args)
        return
    prompts = list(PROMPTS[:args.prompt_limit] if args.prompt_limit else PROMPTS)
    lookup = EXP32.BASE.LatencyTable.load(args.profile)
    selector = EXP32.Selector(lookup, args.saturation_kib)
    vlmflash = EXP32.EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")
    max_required = 0
    from transformers import AutoConfig
    for key in args.models:
        config = AutoConfig.from_pretrained(
            EXP32.cached_snapshot(MODEL_SPECS[key]["repo"]), local_files_only=True
        )
        max_required = max(
            max_required, int(config.intermediate_size) * int(config.hidden_size) * 2
        )
    if args.io_blob.stat().st_size < max_required:
        raise SystemExit(f"I/O blob needs at least {max_required} bytes")

    all_candidates, all_selectors, all_io, all_gemm, all_features, all_e2e, models = (
        [], [], [], [], [], [], []
    )
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

        # Calibrate and run holdout quality for this model before unloading it;
        # retaining all three checkpoints simultaneously exceeds the 6 GB GPU.
        if not args.skip_end_to_end and any(p["split"] == "holdout" for p in prompts):
            model_candidates = pd.DataFrame(candidates)
            model_features = pd.DataFrame(features)
            model_costs = calibration_cost_table(model_candidates)
            _, model_margin_map = calibrate_predictor_margins(
                model_candidates, model_features, args.quality_targets
            )
            model_decisions = build_threshold_decisions(
                model_candidates, model_costs, args.quality_targets,
                model_features, model_margin_map,
            )
            _, model_target_map = calibrate_targets(
                model_decisions, args.calibration_error_ceiling
            )
            all_e2e.extend(run_end_to_end(
                model, tokenizer, model_key, selector, model_costs,
                model_target_map, model_margin_map, prompts, args, vlmflash,
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
        "format": "experiment-34-quality-constrained-variable-r-v1",
        "models": models,
        "prompts": [
            {k: v for k, v in prompt.items() if k != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "budgets": args.budgets, "quality_targets": args.quality_targets,
        "baseline_budget": args.baseline_budget,
        "calibration_error_ceiling": args.calibration_error_ceiling,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions, "io_warmup": args.io_warmup,
        "gemm_repetitions": args.gemm_repetitions,
        "gemm_warmup": args.gemm_warmup, "io_threads": args.io_threads,
        "io_max_read_kib": args.io_max_read_kib,
        "profile": str(args.profile), "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "flash": profile_meta.get("flash", "unknown"),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    aggregate(args)


if __name__ == "__main__":
    main()
