#!/usr/bin/env python3
"""Experiment 36: sweep the Cell-C discretization parameter."""

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
import traceback
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
EXP35_PATH = PROJECT_ROOT / "experiments" / "35_full_frontier" / "run_experiment.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP35 = _load_module("experiment_35_for_36", EXP35_PATH)
EXP32 = EXP35.EXP32
EXP31 = EXP32.EXP31
EXP29 = EXP31.EXP29
EXP34 = EXP35.EXP34
MODEL_SPECS = EXP35.MODEL_SPECS
PROMPTS = EXP35.PROMPTS
LOCAL_PROFILE = EXP35.LOCAL_PROFILE
njit = EXP29.njit

CELL_COUNTS = (1, 2, 4, 8, 16, 32)
CELL_METHODS = tuple(f"cell{count}" for count in CELL_COUNTS)
METHODS = ("paper", *CELL_METHODS)
METHOD_LABELS = {"paper": "Paper"} | {
    f"cell{count}": f"Cell-{count}" for count in CELL_COUNTS
}
METHOD_COLORS = {
    "paper": "#D97706",
    "cell1": "#94A3B8",
    "cell2": "#7C3AED",
    "cell4": "#2563EB",
    "cell8": "#0F766E",
    "cell16": "#16A34A",
    "cell32": "#DC2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument(
        "--budgets", type=float, nargs="+",
        default=[value / 100.0 for value in range(10, 100, 5)],
    )
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
        raise SystemExit("Experiment 36 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if args.budgets != sorted(set(args.budgets)) or any(
        not 0.0 < value < 1.0 for value in args.budgets
    ):
        raise SystemExit("--budgets must be unique, increasing, and inside (0, 1)")
    counts = (
        args.layer_samples, args.selector_repetitions, args.io_repetitions,
        args.gemm_repetitions,
    )
    if any(value < 1 for value in counts):
        raise SystemExit("sample and repetition counts must be positive")


@njit(cache=False)
def _best_k_tiles(values, tile_rows, cell_rows, cells_per_tile, tile_count):
    n = len(values)
    output = np.zeros(n, dtype=np.bool_)
    if tile_count == 0:
        return output, 0, True
    if tile_rows > n:
        return output, 0, False
    candidate_count = (n - tile_rows) // cell_rows + 1
    if candidate_count <= 0:
        return output, 0, False
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
    weights = np.empty(candidate_count, dtype=np.float64)
    for candidate in range(candidate_count):
        start = candidate * cell_rows
        weights[candidate] = prefix[start + tile_rows] - prefix[start]

    negative = -1e300
    dp = np.full((tile_count + 1, candidate_count + 1), negative, dtype=np.float64)
    take = np.zeros((tile_count + 1, candidate_count + 1), dtype=np.uint8)
    for prefix_count in range(candidate_count + 1):
        dp[0, prefix_count] = 0.0
    for count in range(1, tile_count + 1):
        for candidate_prefix in range(1, candidate_count + 1):
            candidate = candidate_prefix - 1
            skip_score = dp[count, candidate_prefix - 1]
            predecessor = candidate_prefix - cells_per_tile
            if predecessor < 0:
                predecessor = 0
            take_score = dp[count - 1, predecessor] + weights[candidate]
            if take_score > skip_score:
                dp[count, candidate_prefix] = take_score
                take[count, candidate_prefix] = 1
            else:
                dp[count, candidate_prefix] = skip_score
    if not np.isfinite(dp[tile_count, candidate_count]):
        return output, candidate_count, False

    count = tile_count
    candidate_prefix = candidate_count
    while count > 0:
        if candidate_prefix <= 0:
            return np.zeros(n, dtype=np.bool_), candidate_count, False
        if take[count, candidate_prefix]:
            candidate = candidate_prefix - 1
            start = candidate * cell_rows
            for index in range(start, start + tile_rows):
                output[index] = True
            count -= 1
            candidate_prefix -= cells_per_tile
            if candidate_prefix < 0:
                candidate_prefix = 0
        else:
            candidate_prefix -= 1
    return output, candidate_count, True


@njit(cache=False)
def _select_cell_kernel(values, row_budget, cell_rows, cells_per_tile, run_costs):
    n = len(values)
    tile_rows = cells_per_tile * cell_rows
    if tile_rows > n or row_budget < tile_rows:
        mask = EXP29._best_contiguous(values, row_budget, cell_rows)
        importance, cost, chunks = EXP29._mask_metrics_lookup(
            mask, values, run_costs
        )
        return mask, 0, tile_rows, 0, 0, chunks, importance / cost, True

    floor_count = row_budget // tile_rows
    floor_mask, candidate_count, valid = _best_k_tiles(
        values, tile_rows, cell_rows, cells_per_tile, floor_count
    )
    if not valid:
        return np.zeros(n, dtype=np.bool_), candidate_count, tile_rows, 0, 0, 0, 0.0, False
    floor_mask, additions, valid = EXP29._expand_endpoints(
        floor_mask, values, row_budget, run_costs
    )
    if not valid:
        return floor_mask, candidate_count, tile_rows, 0, additions, 0, 0.0, False
    floor_importance, floor_cost, floor_chunks = EXP29._mask_metrics_lookup(
        floor_mask, values, run_costs
    )
    best_mask = floor_mask
    best_ratio = floor_importance / floor_cost
    strategy = 1
    repair = additions
    chunks = floor_chunks

    ceil_count = (row_budget + tile_rows - 1) // tile_rows
    if ceil_count != floor_count and ceil_count * tile_rows <= n:
        ceil_mask, _, ceil_valid = _best_k_tiles(
            values, tile_rows, cell_rows, cells_per_tile, ceil_count
        )
        if ceil_valid:
            ceil_mask, deletions, ceil_valid = EXP29._trim_endpoints(
                ceil_mask, values, row_budget, run_costs
            )
            if ceil_valid:
                ceil_importance, ceil_cost, ceil_chunks = EXP29._mask_metrics_lookup(
                    ceil_mask, values, run_costs
                )
                ceil_ratio = ceil_importance / ceil_cost
                if ceil_ratio > best_ratio + 1e-15:
                    best_mask = ceil_mask
                    best_ratio = ceil_ratio
                    strategy = 2
                    repair = deletions
                    chunks = ceil_chunks
    return (
        best_mask, candidate_count, tile_rows, strategy, repair, chunks,
        best_ratio, True,
    )


class CellSweepSelector(EXP32.Selector):
    def select(self, method: str, importance: torch.Tensor, row_budget: int,
               d: int) -> tuple[torch.Tensor, dict]:
        if method == "paper":
            return super().select(method, importance, row_budget, d)
        if method == "tile8_ceil":
            method = "cell8"
        match = re.fullmatch(r"cell(\d+)", method)
        if match is None:
            raise ValueError(f"unknown cell method: {method}")
        cells_per_tile = int(match.group(1))
        n = int(importance.numel())
        context = self.context(n, d)
        normalized = importance / importance.sum().clamp_min(1e-20)
        values = normalized.detach().to("cpu").numpy().astype(np.float64)
        model = context["model"]
        cell_rows = max(
            1, int(math.ceil(float(model["saturation_rows"]) / cells_per_tile))
        )
        row_kib = float(model["saturation_kib"]) / float(model["saturation_rows"])
        run_costs = EXP29.run_costs_for(self.lookup_table, n, row_kib)
        try:
            result = _select_cell_kernel(
                values, int(row_budget), int(cell_rows), int(cells_per_tile),
                run_costs,
            )
            mask_np, candidates, tile_rows, strategy, repair, chunks, ratio, valid = result
            if not valid or int(mask_np.sum()) != int(row_budget):
                raise RuntimeError("Cell-C selector failed to return exact R")
            metadata = {
                "scalarized_calls": 1,
                "outer_iterations": 1,
                "frontier_candidates": int(candidates),
                "eligible_candidates": int(chunks),
                "trim_work_deletions": int(repair if strategy == 2 else 0),
                "minimum_overfill": int(tile_rows),
                "repair_deletions": int(repair if strategy == 2 else 0),
                "repair_additions": int(repair if strategy == 1 else 0),
                "returned_top_r": False,
                "cell_count": cells_per_tile,
                "cell_rows": int(cell_rows),
                "tile_rows": int(tile_rows),
                "tile_strategy": (
                    "contiguous" if strategy == 0 else
                    "floor_expand" if strategy == 1 else "ceil_trim"
                ),
                "lookup_efficiency": float(ratio),
                "fallback_used": False,
                "error": "",
            }
        except Exception as exception:
            mask_np = EXP29.EXP24.top_r_mask(values, int(row_budget))
            metadata = {
                "cell_count": cells_per_tile,
                "fallback_used": True,
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(limit=4),
            }
        mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(
            importance.device
        )
        return mask, metadata


def benchmark_selector(selector, method: str, importance: torch.Tensor,
                       row_budget: int, d: int, repetitions: int):
    selector.select(method, importance, row_budget, d)
    samples, masks = [], []
    metadata = {}
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        mask, metadata = selector.select(method, importance, row_budget, d)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        masks.append(mask.detach().to("cpu").numpy().astype(bool, copy=True))
    deterministic = all(np.array_equal(masks[0], item) for item in masks[1:])
    return masks[0], metadata, samples, deterministic


# Reuse Experiment 35's real-model hook and measurement path with this method set.
EXP35.METHODS = METHODS
EXP35.METHOD_LABELS = METHOD_LABELS
EXP35.benchmark_selector = benchmark_selector


class EndToEndPolicy:
    def __init__(self, selector, selection_type, d, budgets, collector):
        self.selector = selector
        self.selection_type = selection_type
        self.d = int(d)
        self.budgets = budgets
        self.collector = collector
        self.method = "paper"
        self.budget_index = 0

    def configure(self, method: str, budget_index: int) -> None:
        self.method = method
        self.budget_index = int(budget_index)

    def __call__(self, importance, _num_load_rows, _row_size_kib):
        n = int(importance.numel())
        fraction = self.budgets[self.budget_index]
        rows = max(1, min(n - 1, int(round(n * fraction))))
        mask, metadata = self.selector.select(
            self.method, importance, rows, self.d
        )
        self.collector.append({
            "selected_fraction": float(mask.sum()) / n,
            "fallback": bool(metadata.get("fallback_used", False)),
        })
        return self.selection_type(mask, float(importance[mask].sum()), None)


def run_end_to_end(model, tokenizer, model_key, selector, prompts, args, vlmflash):
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
            selector, vlmflash.Selection, module.out_features,
            args.budgets, collector,
        )
        module.nc_policy = policy
        policies.append(policy)
    rows = []
    try:
        for prompt in prompts:
            if prompt["split"] != "holdout":
                continue
            inputs = EXP32.tokenize(tokenizer, prompt["text"], args.max_input_tokens)
            with torch.inference_mode():
                dense = model(**inputs, use_cache=False).logits.detach()
            for method in METHODS:
                for budget_index, fraction in enumerate(args.budgets):
                    collector.clear()
                    for policy in policies:
                        policy.configure(method, budget_index)
                    with torch.inference_mode(), vlmflash.enabled():
                        sparse = model(**inputs, use_cache=False).logits.detach()
                    metrics = EXP32.logits_error(
                        dense, sparse, inputs["input_ids"]
                    )
                    rows.append({
                        "model": model_key,
                        "prompt": prompt["name"],
                        "situation": prompt["situation"],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "budget_index": budget_index,
                        "budget_fraction": fraction,
                        **metrics,
                        "projection_calls": len(collector),
                        "selected_fraction_mean": float(np.mean([
                            item["selected_fraction"] for item in collector
                        ])),
                        "fallback_calls": int(sum(
                            item["fallback"] for item in collector
                        )),
                    })
            print(f"  e2e {model_key} {prompt['name']} complete", flush=True)
            del dense, inputs
    finally:
        handle.detach()
    return rows


def _best_at_error(curve: pd.DataFrame, error_column: str, ceiling: float,
                   time_column: str):
    allowed = curve[curve[error_column] <= ceiling]
    if not len(allowed):
        return None
    return allowed.sort_values(time_column).iloc[0]


def _pareto_points(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = frame.sort_values(["actual_total_ms", "relative_l2_error"])
    best_error = float("inf")
    for row in ordered.itertuples(index=False):
        if row.relative_l2_error < best_error - 1e-12:
            rows.append(row._asdict())
            best_error = row.relative_l2_error
    return pd.DataFrame(rows)


def _interpolate_time(curve: pd.DataFrame, error_column: str,
                      time_column: str, error: float) -> float:
    """Piecewise-linear time at an error value; diagnostic, not measurement."""
    points = (
        curve[[error_column, time_column]]
        .dropna()
        .groupby(error_column, as_index=False)[time_column].min()
        .sort_values(error_column)
    )
    errors = points[error_column].to_numpy(dtype=float)
    times = points[time_column].to_numpy(dtype=float)
    if not len(errors) or error < errors[0] or error > errors[-1]:
        return float("nan")
    return float(np.interp(error, errors, times))


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    holdout = candidates[candidates.split == "holdout"]
    fixed = holdout.groupby(
        ["method", "method_label", "budget_index", "budget_fraction"],
        as_index=False,
    ).agg(
        cases=("case_key", "size"),
        actual_total_ms=("actual_total_ms", "mean"),
        selector_ms=("selector_median_ms", "mean"),
        nonselector_ms=("nonselector_total_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        importance_retention=("importance_retention", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        row_match_rate=("row_match", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    fixed["cell_count"] = fixed.method.str.extract(r"cell(\d+)")[0].astype(float)
    fixed.to_csv(output / "fixed_frontiers.csv", index=False)

    cell_points = fixed[fixed.method.isin(CELL_METHODS)]
    pareto = _pareto_points(cell_points)
    pareto.to_csv(output / "global_cell_pareto.csv", index=False)

    ceilings = [0.50, 0.45, 0.42, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
    comparison_rows = []
    for ceiling in ceilings:
        for family, curve in (
            ("paper", fixed[fixed.method == "paper"]),
            ("cell8", fixed[fixed.method == "cell8"]),
            ("best_cell", cell_points),
        ):
            best = _best_at_error(
                curve, "relative_l2_error", ceiling, "actual_total_ms"
            )
            comparison_rows.append({
                "error_ceiling": ceiling,
                "family": family,
                "feasible": best is not None,
                "method": best.method if best is not None else "",
                "method_label": best.method_label if best is not None else "",
                "budget_fraction": (
                    float(best.budget_fraction) if best is not None else float("nan")
                ),
                "actual_total_ms": (
                    float(best.actual_total_ms) if best is not None else float("nan")
                ),
                "achieved_error": (
                    float(best.relative_l2_error) if best is not None else float("nan")
                ),
            })
    quality_comparison = pd.DataFrame(comparison_rows)
    quality_comparison.to_csv(
        output / "quality_constrained_best_cell.csv", index=False
    )

    interpolated_rows = []
    for ceiling in ceilings:
        for method in CELL_METHODS:
            curve = fixed[fixed.method == method]
            interpolated_rows.append({
                "error_ceiling": ceiling,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "interpolated_total_ms": _interpolate_time(
                    curve, "relative_l2_error", "actual_total_ms", ceiling
                ),
            })
    interpolated = pd.DataFrame(interpolated_rows)
    interpolated.to_csv(
        output / "interpolated_quality_constrained_cells.csv", index=False
    )
    interpolated_best = (
        interpolated.dropna(subset=["interpolated_total_ms"])
        .sort_values(["error_ceiling", "interpolated_total_ms"])
        .groupby("error_ceiling", as_index=False).first()
        .sort_values("error_ceiling", ascending=False)
    )

    model_fixed = holdout.groupby(
        ["model", "method", "method_label", "budget_index", "budget_fraction"],
        as_index=False,
    ).agg(
        actual_total_ms=("actual_total_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
    )
    model_best_rows = []
    for model, group in model_fixed.groupby("model"):
        cells = group[group.method.isin(CELL_METHODS)]
        for ceiling in ceilings:
            best = _best_at_error(
                cells, "relative_l2_error", ceiling, "actual_total_ms"
            )
            if best is not None:
                model_best_rows.append({
                    "model": model,
                    "error_ceiling": ceiling,
                    "method": best.method,
                    "method_label": best.method_label,
                    "budget_fraction": best.budget_fraction,
                    "actual_total_ms": best.actual_total_ms,
                    "achieved_error": best.relative_l2_error,
                })
    model_best = pd.DataFrame(model_best_rows)
    model_best.to_csv(output / "per_model_best_cell.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    for method in METHODS:
        curve = fixed[fixed.method == method].sort_values("budget_fraction")
        ax.plot(
            curve.actual_total_ms, curve.relative_l2_error,
            marker="s" if method == "paper" else "o",
            linestyle="--" if method == "paper" else "-",
            linewidth=2.2 if method in ("paper", "cell8") else 1.5,
            markersize=5 if method in ("paper", "cell8") else 3.8,
            alpha=1.0 if method in ("paper", "cell8") else 0.82,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    if len(interpolated_best):
        ax.plot(
            interpolated_best.interpolated_total_ms,
            interpolated_best.error_ceiling,
            marker="*", markersize=7, linewidth=1.0, linestyle=":",
            color="black", zorder=6, label="Interpolated best Cell-C",
        )
        for row in interpolated_best.itertuples(index=False):
            ax.annotate(
                row.method_label,
                (row.interpolated_total_ms, row.error_ceiling),
                xytext=(3, -10), textcoords="offset points", fontsize=6.5,
                color="black",
            )
    cell8 = fixed[fixed.method == "cell8"]
    for row in cell8.itertuples(index=False):
        pct = int(round(100 * row.budget_fraction))
        if pct in (30, 50, 70, 90, 95):
            ax.annotate(
                f"R={pct}%", (row.actual_total_ms, row.relative_l2_error),
                xytext=(4, 4), textcoords="offset points", fontsize=7,
                color=METHOD_COLORS["cell8"],
            )
    ax.set_xlabel("Measured projection-path latency (ms)")
    ax.set_ylabel("Projection relative L2 error (lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.24)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "cell_parameter_frontiers.png", dpi=200)
    fig.savefig(output / "cell_parameter_frontiers.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    for fraction in (0.50, 0.70, 0.90):
        group = cell_points[np.isclose(cell_points.budget_fraction, fraction)].sort_values(
            "cell_count"
        )
        axes[0].plot(
            group.cell_count, group.actual_total_ms, marker="o",
            label=f"R={100*fraction:.0f}%",
        )
        axes[1].plot(
            group.cell_count, group.relative_l2_error, marker="o",
            label=f"R={100*fraction:.0f}%",
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(CELL_COUNTS, [str(value) for value in CELL_COUNTS])
        ax.set_xlabel("Cells per saturation-length tile (C)")
        ax.grid(alpha=0.24)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Measured projection-path latency (ms)")
    axes[1].set_ylabel("Projection relative L2 error (lower is better)")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "cell_parameter_sensitivity.png", dpi=200)
    fig.savefig(output / "cell_parameter_sensitivity.pdf")
    plt.close(fig)

    e2e_path = output / "end_to_end.csv"
    e2e_summary = pd.DataFrame()
    e2e_comparison = pd.DataFrame()
    if e2e_path.is_file():
        e2e = pd.read_csv(e2e_path)
        e2e_summary = e2e.groupby(
            ["method", "method_label", "budget_index", "budget_fraction"],
            as_index=False,
        ).agg(
            cases=("prompt", "size"),
            dense_to_sparse_kl=("dense_to_sparse_kl", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            nll_delta=("nll_delta", "mean"),
            logit_relative_l2_error=("logit_relative_l2_error", "mean"),
            selected_fraction=("selected_fraction_mean", "mean"),
            fallback_calls=("fallback_calls", "sum"),
        )
        latency_lookup = {
            (row.method, int(row.budget_index)): row.actual_total_ms
            for row in fixed.itertuples(index=False)
        }
        e2e_summary["projection_path_ms"] = [
            latency_lookup[(row.method, int(row.budget_index))]
            for row in e2e_summary.itertuples(index=False)
        ]
        e2e_summary.to_csv(output / "end_to_end_summary.csv", index=False)

        e2e_rows = []
        for ceiling in (12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.6):
            for family, curve in (
                ("paper", e2e_summary[e2e_summary.method == "paper"]),
                ("cell8", e2e_summary[e2e_summary.method == "cell8"]),
                ("best_cell", e2e_summary[e2e_summary.method.isin(CELL_METHODS)]),
            ):
                best = _best_at_error(
                    curve.rename(columns={
                        "dense_to_sparse_kl": "metric_error",
                        "projection_path_ms": "metric_time",
                    }),
                    "metric_error", ceiling, "metric_time",
                )
                e2e_rows.append({
                    "kl_ceiling": ceiling,
                    "family": family,
                    "feasible": best is not None,
                    "method": best.method if best is not None else "",
                    "method_label": best.method_label if best is not None else "",
                    "budget_fraction": (
                        float(best.budget_fraction) if best is not None else float("nan")
                    ),
                    "projection_path_ms": (
                        float(best.metric_time) if best is not None else float("nan")
                    ),
                    "achieved_kl": (
                        float(best.metric_error) if best is not None else float("nan")
                    ),
                })
        e2e_comparison = pd.DataFrame(e2e_rows)
        e2e_comparison.to_csv(
            output / "e2e_quality_constrained_best_cell.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(10.8, 6.8))
        for method in METHODS:
            curve = e2e_summary[e2e_summary.method == method].sort_values(
                "budget_fraction"
            )
            ax.plot(
                curve.projection_path_ms, curve.dense_to_sparse_kl,
                marker="s" if method == "paper" else "o",
                linestyle="--" if method == "paper" else "-",
                linewidth=2.2 if method in ("paper", "cell8") else 1.5,
                markersize=5 if method in ("paper", "cell8") else 3.8,
                alpha=1.0 if method in ("paper", "cell8") else 0.82,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Measured mean projection-path latency (ms)")
        ax.set_ylabel("Dense→sparse logit KL (lower is better)")
        ax.invert_yaxis()
        ax.grid(alpha=0.24)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        fig.tight_layout()
        fig.savefig(output / "end_to_end_cell_frontiers.png", dpi=200)
        fig.savefig(output / "end_to_end_cell_frontiers.pdf")
        plt.close(fig)

    metadata = json.loads((output / "metadata.json").read_text())
    best_cell_only = quality_comparison[
        (quality_comparison.family == "best_cell") & quality_comparison.feasible
    ]
    winner_counts = best_cell_only.method_label.value_counts().to_dict()
    interpolated_winner_counts = (
        interpolated_best.method_label.value_counts().to_dict()
    )
    summary = {
        "format": "experiment-36-cell-parameter-sweep-v1",
        "candidate_cases": int(len(candidates)),
        "cell_counts": list(CELL_COUNTS),
        "all_direct": bool((candidates.direct_rate == 1.0).all()),
        "all_deterministic": bool(candidates.deterministic.all()),
        "fallback_cases": int(candidates.fallback_used.sum()),
        "quality_ceiling_winner_counts": winner_counts,
        "interpolated_winner_counts": interpolated_winner_counts,
        "most_frequent_best_cell": (
            max(winner_counts, key=winner_counts.get) if winner_counts else None
        ),
        "most_frequent_interpolated_best_cell": (
            max(interpolated_winner_counts, key=interpolated_winner_counts.get)
            if interpolated_winner_counts else None
        ),
        "e2e_cases": int(pd.read_csv(e2e_path).shape[0]) if e2e_path.is_file() else 0,
        "gpu": metadata["gpu"],
        "flash": metadata["flash"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        args, fixed, quality_comparison, interpolated, interpolated_best,
        model_best, e2e_comparison, summary,
    )


def write_report(args, fixed, comparison, interpolated, interpolated_best,
                 model_best, e2e_comparison, summary):
    path = args.report_output or HERE / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 36 보고서: Cell-C parameter sweep", "",
        "Cell-8의 숫자 `8`을 고정하지 않고 `C={1,2,4,8,16,32}`를 실제 노트북에서 "
        "측정했다. 각 Cell-C는 saturation 길이를 C개 cell로 나누며, 모든 방법은 "
        "동일하게 `R=10%..95%`를 5% 간격으로 sweep했다. 최종 비교는 특정 R이 "
        "아니라 측정된 quality-latency frontier다.", "",
        "## Projection quality-constrained 결과", "",
        "| error ceiling | Paper | Cell-8 | best measured Cell-C | R | vs Cell-8 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for ceiling in sorted(comparison.error_ceiling.unique(), reverse=True):
        group = comparison[comparison.error_ceiling == ceiling].set_index("family")
        paper = group.loc["paper"]
        cell8 = group.loc["cell8"]
        cell = group.loc["best_cell"]
        if bool(paper.feasible) and bool(cell8.feasible) and bool(cell.feasible):
            saving = 100.0 * (
                cell8.actual_total_ms - cell.actual_total_ms
            ) / cell8.actual_total_ms
            lines.append(
                f"| {ceiling:.2f} | {paper.actual_total_ms:.3f} ms "
                f"| {cell8.actual_total_ms:.3f} ms "
                f"| {cell.method_label} {cell.actual_total_ms:.3f} ms "
                f"| {100*cell.budget_fraction:.0f}% "
                f"| {saving:+.1f}% |"
            )
    winner_text = ", ".join(
        f"{name}: {count}" for name, count in
        summary["quality_ceiling_winner_counts"].items()
    )
    lines.extend([
        "", "Quality ceiling별 최적 Cell-C 횟수는 " + winner_text + ".", "",
        "## 동일 error 보간 진단", "",
        "5% R grid의 빈틈을 승리로 오인하지 않도록 각 Cell-C 곡선을 동일 error에서 "
        "선형 보간했다. 이 값은 실측점이 아니라 곡선 사이 진단값이다.", "",
        "| error | best Cell-C | interpolated time | Cell-8 time | vs Cell-8 |",
        "|---:|---:|---:|---:|---:|",
    ])
    interpolation_lookup = interpolated.set_index(["error_ceiling", "method"])
    for row in interpolated_best.itertuples(index=False):
        cell8_ms = float(interpolation_lookup.loc[
            (row.error_ceiling, "cell8"), "interpolated_total_ms"
        ])
        saving = 100.0 * (
            cell8_ms - row.interpolated_total_ms
        ) / cell8_ms
        lines.append(
            f"| {row.error_ceiling:.2f} | {row.method_label} "
            f"| {row.interpolated_total_ms:.3f} ms | {cell8_ms:.3f} ms "
            f"| {saving:+.1f}% |"
        )
    interpolated_text = ", ".join(
        f"{name}: {count}" for name, count in
        summary["interpolated_winner_counts"].items()
    )
    lines.extend([
        "", "보간 기준 winner 횟수는 " + interpolated_text + ". 따라서 Cell-8은 "
        "강한 baseline이지만 이 노트북의 단일 최적값은 아니다.", "",
        "## 모델별 선택", "",
        "| model | error ceiling | best Cell-C | R | time | error |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in model_best[
        model_best.error_ceiling.isin([0.42, 0.25, 0.15])
    ].itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.error_ceiling:.2f} | {row.method_label} "
            f"| {100*row.budget_fraction:.0f}% | {row.actual_total_ms:.3f} ms "
            f"| {row.achieved_error:.4f} |"
        )
    if len(e2e_comparison):
        lines.extend([
            "", "## End-to-end logit KL", "",
            "| KL ceiling | Paper | Cell-8 | best Cell-C | R | achieved KL |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for ceiling in sorted(e2e_comparison.kl_ceiling.unique(), reverse=True):
            group = e2e_comparison[
                e2e_comparison.kl_ceiling == ceiling
            ].set_index("family")
            paper = group.loc["paper"]
            cell8 = group.loc["cell8"]
            cell = group.loc["best_cell"]
            paper_text = (
                f"{paper.projection_path_ms:.3f} ms" if paper.feasible else "—"
            )
            cell8_text = (
                f"{cell8.projection_path_ms:.3f} ms" if cell8.feasible else "—"
            )
            cell_text = (
                f"{cell.method_label} {cell.projection_path_ms:.3f} ms"
                if cell.feasible else "—"
            )
            lines.append(
                f"| {ceiling:g} | {paper_text} | {cell8_text} | {cell_text} "
                f"| {100*cell.budget_fraction:.0f}% "
                f"| {cell.achieved_kl:.3f} |"
                if cell.feasible else
                f"| {ceiling:g} | {paper_text} | {cell8_text} | — | — | — |"
            )
    lines.extend([
        "", "## 측정 범위", "",
        f"- 후보 case: `{summary['candidate_cases']}`, end-to-end case: "
        f"`{summary['e2e_cases']}`.",
        f"- O_DIRECT fallback: `{summary['fallback_cases']}`건.",
        "- latency는 selector + O_DIRECT/upload + activation gather + compact GEMM의 "
        "projection 경로이며 전체 LLM wall-clock은 아니다.",
        "- 두 error 그래프 모두 작은 error가 위에 오도록 축을 뒤집었다.", "",
        "![Cell parameter frontiers](results_laptop/cell_parameter_frontiers.png)", "",
        "![Cell parameter sensitivity](results_laptop/cell_parameter_sensitivity.png)", "",
        "![End-to-end Cell frontiers](results_laptop/end_to_end_cell_frontiers.png)", "",
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
    selector = CellSweepSelector(lookup, args.saturation_kib)
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
            max_required,
            int(config.intermediate_size) * int(config.hidden_size) * 2,
        )
    if args.io_blob.stat().st_size < max_required:
        raise SystemExit(f"I/O blob needs at least {max_required} bytes")

    # The reused hook expects this field only to record an unused predictor probe.
    args.quality_targets = [0.5]
    all_candidates, all_selectors, all_io, all_gemm = [], [], [], []
    all_e2e, models = [], []
    for model_key in args.models:
        print(f"loading and measuring {MODEL_SPECS[model_key]['label']}", flush=True)
        result = EXP35.run_model(
            model_key, selector, native_reader, prompts, args, vlmflash
        )
        (
            model, tokenizer, candidates, selectors, io_rows, gemm_rows,
            _features, metadata,
        ) = result
        all_candidates.extend(candidates)
        all_selectors.extend(selectors)
        all_io.extend(io_rows)
        all_gemm.extend(gemm_rows)
        models.append(metadata)
        pd.DataFrame(all_candidates).to_csv(
            args.output_dir / "candidates.partial.csv", index=False
        )
        if not args.skip_end_to_end and any(
            prompt["split"] == "holdout" for prompt in prompts
        ):
            all_e2e.extend(run_end_to_end(
                model, tokenizer, model_key, selector, prompts, args, vlmflash
            ))
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(all_candidates).to_csv(
        args.output_dir / "candidates.csv", index=False
    )
    pd.DataFrame(all_selectors).to_csv(
        args.output_dir / "selector_samples.csv", index=False
    )
    pd.DataFrame(all_io).to_csv(
        args.output_dir / "io_aggregates.csv", index=False
    )
    pd.DataFrame(all_gemm).to_csv(
        args.output_dir / "gemm_aggregates.csv", index=False
    )
    if all_e2e:
        pd.DataFrame(all_e2e).to_csv(
            args.output_dir / "end_to_end.csv", index=False
        )
    partial = args.output_dir / "candidates.partial.csv"
    if partial.exists():
        partial.unlink()

    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-36-cell-parameter-sweep-v1",
        "models": models,
        "prompts": [
            {key: value for key, value in prompt.items() if key != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "cell_counts": list(CELL_COUNTS),
        "budgets": args.budgets,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions,
        "io_warmup": args.io_warmup,
        "gemm_repetitions": args.gemm_repetitions,
        "gemm_warmup": args.gemm_warmup,
        "io_threads": args.io_threads,
        "io_max_read_kib": args.io_max_read_kib,
        "profile": str(args.profile),
        "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "flash": profile_meta.get("flash", "unknown"),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    analyze(args)


if __name__ == "__main__":
    main()
