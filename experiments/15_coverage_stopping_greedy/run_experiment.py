#!/usr/bin/env python3
"""Experiment 15: replace Paper greedy's fixed-R stop with coverage stopping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_14 = (
    PROJECT_ROOT / "experiments" / "14_dense_lookup_validation" / "run_experiment.py"
)
DEFAULT_SOURCE = PROJECT_ROOT / "experiments" / "14_dense_lookup_validation" / "results"


def load_experiment_14():
    spec = importlib.util.spec_from_file_location("experiment_14_for_15", EXPERIMENT_14)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 14 from {EXPERIMENT_14}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP14 = load_experiment_14()
EXP13 = EXP14.EXP13
BASE = EXP14.BASE
EXP2 = EXP14.EXP2
EXP11 = EXP14.EXP11
CONFIG = EXP14.CONFIG
SPATIAL_MODES = EXP14.SPATIAL_MODES
SPATIAL_LABELS = EXP14.SPATIAL_LABELS
QUANT_COLOR = EXP14.QUANT_COLOR
COVERAGE_PAPER_COLOR = "#2A9D8F"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--tail-fit-start-kib", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261501)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--input-limit", type=int, default=None,
        help="development-only number of spatial inputs",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def timed_call(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return result, (time.perf_counter_ns() - start) / 1e6


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


class CoverageStoppingPaperOracle:
    """Paper candidate ranking with a cumulative-importance stopping rule."""

    def __init__(self, values: np.ndarray, row_size_kib: float, table,
                 params) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.n = len(self.values)
        start_kib, end_kib, step_kib, jump_cap_kib = params.resolve(table)
        size_start = max(1, int(start_kib / row_size_kib))
        size_end = max(1, int(end_kib / row_size_kib))
        size_step = max(1, int(step_kib / row_size_kib))
        jump_cap = max(1, int(jump_cap_kib / row_size_kib))
        windows = []
        for size in range(size_start, size_end, size_step):
            if size > self.n:
                break
            windows.append((size, table.read_ms(size * row_size_kib)))
        if not windows:
            raise RuntimeError("Paper candidate sweep produced no windows")

        v = torch.from_numpy(self.values.astype(np.float32))
        cumsum = torch.cat([
            torch.zeros(1, dtype=torch.float32), torch.cumsum(v, dim=0)
        ])
        scores, starts, sizes = [], [], []
        for size, cost in windows:
            stride = min(size, jump_cap)
            start = torch.arange(0, self.n - size + 1, stride)
            scores.append(
                (cumsum[start + size] - cumsum[start]) / np.float32(cost)
            )
            starts.append(start)
            sizes.append(torch.full_like(start, size))
        score = torch.cat(scores)
        start = torch.cat(starts)
        size = torch.cat(sizes)
        order = torch.argsort(score, descending=True, stable=True)

        occupied = np.zeros(self.n, dtype=bool)
        accepted_start = []
        accepted_size = []
        cumulative_importance = []
        total = 0.0
        prefix = np.r_[0.0, np.cumsum(self.values)]
        for candidate_start, candidate_size in zip(
            start[order].numpy(), size[order].numpy()
        ):
            candidate_start = int(candidate_start)
            candidate_size = int(candidate_size)
            stop = candidate_start + candidate_size
            if occupied[candidate_start:stop].any():
                continue
            occupied[candidate_start:stop] = True
            accepted_start.append(candidate_start)
            accepted_size.append(candidate_size)
            total += float(prefix[stop] - prefix[candidate_start])
            cumulative_importance.append(total)
        self.starts = np.asarray(accepted_start, dtype=np.int32)
        self.sizes = np.asarray(accepted_size, dtype=np.int32)
        self.cumulative_importance = np.asarray(
            cumulative_importance, dtype=np.float64
        )

    def solve(self, bound: float) -> np.ndarray:
        if not 0 < bound <= float(self.values.sum()) + 1e-12:
            raise ValueError("coverage bound must lie in (0, total importance]")
        count = int(np.searchsorted(
            self.cumulative_importance, bound - 1e-14, side="left"
        ) + 1)
        if count > len(self.starts):
            raise RuntimeError("Paper coverage prefix cannot reach the target")
        mask = np.zeros(self.n, dtype=bool)
        for start, size in zip(self.starts[:count], self.sizes[:count]):
            mask[start:start + size] = True
        return mask


def direct_coverage_greedy(values: np.ndarray, bound: float, row_size_kib: float,
                           table, params) -> np.ndarray:
    """Independent direct implementation used only by the small self-check."""
    oracle = CoverageStoppingPaperOracle(values, row_size_kib, table, params)
    mask = np.zeros(len(values), dtype=bool)
    total = 0.0
    for start, size in zip(oracle.starts, oracle.sizes):
        mask[start:start + size] = True
        total += float(values[start:start + size].sum())
        if total >= bound - 1e-14:
            break
    return mask


def self_check(table, row_size_kib: float, params) -> None:
    rng = np.random.default_rng(1515)
    for _ in range(5):
        values = rng.lognormal(size=257)
        values /= values.sum()
        oracle = CoverageStoppingPaperOracle(values, row_size_kib, table, params)
        previous_rows = 0
        reachable = float(oracle.cumulative_importance[-1])
        for fraction in (0.05, 0.10, 0.30, 0.70, 0.95):
            bound = fraction * reachable
            actual = oracle.solve(bound)
            expected = direct_coverage_greedy(
                values, bound, row_size_kib, table, params
            )
            if not np.array_equal(actual, expected):
                raise RuntimeError("coverage-prefix oracle differs from direct greedy")
            importance = float(values[actual].sum())
            if importance < bound - 1e-12:
                raise RuntimeError("coverage-stopping Paper missed its target")
            rows = int(actual.sum())
            if rows < previous_rows:
                raise RuntimeError("coverage-prefix rows are not monotone")
            previous_rows = rows


def regenerate_inputs(source_summary: dict):
    n = int(source_summary["n"])
    rng = np.random.default_rng(int(source_summary["seed"]))
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    trials = int(source_summary["trials_per_cv"])
    cv_targets = [float(value) for value in source_summary["cv_targets"]]
    for trial in range(trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
            variants = EXP2.spatial_variants(base_values, rng, hotness, 0.95, 0.75)
            for mode, values in variants.items():
                yield trial, target_cv, mode, values


def compact_fields(prefix: str, metrics: dict) -> dict:
    keys = (
        "importance", "rows", "chunks", "run_length_median",
        "rows_beyond_profile_fraction", "two_line_ms", "released_ms",
    )
    return {f"{prefix}_{key}": metrics[key] for key in keys}


def clustered_result(frame: pd.DataFrame, field: str, replicates: int,
                     rng: np.random.Generator) -> dict:
    clusters = (
        frame.groupby(["trial", "target_cv"], sort=True)[field].mean().to_numpy()
    )
    low, high = EXP14.bootstrap_mean_ci(clusters, replicates, rng)
    return {
        "estimate": float(clusters.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "clusters": int(len(clusters)),
    }


def grouped_results(frame: pd.DataFrame, group_key: str, field: str,
                    args, seed_offset: int) -> dict:
    output = {}
    for offset, (key, group) in enumerate(frame.groupby(group_key, sort=True)):
        output[str(key)] = clustered_result(
            group, field, args.bootstrap_replicates,
            np.random.default_rng(args.bootstrap_seed + seed_offset + offset),
        )
    return output


def build_summary(args, source_summary: dict, frame: pd.DataFrame,
                  inputs: pd.DataFrame) -> dict:
    metrics = {}
    for method in ("coverage_paper", "quant"):
        metrics[method] = {
            "saving_vs_paper_lookup_pct": {
                **summarize(frame[f"{method}_saving_vs_paper_lookup_pct"]),
                "cluster_mean_ci95": clustered_result(
                    frame, f"{method}_saving_vs_paper_lookup_pct",
                    args.bootstrap_replicates,
                    np.random.default_rng(args.bootstrap_seed),
                ),
            },
            "strict_win_rate_vs_paper": {
                "case_rate": float(frame[f"{method}_win_vs_paper"].mean()),
                "cluster_mean_ci95": clustered_result(
                    frame, f"{method}_win_vs_paper", args.bootstrap_replicates,
                    np.random.default_rng(args.bootstrap_seed + 1),
                ),
            },
            "importance_overshoot": summarize(
                frame[f"{method}_importance"] - frame.paper_importance
            ),
            "relative_importance_overshoot_pct": summarize(
                100.0 * (
                    frame[f"{method}_importance"] / frame.paper_importance - 1.0
                )
            ),
            "rows": summarize(frame[f"{method}_rows"]),
            "chunks": summarize(frame[f"{method}_chunks"]),
        }
    return {
        "format": "experiment-15-coverage-stopping-greedy-v1",
        "source_experiment": "14_dense_lookup_validation",
        "source_results": str(args.source_results.resolve()),
        "source_sha256": {
            name: file_sha256(args.source_results / name)
            for name in ("summary.json", "paired_trials.csv")
        },
        "n": int(source_summary["n"]),
        "trials_per_cv": int(source_summary["trials_per_cv"]),
        "cv_targets": source_summary["cv_targets"],
        "budget_fractions": source_summary["budget_fractions"],
        "cases": {
            "clusters": int(frame[["trial", "target_cv"]].drop_duplicates().shape[0]),
            "spatial_inputs": int(inputs.shape[0]),
            "paired_budget_cases": int(frame.shape[0]),
        },
        "modified_method": (
            "Paper candidate windows, lookup utility scores, stable ordering, and "
            "non-overlap rule; stop at cumulative importance >= original Paper importance"
        ),
        "methods": metrics,
        "original_paper_structure": {
            "rows": summarize(frame.paper_rows),
            "chunks": summarize(frame.paper_chunks),
        },
        "coverage_paper_vs_quant": {
            "saving_pct": {
                **summarize(frame.coverage_paper_saving_vs_quant_lookup_pct),
                "cluster_mean_ci95": clustered_result(
                    frame, "coverage_paper_saving_vs_quant_lookup_pct",
                    args.bootstrap_replicates,
                    np.random.default_rng(args.bootstrap_seed + 2),
                ),
            },
            "coverage_paper_strict_win_rate": float(
                frame.coverage_paper_win_vs_quant.mean()
            ),
        },
        "by_budget_fraction": {
            method: grouped_results(
                frame, "budget_fraction", f"{method}_saving_vs_paper_lookup_pct",
                args, 100 if method == "coverage_paper" else 200,
            )
            for method in ("coverage_paper", "quant")
        },
        "by_spatial_mode": {
            method: grouped_results(
                frame, "spatial_mode", f"{method}_saving_vs_paper_lookup_pct",
                args, 300 if method == "coverage_paper" else 400,
            )
            for method in ("coverage_paper", "quant")
        },
        "runtime_ms": {
            "coverage_paper_build": summarize(inputs.coverage_paper_build_ms),
            "coverage_paper_query": summarize(frame.coverage_paper_query_ms),
            "paper_query_from_experiment_14": summarize(frame.paper_runtime_ms),
            "quant_build_from_experiment_14": summarize(inputs.quant_build_ms),
        },
    }


def configure_plot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_saving_ci(frame: pd.DataFrame, path: Path, args) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True,
                             constrained_layout=True)
    methods = (
        ("coverage_paper_saving_vs_paper_lookup_pct", "Paper + coverage stop",
         COVERAGE_PAPER_COLOR, 1000),
        ("quant_saving_vs_paper_lookup_pct", "Quant (high-q)", QUANT_COLOR, 2000),
    )
    for mode_index, (ax, mode) in enumerate(zip(axes, SPATIAL_MODES)):
        selected = frame[frame.spatial_mode == mode]
        for field, label, color, seed_offset in methods:
            xs, means, lows, highs = [], [], [], []
            for budget_index, (budget, group) in enumerate(
                selected.groupby("budget_fraction", sort=True)
            ):
                clusters = (
                    group.groupby(["trial", "target_cv"])[field].mean().to_numpy()
                )
                rng = np.random.default_rng(
                    args.bootstrap_seed + seed_offset + mode_index * 100 + budget_index
                )
                low, high = EXP14.bootstrap_mean_ci(
                    clusters, args.bootstrap_replicates, rng
                )
                xs.append(float(budget))
                means.append(float(clusters.mean()))
                lows.append(low)
                highs.append(high)
            ax.plot(xs, means, color=color, marker="o", label=label)
            ax.fill_between(xs, lows, highs, color=color, alpha=0.16)
        ax.axhline(0.0, color="#334155", linestyle="--", linewidth=1.1)
        ax.set_xlabel("Original Paper fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Lookup-latency saving vs original Paper (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_frontiers(frame: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    display_cvs = (1.25, 3.30, 4.55)
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.0), constrained_layout=True)
    methods = (
        ("coverage_paper", "Paper + coverage stop", COVERAGE_PAPER_COLOR, "s"),
        ("quant", "Quant (high-q)", QUANT_COLOR, "P"),
        ("paper", "Original Paper", BASE.PLOT_COLORS["greedy"], "o"),
    )
    for row_index, cv in enumerate(display_cvs):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            selected = frame[
                np.isclose(frame.target_cv, cv) & (frame.spatial_mode == mode)
            ]
            grouped = selected.groupby("budget_fraction", sort=True)
            for prefix, label, color, marker in methods:
                curve = grouped[[
                    f"{prefix}_released_ms", f"{prefix}_importance"
                ]].mean()
                ax.plot(
                    curve[f"{prefix}_released_ms"], curve[f"{prefix}_importance"],
                    color=color, marker=marker, label=label,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Released lookup latency (ms, log scale)")
            ax.set_ylabel("Retained importance")
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={cv:g}")
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_direct_ratio(frame: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True,
                             constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = frame[frame.spatial_mode == mode]
        grouped = selected.groupby("budget_fraction", sort=True)
        x = np.asarray(sorted(selected.budget_fraction.unique()))
        paper = np.ones(len(x))
        coverage_paper = grouped.coverage_paper_vs_paper_lookup_ratio.median().to_numpy()
        quant = grouped.quant_vs_paper_lookup_ratio.median().to_numpy()
        ax.plot(x, paper, color=BASE.PLOT_COLORS["greedy"], marker="o",
                label="Original Paper")
        ax.plot(x, coverage_paper, color=COVERAGE_PAPER_COLOR, marker="s",
                label="Paper + coverage stop")
        ax.plot(x, quant, color=QUANT_COLOR, marker="P", label="Quant (high-q)")
        ax.axhline(1.0, color="#334155", linestyle="--", linewidth=1.1)
        ax.set_xlabel("Original Paper fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Method lookup latency / original Paper latency (median)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_overshoot(frame: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True,
                             constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = frame[frame.spatial_mode == mode].copy()
        selected["coverage_paper_overshoot_pct"] = 100.0 * (
            selected.coverage_paper_importance / selected.paper_importance - 1.0
        )
        selected["quant_overshoot_pct"] = 100.0 * (
            selected.quant_importance / selected.paper_importance - 1.0
        )
        grouped = selected.groupby("budget_fraction", sort=True)
        x = np.asarray(sorted(selected.budget_fraction.unique()))
        ax.plot(
            x, grouped.coverage_paper_overshoot_pct.mean().to_numpy(),
            color=COVERAGE_PAPER_COLOR, marker="s", label="Paper + coverage stop",
        )
        ax.plot(
            x, grouped.quant_overshoot_pct.mean().to_numpy(),
            color=QUANT_COLOR, marker="P", label="Quant (high-q)",
        )
        ax.set_xlabel("Original Paper fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Importance overshoot above Paper target (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(args, source_summary: dict, frame: pd.DataFrame,
            inputs: pd.DataFrame) -> dict:
    summary = build_summary(args, source_summary, frame, inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_saving_ci(frame, args.output_dir / "latency_saving_ci.png", args)
    plot_frontiers(frame, args.output_dir / "importance_latency_frontiers.png")
    plot_direct_ratio(frame, args.output_dir / "latency_ratio.png")
    plot_overshoot(frame, args.output_dir / "importance_overshoot.png")
    return summary


def main() -> None:
    args = parse_args()
    required = ("summary.json", "paired_trials.csv", "input_trials.csv")
    for name in required:
        if not (args.source_results / name).is_file():
            raise SystemExit(f"missing Experiment 14 result: {args.source_results / name}")
    source_summary = json.loads((args.source_results / "summary.json").read_text())
    if source_summary.get("format") != "experiment-14-dense-lookup-validation-v1":
        raise SystemExit("--source-results is not an Experiment 14 result directory")
    if args.bootstrap_replicates < 1000:
        raise SystemExit("--bootstrap-replicates must be at least 1000")

    trials_path = args.output_dir / "paired_trials.csv"
    inputs_path = args.output_dir / "input_trials.csv"
    if args.analyze_only:
        if not trials_path.is_file() or not inputs_path.is_file():
            raise SystemExit("--analyze-only requires saved Experiment 15 CSVs")
        frame = pd.read_csv(trials_path)
        inputs = pd.read_csv(inputs_path)
        summary = analyze(args, source_summary, frame, inputs)
        print(json.dumps(summary["methods"], indent=2))
        return

    source = pd.read_csv(args.source_results / "paired_trials.csv")
    source_inputs = pd.read_csv(args.source_results / "input_trials.csv")
    groups = {
        (int(trial), round(float(cv), 10), mode): group.sort_values("budget_fraction")
        for (trial, cv, mode), group in source.groupby(
            ["trial", "target_cv", "spatial_mode"]
        )
    }
    input_groups = {
        (int(row.trial), round(float(row.target_cv), 10), row.spatial_mode): row
        for row in source_inputs.itertuples(index=False)
    }
    n = int(source_summary["n"])
    row_size_kib = float(CONFIG["cols"]) * 2.0 / 1024.0
    table = BASE.LatencyTable.load(args.profile)
    affine_a, affine_c, _ = BASE.affine_fit(table, row_size_kib)
    affine = (affine_a, affine_c)
    model = EXP13.fit_continuous_two_line(table, row_size_kib, args.saturation_kib)
    policies = EXP11.LatencyPolicies(table, row_size_kib, args.tail_fit_start_kib)
    params = BASE.ChunkParams(
        start_kb=CONFIG["start_kib"], jump_cap_kb=CONFIG["jump_cap_kib"]
    )
    self_check(table, row_size_kib, params)

    output_rows: list[dict] = []
    input_rows: list[dict] = []
    total_inputs = int(source_summary["cases"]["spatial_inputs"])
    completed = 0
    for trial, target_cv, mode, values in regenerate_inputs(source_summary):
        source_group = groups[(trial, round(target_cv, 10), mode)]
        oracle, build_ms = timed_call(
            CoverageStoppingPaperOracle, values, row_size_kib, table, params
        )
        source_input = input_groups[(trial, round(target_cv, 10), mode)]
        input_rows.append({
            "trial": trial,
            "target_cv": target_cv,
            "spatial_mode": mode,
            "coverage_paper_build_ms": build_ms,
            "accepted_greedy_windows": len(oracle.starts),
            "max_prefix_importance": float(oracle.cumulative_importance[-1]),
            "quant_build_ms": float(source_input.quant_build_ms),
        })
        for source_row in source_group.to_dict("records"):
            bound = float(source_row["paper_importance"])
            mask, query_ms = timed_call(oracle.solve, bound)
            coverage_paper = EXP13.mask_metrics(
                mask, values, model, affine, row_size_kib, policies
            )
            if coverage_paper["importance"] < bound - 1e-12:
                raise RuntimeError("coverage-stopping Paper missed target")
            paper_ms = float(source_row["paper_released_ms"])
            quant_ms = float(source_row["quant_released_ms"])
            coverage_ms = coverage_paper["released_ms"]
            output = dict(source_row)
            output.update(compact_fields("coverage_paper", coverage_paper))
            output.update({
                "quant_win_vs_paper": float(source_row["quant_win"]),
                "coverage_paper_query_ms": query_ms,
                "coverage_paper_saving_vs_paper_lookup_pct": 100.0 * (
                    1.0 - coverage_ms / paper_ms
                ),
                "coverage_paper_win_vs_paper": float(coverage_ms < paper_ms - 1e-12),
                "coverage_paper_vs_paper_lookup_ratio": coverage_ms / paper_ms,
                "quant_vs_paper_lookup_ratio": quant_ms / paper_ms,
                "coverage_paper_saving_vs_quant_lookup_pct": 100.0 * (
                    1.0 - coverage_ms / quant_ms
                ),
                "coverage_paper_win_vs_quant": float(coverage_ms < quant_ms - 1e-12),
            })
            output_rows.append(output)
        completed += 1
        if completed == 1 or completed % 25 == 0 or completed == total_inputs:
            print(
                f"completed {completed}/{total_inputs} spatial inputs "
                f"({build_ms:.1f} ms coverage-prefix build)", flush=True,
            )
        if args.input_limit is not None and completed >= args.input_limit:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trials_path, output_rows)
    write_csv(inputs_path, input_rows)
    frame = pd.DataFrame(output_rows)
    inputs = pd.DataFrame(input_rows)
    summary = analyze(args, source_summary, frame, inputs)
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
