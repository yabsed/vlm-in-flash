#!/usr/bin/env python3
"""Experiment 14: dense validation of high-q Quant under released lookup cost."""

from __future__ import annotations

import argparse
import csv
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
EXPERIMENT_13 = (
    PROJECT_ROOT / "experiments" / "13_two_line_lagrangian" / "run_experiment.py"
)


def load_experiment_13():
    spec = importlib.util.spec_from_file_location("experiment_13_for_14", EXPERIMENT_13)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 13 from {EXPERIMENT_13}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP13 = load_experiment_13()
BASE = EXP13.BASE
EXP2 = EXP13.EXP2
EXP11 = EXP13.EXP11
CONFIG = EXP13.CONFIG
SPATIAL_MODES = EXP13.SPATIAL_MODES
SPATIAL_LABELS = EXP13.SPATIAL_LABELS
QUANT_COLOR = EXP13.LAGRANGIAN_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--cv-targets", type=float, nargs="+",
        default=[1.07, 1.25, 1.44, 2.48, 3.30, 4.55],
    )
    parser.add_argument(
        "--budget-fractions", type=float, nargs="+",
        default=[index / 20 for index in range(1, 20)],
    )
    parser.add_argument("--q", type=int, default=131072)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261401)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--tail-fit-start-kib", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument(
        "--input-limit", type=int, default=None,
        help="development-only number of spatial inputs",
    )
    return parser.parse_args()


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


def compact_fields(prefix: str, metrics: dict) -> dict:
    keys = (
        "importance", "rows", "chunks", "run_length_median",
        "rows_beyond_profile_fraction", "two_line_ms", "released_ms",
    )
    return {f"{prefix}_{key}": metrics[key] for key in keys}


def top_masks(values: np.ndarray, order: np.ndarray, prefix: np.ndarray,
              fixed_count: int, bound: float) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.zeros(len(values), dtype=bool)
    fixed[order[:fixed_count]] = True
    matched_count = int(np.searchsorted(prefix, bound - 1e-14, side="left") + 1)
    matched = np.zeros(len(values), dtype=bool)
    matched[order[:min(matched_count, len(values))]] = True
    return fixed, matched


def fast_candidate_index(oracle, bound: float) -> int:
    """First feasible supported point; cost and importance are monotone in lambda."""
    index = int(np.searchsorted(oracle.importance, bound - 1e-14, side="left"))
    if index >= oracle.q:
        return -1
    if oracle.full_cost < oracle.costs[index] - 1e-14:
        return -1
    return index


def solve_supported_index(oracle, index: int) -> dict:
    if index < 0:
        return {
            "mask": np.ones(len(oracle.values), dtype=bool),
            "lambda_index": -1,
            "lambda": math.inf,
            "importance": float(oracle.values.sum()),
            "two_line_ms": float(oracle.full_cost),
        }
    result = EXP13._solve_lambda_mask(
        oracle.values, oracle.multipliers[index],
        oracle.model["a_ms"], oracle.model["c1_ms_per_row"],
        oracle.model["c2_ms_per_row"], oracle.model["saturation_rows"],
        oracle.model["short_max_rows"],
    )
    return {
        "mask": result[0],
        "lambda_index": index,
        "lambda": float(oracle.multipliers[index]),
        "importance": float(result[2]),
        "two_line_ms": float(result[3]),
    }


def validate_monotone_oracle(oracle, bounds: list[float]) -> None:
    importance_tolerance = 2e-12
    cost_tolerance = 2e-12
    if float(np.min(np.diff(oracle.importance))) < -importance_tolerance:
        raise RuntimeError("lambda-grid importance is not monotone")
    if float(np.min(np.diff(oracle.costs))) < -cost_tolerance:
        raise RuntimeError("lambda-grid cost is not monotone")
    for bound in bounds:
        fast = fast_candidate_index(oracle, bound)
        reference = oracle.candidate_index(bound)
        if reference < 0 or oracle.full_cost < oracle.costs[reference] - 1e-14:
            reference = -1
        if fast < 0 and reference < 0:
            continue
        if fast < 0 or reference < 0:
            raise RuntimeError("fast and reference candidate selection disagree")
        for values, label in (
            (oracle.importance, "importance"), (oracle.costs, "cost")
        ):
            if not math.isclose(float(values[fast]), float(values[reference]),
                                rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(f"fast candidate differs in {label}")


def bootstrap_mean_ci(cluster_values: np.ndarray, replicates: int,
                      rng: np.random.Generator) -> tuple[float, float]:
    cluster_values = np.asarray(cluster_values, dtype=np.float64)
    draws = np.empty(replicates, dtype=np.float64)
    batch = 1000
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        indices = rng.integers(
            0, len(cluster_values), size=(stop - start, len(cluster_values))
        )
        draws[start:stop] = cluster_values[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def clustered_result(frame: pd.DataFrame, field: str, replicates: int,
                     rng: np.random.Generator) -> dict:
    cluster_values = (
        frame.groupby(["trial", "target_cv"], sort=True)[field].mean().to_numpy()
    )
    low, high = bootstrap_mean_ci(cluster_values, replicates, rng)
    return {
        "estimate": float(cluster_values.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "clusters": int(len(cluster_values)),
    }


def grouped_results(frame: pd.DataFrame, group_key: str, field: str,
                    replicates: int, seed: int) -> dict:
    output = {}
    for offset, (key, group) in enumerate(frame.groupby(group_key, sort=True)):
        rng = np.random.default_rng(seed + offset)
        output[str(key)] = clustered_result(group, field, replicates, rng)
    return output


def build_summary(args: argparse.Namespace, frame: pd.DataFrame,
                  inputs: pd.DataFrame, model: dict, policies) -> dict:
    rng = np.random.default_rng(args.bootstrap_seed)
    saving = clustered_result(
        frame, "quant_saving_vs_paper_lookup_pct", args.bootstrap_replicates, rng
    )
    win = clustered_result(frame, "quant_win", args.bootstrap_replicates, rng)
    return {
        "format": "experiment-14-dense-lookup-validation-v1",
        "n": int(CONFIG["n"]),
        "model": CONFIG["model"],
        "shape": CONFIG["shape"],
        "seed": args.seed,
        "trials_per_cv": args.trials,
        "cv_targets": sorted(frame.target_cv.unique().tolist()),
        "spatial_modes": list(SPATIAL_MODES),
        "budget_fractions": sorted(frame.budget_fraction.unique().tolist()),
        "q": args.q,
        "cases": {
            "independent_trial_cv_clusters": int(
                frame[["trial", "target_cv"]].drop_duplicates().shape[0]
            ),
            "spatial_inputs": int(inputs.shape[0]),
            "paired_budget_cases": int(frame.shape[0]),
        },
        "primary_latency_assumption": {
            "profile": args.profile,
            "measured_max_kib": policies.max_kib,
            "tail": (
                "constant throughput after saturation, implemented as endpoint-"
                "proportional scaling beyond the final released lookup entry"
            ),
        },
        "two_line_model": model,
        "bootstrap": {
            "unit": "(trial, target_cv); ordering and R budgets remain paired",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "quant_saving_vs_paper_lookup_pct": {
            **summarize(frame.quant_saving_vs_paper_lookup_pct),
            "cluster_mean_ci95": saving,
        },
        "paper_extra_vs_quant_lookup_pct": summarize(
            frame.paper_extra_vs_quant_lookup_pct
        ),
        "quant_strict_win_rate": {
            "case_rate": float(frame.quant_win.mean()),
            "cluster_mean_ci95": win,
        },
        "quant_coverage_overshoot": summarize(
            frame.quant_importance - frame.paper_importance
        ),
        "paper_matched_chunk_count": {
            method: summarize(frame[f"{method}_chunks"])
            for method in ("quant", "paper", "top_matched")
        },
        "by_spatial_mode": grouped_results(
            frame, "spatial_mode", "quant_saving_vs_paper_lookup_pct",
            args.bootstrap_replicates, args.bootstrap_seed + 100,
        ),
        "by_cv": grouped_results(
            frame, "target_cv", "quant_saving_vs_paper_lookup_pct",
            args.bootstrap_replicates, args.bootstrap_seed + 200,
        ),
        "by_budget_fraction": grouped_results(
            frame, "budget_fraction", "quant_saving_vs_paper_lookup_pct",
            args.bootstrap_replicates, args.bootstrap_seed + 300,
        ),
        "runtime_ms": {
            "quant_grid_build_per_spatial_input": summarize(inputs.quant_build_ms),
            "paper_query": summarize(frame.paper_runtime_ms),
        },
        "unique_supported_solutions": summarize(inputs.unique_supported_solutions),
    }


def configure_plot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_method_grid(frame: pd.DataFrame, path: Path, x_fields: dict,
                     y_fields: dict, xlabel: str, ylabel: str,
                     x_log: bool = False) -> None:
    plt = configure_plot()
    display_cvs = (1.25, 3.30, 4.55)
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.0), constrained_layout=True)
    specs = (
        ("quant", "Quant (high-q)", QUANT_COLOR, "P", "-"),
        ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "-"),
        ("top_fixed", "Top-R", BASE.PLOT_COLORS["top_r"], "^", ":"),
    )
    for row_index, cv in enumerate(display_cvs):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            selected = frame[
                np.isclose(frame.target_cv, cv) & (frame.spatial_mode == mode)
            ]
            grouped = selected.groupby("budget_fraction", sort=True)
            for prefix, label, color, marker, linestyle in specs:
                curve = grouped[[x_fields[prefix], y_fields[prefix]]].mean()
                ax.plot(
                    curve[x_fields[prefix]], curve[y_fields[prefix]],
                    color=color, marker=marker, linestyle=linestyle, label=label,
                )
            if x_log:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={cv:g}")
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage_ratios(frame: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True,
                             constrained_layout=True)
    specs = (
        ("quant_lookup_reference_ratio", "Quant (high-q)", QUANT_COLOR, "P"),
        ("paper_vs_quant_lookup_ratio", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o"),
        ("top_matched_vs_quant_lookup_ratio", "Top-R", BASE.PLOT_COLORS["top_r"], "^"),
    )
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = frame[frame.spatial_mode == mode]
        grouped = selected.groupby("budget_fraction", sort=True)
        x = np.asarray(sorted(selected.budget_fraction.unique()), dtype=float)
        for field, label, color, marker in specs:
            median = grouped[field].median().to_numpy()
            ax.plot(x, median, color=color, marker=marker, label=label)
        ax.axhline(1.0, color="#334155", linewidth=1.1, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("Paper fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Lookup latency / high-q Quant latency (median, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_chunk_counts(frame: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    methods = (
        ("quant_chunks", "Quant (high-q)", QUANT_COLOR),
        ("paper_chunks", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
        ("top_matched_chunks", "Top-R", BASE.PLOT_COLORS["top_r"]),
    )
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = frame[frame.spatial_mode == mode]
        upper = max(int(selected[field].max()) for field, _, _ in methods)
        edges = np.unique(np.r_[0.5, np.rint(np.geomspace(1, upper + 1, 48)) + 0.5])
        for field, label, color in methods:
            values = selected[field].to_numpy(dtype=int)
            ax.hist(
                values, bins=edges, weights=np.ones(len(values)) / len(values),
                histtype="step", linewidth=1.9, color=color, label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Chunk count K (log scale)")
        ax.set_ylabel("Fraction of solutions")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_saving_ci(frame: pd.DataFrame, path: Path, replicates: int,
                   seed: int) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True,
                             constrained_layout=True)
    for mode_index, (ax, mode) in enumerate(zip(axes, SPATIAL_MODES)):
        selected = frame[frame.spatial_mode == mode]
        xs, means, lows, highs = [], [], [], []
        for budget_index, (budget, group) in enumerate(
            selected.groupby("budget_fraction", sort=True)
        ):
            clusters = (
                group.groupby(["trial", "target_cv"])
                .quant_saving_vs_paper_lookup_pct.mean().to_numpy()
            )
            rng = np.random.default_rng(seed + mode_index * 100 + budget_index)
            low, high = bootstrap_mean_ci(clusters, replicates, rng)
            xs.append(float(budget))
            means.append(float(clusters.mean()))
            lows.append(low)
            highs.append(high)
        ax.plot(xs, means, color=QUANT_COLOR, marker="o", label="Mean saving")
        ax.fill_between(xs, lows, highs, color=QUANT_COLOR, alpha=0.20,
                        label="Cluster-bootstrap 95% CI")
        ax.axhline(0.0, color="#334155", linestyle="--", linewidth=1.1)
        ax.set_xlabel("Paper fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Quant lookup-latency saving vs Paper (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(args: argparse.Namespace, frame: pd.DataFrame, inputs: pd.DataFrame,
            model: dict, policies) -> dict:
    summary = build_summary(args, frame, inputs, model, policies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_method_grid(
        frame, args.output_dir / "importance_latency_frontiers.png",
        {"quant": "quant_released_ms", "paper": "paper_released_ms",
         "top_fixed": "top_fixed_released_ms"},
        {"quant": "quant_importance", "paper": "paper_importance",
         "top_fixed": "top_fixed_importance"},
        "Released lookup latency (ms, log scale)", "Retained importance", True,
    )
    plot_method_grid(
        frame, args.output_dir / "r_importance.png",
        {"quant": "quant_rows", "paper": "paper_rows",
         "top_fixed": "top_fixed_rows"},
        {"quant": "quant_importance", "paper": "paper_importance",
         "top_fixed": "top_fixed_importance"},
        "Selected rows R", "Retained importance",
    )
    plot_method_grid(
        frame, args.output_dir / "r_latency.png",
        {"quant": "quant_rows", "paper": "paper_rows",
         "top_fixed": "top_fixed_rows"},
        {"quant": "quant_released_ms", "paper": "paper_released_ms",
         "top_fixed": "top_fixed_released_ms"},
        "Selected rows R", "Released lookup latency (ms)",
    )
    plot_coverage_ratios(frame, args.output_dir / "coverage_opt_ratio.png")
    plot_chunk_counts(frame, args.output_dir / "chunk_count.png")
    plot_saving_ci(
        frame, args.output_dir / "latency_saving_ci.png",
        args.bootstrap_replicates, args.bootstrap_seed + 1000,
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.q < 2 or args.bootstrap_replicates < 1000:
        raise SystemExit("trials/q/bootstrap-replicates are too small")
    cv_targets = sorted(set(args.cv_targets))
    budgets_fraction = sorted(set(args.budget_fractions))
    if not cv_targets or cv_targets[0] <= 0:
        raise SystemExit("CV targets must be positive")
    if not budgets_fraction or budgets_fraction[0] <= 0 or budgets_fraction[-1] >= 1:
        raise SystemExit("budget fractions must lie strictly between zero and one")

    n = int(CONFIG["n"])
    row_size_kib = float(CONFIG["cols"]) * 2.0 / 1024.0
    table = BASE.LatencyTable.load(args.profile)
    affine_a, affine_c, _ = BASE.affine_fit(table, row_size_kib)
    affine = (affine_a, affine_c)
    model = EXP13.fit_continuous_two_line(table, row_size_kib, args.saturation_kib)
    policies = EXP11.LatencyPolicies(table, row_size_kib, args.tail_fit_start_kib)

    trials_path = args.output_dir / "paired_trials.csv"
    inputs_path = args.output_dir / "input_trials.csv"
    if args.analyze_only:
        if not trials_path.is_file() or not inputs_path.is_file():
            raise SystemExit("--analyze-only requires paired_trials.csv and input_trials.csv")
        frame = pd.read_csv(trials_path)
        inputs = pd.read_csv(inputs_path)
        summary = analyze(args, frame, inputs, model, policies)
        print(json.dumps(summary["quant_saving_vs_paper_lookup_pct"], indent=2))
        return

    if not args.skip_self_check:
        EXP13.self_check(model)
        EXP2.self_check()

    warm_values = np.full(16, 1 / 16, dtype=np.float64)
    warm_grid = np.asarray([0.0, 0.1, 1.0, 10.0])
    EXP13._lambda_grid_kernel(
        warm_values, warm_grid, model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"],
        model["short_max_rows"],
    )

    params = BASE.ChunkParams(
        start_kb=CONFIG["start_kib"], jump_cap_kb=CONFIG["jump_cap_kib"]
    )
    budgets = [int(round(n * fraction)) for fraction in budgets_fraction]
    rng = np.random.default_rng(args.seed)
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    rows: list[dict] = []
    input_rows: list[dict] = []
    completed = 0
    total_inputs = args.trials * len(cv_targets) * len(SPATIAL_MODES)
    validated_fast_selector = False

    for trial in range(args.trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
            variants = EXP2.spatial_variants(base_values, rng, hotness, 0.95, 0.75)
            for mode, values in variants.items():
                oracle, build_ms = timed_call(
                    EXP13.TwoLineLagrangianOracle, values, model, args.q
                )
                order = np.argsort(-values, kind="stable")
                top_prefix = np.cumsum(values[order])
                values_t = torch.from_numpy(values.astype(np.float32))

                paper_solutions = []
                bounds = []
                for fraction, budget in zip(budgets_fraction, budgets):
                    paper_selected, paper_runtime_ms = timed_call(
                        BASE.select_chunks, values_t, budget, row_size_kib, table,
                        params=params, impl="torch",
                    )
                    paper_mask = paper_selected.mask.cpu().numpy()
                    paper = EXP13.mask_metrics(
                        paper_mask, values, model, affine, row_size_kib, policies
                    )
                    paper_solutions.append(
                        (fraction, budget, paper, paper_runtime_ms)
                    )
                    bounds.append(float(paper["importance"]))

                if not validated_fast_selector:
                    validate_monotone_oracle(oracle, bounds)
                    validated_fast_selector = True

                quant_cache: dict[int, dict] = {}
                for fraction, budget, paper, paper_runtime_ms in paper_solutions:
                    bound = float(paper["importance"])
                    quant_index = fast_candidate_index(oracle, bound)
                    if quant_index not in quant_cache:
                        solution = solve_supported_index(oracle, quant_index)
                        metrics = EXP13.mask_metrics(
                            solution["mask"], values, model, affine,
                            row_size_kib, policies,
                        )
                        metrics["lambda_index"] = solution["lambda_index"]
                        metrics["lambda"] = solution["lambda"]
                        quant_cache[quant_index] = metrics
                    quant = quant_cache[quant_index]
                    if quant["importance"] < bound - 1e-12:
                        raise RuntimeError("Quant missed Paper importance target")

                    top_fixed_mask, top_matched_mask = top_masks(
                        values, order, top_prefix, budget, bound
                    )
                    top_fixed = EXP13.mask_metrics(
                        top_fixed_mask, values, model, affine, row_size_kib, policies
                    )
                    top_matched = EXP13.mask_metrics(
                        top_matched_mask, values, model, affine, row_size_kib, policies
                    )
                    quant_saving = 100.0 * (
                        1.0 - quant["released_ms"] / paper["released_ms"]
                    )
                    rows.append({
                        "trial": trial,
                        "target_cv": target_cv,
                        "spatial_mode": mode,
                        "budget_fraction": fraction,
                        "budget_rows": budget,
                        "paper_runtime_ms": paper_runtime_ms,
                        **compact_fields("paper", paper),
                        **compact_fields("quant", quant),
                        "quant_lambda_index": quant["lambda_index"],
                        "quant_lambda": quant["lambda"],
                        **compact_fields("top_fixed", top_fixed),
                        **compact_fields("top_matched", top_matched),
                        "quant_saving_vs_paper_lookup_pct": quant_saving,
                        "paper_extra_vs_quant_lookup_pct": 100.0 * (
                            paper["released_ms"] / quant["released_ms"] - 1.0
                        ),
                        "quant_win": float(quant_saving > 1e-10),
                        "quant_lookup_reference_ratio": 1.0,
                        "paper_vs_quant_lookup_ratio": (
                            paper["released_ms"] / quant["released_ms"]
                        ),
                        "top_matched_vs_quant_lookup_ratio": (
                            top_matched["released_ms"] / quant["released_ms"]
                        ),
                    })

                unique_solutions = len(set(zip(
                    np.round(oracle.importance, 14), np.round(oracle.costs, 14),
                    oracle.rows, oracle.chunks,
                )))
                input_rows.append({
                    "trial": trial,
                    "target_cv": target_cv,
                    "spatial_mode": mode,
                    "actual_cv": EXP2.coefficient_of_variation(values),
                    "q": args.q,
                    "quant_build_ms": build_ms,
                    "unique_supported_solutions": unique_solutions,
                })
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total_inputs:
                    print(
                        f"completed {completed}/{total_inputs} spatial inputs "
                        f"({build_ms / 1000:.2f}s grid)", flush=True,
                    )
                if args.input_limit is not None and completed >= args.input_limit:
                    break
            if args.input_limit is not None and completed >= args.input_limit:
                break
        if args.input_limit is not None and completed >= args.input_limit:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trials_path, rows)
    write_csv(inputs_path, input_rows)
    frame = pd.DataFrame(rows)
    inputs = pd.DataFrame(input_rows)
    summary = analyze(args, frame, inputs, model, policies)
    print(json.dumps(summary["quant_saving_vs_paper_lookup_pct"], indent=2))


if __name__ == "__main__":
    main()
