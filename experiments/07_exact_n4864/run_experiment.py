#!/usr/bin/env python3
"""Experiment 07: exact DP-Cover at the realistic N=4,864.

Compare exactly four methods: Paper greedy, exact fixed-R DP (DP-R), Top-R,
and an exact coverage DP (DP-Cover).  Single interval and Quantized Pareto are
intentionally absent.  DP-Cover stores two rolling O(N^2) metric tables and
does not retain O(N^3) backtracking parents.
"""

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
import torch

try:
    from numba import njit
except ImportError as error:  # pragma: no cover
    raise SystemExit("Experiment 07 requires numba") from error


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_06 = PROJECT_ROOT / "experiments" / "06_quant_scale" / "run_experiment.py"


def load_experiment_06():
    spec = importlib.util.spec_from_file_location("experiment_06", EXPERIMENT_06)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 06 from {EXPERIMENT_06}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP6 = load_experiment_06()
EXP5 = EXP6.EXP5
EXP2 = EXP6.EXP2
BASE = EXP6.BASE
SPATIAL_MODES = EXP6.SPATIAL_MODES
SPATIAL_LABELS = EXP6.SPATIAL_LABELS
CONFIG = next(item for item in EXP6.MATRIX_CONFIGS if item["n"] == 4864)


@njit(cache=True)
def _exact_coverage_metrics_kernel(values):
    """Exact (R,K,ending-state) DP with two in-place O(N^2) tables."""
    n = len(values)
    kmax = (n + 1) // 2
    end_zero = np.full((n + 1, kmax + 1), -np.inf, dtype=np.float64)
    end_one = np.full((n + 1, kmax + 1), -np.inf, dtype=np.float64)
    end_zero[0, 0] = 0.0

    for length in range(1, n + 1):
        value = values[length - 1]
        # Descending R preserves prefix-(length-1) entries at R-1.
        for rows in range(length, -1, -1):
            zero_kmax = min(rows, length - rows)
            one_kmax = min(rows, length - rows + 1)
            for chunks in range(zero_kmax + 1):
                skip_zero = end_zero[rows, chunks]
                skip_one = end_one[rows, chunks]
                end_zero[rows, chunks] = (
                    skip_zero if skip_zero >= skip_one else skip_one
                )
            end_one[rows, 0] = -np.inf
            if rows > 0:
                for chunks in range(1, one_kmax + 1):
                    continue_run = end_one[rows - 1, chunks]
                    start_run = end_zero[rows - 1, chunks - 1]
                    previous = (
                        continue_run if continue_run >= start_run else start_run
                    )
                    end_one[rows, chunks] = value + previous
    return end_zero, end_one


@njit(cache=True)
def _solve_exact_metrics(end_zero, end_one, bound, a_ms, c_ms_per_row):
    n = end_zero.shape[0] - 1
    best_cost = np.inf
    best_importance = -np.inf
    best_rows = -1
    best_chunks = -1
    for rows in range(n + 1):
        max_chunks = min((n + 1) // 2, rows)
        for chunks in range(max_chunks + 1):
            z = end_zero[rows, chunks]
            o = end_one[rows, chunks]
            importance = z if z >= o else o
            if importance < bound - 1e-14:
                continue
            cost = c_ms_per_row * rows + a_ms * chunks
            if cost < best_cost - 1e-15 or (
                abs(cost - best_cost) <= 1e-15
                and importance > best_importance
            ):
                best_cost = cost
                best_importance = importance
                best_rows = rows
                best_chunks = chunks
    return best_importance, best_rows, best_chunks, best_cost


class ExactCoverageMetricsOracle:
    def __init__(self, values, a_ms, c_ms_per_row):
        self.values = np.asarray(values, dtype=np.float64)
        self.total = float(self.values.sum())
        self.a_ms = float(a_ms)
        self.c_ms_per_row = float(c_ms_per_row)
        self.end_zero, self.end_one = _exact_coverage_metrics_kernel(self.values)
        self.buffer_bytes = self.end_zero.nbytes + self.end_one.nbytes

    def solve(self, bound):
        if not 0 < bound <= self.total + 1e-12:
            raise ValueError("coverage bound must lie in (0, total importance]")
        importance, rows, chunks, cost = _solve_exact_metrics(
            self.end_zero, self.end_one, bound, self.a_ms, self.c_ms_per_row
        )
        if rows < 0:
            raise RuntimeError("exact coverage metric DP found no feasible state")
        return {
            "importance": float(importance),
            "rows": int(rows),
            "chunks": int(chunks),
            "aff_ms": float(cost),
        }


def timed_call(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return result, (time.perf_counter_ns() - start) / 1e6


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fields(prefix, solution, runtime_ms=None):
    output = {
        f"{prefix}_importance": solution["importance"],
        f"{prefix}_rows": solution["rows"],
        f"{prefix}_chunks": solution["chunks"],
        f"{prefix}_aff_ms": solution["aff_ms"],
    }
    if runtime_ms is not None:
        output[f"{prefix}_runtime_ms"] = runtime_ms
    return output


def distribution(values):
    return EXP2.distribution_stats([float(value) for value in values])


def aggregate_curve(rows, target_cv, spatial_mode, x_key, metrics):
    selected = [
        row for row in rows
        if float(row["target_cv"]) == target_cv
        and row["spatial_mode"] == spatial_mode
    ]
    return EXP2.summarize(selected, (x_key,), metrics)


def plot_method_grid(coverage_rows, fixed_rows, path, x_fields, y_fields,
                     xlabel, ylabel, x_log=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    cvs = sorted({float(row["target_cv"]) for row in coverage_rows})
    display_cvs = [min(cvs, key=lambda value: abs(value - wanted))
                   for wanted in (1.25, 3.30, 4.55)]
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.0), constrained_layout=True)
    for row_index, cv in enumerate(display_cvs):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            coverage = aggregate_curve(
                coverage_rows, cv, mode, "coverage_target",
                (x_fields["cover"], y_fields["cover"]),
            )
            fixed = aggregate_curve(
                fixed_rows, cv, mode, "budget_fraction",
                tuple({
                    x_fields["paper"], x_fields["dpr"], x_fields["top"],
                    y_fields["paper"], y_fields["dpr"], y_fields["top"],
                }),
            )
            specs = (
                (coverage, "cover", "DP-Cover exact", BASE.PLOT_COLORS["coverage"], "X", "--"),
                (fixed, "paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "-"),
                (fixed, "dpr", "DP-R exact", BASE.PLOT_COLORS["fixed"], "s", "-"),
                (fixed, "top", "Top-R", BASE.PLOT_COLORS["top_r"], "^", ":"),
            )
            for points, prefix, label, color, marker, linestyle in specs:
                ax.plot(
                    [point[x_fields[prefix]]["mean"] for point in points],
                    [point[y_fields[prefix]]["mean"] for point in points],
                    color=color, marker=marker, linestyle=linestyle, label=label,
                )
            if x_log:
                ax.set_xscale("log")
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={cv:g}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage_ratios(fixed_rows, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    rows = [row for row in fixed_rows if float(row["target_cv"]) < 9.0]
    summaries = EXP2.summarize(
        rows, ("spatial_mode", "budget_fraction"),
        ("paper_cover_ratio", "dpr_cover_ratio", "top_cover_ratio"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True,
                             constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        points = [item for item in summaries if item["spatial_mode"] == mode]
        points.sort(key=lambda item: item["budget_fraction"])
        for key, label, color, marker in (
            ("paper_cover_ratio", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o"),
            ("dpr_cover_ratio", "DP-R exact", BASE.PLOT_COLORS["fixed"], "s"),
            ("top_cover_ratio", "Top-R", BASE.PLOT_COLORS["top_r"], "^"),
        ):
            ax.plot([item["budget_fraction"] for item in points],
                    [item[key]["median"] for item in points],
                    color=color, marker=marker, label=label)
        ax.axhline(1.0, color=BASE.PLOT_COLORS["coverage"], linewidth=1.2,
                   linestyle="--", label="DP-Cover optimum")
        ax.set_yscale("log")
        ax.set_xlabel("Fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Method latency / exact DP-Cover latency (median, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_chunk_histograms(fixed_rows, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    rows = [row for row in fixed_rows if float(row["target_cv"]) < 9.0]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    methods = (
        ("paper_cover_chunks", "DP-Cover exact", BASE.PLOT_COLORS["coverage"]),
        ("paper_chunks", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
        ("dpr_chunks", "DP-R exact", BASE.PLOT_COLORS["fixed"]),
        ("top_chunks", "Top-R", BASE.PLOT_COLORS["top_r"]),
    )
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = [row for row in rows if row["spatial_mode"] == mode]
        values_by_method = {
            label: np.asarray([int(row[key]) for row in selected])
            for key, label, _ in methods
        }
        upper = max(int(values.max()) for values in values_by_method.values())
        edges = np.unique(np.r_[0.5, np.rint(np.geomspace(1, upper + 1, 42)) + 0.5])
        for key, label, color in methods:
            values = values_by_method[label]
            ax.hist(values, bins=edges,
                    weights=np.ones(len(values), dtype=float) / len(values),
                    histtype="step", linewidth=1.8, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Chunk count K (log scale)")
        ax.set_ylabel("Fraction of solutions")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def self_check():
    rng = np.random.default_rng(707)
    ExactCoverageMetricsOracle(np.full(8, 1 / 8), 0.011, 0.0007).solve(0.5)
    for n in (8, 17, 32, 65):
        values = rng.lognormal(size=n)
        values /= values.sum()
        reference = BASE.ExactCoverageOracle(values, 0.011, 0.0007)
        rolling = ExactCoverageMetricsOracle(values, 0.011, 0.0007)
        if not np.allclose(rolling.end_zero, reference.end_zero, equal_nan=True):
            raise RuntimeError("rolling end-zero table differs from reference")
        if not np.allclose(rolling.end_one, reference.end_one, equal_nan=True):
            raise RuntimeError("rolling end-one table differs from reference")
        for target in (0.1, 0.5, 0.9, 0.99):
            expected = reference.solve(target)
            actual = rolling.solve(target)
            for key in ("importance", "rows", "chunks", "aff_ms"):
                if not math.isclose(float(expected[key]), float(actual[key]),
                                    rel_tol=1e-11, abs_tol=1e-13):
                    raise RuntimeError(f"rolling exact solver differs on {key}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--cv-targets", type=float, nargs="+",
                        default=[1.07, 1.25, 1.44, 2.48, 3.30, 4.55, 9.19])
    parser.add_argument("--coverage-targets", type=float, nargs="+",
                        default=[0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99])
    parser.add_argument("--budget-fractions", type=float, nargs="+",
                        default=[0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875])
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--dtype-bytes", type=float, default=2.0)
    parser.add_argument("--local-rho", type=float, default=0.95)
    parser.add_argument("--hot-rho", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    if not args.skip_self_check:
        self_check()
        EXP2.self_check()

    n = int(CONFIG["n"])
    row_size_kib = float(CONFIG["cols"]) * args.dtype_bytes / 1024.0
    table = BASE.LatencyTable.load(args.profile)
    a_ms, c_ms, fit_r2 = BASE.affine_fit(table, row_size_kib)
    params = BASE.ChunkParams(start_kb=CONFIG["start_kib"],
                              jump_cap_kb=CONFIG["jump_cap_kib"])
    cv_targets = sorted(set(args.cv_targets))
    coverage_targets = sorted(set(args.coverage_targets))
    budget_fractions = sorted(set(args.budget_fractions))
    budgets = [int(round(n * fraction)) for fraction in budget_fractions]
    rng = np.random.default_rng(args.seed)
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    coverage_rows = []
    fixed_rows = []
    input_rows = []
    total_inputs = args.trials * len(cv_targets) * len(SPATIAL_MODES)
    completed = 0

    # Compile measured kernels before the first recorded input.
    ExactCoverageMetricsOracle(np.full(8, 1 / 8), a_ms, c_ms).solve(0.5)

    for trial in range(args.trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
            variants = EXP2.spatial_variants(
                base_values, rng, hotness, args.local_rho, args.hot_rho
            )
            for mode, values in variants.items():
                oracle, cover_build_ms = timed_call(
                    ExactCoverageMetricsOracle, values, a_ms, c_ms
                )
                top_oracle, top_build_ms = timed_call(
                    EXP6.TopROracle, values, a_ms, c_ms
                )
                lag1, first_half = EXP2.spatial_stats(values)
                input_rows.append({
                    "trial": trial, "target_cv": target_cv,
                    "spatial_mode": mode,
                    "actual_cv": EXP2.coefficient_of_variation(values),
                    "lag1_corr": lag1, "first_half_mass": first_half,
                    "cover_build_ms": cover_build_ms,
                    "cover_buffer_gib": oracle.buffer_bytes / 2**30,
                    "top_build_ms": top_build_ms,
                })

                total = float(values.sum())
                for alpha in coverage_targets:
                    exact, exact_ms = timed_call(oracle.solve, alpha * total)
                    coverage_rows.append({
                        "trial": trial, "target_cv": target_cv,
                        "spatial_mode": mode, "coverage_target": alpha,
                        "target_importance": alpha * total,
                        **fields("cover", exact, exact_ms),
                        "cover_build_ms": cover_build_ms,
                    })

                values_t = torch.from_numpy(values.astype(np.float32))
                for fraction, budget in zip(budget_fractions, budgets):
                    paper, paper_ms = timed_call(
                        BASE.select_chunks, values_t, budget, row_size_kib, table,
                        params=params, impl="torch",
                    )
                    paper_mask = paper.mask.cpu().numpy()
                    paper_rows, paper_chunks = BASE.mask_stats(paper_mask)
                    paper_solution = {
                        "importance": float(values[paper_mask].sum()),
                        "rows": paper_rows, "chunks": paper_chunks,
                        "aff_ms": BASE.affine_latency(paper_mask, a_ms, c_ms),
                    }

                    (dpr_mask, dpr_iterations, dpr_residual), dpr_ms = timed_call(
                        BASE.exact_fixed_r_affine, values, budget, a_ms, c_ms
                    )
                    dpr_rows, dpr_chunks = BASE.mask_stats(dpr_mask)
                    dpr_solution = {
                        "importance": float(values[dpr_mask].sum()),
                        "rows": dpr_rows, "chunks": dpr_chunks,
                        "aff_ms": BASE.affine_latency(dpr_mask, a_ms, c_ms),
                    }

                    top_solution, top_ms = timed_call(top_oracle.fixed, budget)
                    paper_cover, paper_cover_ms = timed_call(
                        oracle.solve, paper_solution["importance"]
                    )
                    dpr_cover, dpr_cover_ms = timed_call(
                        oracle.solve, dpr_solution["importance"]
                    )
                    top_cover, top_cover_ms = timed_call(
                        oracle.solve, top_solution["importance"]
                    )
                    for name, solution, exact in (
                        ("Paper", paper_solution, paper_cover),
                        ("DP-R", dpr_solution, dpr_cover),
                        ("Top-R", top_solution, top_cover),
                    ):
                        if solution["aff_ms"] < exact["aff_ms"] - 1e-10:
                            raise RuntimeError(f"{name} beat exact DP-Cover")

                    fixed_rows.append({
                        "trial": trial, "target_cv": target_cv,
                        "spatial_mode": mode, "budget_fraction": fraction,
                        "budget_rows": budget,
                        **fields("paper", paper_solution, paper_ms),
                        **fields("dpr", dpr_solution, dpr_ms),
                        **fields("top", top_solution, top_ms),
                        **fields("paper_cover", paper_cover, paper_cover_ms),
                        **fields("dpr_cover", dpr_cover, dpr_cover_ms),
                        **fields("top_cover", top_cover, top_cover_ms),
                        "paper_cover_ratio": paper_solution["aff_ms"] / paper_cover["aff_ms"],
                        "dpr_cover_ratio": dpr_solution["aff_ms"] / dpr_cover["aff_ms"],
                        "top_cover_ratio": top_solution["aff_ms"] / top_cover["aff_ms"],
                        "paper_ratio": paper_solution["importance"] / paper_solution["aff_ms"],
                        "dpr_ratio": dpr_solution["importance"] / dpr_solution["aff_ms"],
                        "top_ratio": top_solution["importance"] / top_solution["aff_ms"],
                        "dpr_gain_over_paper_pct": 100.0 * (
                            (dpr_solution["importance"] / dpr_solution["aff_ms"])
                            / (paper_solution["importance"] / paper_solution["aff_ms"]) - 1.0
                        ),
                        "top_importance_gain_over_paper_pct": 100.0 * (
                            top_solution["importance"] / paper_solution["importance"] - 1.0
                        ),
                        "dpr_iterations": dpr_iterations,
                        "dpr_residual": dpr_residual,
                    })

                completed += 1
                print(f"completed {completed}/{total_inputs} inputs", flush=True)
                del oracle

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "coverage_trials.csv", coverage_rows)
    write_csv(args.output_dir / "fixed_r_trials.csv", fixed_rows)
    write_csv(args.output_dir / "input_trials.csv", input_rows)

    vlm_fixed = [row for row in fixed_rows if float(row["target_cv"]) < 9.0]
    vlm_inputs = [row for row in input_rows if float(row["target_cv"]) < 9.0]
    summary = {
        "format": "experiment-07-exact-n4864-v1",
        "n": n, "model": CONFIG["model"], "shape": CONFIG["shape"],
        "trials_per_cv": args.trials, "seed": args.seed,
        "cv_targets": cv_targets, "coverage_targets": coverage_targets,
        "budget_fractions": budget_fractions,
        "methods": ["DP-Cover exact", "Paper greedy", "DP-R exact", "Top-R"],
        "excluded_methods": ["Single interval", "Quantized Pareto"],
        "affine_fit": {
            "a_ms_per_chunk": a_ms, "c_ms_per_row": c_ms,
            "r_squared": fit_r2, "chunk_open_equivalent_rows": a_ms / c_ms,
        },
        "dp_cover": {
            "build_ms": distribution([row["cover_build_ms"] for row in vlm_inputs]),
            "buffer_gib": distribution([row["cover_buffer_gib"] for row in vlm_inputs]),
        },
        "method_runtime_ms": {
            "paper": distribution([row["paper_runtime_ms"] for row in vlm_fixed]),
            "dpr": distribution([row["dpr_runtime_ms"] for row in vlm_fixed]),
            "top_query": distribution([row["top_runtime_ms"] for row in vlm_fixed]),
            "top_build": distribution([row["top_build_ms"] for row in vlm_inputs]),
        },
        "coverage_ratio": {
            "paper": distribution([row["paper_cover_ratio"] for row in vlm_fixed]),
            "dpr": distribution([row["dpr_cover_ratio"] for row in vlm_fixed]),
            "top": distribution([row["top_cover_ratio"] for row in vlm_fixed]),
        },
        "fixed_r": {
            "dpr_gain_over_paper_pct": distribution([
                row["dpr_gain_over_paper_pct"] for row in vlm_fixed
            ]),
            "top_importance_gain_over_paper_pct": distribution([
                row["top_importance_gain_over_paper_pct"] for row in vlm_fixed
            ]),
            "paper_chunks": distribution([row["paper_chunks"] for row in vlm_fixed]),
            "dpr_chunks": distribution([row["dpr_chunks"] for row in vlm_fixed]),
            "top_chunks": distribution([row["top_chunks"] for row in vlm_fixed]),
            "cover_at_paper_chunks": distribution([
                row["paper_cover_chunks"] for row in vlm_fixed
            ]),
        },
        "timing_scope": "single-process CPU reference",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_method_grid(
        coverage_rows, fixed_rows, args.output_dir / "importance_latency_frontiers.png",
        {"cover": "cover_aff_ms", "paper": "paper_aff_ms",
         "dpr": "dpr_aff_ms", "top": "top_aff_ms"},
        {"cover": "cover_importance", "paper": "paper_importance",
         "dpr": "dpr_importance", "top": "top_importance"},
        "Affine latency (ms, log scale)", "Retained importance", True,
    )
    plot_method_grid(
        coverage_rows, fixed_rows, args.output_dir / "r_importance.png",
        {"cover": "cover_rows", "paper": "paper_rows",
         "dpr": "dpr_rows", "top": "top_rows"},
        {"cover": "cover_importance", "paper": "paper_importance",
         "dpr": "dpr_importance", "top": "top_importance"},
        "Selected rows R", "Retained importance",
    )
    plot_method_grid(
        coverage_rows, fixed_rows, args.output_dir / "r_latency.png",
        {"cover": "cover_rows", "paper": "paper_rows",
         "dpr": "dpr_rows", "top": "top_rows"},
        {"cover": "cover_aff_ms", "paper": "paper_aff_ms",
         "dpr": "dpr_aff_ms", "top": "top_aff_ms"},
        "Selected rows R", "Affine latency (ms)",
    )
    plot_coverage_ratios(fixed_rows, args.output_dir / "coverage_optimality_ratio.png")
    plot_chunk_histograms(fixed_rows, args.output_dir / "chunk_count_histograms.png")
    print(f"wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
