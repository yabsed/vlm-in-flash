#!/usr/bin/env python3
"""Experiment 05: realistic down-projection channel counts.

Run Paper greedy, the O(N) single-interval solver, and Quantized Pareto q=1024
at the actual down-projection row counts and FP16 row sizes represented in the
paper.  Exact Coverage DP is intentionally excluded because its current
O(N^3) implementation cannot scale to these matrix sizes.
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
except ImportError as error:  # pragma: no cover - environment diagnostic
    raise SystemExit("Experiment 05 requires numba for the q=1024 large-N DP") from error


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_04 = PROJECT_ROOT / "experiments" / "04_single_interval" / "run_experiment.py"


def load_experiment_04():
    spec = importlib.util.spec_from_file_location("experiment_04", EXPERIMENT_04)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 04 from {EXPERIMENT_04}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP4 = load_experiment_04()
EXP3 = EXP4.EXP3
EXP2 = EXP4.EXP2
BASE = EXP4.BASE
SPATIAL_MODES = EXP4.SPATIAL_MODES
SPATIAL_LABELS = EXP4.SPATIAL_LABELS
SINGLE_COLOR = EXP4.SINGLE_COLOR


# The paper's AGX Table 2 settings for the four down-projection shapes.
# Rows are selectable input channels; cols determine one FP16 row's byte size.
MATRIX_CONFIGS = (
    {
        "n": 4864,
        "cols": 896,
        "model": "LLaVA-OneVision-0.5B",
        "shape": "4864x896",
        "start_kib": 12.0,
        "jump_cap_kib": 16.0,
    },
    {
        "n": 8960,
        "cols": 1536,
        "model": "NVILA-Lite-2B",
        "shape": "8960x1536",
        "start_kib": 16.0,
        "jump_cap_kib": 16.0,
    },
    {
        "n": 14336,
        "cols": 4096,
        "model": "VILA1.5-8B",
        "shape": "14336x4096",
        "start_kib": 32.0,
        "jump_cap_kib": 32.0,
    },
    {
        "n": 18944,
        "cols": 3584,
        "model": "LLaVA-OneVision-7B / LongVA-7B",
        "shape": "18944x3584",
        "start_kib": 32.0,
        "jump_cap_kib": 32.0,
    },
)


@njit(cache=True)
def _quantized_pareto_kernel(values, a_ms, c_ms_per_row, q):
    """Numba implementation equivalent to Experiment 01's O(qN) DP."""
    n = len(values)
    total = values.sum()
    costs = np.full((2, q + 1), np.inf, dtype=np.float64)
    importance = np.full((2, q + 1), -np.inf, dtype=np.float64)
    new_costs = np.empty((2, q + 1), dtype=np.float64)
    new_importance = np.empty((2, q + 1), dtype=np.float64)
    costs[0, 0] = 0.0
    importance[0, 0] = 0.0

    parent_state = np.full((n + 1, 2, q + 1), 255, dtype=np.uint8)
    parent_bucket = np.full((n + 1, 2, q + 1), -1, dtype=np.int16)
    peak_states = 1

    for i in range(1, n + 1):
        value = values[i - 1]
        new_costs[:, :] = np.inf
        new_importance[:, :] = -np.inf
        for previous_state in range(2):
            for previous_bucket in range(q + 1):
                old_cost = costs[previous_state, previous_bucket]
                if not np.isfinite(old_cost):
                    continue
                old_importance = importance[previous_state, previous_bucket]

                skip_bucket = int(math.floor(q * old_importance / total + 1e-12))
                if skip_bucket < 0:
                    skip_bucket = 0
                elif skip_bucket > q:
                    skip_bucket = q
                incumbent = new_costs[0, skip_bucket]
                if old_cost < incumbent - 1e-15 or (
                    abs(old_cost - incumbent) <= 1e-15
                    and old_importance > new_importance[0, skip_bucket]
                ):
                    new_costs[0, skip_bucket] = old_cost
                    new_importance[0, skip_bucket] = old_importance
                    parent_state[i, 0, skip_bucket] = previous_state
                    parent_bucket[i, 0, skip_bucket] = previous_bucket

                selected_importance = old_importance + value
                selected_cost = old_cost + c_ms_per_row
                if previous_state == 0:
                    selected_cost += a_ms
                selected_bucket = int(
                    math.floor(q * selected_importance / total + 1e-12)
                )
                if selected_bucket < 0:
                    selected_bucket = 0
                elif selected_bucket > q:
                    selected_bucket = q
                incumbent = new_costs[1, selected_bucket]
                if selected_cost < incumbent - 1e-15 or (
                    abs(selected_cost - incumbent) <= 1e-15
                    and selected_importance > new_importance[1, selected_bucket]
                ):
                    new_costs[1, selected_bucket] = selected_cost
                    new_importance[1, selected_bucket] = selected_importance
                    parent_state[i, 1, selected_bucket] = previous_state
                    parent_bucket[i, 1, selected_bucket] = previous_bucket

        live_states = 0
        for ending in range(2):
            best_higher_cost = np.inf
            for bucket in range(q, -1, -1):
                candidate_cost = new_costs[ending, bucket]
                if not np.isfinite(candidate_cost):
                    continue
                if candidate_cost >= best_higher_cost - 1e-15:
                    new_costs[ending, bucket] = np.inf
                    new_importance[ending, bucket] = -np.inf
                else:
                    best_higher_cost = candidate_cost
                    live_states += 1

        temp_costs = costs
        costs = new_costs
        new_costs = temp_costs
        temp_importance = importance
        importance = new_importance
        new_importance = temp_importance
        if live_states > peak_states:
            peak_states = live_states

    return costs, importance, parent_state, parent_bucket, peak_states


class FastQuantizedParetoOracle:
    """q-Pareto oracle with a compiled build loop for realistic N."""

    def __init__(self, values: np.ndarray, a_ms: float, c_ms_per_row: float, q: int):
        if q < 2 or q > np.iinfo(np.int16).max:
            raise ValueError("q must lie in [2, 32767]")
        self.values = np.asarray(values, dtype=np.float64)
        self.a_ms = float(a_ms)
        self.c_ms_per_row = float(c_ms_per_row)
        self.q = int(q)
        self.total = float(self.values.sum())
        if self.total <= 0:
            raise ValueError("coverage importance must have positive total mass")
        (
            self.costs,
            self.importance,
            self.parent_state,
            self.parent_bucket,
            self.peak_states,
        ) = _quantized_pareto_kernel(
            self.values, self.a_ms, self.c_ms_per_row, self.q
        )

    def solve(self, bound: float) -> dict:
        if bound <= 0 or bound > self.total + 1e-12:
            raise ValueError("coverage bound must lie in (0, total importance]")
        best_key = (math.inf, math.inf)
        best_state = None
        best_bucket = None
        for ending in (0, 1):
            for bucket in np.flatnonzero(np.isfinite(self.costs[ending])):
                actual_importance = float(self.importance[ending, bucket])
                if actual_importance < bound - 1e-14:
                    continue
                key = (float(self.costs[ending, bucket]), -actual_importance)
                if key < best_key:
                    best_key = key
                    best_state = ending
                    best_bucket = int(bucket)

        full_cost = self.a_ms + self.c_ms_per_row * len(self.values)
        if best_state is None or full_cost < best_key[0] - 1e-15:
            mask = np.ones(len(self.values), dtype=bool)
        else:
            mask = np.zeros(len(self.values), dtype=bool)
            state = int(best_state)
            bucket = int(best_bucket)
            for i in range(len(self.values), 0, -1):
                if state == 1:
                    mask[i - 1] = True
                previous_state = int(self.parent_state[i, state, bucket])
                previous_bucket = int(self.parent_bucket[i, state, bucket])
                if previous_state == 255 or previous_bucket < 0:
                    raise RuntimeError("q-Pareto backtracking reached an invalid state")
                state, bucket = previous_state, previous_bucket

        rows, chunks = BASE.mask_stats(mask)
        achieved = float(self.values[mask].sum())
        if achieved < bound - 1e-12:
            raise RuntimeError("q-Pareto returned an infeasible mask")
        return {
            "mask": mask,
            "importance": achieved,
            "rows": rows,
            "chunks": chunks,
            "aff_ms": self.a_ms * chunks + self.c_ms_per_row * rows,
            "peak_states": int(self.peak_states),
        }


def timed_call(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return result, elapsed_ms


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def solve_pair(values, target, oracle, a_ms, c_ms_per_row):
    single, single_ms = timed_call(
        EXP4.minimum_single_interval, values, target, a_ms, c_ms_per_row
    )
    quant, quant_solve_ms = timed_call(oracle.solve, target)
    if single["importance"] < target - 1e-12 or quant["importance"] < target - 1e-12:
        raise RuntimeError("a coverage solver returned an infeasible mask")
    relative = 100.0 * (single["aff_ms"] / quant["aff_ms"] - 1.0)
    return {
        "target_importance": target,
        "single_importance": single["importance"],
        "single_overshoot": single["importance"] - target,
        "single_rows": single["rows"],
        "single_chunks": single["chunks"],
        "single_aff_ms": single["aff_ms"],
        "single_runtime_ms": single_ms,
        "quant_importance": quant["importance"],
        "quant_overshoot": quant["importance"] - target,
        "quant_rows": quant["rows"],
        "quant_chunks": quant["chunks"],
        "quant_aff_ms": quant["aff_ms"],
        "quant_solve_ms": quant_solve_ms,
        "single_extra_vs_quant_pct": relative,
        "quant_beats_single": relative > 1e-10,
    }


def aggregate_curve(rows, target_cv, spatial_mode, x_key, metrics):
    selected = [
        row
        for row in rows
        if float(row["target_cv"]) == target_cv
        and row["spatial_mode"] == spatial_mode
    ]
    return EXP2.summarize(selected, (x_key,), metrics)


def plot_method_grid(
    coverage_rows,
    paper_rows,
    path,
    q,
    title_suffix,
    x_fields,
    y_fields,
    xlabel,
    ylabel,
    x_log=False,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    available_cv = sorted({float(row["target_cv"]) for row in coverage_rows})
    representative_cv = [
        min(available_cv, key=lambda value: abs(value - desired))
        for desired in (1.25, 3.30, 9.19)
    ]
    representative_cv = list(dict.fromkeys(representative_cv))
    fig, axes = plt.subplots(
        len(representative_cv), 3, figsize=(14.8, 4.0 * len(representative_cv)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, cv in enumerate(representative_cv):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            coverage = aggregate_curve(
                coverage_rows,
                cv,
                mode,
                "coverage_target",
                tuple(
                    {
                        x_fields["single"], x_fields["quant"],
                        y_fields["single"], y_fields["quant"],
                    }
                ),
            )
            paper = aggregate_curve(
                paper_rows,
                cv,
                mode,
                "budget_fraction",
                (x_fields["paper"], y_fields["paper"]),
            )
            for points, prefix, label, color, marker, linestyle in (
                (paper, "paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "-"),
                (coverage, "single", "Single interval O(N)", SINGLE_COLOR, "s", "-"),
                (coverage, "quant", f"Quantized Pareto q={q}", BASE.PLOT_COLORS["pareto"], "P", "--"),
            ):
                ax.plot(
                    [item[x_fields[prefix]]["mean"] for item in points],
                    [item[y_fields[prefix]]["mean"] for item in points],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                )
            if x_log:
                ax.set_xscale("log")
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={cv:g}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_quant_chunks(coverage_rows, path, title):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    cvs = sorted({float(row["target_cv"]) for row in coverage_rows})
    targets = sorted({float(row["coverage_target"]) for row in coverage_rows})
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.0), sharey=True, constrained_layout=True)
    image_handle = None
    summaries = EXP2.summarize(
        coverage_rows,
        ("target_cv", "spatial_mode", "coverage_target"),
        ("quant_chunks",),
    )
    lookup = {
        (item["target_cv"], item["spatial_mode"], item["coverage_target"]): item
        for item in summaries
    }
    vmax = max(item["quant_chunks"]["p95"] for item in summaries)
    for ax, mode in zip(axes, SPATIAL_MODES):
        means = np.zeros((len(cvs), len(targets)))
        for row_index, cv in enumerate(cvs):
            for column_index, target in enumerate(targets):
                item = lookup[(cv, mode, target)]["quant_chunks"]
                means[row_index, column_index] = item["mean"]
        image_handle = ax.imshow(
            means, cmap="magma", vmin=1.0, vmax=max(2.0, vmax),
            aspect="auto", interpolation="nearest",
        )
        for row_index, cv in enumerate(cvs):
            for column_index, target in enumerate(targets):
                stats = lookup[(cv, mode, target)]["quant_chunks"]
                ax.text(
                    column_index, row_index,
                    f"mean {stats['mean']:.1f}\np95 {stats['p95']:.0f}",
                    ha="center", va="center", fontsize=6.8,
                    color="white" if means[row_index, column_index] < 0.65 * vmax else "#17202A",
                )
        ax.set_xticks(range(len(targets)), [f"{target:g}" for target in targets])
        ax.set_yticks(range(len(cvs)), [f"{cv:g}" for cv in cvs])
        ax.set_xlabel("Coverage target")
        ax.set_title(SPATIAL_LABELS[mode])
        ax.grid(False)
    axes[0].set_ylabel("Target CV")
    colorbar = fig.colorbar(image_handle, ax=axes, shrink=0.88, pad=0.015)
    colorbar.set_label("Mean Quantized-Pareto chunks")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_scaling(per_n_summaries, path, q):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    points = sorted(per_n_summaries, key=lambda item: item["n"])
    n = np.asarray([item["n"] for item in points])
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    ax = axes[0]
    for key, label, color, marker in (
        ("paper_runtime_ms", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o"),
        ("single_runtime_ms", "Single interval", SINGLE_COLOR, "s"),
        ("quant_single_query_ms", f"Quantized q={q}", BASE.PLOT_COLORS["pareto"], "P"),
    ):
        ax.plot(n, [item["paper_matched_vlm"][key]["median"] for item in points],
                marker=marker, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Channels N")
    ax.set_ylabel("CPU time per target (ms, median; log)")
    ax.set_title("A. Reference computation cost")
    BASE.polish_axis(ax)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for mode, marker in zip(SPATIAL_MODES, ("o", "s", "D")):
        ax.plot(
            n,
            [item["coverage_by_ordering_vlm"][mode]["single_extra_vs_quant_pct"]["median"] for item in points],
            marker=marker,
            label=SPATIAL_LABELS[mode],
        )
    ax.axhline(0, color="#455A64", linewidth=1.0)
    ax.set_xlabel("Channels N")
    ax.set_ylabel("Single latency relative to q=1024 (%, median)")
    ax.set_title("B. Typical common-target comparison")
    BASE.polish_axis(ax)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for mode, marker in zip(SPATIAL_MODES, ("o", "s", "D")):
        ax.plot(
            n,
            [item["coverage_by_ordering_vlm"][mode]["quant_chunks"]["mean"] for item in points],
            marker=marker,
            label=SPATIAL_LABELS[mode],
        )
    ax.set_xlabel("Channels N")
    ax.set_ylabel("Mean q=1024 chunk count")
    ax.set_title("C. Structure selected by q-Pareto")
    BASE.polish_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_quant_reliability(per_n_coverage, path, q):
    """Show when fixed q loses the partial frontier and returns the full mask."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    n_values = np.asarray(sorted(per_n_coverage))
    representative_targets = (0.30, 0.70, 0.95)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(representative_targets)))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    ax = axes[0]
    for target, color in zip(representative_targets, colors):
        rates = []
        for n in n_values:
            rows = [
                row for row in per_n_coverage[int(n)]
                if float(row["coverage_target"]) == target
            ]
            rates.append(100 * np.mean([bool(row["quant_full_mask"]) for row in rows]))
        ax.plot(n_values, rates, marker="o", color=color, label=f"alpha={target:g}")
    ax.set_xlabel("Channels N")
    ax.set_ylabel("q-Pareto full-mask fallback (%)")
    ax.set_title("A. Frontier collapse at fixed q")
    BASE.polish_axis(ax)
    ax.legend(frameon=False)

    ax = axes[1]
    for target, color in zip(representative_targets, colors):
        overshoot = []
        for n in n_values:
            rows = [
                row for row in per_n_coverage[int(n)]
                if float(row["coverage_target"]) == target
            ]
            overshoot.append(np.mean([float(row["quant_overshoot"]) for row in rows]))
        ax.plot(n_values, overshoot, marker="o", color=color, label=f"alpha={target:g}")
    ax.set_xlabel("Channels N")
    ax.set_ylabel("Mean q-Pareto importance overshoot")
    ax.set_title("B. Cost of lost resolution")
    BASE.polish_axis(ax)
    ax.legend(frameon=False)

    ax = axes[2]
    for mode, marker in zip(SPATIAL_MODES, ("o", "s", "D")):
        rates = []
        for n in n_values:
            rows = [row for row in per_n_coverage[int(n)] if row["spatial_mode"] == mode]
            rates.append(100 * np.mean([bool(row["quant_beats_single"]) for row in rows]))
        ax.plot(n_values, rates, marker=marker, label=SPATIAL_LABELS[mode])
    ax.set_xlabel("Channels N")
    ax.set_ylabel("Cases where q=1024 beats single (%)")
    ax.set_title("C. q-Pareto usefulness despite collapse")
    BASE.polish_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Quantized Pareto reliability at q={q}", fontsize=13)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate_fast_oracle():
    rng = np.random.default_rng(51)
    # Trigger compilation before any measured call.
    FastQuantizedParetoOracle(np.full(8, 1 / 8), 0.01, 0.001, 8).solve(0.5)
    for n in (8, 17, 31):
        values = rng.lognormal(size=n)
        values /= values.sum()
        for q in (8, 16, 32):
            reference = BASE.QuantizedParetoOracle(values, 0.011, 0.0007, q)
            fast = FastQuantizedParetoOracle(values, 0.011, 0.0007, q)
            for target in (0.1, 0.5, 0.9, 0.99):
                expected = reference.solve(target)
                actual = fast.solve(target)
                if not np.array_equal(expected["mask"], actual["mask"]):
                    raise RuntimeError("compiled q-Pareto mask differs from reference")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--q", type=int, default=1024)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--dtype-bytes", type=float, default=2.0)
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=[item["n"] for item in MATRIX_CONFIGS]
    )
    parser.add_argument(
        "--cv-targets", type=float, nargs="+",
        default=[1.07, 1.25, 1.44, 2.48, 3.30, 4.55, 9.19],
    )
    parser.add_argument(
        "--budget-fractions", type=float, nargs="+",
        default=[0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875],
    )
    parser.add_argument(
        "--coverage-targets", type=float, nargs="+",
        default=[0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99],
    )
    parser.add_argument("--local-rho", type=float, default=0.95)
    parser.add_argument("--hot-rho", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials < 1 or args.q < 2:
        raise SystemExit("--trials must be positive and --q at least 2")
    config_by_n = {item["n"]: item for item in MATRIX_CONFIGS}
    unknown = sorted(set(args.n_values) - set(config_by_n))
    if unknown:
        raise SystemExit(f"no real-matrix metadata for N={unknown}")
    configs = [config_by_n[n] for n in sorted(set(args.n_values))]
    cv_targets = sorted(set(args.cv_targets))
    coverage_targets = sorted(set(args.coverage_targets))
    budget_fractions = sorted(set(args.budget_fractions))
    if not args.skip_self_check:
        validate_fast_oracle()
        EXP2.self_check()

    table = BASE.LatencyTable.load(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_n_summary = []
    per_n_coverage = {}

    for config in configs:
        n = int(config["n"])
        row_size_kib = float(config["cols"]) * args.dtype_bytes / 1024.0
        a_ms, c_ms_per_row, fit_r2 = BASE.affine_fit(table, row_size_kib)
        params = BASE.ChunkParams(
            start_kb=config["start_kib"], jump_cap_kb=config["jump_cap_kib"]
        )
        budgets = [int(round(fraction * n)) for fraction in budget_fractions]
        rng = np.random.default_rng(args.seed + n)
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()
        coverage_rows = []
        paper_rows = []
        input_rows = []
        total_multisets = args.trials * len(cv_targets)
        completed = 0

        for trial in range(args.trials):
            for target_cv in cv_targets:
                base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
                variants = EXP2.spatial_variants(
                    base_values, rng, hotness, args.local_rho, args.hot_rho
                )
                for mode, values in variants.items():
                    oracle, quant_build_ms = timed_call(
                        FastQuantizedParetoOracle,
                        values,
                        a_ms,
                        c_ms_per_row,
                        args.q,
                    )
                    lag1, first_half = EXP2.spatial_stats(values)
                    input_rows.append(
                        {
                            "trial": trial,
                            "target_cv": target_cv,
                            "spatial_mode": mode,
                            "actual_cv": EXP2.coefficient_of_variation(values),
                            "lag1_corr": lag1,
                            "first_half_mass": first_half,
                            "quant_build_ms": quant_build_ms,
                            "quant_peak_states": oracle.peak_states,
                        }
                    )
                    query_count = len(coverage_targets) + len(budgets)
                    for target in coverage_targets:
                        solved = solve_pair(values, target, oracle, a_ms, c_ms_per_row)
                        solved["quant_build_ms"] = quant_build_ms
                        solved["quant_single_query_ms"] = quant_build_ms + solved["quant_solve_ms"]
                        solved["quant_amortized_ms"] = quant_build_ms / query_count + solved["quant_solve_ms"]
                        solved["quant_full_mask"] = solved["quant_rows"] == n
                        coverage_rows.append(
                            {
                                "trial": trial,
                                "target_cv": target_cv,
                                "spatial_mode": mode,
                                "coverage_target": target,
                                **solved,
                            }
                        )

                    values_t = torch.from_numpy(values.astype(np.float32))
                    for budget, fraction in zip(budgets, budget_fractions):
                        paper, paper_runtime_ms = timed_call(
                            BASE.select_chunks,
                            values_t,
                            budget,
                            row_size_kib,
                            table,
                            params=params,
                            impl="torch",
                        )
                        paper_mask = paper.mask.cpu().numpy()
                        paper_row_count, paper_chunks = BASE.mask_stats(paper_mask)
                        if paper_row_count == 0:
                            raise RuntimeError("Paper greedy selected an empty mask")
                        paper_importance = float(values[paper_mask].sum())
                        paper_aff_ms = BASE.affine_latency(paper_mask, a_ms, c_ms_per_row)
                        solved = solve_pair(
                            values, paper_importance, oracle, a_ms, c_ms_per_row
                        )
                        solved["quant_build_ms"] = quant_build_ms
                        solved["quant_single_query_ms"] = quant_build_ms + solved["quant_solve_ms"]
                        solved["quant_amortized_ms"] = quant_build_ms / query_count + solved["quant_solve_ms"]
                        solved["quant_full_mask"] = solved["quant_rows"] == n
                        paper_rows.append(
                            {
                                "trial": trial,
                                "target_cv": target_cv,
                                "spatial_mode": mode,
                                "budget_rows": budget,
                                "budget_fraction": fraction,
                                "paper_importance": paper_importance,
                                "paper_rows": paper_row_count,
                                "paper_chunks": paper_chunks,
                                "paper_aff_ms": paper_aff_ms,
                                "paper_runtime_ms": paper_runtime_ms,
                                "paper_extra_vs_quant_pct": 100.0
                                * (paper_aff_ms / solved["quant_aff_ms"] - 1.0),
                                "paper_extra_vs_single_pct": 100.0
                                * (paper_aff_ms / solved["single_aff_ms"] - 1.0),
                                "quant_beats_paper": paper_aff_ms > solved["quant_aff_ms"] + 1e-15,
                                **solved,
                            }
                        )
                completed += 1
                if completed % max(1, total_multisets // 10) == 0:
                    print(f"N={n}: completed {completed}/{total_multisets} multisets", flush=True)

        result_dir = args.output_dir / f"n_{n}"
        result_dir.mkdir(parents=True, exist_ok=True)
        write_csv(result_dir / "coverage_trials.csv", coverage_rows)
        write_csv(result_dir / "paper_matched_trials.csv", paper_rows)
        write_csv(result_dir / "input_trials.csv", input_rows)

        vlm_coverage = [r for r in coverage_rows if float(r["target_cv"]) < 9.0]
        vlm_paper = [r for r in paper_rows if float(r["target_cv"]) < 9.0]
        coverage_metrics = (
            "single_rows", "single_aff_ms", "single_runtime_ms", "single_overshoot",
            "quant_rows", "quant_chunks", "quant_aff_ms", "quant_build_ms",
            "quant_solve_ms", "quant_single_query_ms", "quant_amortized_ms",
            "quant_overshoot", "quant_full_mask", "quant_beats_single",
            "single_extra_vs_quant_pct",
        )
        paper_metrics = (
            "paper_rows", "paper_chunks", "paper_aff_ms", "paper_runtime_ms",
            "paper_extra_vs_quant_pct", "paper_extra_vs_single_pct",
            "quant_beats_paper",
            *coverage_metrics,
        )
        coverage_vlm = {
            key: EXP2.distribution_stats([float(row[key]) for row in vlm_coverage])
            for key in coverage_metrics
        }
        paper_vlm = {
            key: EXP2.distribution_stats([float(row[key]) for row in vlm_paper])
            for key in paper_metrics
        }
        by_ordering_rows = EXP2.summarize(
            vlm_coverage, ("spatial_mode",), coverage_metrics
        )
        by_ordering = {item["spatial_mode"]: item for item in by_ordering_rows}
        metadata = {
            "format": "experiment-05-realistic-n-v1",
            "trials_per_cv": args.trials,
            "seed": args.seed + n,
            "n": n,
            "model": config["model"],
            "shape_rows_x_cols": config["shape"],
            "dtype_bytes": args.dtype_bytes,
            "row_size_kib": row_size_kib,
            "paper_chunk_params_kib": {
                "start": config["start_kib"],
                "jump_cap": config["jump_cap_kib"],
            },
            "q": args.q,
            "cv_targets": cv_targets,
            "coverage_targets": coverage_targets,
            "budget_fractions": budget_fractions,
            "spatial_modes": list(SPATIAL_MODES),
            "affine_fit": {
                "a_ms_per_chunk": a_ms,
                "c_ms_per_row": c_ms_per_row,
                "r_squared": fit_r2,
                "chunk_open_equivalent_rows": a_ms / c_ms_per_row,
            },
            "paper_work_proxy": EXP3.candidate_work(n, row_size_kib, table, params),
            "coverage_vlm": coverage_vlm,
            "paper_matched_vlm": paper_vlm,
            "coverage_by_ordering_vlm": by_ordering,
            "timing_scope": "single-process CPU; q build is compiled numba and one-target time includes build",
        }
        (result_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
        all_n_summary.append(metadata)
        per_n_coverage[n] = vlm_coverage

        title = f"{config['model']} down projection ({config['shape']}, row={row_size_kib:g} KiB)"
        plot_method_grid(
            coverage_rows, paper_rows, result_dir / "importance_latency_frontiers.png",
            args.q, title,
            {"paper": "paper_aff_ms", "single": "single_aff_ms", "quant": "quant_aff_ms"},
            {"paper": "paper_importance", "single": "single_importance", "quant": "quant_importance"},
            "Affine latency (ms, log scale)", "Retained importance", True,
        )
        plot_method_grid(
            coverage_rows, paper_rows, result_dir / "r_importance.png",
            args.q, title,
            {"paper": "paper_rows", "single": "single_rows", "quant": "quant_rows"},
            {"paper": "paper_importance", "single": "single_importance", "quant": "quant_importance"},
            "Selected rows R", "Retained importance",
        )
        plot_method_grid(
            coverage_rows, paper_rows, result_dir / "r_latency.png",
            args.q, title,
            {"paper": "paper_rows", "single": "single_rows", "quant": "quant_rows"},
            {"paper": "paper_aff_ms", "single": "single_aff_ms", "quant": "quant_aff_ms"},
            "Selected rows R", "Affine latency (ms)",
        )
        plot_quant_chunks(
            coverage_rows, result_dir / "quant_chunk_distribution.png",
            f"Quantized Pareto q={args.q} chunks: N={n}, row={row_size_kib:g} KiB",
        )
        print(f"N={n}: wrote results to {result_dir}", flush=True)

    aggregate = {
        "format": "experiment-05-realistic-n-aggregate-v1",
        "trials_per_cv": args.trials,
        "q": args.q,
        "n_values": [item["n"] for item in all_n_summary],
        "per_n": all_n_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    plot_scaling(all_n_summary, args.output_dir / "n_scaling.png", args.q)
    plot_quant_reliability(
        per_n_coverage, args.output_dir / "quant_reliability.png", args.q
    )
    print(f"wrote aggregate results to {args.output_dir}")


if __name__ == "__main__":
    main()
