#!/usr/bin/env python3
"""Experiment 06: Top-R and the q scale needed at realistic channel counts.

The experiment keeps Experiment 05's real down-projection dimensions, adds
Top-R, and sweeps q for Quantized Pareto.  Exact Coverage is not used.  The
q-sweep implementation retains cost, importance, row count, and chunk count
but deliberately omits mask backtracking, reducing memory from O(Nq) to O(q).
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
    raise SystemExit("Experiment 06 requires numba") from error


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_05 = PROJECT_ROOT / "experiments" / "05_realistic_n" / "run_experiment.py"


def load_experiment_05():
    spec = importlib.util.spec_from_file_location("experiment_05", EXPERIMENT_05)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 05 from {EXPERIMENT_05}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP5 = load_experiment_05()
EXP4 = EXP5.EXP4
EXP3 = EXP5.EXP3
EXP2 = EXP5.EXP2
BASE = EXP5.BASE
MATRIX_CONFIGS = EXP5.MATRIX_CONFIGS
SPATIAL_MODES = EXP5.SPATIAL_MODES
SPATIAL_LABELS = EXP5.SPATIAL_LABELS
SINGLE_COLOR = EXP5.SINGLE_COLOR
TOP_COLOR = BASE.PLOT_COLORS["top_r"]

DEFAULT_Q_BY_N = {
    4864: [1024, 2048, 4096, 8192, 16384, 32768],
    8960: [1024, 2048, 4096, 8192, 16384, 32768, 65536],
    14336: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    18944: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
}


@njit(cache=True)
def _quantized_metric_kernel(values, a_ms, c_ms_per_row, q):
    """The Experiment 05 q recurrence without O(Nq) parent storage."""
    total = values.sum()
    costs = np.full((2, q + 1), np.inf, dtype=np.float64)
    importance = np.full((2, q + 1), -np.inf, dtype=np.float64)
    rows = np.full((2, q + 1), -1, dtype=np.int32)
    chunks = np.full((2, q + 1), -1, dtype=np.int32)
    new_costs = np.empty((2, q + 1), dtype=np.float64)
    new_importance = np.empty((2, q + 1), dtype=np.float64)
    new_rows = np.empty((2, q + 1), dtype=np.int32)
    new_chunks = np.empty((2, q + 1), dtype=np.int32)
    costs[0, 0] = 0.0
    importance[0, 0] = 0.0
    rows[0, 0] = 0
    chunks[0, 0] = 0
    peak_states = 1

    for value in values:
        new_costs[:, :] = np.inf
        new_importance[:, :] = -np.inf
        new_rows[:, :] = -1
        new_chunks[:, :] = -1
        for previous_state in range(2):
            for previous_bucket in range(q + 1):
                old_cost = costs[previous_state, previous_bucket]
                if not np.isfinite(old_cost):
                    continue
                old_importance = importance[previous_state, previous_bucket]
                old_rows = rows[previous_state, previous_bucket]
                old_chunks = chunks[previous_state, previous_bucket]

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
                    new_rows[0, skip_bucket] = old_rows
                    new_chunks[0, skip_bucket] = old_chunks

                selected_importance = old_importance + value
                selected_cost = old_cost + c_ms_per_row
                selected_chunks = old_chunks
                if previous_state == 0:
                    selected_cost += a_ms
                    selected_chunks += 1
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
                    new_rows[1, selected_bucket] = old_rows + 1
                    new_chunks[1, selected_bucket] = selected_chunks

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
                    new_rows[ending, bucket] = -1
                    new_chunks[ending, bucket] = -1
                else:
                    best_higher_cost = candidate_cost
                    live_states += 1

        old_costs = costs
        costs = new_costs
        new_costs = old_costs
        old_importance_array = importance
        importance = new_importance
        new_importance = old_importance_array
        old_rows_array = rows
        rows = new_rows
        new_rows = old_rows_array
        old_chunks_array = chunks
        chunks = new_chunks
        new_chunks = old_chunks_array
        if live_states > peak_states:
            peak_states = live_states

    return costs, importance, rows, chunks, peak_states


class QuantizedMetricOracle:
    """Quantized Pareto metrics with O(q) memory and no returned mask."""

    def __init__(self, values, a_ms, c_ms_per_row, q):
        self.values = np.asarray(values, dtype=np.float64)
        self.a_ms = float(a_ms)
        self.c_ms_per_row = float(c_ms_per_row)
        self.q = int(q)
        self.total = float(self.values.sum())
        if self.q < 2 or self.total <= 0:
            raise ValueError("q must be at least 2 and total importance positive")
        (
            self.costs,
            self.importance,
            self.rows,
            self.chunks,
            self.peak_states,
        ) = _quantized_metric_kernel(
            self.values, self.a_ms, self.c_ms_per_row, self.q
        )

    def solve(self, bound):
        if not 0 < bound <= self.total + 1e-12:
            raise ValueError("coverage bound must lie in (0, total importance]")
        best_key = (math.inf, math.inf)
        best = None
        for ending in (0, 1):
            for bucket in np.flatnonzero(np.isfinite(self.costs[ending])):
                achieved = float(self.importance[ending, bucket])
                if achieved < bound - 1e-14:
                    continue
                key = (float(self.costs[ending, bucket]), -achieved)
                if key < best_key:
                    best_key = key
                    best = (ending, int(bucket))
        full_cost = self.a_ms + self.c_ms_per_row * len(self.values)
        if best is None or full_cost < best_key[0] - 1e-15:
            return {
                "importance": self.total,
                "rows": len(self.values),
                "chunks": 1,
                "aff_ms": full_cost,
                "peak_states": int(self.peak_states),
            }
        ending, bucket = best
        return {
            "importance": float(self.importance[ending, bucket]),
            "rows": int(self.rows[ending, bucket]),
            "chunks": int(self.chunks[ending, bucket]),
            "aff_ms": float(self.costs[ending, bucket]),
            "peak_states": int(self.peak_states),
        }


class TopROracle:
    """Smallest Top-R prefix that reaches a coverage target."""

    def __init__(self, values, a_ms, c_ms_per_row):
        self.values = np.asarray(values, dtype=np.float64)
        self.order = np.argsort(-self.values, kind="stable")
        self.prefix = np.cumsum(self.values[self.order])
        self.a_ms = float(a_ms)
        self.c_ms_per_row = float(c_ms_per_row)

    def fixed(self, count):
        count = int(count)
        if not 1 <= count <= len(self.values):
            raise ValueError("Top-R count outside valid range")
        return self._make(count)

    def solve(self, bound):
        count = int(np.searchsorted(self.prefix, bound - 1e-14, side="left") + 1)
        return self._make(min(count, len(self.values)))

    def _make(self, count):
        mask = np.zeros(len(self.values), dtype=bool)
        mask[self.order[:count]] = True
        rows, chunks = BASE.mask_stats(mask)
        return {
            "importance": float(self.values[mask].sum()),
            "rows": rows,
            "chunks": chunks,
            "aff_ms": self.a_ms * chunks + self.c_ms_per_row * rows,
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


def solution_fields(prefix, solution, runtime_ms=None):
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


def grouped(rows, keys, metrics):
    return EXP2.summarize(rows, keys, metrics)


def aggregate_curve(rows, cv, mode, x_key, metrics):
    selected = [
        row for row in rows
        if float(row["target_cv"]) == cv and row["spatial_mode"] == mode
    ]
    return grouped(selected, (x_key,), metrics)


def plot_method_grid(coverage_rows, fixed_rows, path, max_q, x_fields, y_fields,
                     xlabel, ylabel, x_log=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    coverage_rows = [row for row in coverage_rows if int(row["q"]) == max_q]
    cvs = sorted({float(row["target_cv"]) for row in coverage_rows})
    display_cvs = [min(cvs, key=lambda value: abs(value - wanted))
                   for wanted in (1.25, 3.30, 4.55)]
    display_cvs = list(dict.fromkeys(display_cvs))
    fig, axes = plt.subplots(len(display_cvs), 3, figsize=(14.8, 4 * len(display_cvs)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, cv in enumerate(display_cvs):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            coverage = aggregate_curve(
                coverage_rows, cv, mode, "coverage_target",
                tuple({
                    x_fields["single"], x_fields["quant"],
                    y_fields["single"], y_fields["quant"],
                }),
            )
            fixed = aggregate_curve(
                fixed_rows, cv, mode, "budget_fraction",
                (x_fields["paper"], y_fields["paper"],
                 x_fields["top_fixed"], y_fields["top_fixed"]),
            )
            specs = (
                (fixed, "paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "-"),
                (fixed, "top_fixed", "Top-R", TOP_COLOR, "^", ":"),
                (coverage, "single", "Single interval", SINGLE_COLOR, "s", "-"),
                (coverage, "quant", f"Quantized q={max_q}", BASE.PLOT_COLORS["pareto"], "P", "--"),
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


def plot_chunk_histograms(matched_rows, path, max_q, n):
    """Actual histograms: horizontal K, vertical fraction of solutions."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    rows = [
        row for row in matched_rows
        if int(row["q"]) == max_q and float(row["target_cv"]) < 9.0
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    for column, mode in enumerate(SPATIAL_MODES):
        selected = [row for row in rows if row["spatial_mode"] == mode]
        quant = np.asarray([int(row["quant_chunks"]) for row in selected])
        ax = axes[0, column]
        maximum = int(quant.max())
        if maximum <= 200:
            bins = np.arange(0.5, maximum + 1.5, 1.0)
        else:
            bins = np.unique(np.rint(np.geomspace(1, maximum + 1, 35))).astype(float)
            if len(bins) < 2:
                bins = np.asarray([0.5, 1.5])
        ax.hist(quant, bins=bins, weights=np.ones(len(quant)) / len(quant),
                color=BASE.PLOT_COLORS["pareto"], alpha=0.82, edgecolor="white")
        if maximum > 200:
            ax.set_xscale("log")
        ax.set_xlabel("Chunk count K")
        ax.set_ylabel("Fraction of solutions")
        ax.set_title(f"{SPATIAL_LABELS[mode]}: Quantized q={max_q}")
        BASE.polish_axis(ax)

        ax = axes[1, column]
        method_values = {
            "Paper greedy": np.asarray([int(row["paper_chunks"]) for row in selected]),
            "Top-R (matched target)": np.asarray([int(row["top_chunks"]) for row in selected]),
            f"Quantized q={max_q}": quant,
        }
        upper = max(int(values.max()) for values in method_values.values())
        edges = np.unique(np.rint(np.geomspace(1, upper + 1, 40))).astype(float)
        edges = np.unique(np.r_[0.5, edges + 0.5])
        for (label, values), color in zip(
            method_values.items(),
            (BASE.PLOT_COLORS["greedy"], TOP_COLOR, BASE.PLOT_COLORS["pareto"]),
        ):
            ax.hist(values, bins=edges, weights=np.ones(len(values)) / len(values),
                    histtype="step", linewidth=1.8, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Chunk count K (log scale)")
        ax.set_ylabel("Fraction of solutions")
        ax.set_title(f"{SPATIAL_LABELS[mode]}: method comparison")
        BASE.polish_axis(ax)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"Chunk-count histograms at Paper-matched targets (N={n:,})",
                 fontsize=13, y=1.02)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_q_tradeoff(per_n_q, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    specs = (
        ("nonworse_rate", "Raw q non-worse than Paper (%)", 100.0),
        ("strict_win_rate", "Raw q strictly beats Paper (%)", 100.0),
        ("full_mask_rate", "Full-mask solutions (%)", 100.0),
        ("build_ms_median", "q frontier build time (ms)", 1.0),
    )
    for ax, (metric, ylabel, scale) in zip(axes.flat, specs):
        for n, entries in sorted(per_n_q.items()):
            points = sorted(entries, key=lambda item: item["q"])
            ax.plot([item["q"] for item in points],
                    [scale * item[metric] for item in points], marker="o", label=f"N={n:,}")
        ax.set_xscale("log", base=2)
        if metric == "build_ms_median":
            ax.set_yscale("log")
        else:
            ax.set_ylim(-3, 103)
        ax.set_xlabel("Quantization buckets q")
        ax.set_ylabel(ylabel)
        BASE.polish_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate_metric_oracle():
    rng = np.random.default_rng(606)
    QuantizedMetricOracle(np.full(8, 1 / 8), 0.01, 0.001, 8).solve(0.5)
    for n in (8, 17, 31):
        values = rng.lognormal(size=n)
        values /= values.sum()
        for q in (8, 16, 32):
            reference = EXP5.FastQuantizedParetoOracle(values, 0.011, 0.0007, q)
            metric = QuantizedMetricOracle(values, 0.011, 0.0007, q)
            for target in (0.1, 0.5, 0.9, 0.99):
                expected = reference.solve(target)
                actual = metric.solve(target)
                for key in ("importance", "rows", "chunks", "aff_ms"):
                    if not math.isclose(float(expected[key]), float(actual[key]),
                                        rel_tol=1e-11, abs_tol=1e-13):
                        raise RuntimeError(f"metric q oracle differs on {key}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--q-values", type=int, nargs="+", default=None,
                        help="override the per-N default q grids")
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[item["n"] for item in MATRIX_CONFIGS])
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
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_q_values = sorted(set(args.q_values)) if args.q_values else None
    if args.trials < 1 or (
        requested_q_values is not None and requested_q_values[0] < 2
    ):
        raise SystemExit("trials must be positive and q values at least 2")
    config_by_n = {item["n"]: item for item in MATRIX_CONFIGS}
    unknown = sorted(set(args.n_values) - set(config_by_n))
    if unknown:
        raise SystemExit(f"no real-matrix metadata for N={unknown}")
    configs = [config_by_n[n] for n in sorted(set(args.n_values))]
    cv_targets = sorted(set(args.cv_targets))
    coverage_targets = sorted(set(args.coverage_targets))
    budget_fractions = sorted(set(args.budget_fractions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_summaries = [
            json.loads((args.output_dir / f"n_{item['n']}" / "summary.json").read_text())
            for item in configs
        ]
        for item in aggregate_summaries:
            n = int(item["n"])
            result_dir = args.output_dir / f"n_{n}"
            with (result_dir / "coverage_trials.csv").open() as handle:
                coverage_rows = list(csv.DictReader(handle))
            with (result_dir / "paper_matched_trials.csv").open() as handle:
                matched_rows = list(csv.DictReader(handle))
            with (result_dir / "fixed_r_trials.csv").open() as handle:
                fixed_rows = list(csv.DictReader(handle))
            stable = [
                entry["q"] for entry in item["q_summary_vlm"]
                if entry["stable_better"]
            ]
            selected_q = min(stable) if stable else int(item["max_q"])
            selected_rows = [
                row for row in matched_rows
                if int(row["q"]) == selected_q and float(row["target_cv"]) < 9.0
            ]
            item.pop("max_q_method_comparison_vlm", None)
            item["selected_q"] = selected_q
            item["selected_q_method_comparison_vlm"] = {
                "paper_extra_vs_top_pct": distribution([
                    row["paper_extra_vs_top_pct"] for row in selected_rows
                ]),
                "paper_extra_vs_single_pct": distribution([
                    row["paper_extra_vs_single_pct"] for row in selected_rows
                ]),
                "paper_extra_vs_quant_pct": distribution([
                    row["paper_extra_vs_quant_pct"] for row in selected_rows
                ]),
                "top_chunks": distribution([row["top_chunks"] for row in selected_rows]),
                "paper_chunks": distribution([row["paper_chunks"] for row in selected_rows]),
                "quant_chunks": distribution([row["quant_chunks"] for row in selected_rows]),
            }
            (result_dir / "summary.json").write_text(json.dumps(item, indent=2) + "\n")
            plot_method_grid(
                coverage_rows, fixed_rows,
                result_dir / "importance_latency_frontiers.png", selected_q,
                {"paper": "paper_aff_ms", "top_fixed": "top_fixed_aff_ms",
                 "single": "single_aff_ms", "quant": "quant_aff_ms"},
                {"paper": "paper_importance", "top_fixed": "top_fixed_importance",
                 "single": "single_importance", "quant": "quant_importance"},
                "Affine latency (ms, log scale)", "Retained importance", True,
            )
            plot_method_grid(
                coverage_rows, fixed_rows, result_dir / "r_importance.png", selected_q,
                {"paper": "paper_rows", "top_fixed": "top_fixed_rows",
                 "single": "single_rows", "quant": "quant_rows"},
                {"paper": "paper_importance", "top_fixed": "top_fixed_importance",
                 "single": "single_importance", "quant": "quant_importance"},
                "Selected rows R", "Retained importance",
            )
            plot_method_grid(
                coverage_rows, fixed_rows, result_dir / "r_latency.png", selected_q,
                {"paper": "paper_rows", "top_fixed": "top_fixed_rows",
                 "single": "single_rows", "quant": "quant_rows"},
                {"paper": "paper_aff_ms", "top_fixed": "top_fixed_aff_ms",
                 "single": "single_aff_ms", "quant": "quant_aff_ms"},
                "Selected rows R", "Affine latency (ms)",
            )
            plot_chunk_histograms(
                matched_rows, result_dir / "chunk_count_histograms.png", selected_q, n
            )
        per_n_q = {item["n"]: item["q_summary_vlm"] for item in aggregate_summaries}
        aggregate = {
            "format": "experiment-06-quant-scale-aggregate-v1",
            "trials_per_cv": aggregate_summaries[0]["trials_per_cv"],
            "q_values": sorted({q for item in aggregate_summaries for q in item["q_values"]}),
            "n_values": [item["n"] for item in aggregate_summaries],
            "per_n": aggregate_summaries,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
        plot_q_tradeoff(per_n_q, args.output_dir / "q_scale_tradeoff.png")
        print(f"aggregated existing results in {args.output_dir}", flush=True)
        return
    if not args.skip_self_check:
        validate_metric_oracle()
        EXP2.self_check()

    table = BASE.LatencyTable.load(args.profile)
    aggregate_summaries = []
    per_n_q = {}

    for config in configs:
        n = int(config["n"])
        q_values = requested_q_values or DEFAULT_Q_BY_N[n]
        result_dir = args.output_dir / f"n_{n}"
        result_dir.mkdir(parents=True, exist_ok=True)
        row_size_kib = float(config["cols"]) * args.dtype_bytes / 1024.0
        a_ms, c_ms, fit_r2 = BASE.affine_fit(table, row_size_kib)
        params = BASE.ChunkParams(start_kb=config["start_kib"],
                                  jump_cap_kb=config["jump_cap_kib"])
        budgets = [int(round(n * fraction)) for fraction in budget_fractions]
        rng = np.random.default_rng(args.seed + n)
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()
        coverage_rows = []
        matched_rows = []
        fixed_rows = []
        input_q_rows = []
        total_inputs = args.trials * len(cv_targets) * len(SPATIAL_MODES)
        completed = 0

        for trial in range(args.trials):
            for target_cv in cv_targets:
                base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
                variants = EXP2.spatial_variants(
                    base_values, rng, hotness, args.local_rho, args.hot_rho
                )
                for mode, values in variants.items():
                    total = float(values.sum())
                    top_oracle, top_build_ms = timed_call(TopROracle, values, a_ms, c_ms)
                    common_base = []
                    for alpha in coverage_targets:
                        target = alpha * total
                        single, single_ms = timed_call(
                            EXP4.minimum_single_interval, values, target, a_ms, c_ms
                        )
                        top, top_ms = timed_call(top_oracle.solve, target)
                        common_base.append((alpha, target, single, single_ms, top, top_ms))

                    values_t = torch.from_numpy(values.astype(np.float32))
                    paper_base = []
                    for fraction, budget in zip(budget_fractions, budgets):
                        paper, paper_ms = timed_call(
                            BASE.select_chunks, values_t, budget, row_size_kib, table,
                            params=params, impl="torch",
                        )
                        paper_mask = paper.mask.cpu().numpy()
                        paper_count, paper_chunks = BASE.mask_stats(paper_mask)
                        paper_solution = {
                            "importance": float(values[paper_mask].sum()),
                            "rows": paper_count,
                            "chunks": paper_chunks,
                            "aff_ms": BASE.affine_latency(paper_mask, a_ms, c_ms),
                        }
                        top_fixed, top_fixed_ms = timed_call(top_oracle.fixed, budget)
                        fixed_rows.append({
                            "trial": trial, "target_cv": target_cv,
                            "spatial_mode": mode, "budget_fraction": fraction,
                            "budget_rows": budget,
                            **solution_fields("paper", paper_solution, paper_ms),
                            **solution_fields("top_fixed", top_fixed, top_fixed_ms),
                            "top_fixed_importance_gain_pct": 100.0 *
                            (top_fixed["importance"] / paper_solution["importance"] - 1.0),
                        })
                        single, single_ms = timed_call(
                            EXP4.minimum_single_interval, values,
                            paper_solution["importance"], a_ms, c_ms,
                        )
                        top, top_ms = timed_call(
                            top_oracle.solve, paper_solution["importance"]
                        )
                        paper_base.append((fraction, budget, paper_solution, paper_ms,
                                           single, single_ms, top, top_ms))

                    query_count = len(common_base) + len(paper_base)
                    for q in q_values:
                        oracle, build_ms = timed_call(
                            QuantizedMetricOracle, values, a_ms, c_ms, q
                        )
                        input_q_rows.append({
                            "trial": trial, "target_cv": target_cv,
                            "spatial_mode": mode, "q": q,
                            "top_build_ms": top_build_ms,
                            "quant_build_ms": build_ms,
                            "quant_peak_states": oracle.peak_states,
                            "quant_state_slots": 2 * (q + 1),
                        })
                        for alpha, target, single, single_ms, top, top_ms in common_base:
                            quant, quant_ms = timed_call(oracle.solve, target)
                            coverage_rows.append({
                                "trial": trial, "target_cv": target_cv,
                                "spatial_mode": mode, "coverage_target": alpha,
                                "target_importance": target, "q": q,
                                **solution_fields("single", single, single_ms),
                                **solution_fields("top", top, top_ms),
                                **solution_fields("quant", quant, quant_ms),
                                "quant_build_ms": build_ms,
                                "quant_one_query_ms": build_ms + quant_ms,
                                "quant_amortized_ms": build_ms / query_count + quant_ms,
                                "quant_full_mask": quant["rows"] == n,
                            })
                        for (fraction, budget, paper, paper_ms, single, single_ms,
                             top, top_ms) in paper_base:
                            quant, quant_ms = timed_call(oracle.solve, paper["importance"])
                            gain = 100.0 * (paper["aff_ms"] / quant["aff_ms"] - 1.0)
                            matched_rows.append({
                                "trial": trial, "target_cv": target_cv,
                                "spatial_mode": mode, "budget_fraction": fraction,
                                "budget_rows": budget, "q": q,
                                **solution_fields("paper", paper, paper_ms),
                                **solution_fields("single", single, single_ms),
                                **solution_fields("top", top, top_ms),
                                **solution_fields("quant", quant, quant_ms),
                                "quant_build_ms": build_ms,
                                "quant_one_query_ms": build_ms + quant_ms,
                                "quant_amortized_ms": build_ms / query_count + quant_ms,
                                "quant_full_mask": quant["rows"] == n,
                                "paper_extra_vs_quant_pct": gain,
                                "quant_strictly_beats_paper": gain > 1e-10,
                                "quant_nonworse_than_paper": gain >= -1e-10,
                                "paper_extra_vs_top_pct": 100.0 *
                                (paper["aff_ms"] / top["aff_ms"] - 1.0),
                                "paper_extra_vs_single_pct": 100.0 *
                                (paper["aff_ms"] / single["aff_ms"] - 1.0),
                            })
                        del oracle

                    completed += 1
                    print(f"N={n}: completed {completed}/{total_inputs} inputs", flush=True)

        write_csv(result_dir / "coverage_trials.csv", coverage_rows)
        write_csv(result_dir / "paper_matched_trials.csv", matched_rows)
        write_csv(result_dir / "fixed_r_trials.csv", fixed_rows)
        write_csv(result_dir / "input_q_trials.csv", input_q_rows)

        vlm_matched = [row for row in matched_rows if float(row["target_cv"]) < 9.0]
        vlm_inputs = [row for row in input_q_rows if float(row["target_cv"]) < 9.0]
        q_summary = []
        for q in q_values:
            selected = [row for row in vlm_matched if int(row["q"]) == q]
            builds = [row for row in vlm_inputs if int(row["q"]) == q]
            gains = [float(row["paper_extra_vs_quant_pct"]) for row in selected]
            nonworse = np.mean([bool(row["quant_nonworse_than_paper"]) for row in selected])
            strict = np.mean([bool(row["quant_strictly_beats_paper"]) for row in selected])
            full = np.mean([bool(row["quant_full_mask"]) for row in selected])
            gain_stats = distribution(gains)
            entry = {
                "q": q,
                "q_over_n": q / n,
                "cases": len(selected),
                "nonworse_rate": float(nonworse),
                "strict_win_rate": float(strict),
                "full_mask_rate": float(full),
                "paper_extra_vs_quant_pct": gain_stats,
                "build_ms": distribution([row["quant_build_ms"] for row in builds]),
                "build_ms_median": float(np.median(
                    [float(row["quant_build_ms"]) for row in builds]
                )),
                "one_query_ms": distribution([
                    row["quant_one_query_ms"] for row in selected
                ]),
                "stable_better": bool(
                    nonworse >= 1.0 - 1e-12 and strict >= 0.95
                    and gain_stats["p05"] >= -1e-10
                ),
            }
            q_summary.append(entry)
        stable = [item["q"] for item in q_summary if item["stable_better"]]
        max_q = max(q_values)
        selected_q = min(stable) if stable else max_q
        selected_rows = [
            row for row in vlm_matched if int(row["q"]) == selected_q
        ]
        metadata = {
            "format": "experiment-06-quant-scale-v1",
            "trials_per_cv": args.trials,
            "seed": args.seed + n,
            "n": n,
            "model": config["model"],
            "shape_rows_x_cols": config["shape"],
            "row_size_kib": row_size_kib,
            "q_values": q_values,
            "max_q": max_q,
            "selected_q": selected_q,
            "minimum_stable_q": min(stable) if stable else None,
            "stability_definition": (
                "100% non-worse than Paper, >=95% strict wins, and p05 gain >=0 "
                "on VLM-CV Paper-matched cases"
            ),
            "cv_targets": cv_targets,
            "coverage_targets": coverage_targets,
            "budget_fractions": budget_fractions,
            "affine_fit": {
                "a_ms_per_chunk": a_ms,
                "c_ms_per_row": c_ms,
                "r_squared": fit_r2,
                "chunk_open_equivalent_rows": a_ms / c_ms,
            },
            "q_summary_vlm": q_summary,
            "selected_q_method_comparison_vlm": {
                "paper_extra_vs_top_pct": distribution([
                    row["paper_extra_vs_top_pct"] for row in selected_rows
                ]),
                "paper_extra_vs_single_pct": distribution([
                    row["paper_extra_vs_single_pct"] for row in selected_rows
                ]),
                "paper_extra_vs_quant_pct": distribution([
                    row["paper_extra_vs_quant_pct"] for row in selected_rows
                ]),
                "top_chunks": distribution([row["top_chunks"] for row in selected_rows]),
                "paper_chunks": distribution([row["paper_chunks"] for row in selected_rows]),
                "quant_chunks": distribution([row["quant_chunks"] for row in selected_rows]),
            },
            "timing_scope": "single-process CPU; q one-query time includes frontier build",
        }
        (result_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
        aggregate_summaries.append(metadata)
        per_n_q[n] = q_summary

        plot_method_grid(
            coverage_rows, fixed_rows, result_dir / "importance_latency_frontiers.png",
            selected_q,
            {"paper": "paper_aff_ms", "top_fixed": "top_fixed_aff_ms",
             "single": "single_aff_ms", "quant": "quant_aff_ms"},
            {"paper": "paper_importance", "top_fixed": "top_fixed_importance",
             "single": "single_importance", "quant": "quant_importance"},
            "Affine latency (ms, log scale)", "Retained importance", True,
        )
        plot_method_grid(
            coverage_rows, fixed_rows, result_dir / "r_importance.png", selected_q,
            {"paper": "paper_rows", "top_fixed": "top_fixed_rows",
             "single": "single_rows", "quant": "quant_rows"},
            {"paper": "paper_importance", "top_fixed": "top_fixed_importance",
             "single": "single_importance", "quant": "quant_importance"},
            "Selected rows R", "Retained importance",
        )
        plot_method_grid(
            coverage_rows, fixed_rows, result_dir / "r_latency.png", selected_q,
            {"paper": "paper_rows", "top_fixed": "top_fixed_rows",
             "single": "single_rows", "quant": "quant_rows"},
            {"paper": "paper_aff_ms", "top_fixed": "top_fixed_aff_ms",
             "single": "single_aff_ms", "quant": "quant_aff_ms"},
            "Selected rows R", "Affine latency (ms)",
        )
        plot_chunk_histograms(
            matched_rows, result_dir / "chunk_count_histograms.png", selected_q, n
        )
        print(f"N={n}: wrote {result_dir}", flush=True)

    if args.skip_aggregate:
        return
    aggregate = {
        "format": "experiment-06-quant-scale-aggregate-v1",
        "trials_per_cv": args.trials,
        "q_values": sorted({q for item in aggregate_summaries for q in item["q_values"]}),
        "n_values": [item["n"] for item in aggregate_summaries],
        "per_n": aggregate_summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    plot_q_tradeoff(per_n_q, args.output_dir / "q_scale_tradeoff.png")
    print(f"wrote aggregate results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
