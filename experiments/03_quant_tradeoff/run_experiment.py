#!/usr/bin/env python3
"""Experiment 03: Quantized Pareto cost--accuracy trade-off.

Sweep the number of importance buckets q while holding the Experiment 02
CV-calibrated inputs fixed.  Exact Coverage DP supplies the accuracy oracle.
Paper greedy is timed on the same inputs and is also compared with Quantized
Pareto at the importance targets actually achieved by the greedy masks.
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


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_02 = PROJECT_ROOT / "experiments" / "02_cv_sweep" / "run_experiment.py"


def load_experiment_02():
    spec = importlib.util.spec_from_file_location("experiment_02", EXPERIMENT_02)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 02 from {EXPERIMENT_02}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP2 = load_experiment_02()
BASE = EXP2.BASE
SPATIAL_MODES = EXP2.SPATIAL_MODES
SPATIAL_LABELS = EXP2.SPATIAL_LABELS


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def candidate_work(
    n: int,
    row_size_kib: float,
    table,
    params,
) -> dict:
    """Return transparent operation-count proxies for Paper greedy."""
    from vlmflash.policy import _window_sweep

    windows, jump_cap = _window_sweep(n, row_size_kib, table, params)
    candidate_count = 0
    overlap_cell_upper = 0
    for size, _ in windows:
        stride = min(size, jump_cap)
        count = 1 + (n - size) // stride
        candidate_count += count
        overlap_cell_upper += count * size
    sort_comparison_proxy = (
        candidate_count * math.log2(candidate_count) if candidate_count > 1 else 0.0
    )
    return {
        "window_count": len(windows),
        "candidate_count": candidate_count,
        "sort_comparison_proxy": sort_comparison_proxy,
        "overlap_cell_upper": overlap_cell_upper,
    }


def timed_call(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return result, elapsed_ms


def plot_accuracy_by_cv(summary: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    cvs = sorted({float(item["target_cv"]) for item in summary})
    colors = plt.cm.viridis(np.linspace(0.06, 0.94, len(cvs)))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        for cv, color in zip(cvs, colors):
            points = sorted(
                EXP2.select(summary, spatial_mode=mode, target_cv=cv),
                key=lambda item: item["q"],
            )
            x = np.asarray([item["q"] for item in points])
            y = np.asarray([item["requested_gap_pct"]["mean"] for item in points])
            ax.plot(x, np.maximum(y, 1e-4), marker="o", color=color, label=f"CV={cv:g}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(SPATIAL_LABELS[mode])
        ax.set_xlabel("Importance buckets q")
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Mean latency gap to Exact Coverage (%)")
    axes[-1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Quantized Pareto accuracy as q increases", fontsize=13)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_cost_accuracy(
    q_runtime_summary: list[dict],
    matched_summary: list[dict],
    paper_runtime: dict,
    paper_gap: dict,
    path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    q_runtime_summary = sorted(q_runtime_summary, key=lambda item: item["q"])
    matched_summary = sorted(matched_summary, key=lambda item: item["q"])
    q = np.asarray([item["q"] for item in q_runtime_summary])
    runtime = np.asarray([item["single_query_ms"]["median"] for item in q_runtime_summary])
    runtime_lo = np.asarray([item["single_query_ms"]["p05"] for item in q_runtime_summary])
    runtime_hi = np.asarray([item["single_query_ms"]["p95"] for item in q_runtime_summary])
    gap = np.asarray([item["quant_gap_pct"]["mean"] for item in matched_summary])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    ax = axes[0]
    ax.plot(q, runtime, marker="P", color=BASE.PLOT_COLORS["pareto"], label="Quantized Pareto")
    ax.fill_between(q, runtime_lo, runtime_hi, color=BASE.PLOT_COLORS["pareto"], alpha=0.12)
    ax.axhline(
        paper_runtime["median"],
        color=BASE.PLOT_COLORS["greedy"],
        linestyle="--",
        label="Paper greedy median",
    )
    ax.axhspan(
        paper_runtime["p05"],
        paper_runtime["p95"],
        color=BASE.PLOT_COLORS["greedy"],
        alpha=0.10,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Importance buckets q")
    ax.set_ylabel("CPU time for one target (ms, log scale)")
    ax.set_title("A. Measured reference runtime")
    BASE.polish_axis(ax)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(runtime, gap, marker="P", color=BASE.PLOT_COLORS["pareto"], label="Quantized Pareto")
    for x_value, y_value, q_value in zip(runtime, gap, q):
        ax.annotate(f"q={int(q_value)}", (x_value, y_value), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.scatter(
        [paper_runtime["median"]],
        [paper_gap["mean"]],
        marker="o",
        s=58,
        color=BASE.PLOT_COLORS["greedy"],
        label="Paper greedy",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median CPU time for one target (ms, log scale)")
    ax.set_ylabel("Mean latency gap to Exact Coverage (%, log scale)")
    ax.set_title("B. Cost–accuracy trade-off on greedy targets")
    BASE.polish_axis(ax)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_frontiers_by_q(
    greedy_summary: list[dict],
    exact_summary: list[dict],
    quant_summary: list[dict],
    q_values: list[int],
    path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    available_cv = sorted({float(item["target_cv"]) for item in exact_summary})
    representative_cv = [
        min(available_cv, key=lambda value: abs(value - desired))
        for desired in (1.25, 3.30, 9.19)
    ]
    representative_cv = list(dict.fromkeys(representative_cv))
    # Keep the frontier readable while showing the low-q regime around the
    # paired Paper-greedy quality crossover.
    display_q = [q for q in (8, 16, 32, 64, 128) if q in q_values]
    if not display_q:
        display_q = q_values[:5]
    q_colors = plt.cm.plasma(np.linspace(0.12, 0.82, len(display_q)))

    fig, axes = plt.subplots(
        len(representative_cv),
        3,
        figsize=(14.8, 4.0 * len(representative_cv)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, cv in enumerate(representative_cv):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            greedy_points = sorted(
                EXP2.select(greedy_summary, target_cv=cv, spatial_mode=mode),
                key=lambda item: item["budget_fraction"],
            )
            ax.plot(
                [item["greedy_aff_ms"]["mean"] for item in greedy_points],
                [item["greedy_importance"]["mean"] for item in greedy_points],
                marker="o",
                color=BASE.PLOT_COLORS["greedy"],
                label="Paper greedy",
            )
            exact_points = sorted(
                EXP2.select(exact_summary, target_cv=cv, spatial_mode=mode),
                key=lambda item: item["coverage_target"],
            )
            ax.plot(
                [item["exact_aff_ms"]["mean"] for item in exact_points],
                [item["exact_achieved"]["mean"] for item in exact_points],
                marker="X",
                linestyle="--",
                color=BASE.PLOT_COLORS["coverage"],
                label="Exact Coverage DP",
            )
            for q, color in zip(display_q, q_colors):
                quant_points = sorted(
                    EXP2.select(quant_summary, target_cv=cv, spatial_mode=mode, q=q),
                    key=lambda item: item["coverage_target"],
                )
                ax.plot(
                    [item["quant_aff_ms"]["mean"] for item in quant_points],
                    [item["quant_achieved"]["mean"] for item in quant_points],
                    marker=".",
                    color=color,
                    alpha=0.92,
                    label=f"Quantized q={q}",
                )
            ax.set_xscale("log")
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={cv:g}")
            ax.set_xlabel("Affine latency (ms, log scale)")
            ax.set_ylabel("Retained importance")
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--row-size-kib", type=float, default=1.0)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument(
        "--cv-targets",
        type=float,
        nargs="+",
        default=[1.07, 1.25, 1.44, 2.48, 3.30, 4.55, 9.19],
    )
    parser.add_argument(
        "--q-values",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
    )
    parser.add_argument(
        "--budget-rows", type=int, nargs="+", default=[32, 64, 96, 128, 160, 192, 224]
    )
    parser.add_argument(
        "--coverage-targets",
        type=float,
        nargs="+",
        default=[0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99],
    )
    parser.add_argument("--local-rho", type=float, default=0.95)
    parser.add_argument("--hot-rho", type=float, default=0.75)
    parser.add_argument("--start-kib", type=float, default=8.0)
    parser.add_argument("--jump-cap-kib", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.n < 2:
        raise SystemExit("--trials must be positive and --n must be at least 2")
    cv_targets = sorted(set(args.cv_targets))
    q_values = sorted(set(args.q_values))
    budgets = sorted(set(args.budget_rows))
    coverage_targets = sorted(set(args.coverage_targets))
    if cv_targets[0] <= 0 or cv_targets[-1] >= math.sqrt(args.n - 1):
        raise SystemExit("every target CV must lie in (0, sqrt(n-1))")
    if not q_values or q_values[0] < 2:
        raise SystemExit("every q must be at least 2")
    if not budgets or budgets[0] < 1 or budgets[-1] > args.n:
        raise SystemExit("every row budget must lie in [1, n]")
    if not coverage_targets or coverage_targets[0] <= 0 or coverage_targets[-1] > 1:
        raise SystemExit("coverage targets must lie in (0, 1]")
    if not args.skip_self_check:
        EXP2.self_check()

    table = BASE.LatencyTable.load(args.profile)
    a_ms, c_ms_per_row, fit_r2 = BASE.affine_fit(table, args.row_size_kib)
    params = BASE.ChunkParams(start_kb=args.start_kib, jump_cap_kb=args.jump_cap_kib)
    paper_work = candidate_work(args.n, args.row_size_kib, table, params)
    data_rng = np.random.default_rng(args.seed)
    order_rng = np.random.default_rng(args.seed + 1)
    hotness = np.linspace(1.0, -1.0, args.n)
    hotness = (hotness - hotness.mean()) / hotness.std()

    # Warm timing paths without retaining their measurements.
    warm_values = np.full(args.n, 1.0 / args.n, dtype=np.float64)
    BASE.QuantizedParetoOracle(warm_values, a_ms, c_ms_per_row, q_values[0]).solve(0.5)
    BASE.select_chunks(
        torch.from_numpy(warm_values.astype(np.float32)),
        budgets[0],
        args.row_size_kib,
        table,
        params=params,
        impl="torch",
    )

    exact_rows: list[dict] = []
    frontier_rows: list[dict] = []
    q_rows: list[dict] = []
    greedy_rows: list[dict] = []
    matched_rows: list[dict] = []
    total_multisets = args.trials * len(cv_targets)
    completed = 0

    for trial in range(args.trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(data_rng, args.n, target_cv)
            variants = EXP2.spatial_variants(
                base_values, data_rng, hotness, args.local_rho, args.hot_rho
            )
            for spatial_mode, values in variants.items():
                values_t = torch.from_numpy(values.astype(np.float32))
                exact_oracle = BASE.ExactCoverageOracle(values, a_ms, c_ms_per_row)
                exact_by_target = {}
                for coverage_target in coverage_targets:
                    exact = exact_oracle.solve(coverage_target)
                    exact_by_target[coverage_target] = exact
                    exact_rows.append(
                        {
                            "trial": trial,
                            "target_cv": target_cv,
                            "spatial_mode": spatial_mode,
                            "coverage_target": coverage_target,
                            "exact_achieved": exact["importance"],
                            "exact_rows": exact["rows"],
                            "exact_chunks": exact["chunks"],
                            "exact_aff_ms": exact["aff_ms"],
                        }
                    )

                greedy_targets = []
                for budget in budgets:
                    greedy, greedy_runtime_ms = timed_call(
                        BASE.select_chunks,
                        values_t,
                        budget,
                        args.row_size_kib,
                        table,
                        params=params,
                        impl="torch",
                    )
                    mask = greedy.mask.cpu().numpy()
                    importance = float(values[mask].sum())
                    latency = BASE.affine_latency(mask, a_ms, c_ms_per_row)
                    selected_rows, chunks = BASE.mask_stats(mask)
                    if selected_rows == 0:
                        raise RuntimeError("Paper greedy selected an empty mask")
                    exact = exact_oracle.solve(importance)
                    gap = 100 * (latency / exact["aff_ms"] - 1)
                    row = {
                        "trial": trial,
                        "target_cv": target_cv,
                        "spatial_mode": spatial_mode,
                        "budget_rows": budget,
                        "budget_fraction": budget / args.n,
                        "greedy_rows": selected_rows,
                        "greedy_chunks": chunks,
                        "greedy_importance": importance,
                        "greedy_aff_ms": latency,
                        "greedy_runtime_ms": greedy_runtime_ms,
                        "exact_at_greedy_aff_ms": exact["aff_ms"],
                        "greedy_gap_pct": gap,
                    }
                    greedy_rows.append(row)
                    greedy_targets.append(row)

                q_order = list(order_rng.permutation(q_values))
                for q in q_order:
                    oracle, build_ms = timed_call(
                        BASE.QuantizedParetoOracle,
                        values,
                        a_ms,
                        c_ms_per_row,
                        int(q),
                    )
                    coverage_solve_times = []
                    requested_gaps = []
                    achieved_gaps = []
                    overshoots = []
                    for coverage_target in coverage_targets:
                        quant, solve_ms = timed_call(oracle.solve, coverage_target)
                        coverage_solve_times.append(solve_ms)
                        exact_requested = exact_by_target[coverage_target]
                        exact_achieved = exact_oracle.solve(quant["importance"])
                        requested_gap = 100 * (
                            quant["aff_ms"] / exact_requested["aff_ms"] - 1
                        )
                        achieved_gap = 100 * (
                            quant["aff_ms"] / exact_achieved["aff_ms"] - 1
                        )
                        if requested_gap < -1e-9 or achieved_gap < -1e-9:
                            raise RuntimeError("Quantized Pareto beat Exact Coverage")
                        overshoot = quant["importance"] - coverage_target
                        requested_gaps.append(requested_gap)
                        achieved_gaps.append(achieved_gap)
                        overshoots.append(overshoot)
                        frontier_rows.append(
                            {
                                "trial": trial,
                                "target_cv": target_cv,
                                "spatial_mode": spatial_mode,
                                "q": int(q),
                                "coverage_target": coverage_target,
                                "quant_achieved": quant["importance"],
                                "quant_overshoot": overshoot,
                                "quant_rows": quant["rows"],
                                "quant_chunks": quant["chunks"],
                                "quant_aff_ms": quant["aff_ms"],
                                "solve_ms": solve_ms,
                                "exact_requested_aff_ms": exact_requested["aff_ms"],
                                "exact_at_achieved_aff_ms": exact_achieved["aff_ms"],
                                "requested_gap_pct": requested_gap,
                                "achieved_gap_pct": achieved_gap,
                            }
                        )

                    greedy_target_solve_times = []
                    greedy_target_gaps = []
                    for greedy_row in greedy_targets:
                        quant, solve_ms = timed_call(
                            oracle.solve, greedy_row["greedy_importance"]
                        )
                        greedy_target_solve_times.append(solve_ms)
                        quant_gap = 100 * (
                            quant["aff_ms"] / greedy_row["exact_at_greedy_aff_ms"] - 1
                        )
                        if quant_gap < -1e-9:
                            raise RuntimeError("Quantized Pareto beat Exact Coverage")
                        greedy_target_gaps.append(quant_gap)
                        matched_rows.append(
                            {
                                "trial": trial,
                                "target_cv": target_cv,
                                "spatial_mode": spatial_mode,
                                "q": int(q),
                                "budget_rows": greedy_row["budget_rows"],
                                "target_importance": greedy_row["greedy_importance"],
                                "paper_aff_ms": greedy_row["greedy_aff_ms"],
                                "paper_runtime_ms": greedy_row["greedy_runtime_ms"],
                                "paper_gap_pct": greedy_row["greedy_gap_pct"],
                                "quant_aff_ms": quant["aff_ms"],
                                "quant_solve_ms": solve_ms,
                                "quant_single_query_ms": build_ms + solve_ms,
                                "quant_gap_pct": quant_gap,
                                "paper_extra_vs_quant_pct": 100
                                * (greedy_row["greedy_aff_ms"] / quant["aff_ms"] - 1),
                            }
                        )

                    all_solve_times = coverage_solve_times + greedy_target_solve_times
                    q_rows.append(
                        {
                            "trial": trial,
                            "target_cv": target_cv,
                            "spatial_mode": spatial_mode,
                            "q": int(q),
                            "build_ms": build_ms,
                            "mean_solve_ms": float(np.mean(all_solve_times)),
                            "single_query_ms": build_ms + float(np.mean(all_solve_times)),
                            "seven_target_amortized_ms": (
                                build_ms + float(np.sum(coverage_solve_times))
                            )
                            / len(coverage_targets),
                            "peak_states": oracle.peak_states,
                            "dp_state_slots": 2 * args.n * (int(q) + 1),
                            "requested_gap_pct": float(np.mean(requested_gaps)),
                            "achieved_gap_pct": float(np.mean(achieved_gaps)),
                            "mean_overshoot": float(np.mean(overshoots)),
                            "greedy_target_gap_pct": float(np.mean(greedy_target_gaps)),
                        }
                    )
            completed += 1
            if completed % max(1, total_multisets // 20) == 0:
                print(
                    f"completed {completed}/{total_multisets} CV-calibrated multisets",
                    flush=True,
                )

    exact_summary = EXP2.summarize(
        exact_rows,
        ("target_cv", "spatial_mode", "coverage_target"),
        ("exact_achieved", "exact_rows", "exact_chunks", "exact_aff_ms"),
    )
    quant_summary = EXP2.summarize(
        frontier_rows,
        ("target_cv", "spatial_mode", "q", "coverage_target"),
        (
            "quant_achieved",
            "quant_overshoot",
            "quant_rows",
            "quant_chunks",
            "quant_aff_ms",
            "solve_ms",
            "requested_gap_pct",
            "achieved_gap_pct",
        ),
    )
    accuracy_summary = EXP2.summarize(
        frontier_rows,
        ("target_cv", "spatial_mode", "q"),
        ("requested_gap_pct", "achieved_gap_pct", "quant_overshoot"),
    )
    greedy_summary = EXP2.summarize(
        greedy_rows,
        ("target_cv", "spatial_mode", "budget_fraction"),
        (
            "greedy_importance",
            "greedy_aff_ms",
            "greedy_runtime_ms",
            "greedy_gap_pct",
        ),
    )
    vlm_q_rows = [row for row in q_rows if float(row["target_cv"]) < 9.0]
    q_runtime_summary = EXP2.summarize(
        vlm_q_rows,
        ("q",),
        (
            "build_ms",
            "mean_solve_ms",
            "single_query_ms",
            "seven_target_amortized_ms",
            "peak_states",
            "dp_state_slots",
            "requested_gap_pct",
            "achieved_gap_pct",
            "mean_overshoot",
            "greedy_target_gap_pct",
        ),
    )
    vlm_matched = [row for row in matched_rows if float(row["target_cv"]) < 9.0]
    matched_summary = EXP2.summarize(
        vlm_matched,
        ("q",),
        (
            "quant_single_query_ms",
            "quant_gap_pct",
            "paper_extra_vs_quant_pct",
        ),
    )
    vlm_greedy = [row for row in greedy_rows if float(row["target_cv"]) < 9.0]
    paper_runtime = EXP2.distribution_stats(
        [float(row["greedy_runtime_ms"]) for row in vlm_greedy]
    )
    paper_gap = EXP2.distribution_stats(
        [float(row["greedy_gap_pct"]) for row in vlm_greedy]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "exact": args.output_dir / "exact_frontier_trials.csv",
        "frontier": args.output_dir / "quant_frontier_trials.csv",
        "q": args.output_dir / "q_runtime_trials.csv",
        "greedy": args.output_dir / "paper_greedy_trials.csv",
        "matched": args.output_dir / "paper_matched_trials.csv",
        "summary": args.output_dir / "summary.json",
        "accuracy_plot": args.output_dir / "q_accuracy_by_cv.png",
        "tradeoff_plot": args.output_dir / "q_cost_accuracy.png",
        "frontier_plot": args.output_dir / "importance_latency_frontiers_by_q.png",
    }
    write_csv(paths["exact"], exact_rows)
    write_csv(paths["frontier"], frontier_rows)
    write_csv(paths["q"], q_rows)
    write_csv(paths["greedy"], greedy_rows)
    write_csv(paths["matched"], matched_rows)
    metadata = {
        "format": "experiment-03-quant-tradeoff-v1",
        "seed": args.seed,
        "trials_per_cv": args.trials,
        "n": args.n,
        "cv_targets": cv_targets,
        "q_values": q_values,
        "budgets": budgets,
        "coverage_targets": coverage_targets,
        "spatial_modes": list(SPATIAL_MODES),
        "local_rho": args.local_rho,
        "hot_rho": args.hot_rho,
        "row_size_kib": args.row_size_kib,
        "profile": args.profile,
        "affine_fit": {
            "a_ms_per_chunk": a_ms,
            "c_ms_per_row": c_ms_per_row,
            "r_squared": fit_r2,
        },
        "paper_work_proxy": paper_work,
        "quant_work_proxy_definition": "2*N*(q+1) ending-state bucket slots",
        "timing_scope": "single-process CPU reference; includes Python implementation effects",
        "paper_runtime_vlm": paper_runtime,
        "paper_gap_vlm": paper_gap,
        "q_runtime_summary_vlm": q_runtime_summary,
        "matched_summary_vlm": matched_summary,
        "accuracy_summary": accuracy_summary,
        "greedy_summary": greedy_summary,
        "exact_summary": exact_summary,
        "quant_summary": quant_summary,
    }
    paths["summary"].write_text(json.dumps(metadata, indent=2) + "\n")
    plot_accuracy_by_cv(accuracy_summary, paths["accuracy_plot"])
    plot_cost_accuracy(
        q_runtime_summary,
        matched_summary,
        paper_runtime,
        paper_gap,
        paths["tradeoff_plot"],
    )
    plot_frontiers_by_q(
        greedy_summary,
        exact_summary,
        quant_summary,
        q_values,
        paths["frontier_plot"],
    )

    for path in paths.values():
        if path.suffix == ".png":
            print(f"wrote {path} and {path.with_suffix('.pdf')}")
        else:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
