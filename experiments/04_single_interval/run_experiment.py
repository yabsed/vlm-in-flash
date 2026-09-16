#!/usr/bin/env python3
"""Experiment 04: test the O(N) single-interval coverage solver.

The coverage problem is

    minimize a*K(M) + c*R(M)  subject to I(M) >= alpha.

For nonnegative importance, the best one-chunk solution is the shortest
interval reaching the target and is found exactly in O(N) with two pointers.
This experiment compares that simple solver with Paper greedy and the Exact
Coverage DP, then measures the exact optimum's chunk-count distribution.
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
EXPERIMENT_03 = PROJECT_ROOT / "experiments" / "03_quant_tradeoff" / "run_experiment.py"


def load_experiment_03():
    spec = importlib.util.spec_from_file_location("experiment_03", EXPERIMENT_03)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 03 from {EXPERIMENT_03}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP3 = load_experiment_03()
EXP2 = EXP3.EXP2
BASE = EXP3.BASE
SPATIAL_MODES = EXP3.SPATIAL_MODES
SPATIAL_LABELS = EXP3.SPATIAL_LABELS
SINGLE_COLOR = "#3366A6"


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


def minimum_single_interval(
    values: np.ndarray,
    bound: float,
    a_ms: float,
    c_ms_per_row: float,
) -> dict:
    """Return the cheapest one-chunk feasible mask in O(N) time."""
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("the two-pointer solver requires nonnegative importance")
    total_importance = float(values.sum())
    if not 0 < bound <= total_importance + 1e-12:
        raise ValueError("coverage bound must lie in (0, total importance]")

    tolerance = 1e-14 * max(1.0, total_importance)
    left = 0
    window_sum = 0.0
    best_left = 0
    best_right = len(values)
    best_length = len(values) + 1
    best_importance = -math.inf

    for right, value in enumerate(values):
        window_sum += float(value)
        while (
            left <= right
            and window_sum - float(values[left]) >= bound - tolerance
        ):
            window_sum -= float(values[left])
            left += 1
        if window_sum >= bound - tolerance:
            length = right - left + 1
            if length < best_length or (
                length == best_length and window_sum > best_importance
            ):
                best_left = left
                best_right = right + 1
                best_length = length
                best_importance = window_sum

    if best_length > len(values):
        raise RuntimeError("the single-interval scan did not find a feasible interval")
    mask = np.zeros(len(values), dtype=bool)
    mask[best_left:best_right] = True
    return {
        "mask": mask,
        "importance": float(values[mask].sum()),
        "rows": int(best_length),
        "chunks": 1,
        "aff_ms": float(a_ms + c_ms_per_row * best_length),
        "left": int(best_left),
        "right": int(best_right),
    }


def gap_pct(candidate: dict, exact: dict) -> float:
    return 100.0 * (float(candidate["aff_ms"]) / float(exact["aff_ms"]) - 1.0)


def is_exact(candidate: dict, exact: dict) -> bool:
    return math.isclose(
        float(candidate["aff_ms"]),
        float(exact["aff_ms"]),
        rel_tol=1e-11,
        abs_tol=1e-13,
    )


def evaluate_target(
    values: np.ndarray,
    bound: float,
    exact_oracle,
    exact_build_ms: float,
    a_ms: float,
    c_ms_per_row: float,
) -> dict:
    exact, exact_solve_ms = timed_call(exact_oracle.solve, bound)
    single, single_runtime_ms = timed_call(
        minimum_single_interval, values, bound, a_ms, c_ms_per_row
    )
    single_gap = gap_pct(single, exact)
    if single_gap < -1e-8:
        raise RuntimeError("single interval beat Exact Coverage")
    return {
        "target_importance": bound,
        "exact_importance": exact["importance"],
        "exact_rows": exact["rows"],
        "exact_chunks": exact["chunks"],
        "exact_aff_ms": exact["aff_ms"],
        "exact_build_ms": exact_build_ms,
        "exact_solve_ms": exact_solve_ms,
        "exact_single_query_ms": exact_build_ms + exact_solve_ms,
        "single_importance": single["importance"],
        "single_rows": single["rows"],
        "single_chunks": 1,
        "single_aff_ms": single["aff_ms"],
        "single_runtime_ms": single_runtime_ms,
        "single_gap_pct": max(0.0, single_gap),
        "single_exact": is_exact(single, exact),
    }


def aggregate_curve(
    rows: list[dict],
    target_cv: float,
    spatial_mode: str,
    x_key: str,
    metric_keys: tuple[str, ...],
) -> list[dict]:
    selected = [
        row
        for row in rows
        if float(row["target_cv"]) == target_cv
        and row["spatial_mode"] == spatial_mode
    ]
    return EXP2.summarize(selected, (x_key,), metric_keys)


def plot_method_grid(
    coverage_rows: list[dict],
    paper_rows: list[dict],
    path: Path,
    x_fields: dict[str, str],
    y_fields: dict[str, str],
    xlabel: str,
    ylabel: str,
    x_log: bool = False,
) -> None:
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
        len(representative_cv),
        3,
        figsize=(14.8, 4.0 * len(representative_cv)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, target_cv in enumerate(representative_cv):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            coverage = aggregate_curve(
                coverage_rows,
                target_cv,
                mode,
                "coverage_target",
                tuple(
                    {
                        x_fields["exact"],
                        x_fields["single"],
                        y_fields["exact"],
                        y_fields["single"],
                    }
                ),
            )
            paper = aggregate_curve(
                paper_rows,
                target_cv,
                mode,
                "budget_fraction",
                (x_fields["paper"], y_fields["paper"]),
            )
            for points, prefix, label, color, marker, linestyle in (
                (
                    paper,
                    "paper",
                    "Paper greedy",
                    BASE.PLOT_COLORS["greedy"],
                    "o",
                    "-",
                ),
                (
                    coverage,
                    "single",
                    "Single interval O(N)",
                    SINGLE_COLOR,
                    "s",
                    "-",
                ),
                (
                    coverage,
                    "exact",
                    "Exact Coverage DP",
                    BASE.PLOT_COLORS["coverage"],
                    "X",
                    "--",
                ),
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
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={target_cv:g}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def chunk_distribution(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[int]] = {}
    for row in rows:
        key = (
            float(row["target_cv"]),
            row["spatial_mode"],
            float(row["coverage_target"]),
        )
        groups.setdefault(key, []).append(int(row["exact_chunks"]))
    output = []
    for (target_cv, mode, target), chunks in sorted(groups.items()):
        counts = {
            str(k): int(sum(value == k for value in chunks))
            for k in sorted(set(chunks))
        }
        output.append(
            {
                "target_cv": target_cv,
                "spatial_mode": mode,
                "coverage_target": target,
                "trials": len(chunks),
                "counts": counts,
                "fractions": {
                    key: value / len(chunks) for key, value in counts.items()
                },
            }
        )
    return output


def plot_chunk_distribution(distribution: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    cvs = sorted({float(item["target_cv"]) for item in distribution})
    targets = sorted({float(item["coverage_target"]) for item in distribution})
    lookup = {
        (
            float(item["target_cv"]),
            item["spatial_mode"],
            float(item["coverage_target"]),
        ): item
        for item in distribution
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(16.0, 6.0), sharey=True, constrained_layout=True
    )
    image_handle = None
    for ax, mode in zip(axes, SPATIAL_MODES):
        one_chunk = np.zeros((len(cvs), len(targets)), dtype=np.float64)
        for row_index, cv in enumerate(cvs):
            for column_index, target in enumerate(targets):
                item = lookup[(cv, mode, target)]
                one_chunk[row_index, column_index] = item["fractions"].get("1", 0.0)
        image_handle = ax.imshow(
            one_chunk,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="nearest",
        )
        for row_index, cv in enumerate(cvs):
            for column_index, target in enumerate(targets):
                item = lookup[(cv, mode, target)]
                pieces = [
                    f"K{k}: {100 * fraction:.0f}%"
                    for k, fraction in item["fractions"].items()
                    if fraction > 0
                ]
                ax.text(
                    column_index,
                    row_index,
                    "\n".join(pieces),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if one_chunk[row_index, column_index] >= 0.62 else "#17202A",
                )
        ax.set_xticks(range(len(targets)), [f"{target:g}" for target in targets])
        ax.set_yticks(range(len(cvs)), [f"{cv:g}" for cv in cvs])
        ax.set_xlabel("Coverage target")
        ax.set_title(SPATIAL_LABELS[mode])
        ax.grid(False)
    axes[0].set_ylabel("Target CV")
    colorbar = fig.colorbar(image_handle, ax=axes, shrink=0.88, pad=0.015)
    colorbar.set_label("Fraction whose exact optimum has one chunk")
    fig.suptitle("Exact Coverage optimum: chunk-count distribution", fontsize=13)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def self_check() -> None:
    cases = [
        (np.asarray([0.1, 0.2, 0.5, 0.2]), 0.6),
        (np.asarray([0.4, 0.0, 0.0, 0.4, 0.2]), 0.8),
        (np.asarray([0.25, 0.25, 0.25, 0.25]), 1.0),
    ]
    a_ms, c_ms = 0.011, 0.00014
    for values, bound in cases:
        found = minimum_single_interval(values, bound, a_ms, c_ms)
        brute = []
        for left in range(len(values)):
            for right in range(left + 1, len(values) + 1):
                importance = float(values[left:right].sum())
                if importance >= bound - 1e-14:
                    brute.append((right - left, -importance, left, right))
        assert found["rows"] == min(brute)[0]
    EXP2.self_check()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260919)
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
    budgets = sorted(set(args.budget_rows))
    coverage_targets = sorted(set(args.coverage_targets))
    if cv_targets[0] <= 0 or cv_targets[-1] >= math.sqrt(args.n - 1):
        raise SystemExit("every target CV must lie in (0, sqrt(n-1))")
    if not budgets or budgets[0] < 1 or budgets[-1] > args.n:
        raise SystemExit("every row budget must lie in [1, n]")
    if not coverage_targets or coverage_targets[0] <= 0 or coverage_targets[-1] > 1:
        raise SystemExit("coverage targets must lie in (0, 1]")
    if not args.skip_self_check:
        self_check()

    table = BASE.LatencyTable.load(args.profile)
    a_ms, c_ms_per_row, fit_r2 = BASE.affine_fit(table, args.row_size_kib)
    params = BASE.ChunkParams(start_kb=args.start_kib, jump_cap_kb=args.jump_cap_kib)
    paper_work = EXP3.candidate_work(args.n, args.row_size_kib, table, params)
    data_rng = np.random.default_rng(args.seed)
    hotness = np.linspace(1.0, -1.0, args.n)
    hotness = (hotness - hotness.mean()) / hotness.std()

    warm = np.full(args.n, 1.0 / args.n, dtype=np.float64)
    minimum_single_interval(warm, 0.5, a_ms, c_ms_per_row)
    BASE.select_chunks(
        torch.from_numpy(warm.astype(np.float32)),
        budgets[0],
        args.row_size_kib,
        table,
        params=params,
        impl="torch",
    )

    coverage_rows: list[dict] = []
    paper_rows: list[dict] = []
    input_rows: list[dict] = []
    total_multisets = args.trials * len(cv_targets)
    completed = 0

    for trial in range(args.trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(data_rng, args.n, target_cv)
            variants = EXP2.spatial_variants(
                base_values, data_rng, hotness, args.local_rho, args.hot_rho
            )
            for spatial_mode, values in variants.items():
                exact_oracle, exact_build_ms = timed_call(
                    BASE.ExactCoverageOracle, values, a_ms, c_ms_per_row
                )
                lag1_corr, first_half_mass = EXP2.spatial_stats(values)
                input_rows.append(
                    {
                        "trial": trial,
                        "target_cv": target_cv,
                        "spatial_mode": spatial_mode,
                        "actual_cv": EXP2.coefficient_of_variation(values),
                        "lag1_corr": lag1_corr,
                        "first_half_mass": first_half_mass,
                        "exact_build_ms": exact_build_ms,
                    }
                )
                for coverage_target in coverage_targets:
                    coverage_rows.append(
                        {
                            "trial": trial,
                            "target_cv": target_cv,
                            "spatial_mode": spatial_mode,
                            "coverage_target": coverage_target,
                            **evaluate_target(
                                values,
                                coverage_target,
                                exact_oracle,
                                exact_build_ms,
                                a_ms,
                                c_ms_per_row,
                            ),
                        }
                    )

                values_t = torch.from_numpy(values.astype(np.float32))
                for budget in budgets:
                    paper, paper_runtime_ms = timed_call(
                        BASE.select_chunks,
                        values_t,
                        budget,
                        args.row_size_kib,
                        table,
                        params=params,
                        impl="torch",
                    )
                    paper_mask = paper.mask.cpu().numpy()
                    paper_rows_count, paper_chunks = BASE.mask_stats(paper_mask)
                    if paper_rows_count == 0:
                        raise RuntimeError("Paper greedy selected an empty mask")
                    paper_importance = float(values[paper_mask].sum())
                    paper_aff_ms = BASE.affine_latency(paper_mask, a_ms, c_ms_per_row)
                    evaluated = evaluate_target(
                        values,
                        paper_importance,
                        exact_oracle,
                        exact_build_ms,
                        a_ms,
                        c_ms_per_row,
                    )
                    paper_gap = 100.0 * (
                        paper_aff_ms / evaluated["exact_aff_ms"] - 1.0
                    )
                    paper_rows.append(
                        {
                            "trial": trial,
                            "target_cv": target_cv,
                            "spatial_mode": spatial_mode,
                            "budget_rows": budget,
                            "budget_fraction": budget / args.n,
                            "paper_importance": paper_importance,
                            "paper_rows": paper_rows_count,
                            "paper_chunks": paper_chunks,
                            "paper_aff_ms": paper_aff_ms,
                            "paper_runtime_ms": paper_runtime_ms,
                            "paper_gap_pct": max(0.0, paper_gap),
                            "paper_extra_vs_single_pct": 100.0
                            * (paper_aff_ms / evaluated["single_aff_ms"] - 1.0),
                            **evaluated,
                        }
                    )
            completed += 1
            if completed % max(1, total_multisets // 20) == 0:
                print(
                    f"completed {completed}/{total_multisets} CV-calibrated multisets",
                    flush=True,
                )

    vlm_coverage = [row for row in coverage_rows if float(row["target_cv"]) < 9.0]
    vlm_paper = [row for row in paper_rows if float(row["target_cv"]) < 9.0]
    coverage_metrics = (
        "exact_rows",
        "exact_chunks",
        "exact_aff_ms",
        "exact_single_query_ms",
        "single_rows",
        "single_aff_ms",
        "single_gap_pct",
        "single_runtime_ms",
        "single_exact",
    )
    paper_metrics = (
        "paper_rows",
        "paper_chunks",
        "paper_aff_ms",
        "paper_gap_pct",
        "paper_runtime_ms",
        "paper_extra_vs_single_pct",
        "exact_chunks",
        "single_gap_pct",
        "single_runtime_ms",
        "single_exact",
    )
    coverage_overall_vlm = {
        metric: EXP2.distribution_stats([float(row[metric]) for row in vlm_coverage])
        for metric in coverage_metrics
    }
    paper_matched_overall_vlm = {
        metric: EXP2.distribution_stats([float(row[metric]) for row in vlm_paper])
        for metric in paper_metrics
    }
    coverage_by_target_vlm = EXP2.summarize(
        vlm_coverage, ("coverage_target",), coverage_metrics
    )
    coverage_by_ordering_vlm = EXP2.summarize(
        vlm_coverage, ("spatial_mode",), coverage_metrics
    )
    paper_by_ordering_vlm = EXP2.summarize(
        vlm_paper, ("spatial_mode",), paper_metrics
    )
    exact_chunk_distribution = chunk_distribution(coverage_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": args.output_dir / "coverage_trials.csv",
        "paper": args.output_dir / "paper_matched_trials.csv",
        "inputs": args.output_dir / "input_trials.csv",
        "summary": args.output_dir / "summary.json",
        "frontier_plot": args.output_dir / "importance_latency_frontiers.png",
        "r_importance_plot": args.output_dir / "r_importance.png",
        "r_latency_plot": args.output_dir / "r_latency.png",
        "chunk_plot": args.output_dir / "optimal_chunk_distribution.png",
    }
    write_csv(paths["coverage"], coverage_rows)
    write_csv(paths["paper"], paper_rows)
    write_csv(paths["inputs"], input_rows)
    metadata = {
        "format": "experiment-04-single-interval-v1",
        "seed": args.seed,
        "trials_per_cv": args.trials,
        "n": args.n,
        "cv_targets": cv_targets,
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
            "chunk_open_equivalent_rows": a_ms / c_ms_per_row,
        },
        "complexity": {
            "single_interval": "O(N) time, O(1) auxiliary space per target",
            "exact_oracle_implementation": "O(N^3) state updates and O(N^3) parent storage",
        },
        "paper_work_proxy": paper_work,
        "timing_scope": "single-process CPU reference; one target includes every method's own construction",
        "coverage_overall_vlm": coverage_overall_vlm,
        "paper_matched_overall_vlm": paper_matched_overall_vlm,
        "coverage_by_target_vlm": coverage_by_target_vlm,
        "coverage_by_ordering_vlm": coverage_by_ordering_vlm,
        "paper_by_ordering_vlm": paper_by_ordering_vlm,
        "exact_chunk_distribution": exact_chunk_distribution,
    }
    paths["summary"].write_text(json.dumps(metadata, indent=2) + "\n")

    plot_method_grid(
        coverage_rows,
        paper_rows,
        paths["frontier_plot"],
        x_fields={
            "paper": "paper_aff_ms",
            "exact": "exact_aff_ms",
            "single": "single_aff_ms",
        },
        y_fields={
            "paper": "paper_importance",
            "exact": "exact_importance",
            "single": "single_importance",
        },
        xlabel="Affine latency (ms, log scale)",
        ylabel="Retained importance",
        x_log=True,
    )
    plot_method_grid(
        coverage_rows,
        paper_rows,
        paths["r_importance_plot"],
        x_fields={"paper": "paper_rows", "exact": "exact_rows", "single": "single_rows"},
        y_fields={
            "paper": "paper_importance",
            "exact": "exact_importance",
            "single": "single_importance",
        },
        xlabel="Selected rows R",
        ylabel="Retained importance",
    )
    plot_method_grid(
        coverage_rows,
        paper_rows,
        paths["r_latency_plot"],
        x_fields={"paper": "paper_rows", "exact": "exact_rows", "single": "single_rows"},
        y_fields={
            "paper": "paper_aff_ms",
            "exact": "exact_aff_ms",
            "single": "single_aff_ms",
        },
        xlabel="Selected rows R",
        ylabel="Affine latency (ms)",
    )
    plot_chunk_distribution(exact_chunk_distribution, paths["chunk_plot"])

    for path in paths.values():
        if path.suffix == ".png":
            print(f"wrote {path} and {path.with_suffix('.pdf')}")
        else:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
