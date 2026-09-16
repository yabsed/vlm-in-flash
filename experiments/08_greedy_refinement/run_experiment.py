#!/usr/bin/env python3
"""Experiment 08: add a cheap greedy refinement to Experiment 07.

The expensive Experiment 07 baselines and exact DP results are reused from
CSV. Inputs and Paper masks are deterministically replayed, then two
coverage-preserving operations are applied:

1. merge any gap g for which c*g < a, and
2. trim low-importance boundary rows without dropping below Paper importance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_07 = PROJECT_ROOT / "experiments" / "07_exact_n4864" / "run_experiment.py"
DEFAULT_SOURCE = PROJECT_ROOT / "experiments" / "07_exact_n4864" / "results"
REFINED_COLOR = "#E6A700"
REFINED_LABEL = "Paper + merge + trim"


def load_experiment_07():
    spec = importlib.util.spec_from_file_location("experiment_07", EXPERIMENT_07)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 07 from {EXPERIMENT_07}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP7 = load_experiment_07()
BASE = EXP7.BASE
EXP2 = EXP7.EXP2
SPATIAL_MODES = EXP7.SPATIAL_MODES
SPATIAL_LABELS = EXP7.SPATIAL_LABELS
CONFIG = EXP7.CONFIG


INTEGER_COLUMNS = {
    "trial", "budget_rows", "paper_rows", "paper_chunks", "dpr_rows",
    "dpr_chunks", "top_rows", "top_chunks", "paper_cover_rows",
    "paper_cover_chunks", "dpr_cover_rows", "dpr_cover_chunks",
    "top_cover_rows", "top_cover_chunks", "dpr_iterations", "cover_rows",
    "cover_chunks",
}
TEXT_COLUMNS = {"spatial_mode"}


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in TEXT_COLUMNS:
                    row[key] = value
                elif key in INTEGER_COLUMNS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def timed_call(function, *args):
    start = time.perf_counter_ns()
    result = function(*args)
    return result, (time.perf_counter_ns() - start) / 1e6


def merge_profitable_gaps(mask: np.ndarray, a_ms: float,
                          c_ms_per_row: float) -> np.ndarray:
    """Fill every internal gap whose transfer cost is below one chunk open."""
    merged = np.asarray(mask, dtype=bool).copy()
    edges = np.diff(np.r_[False, merged, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for left_end, right_start in zip(ends[:-1], starts[1:]):
        gap = int(right_start - left_end)
        if c_ms_per_row * gap < a_ms - 1e-15:
            merged[left_end:right_start] = True
    return merged


def trim_boundaries(mask: np.ndarray, values: np.ndarray, target: float,
                    a_ms: float, c_ms_per_row: float) -> np.ndarray:
    """Greedily remove efficient boundary rows while preserving coverage.

    A normal boundary removal saves c. Removing a singleton also closes its
    chunk and saves a+c. The min-heap ranks the currently exposed boundary by
    lost importance per saved millisecond.
    """
    trimmed = np.asarray(mask, dtype=bool).copy()
    importance = float(values[trimmed].sum())
    surplus = importance - float(target)
    if surplus < -1e-12:
        raise RuntimeError("refinement input does not meet its coverage target")
    surplus = max(0.0, surplus)
    heap: list[tuple[float, int, int]] = []

    def offer(index: int) -> None:
        if not 0 <= index < len(trimmed) or not trimmed[index]:
            return
        neighbors = int(index > 0 and trimmed[index - 1])
        neighbors += int(index + 1 < len(trimmed) and trimmed[index + 1])
        if neighbors > 1:
            return
        saving = c_ms_per_row + (a_ms if neighbors == 0 else 0.0)
        heapq.heappush(heap, (float(values[index]) / saving, index, neighbors))

    for index in np.flatnonzero(trimmed):
        offer(int(index))

    while heap:
        _, index, recorded_neighbors = heapq.heappop(heap)
        if not trimmed[index]:
            continue
        neighbors = int(index > 0 and trimmed[index - 1])
        neighbors += int(index + 1 < len(trimmed) and trimmed[index + 1])
        if neighbors > 1:
            continue
        if neighbors != recorded_neighbors:
            offer(index)
            continue
        value = float(values[index])
        if value > surplus + 1e-14:
            continue
        trimmed[index] = False
        surplus -= value
        offer(index - 1)
        offer(index + 1)

    if float(values[trimmed].sum()) < target - 1e-12:
        raise RuntimeError("boundary trimming violated the coverage target")
    return trimmed


def refine_mask(mask: np.ndarray, values: np.ndarray, target: float,
                a_ms: float, c_ms_per_row: float) -> tuple[np.ndarray, np.ndarray]:
    merged = merge_profitable_gaps(mask, a_ms, c_ms_per_row)
    refined = trim_boundaries(merged, values, target, a_ms, c_ms_per_row)
    return merged, refined


def mask_metrics(mask: np.ndarray, values: np.ndarray,
                 a_ms: float, c_ms_per_row: float) -> dict:
    rows, chunks = BASE.mask_stats(mask)
    return {
        "importance": float(values[mask].sum()),
        "rows": rows,
        "chunks": chunks,
        "aff_ms": float(a_ms * chunks + c_ms_per_row * rows),
    }


def distribution(values) -> dict:
    return EXP2.distribution_stats([float(value) for value in values])


def row_key(row: dict) -> tuple:
    return (
        int(row["trial"]), round(float(row["target_cv"]), 10),
        row["spatial_mode"], round(float(row["budget_fraction"]), 10),
    )


def input_key(row: dict) -> tuple:
    return int(row["trial"]), round(float(row["target_cv"]), 10), row["spatial_mode"]


def assert_close(actual: float, expected: float, label: str,
                 tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"Experiment 07 replay mismatch for {label}: {actual} != {expected}")


def self_check() -> None:
    values = np.asarray([0.2, 0.01, 0.01, 0.3, 0.02, 0.46])
    mask = np.asarray([1, 0, 0, 1, 0, 1], dtype=bool)
    merged = merge_profitable_gaps(mask, 3.1, 1.0)
    if not np.all(merged):
        raise RuntimeError("profitable-gap self-check failed")
    refined = trim_boundaries(merged, values, float(values[mask].sum()), 3.1, 1.0)
    if values[refined].sum() < values[mask].sum() - 1e-12:
        raise RuntimeError("trimming self-check violated coverage")
    rng = np.random.default_rng(808)
    for n in (17, 64, 129):
        for _ in range(20):
            random_values = rng.random(n)
            random_mask = rng.random(n) < 0.35
            if not random_mask.any():
                random_mask[rng.integers(n)] = True
            target = float(random_values[random_mask].sum())
            before = BASE.affine_latency(random_mask, 0.011, 0.0003)
            _, result = refine_mask(random_mask, random_values, target, 0.011, 0.0003)
            after = BASE.affine_latency(result, 0.011, 0.0003)
            if after > before + 1e-12 or random_values[result].sum() < target - 1e-12:
                raise RuntimeError("randomized refinement invariant failed")


def aggregate_curve(rows: list[dict], target_cv: float, spatial_mode: str,
                    x_key: str, metrics: tuple[str, ...]) -> list[dict]:
    selected = [
        row for row in rows
        if float(row["target_cv"]) == target_cv
        and row["spatial_mode"] == spatial_mode
    ]
    return EXP2.summarize(selected, (x_key,), metrics)


def plot_method_grid(coverage_rows: list[dict], fixed_rows: list[dict], path: Path,
                     x_fields: dict, y_fields: dict, xlabel: str, ylabel: str,
                     x_log: bool = False) -> None:
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
            fixed_metrics = tuple({
                *(x_fields[key] for key in ("paper", "refined", "dpr", "top")),
                *(y_fields[key] for key in ("paper", "refined", "dpr", "top")),
            })
            fixed = aggregate_curve(
                fixed_rows, cv, mode, "budget_fraction", fixed_metrics
            )
            specs = (
                (coverage, "cover", "DP-Cover exact", BASE.PLOT_COLORS["coverage"], "X", "--"),
                (fixed, "paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "-"),
                (fixed, "refined", REFINED_LABEL, REFINED_COLOR, "P", "-"),
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
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage_ratios(fixed_rows: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    rows = [row for row in fixed_rows if float(row["target_cv"]) < 9.0]
    summaries = EXP2.summarize(
        rows, ("spatial_mode", "budget_fraction"),
        ("paper_cover_ratio", "refined_target_ratio", "dpr_cover_ratio", "top_cover_ratio"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True,
                             constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        points = [item for item in summaries if item["spatial_mode"] == mode]
        points.sort(key=lambda item: item["budget_fraction"])
        for key, label, color, marker in (
            ("paper_cover_ratio", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o"),
            ("refined_target_ratio", REFINED_LABEL, REFINED_COLOR, "P"),
            ("dpr_cover_ratio", "DP-R exact", BASE.PLOT_COLORS["fixed"], "s"),
            ("top_cover_ratio", "Top-R", BASE.PLOT_COLORS["top_r"], "^"),
        ):
            ax.plot([item["budget_fraction"] for item in points],
                    [item[key]["median"] for item in points],
                    color=color, marker=marker, label=label)
        ax.axhline(1.0, color=BASE.PLOT_COLORS["coverage"], linewidth=1.2,
                   linestyle="--", label="DP-Cover optimum")
        ax.set_yscale("log")
        ax.set_xlabel("Paper row budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Method latency / exact target latency (median, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_chunk_histograms(fixed_rows: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    rows = [row for row in fixed_rows if float(row["target_cv"]) < 9.0]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    methods = (
        ("paper_cover_chunks", "DP-Cover exact", BASE.PLOT_COLORS["coverage"]),
        ("paper_chunks", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
        ("refined_chunks", REFINED_LABEL, REFINED_COLOR),
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
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_self_check:
        self_check()

    required = ("summary.json", "fixed_r_trials.csv", "coverage_trials.csv",
                "input_trials.csv")
    for name in required:
        if not (args.source_results / name).is_file():
            raise SystemExit(f"missing Experiment 07 result: {args.source_results / name}")

    source_summary = json.loads((args.source_results / "summary.json").read_text())
    if source_summary.get("format") != "experiment-07-exact-n4864-v1":
        raise SystemExit("--source-results is not an Experiment 07 result directory")
    if int(source_summary["n"]) != 4864:
        raise SystemExit("Experiment 08 requires the N=4,864 Experiment 07 results")

    source_fixed = read_csv(args.source_results / "fixed_r_trials.csv")
    coverage_rows = read_csv(args.source_results / "coverage_trials.csv")
    source_inputs = read_csv(args.source_results / "input_trials.csv")
    fixed_by_key = {row_key(row): row for row in source_fixed}
    inputs_by_key = {input_key(row): row for row in source_inputs}
    if len(fixed_by_key) != len(source_fixed) or len(inputs_by_key) != len(source_inputs):
        raise RuntimeError("duplicate keys in Experiment 07 source results")

    n = int(source_summary["n"])
    row_size_kib = float(CONFIG["cols"]) * 2.0 / 1024.0
    table = BASE.LatencyTable.load("orin-agx")
    a_ms = float(source_summary["affine_fit"]["a_ms_per_chunk"])
    c_ms = float(source_summary["affine_fit"]["c_ms_per_row"])
    params = BASE.ChunkParams(start_kb=CONFIG["start_kib"],
                              jump_cap_kb=CONFIG["jump_cap_kib"])
    trials = int(source_summary["trials_per_cv"])
    cv_targets = [float(value) for value in source_summary["cv_targets"]]
    budget_fractions = [float(value) for value in source_summary["budget_fractions"]]
    rng = np.random.default_rng(int(source_summary["seed"]))
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    output_rows = []

    for trial in range(trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
            variants = EXP2.spatial_variants(base_values, rng, hotness, 0.95, 0.75)
            for mode, values in variants.items():
                source_input = inputs_by_key[(trial, round(target_cv, 10), mode)]
                actual_cv = EXP2.coefficient_of_variation(values)
                lag1, first_half = EXP2.spatial_stats(values)
                assert_close(actual_cv, source_input["actual_cv"], "actual CV")
                assert_close(lag1, source_input["lag1_corr"], "lag-1 correlation")
                assert_close(first_half, source_input["first_half_mass"], "first-half mass")
                values_t = torch.from_numpy(values.astype(np.float32))

                for fraction in budget_fractions:
                    key = (trial, round(target_cv, 10), mode, round(fraction, 10))
                    source = fixed_by_key[key]
                    budget = int(source["budget_rows"])
                    paper, replay_paper_ms = timed_call(
                        BASE.select_chunks, values_t, budget, row_size_kib, table,
                        params, "torch",
                    )
                    paper_mask = paper.mask.cpu().numpy()
                    replay = mask_metrics(paper_mask, values, a_ms, c_ms)
                    for metric in ("importance", "aff_ms"):
                        assert_close(replay[metric], source[f"paper_{metric}"], f"paper {metric}")
                    if replay["rows"] != source["paper_rows"] or replay["chunks"] != source["paper_chunks"]:
                        raise RuntimeError("Experiment 07 Paper mask statistics did not replay")

                    target = float(source["paper_importance"])
                    (merged_mask, refined_mask), refine_post_ms = timed_call(
                        refine_mask, paper_mask, values, target, a_ms, c_ms
                    )
                    merged = mask_metrics(merged_mask, values, a_ms, c_ms)
                    refined = mask_metrics(refined_mask, values, a_ms, c_ms)
                    exact_target_ms = float(source["paper_cover_aff_ms"])
                    if merged["aff_ms"] > replay["aff_ms"] + 1e-12:
                        raise RuntimeError("gap merging increased latency")
                    if refined["aff_ms"] > merged["aff_ms"] + 1e-12:
                        raise RuntimeError("boundary trimming increased latency")
                    if refined["importance"] < target - 1e-12:
                        raise RuntimeError("refined solution missed Paper importance")
                    if refined["aff_ms"] < exact_target_ms - 1e-10:
                        raise RuntimeError("refined solution beat reused exact DP-Cover result")

                    row = dict(source)
                    row.update({
                        "refinement_target_importance": target,
                        "merge_importance": merged["importance"],
                        "merge_rows": merged["rows"],
                        "merge_chunks": merged["chunks"],
                        "merge_aff_ms": merged["aff_ms"],
                        "merge_target_ratio": merged["aff_ms"] / exact_target_ms,
                        "merge_saving_vs_paper_pct": 100.0 * (1.0 - merged["aff_ms"] / replay["aff_ms"]),
                        "refined_importance": refined["importance"],
                        "refined_rows": refined["rows"],
                        "refined_chunks": refined["chunks"],
                        "refined_aff_ms": refined["aff_ms"],
                        "refined_ratio": refined["importance"] / refined["aff_ms"],
                        "refined_target_ratio": refined["aff_ms"] / exact_target_ms,
                        "refined_target_gap_pct": 100.0 * (refined["aff_ms"] / exact_target_ms - 1.0),
                        "refined_saving_vs_paper_pct": 100.0 * (1.0 - refined["aff_ms"] / replay["aff_ms"]),
                        "refined_excess_gap_recovered_pct": 100.0 * (
                            (replay["aff_ms"] - refined["aff_ms"])
                            / (replay["aff_ms"] - exact_target_ms)
                        ),
                        "refined_chunk_reduction_vs_paper_pct": 100.0 * (
                            1.0 - refined["chunks"] / replay["chunks"]
                        ),
                        "refined_importance_overshoot": refined["importance"] - target,
                        "replay_paper_runtime_ms": replay_paper_ms,
                        "refine_post_runtime_ms": refine_post_ms,
                        "refined_total_runtime_ms": replay_paper_ms + refine_post_ms,
                    })
                    output_rows.append(row)

    if len(output_rows) != len(source_fixed):
        raise RuntimeError("Experiment 08 did not reproduce every Experiment 07 fixed-R row")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fixed_r_trials.csv", output_rows)
    shutil.copy2(args.source_results / "coverage_trials.csv",
                 args.output_dir / "coverage_trials.csv")
    shutil.copy2(args.source_results / "input_trials.csv",
                 args.output_dir / "input_trials.csv")

    vlm = [row for row in output_rows if float(row["target_cv"]) < 9.0]
    refined_wins = [row["refined_aff_ms"] < row["paper_aff_ms"] - 1e-12 for row in vlm]
    summary = {
        "format": "experiment-08-greedy-refinement-v1",
        "source_experiment": "07_exact_n4864",
        "source_results": str(args.source_results.resolve()),
        "source_sha256": {
            name: file_sha256(args.source_results / name) for name in required
        },
        "reused_expensive_results": ["DP-Cover exact", "DP-R exact", "Top-R"],
        "n": n,
        "model": source_summary["model"],
        "shape": source_summary["shape"],
        "trials_per_cv": trials,
        "seed": source_summary["seed"],
        "cv_targets": cv_targets,
        "coverage_targets": source_summary["coverage_targets"],
        "budget_fractions": budget_fractions,
        "methods": ["DP-Cover exact", "Paper greedy", REFINED_LABEL,
                    "DP-R exact", "Top-R"],
        "affine_fit": source_summary["affine_fit"],
        "refinement": {
            "coverage_target": "importance attained by Paper greedy",
            "largest_strictly_profitable_gap_rows": int(math.ceil(a_ms / c_ms) - 1),
            "complexity_after_paper": "O(N log N) time, O(N) memory",
            "strict_win_rate": float(np.mean(refined_wins)),
            "nonworse_rate": float(np.mean([
                row["refined_aff_ms"] <= row["paper_aff_ms"] + 1e-12 for row in vlm
            ])),
            "exact_target_rate": float(np.mean([
                row["refined_target_ratio"] <= 1.0 + 1e-9 for row in vlm
            ])),
            "merge_saving_vs_paper_pct": distribution(
                row["merge_saving_vs_paper_pct"] for row in vlm
            ),
            "saving_vs_paper_pct": distribution(
                row["refined_saving_vs_paper_pct"] for row in vlm
            ),
            "paper_excess_gap_recovered_pct": distribution(
                row["refined_excess_gap_recovered_pct"] for row in vlm
            ),
            "target_gap_to_exact_pct": distribution(
                row["refined_target_gap_pct"] for row in vlm
            ),
            "target_ratio_to_exact": distribution(
                row["refined_target_ratio"] for row in vlm
            ),
            "importance_overshoot": distribution(
                row["refined_importance_overshoot"] for row in vlm
            ),
            "chunks": distribution(row["refined_chunks"] for row in vlm),
            "chunk_reduction_vs_paper_pct": distribution(
                row["refined_chunk_reduction_vs_paper_pct"] for row in vlm
            ),
            "rows": distribution(row["refined_rows"] for row in vlm),
            "replayed_paper_runtime_ms": distribution(
                row["replay_paper_runtime_ms"] for row in vlm
            ),
            "post_runtime_ms": distribution(row["refine_post_runtime_ms"] for row in vlm),
            "replayed_total_runtime_ms": distribution(
                row["refined_total_runtime_ms"] for row in vlm
            ),
        },
        "experiment_07_reference": {
            "dp_cover": source_summary["dp_cover"],
            "method_runtime_ms": source_summary["method_runtime_ms"],
            "coverage_ratio": source_summary["coverage_ratio"],
        },
        "timing_scope": "single-process CPU; exact and baseline metrics reused from Experiment 07",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_method_grid(
        coverage_rows, output_rows, args.output_dir / "importance_latency_frontiers.png",
        {"cover": "cover_aff_ms", "paper": "paper_aff_ms", "refined": "refined_aff_ms",
         "dpr": "dpr_aff_ms", "top": "top_aff_ms"},
        {"cover": "cover_importance", "paper": "paper_importance",
         "refined": "refined_importance", "dpr": "dpr_importance",
         "top": "top_importance"},
        "Affine latency (ms, log scale)", "Retained importance", True,
    )
    plot_method_grid(
        coverage_rows, output_rows, args.output_dir / "r_importance.png",
        {"cover": "cover_rows", "paper": "paper_rows", "refined": "refined_rows",
         "dpr": "dpr_rows", "top": "top_rows"},
        {"cover": "cover_importance", "paper": "paper_importance",
         "refined": "refined_importance", "dpr": "dpr_importance",
         "top": "top_importance"},
        "Selected rows R", "Retained importance",
    )
    plot_method_grid(
        coverage_rows, output_rows, args.output_dir / "r_latency.png",
        {"cover": "cover_rows", "paper": "paper_rows", "refined": "refined_rows",
         "dpr": "dpr_rows", "top": "top_rows"},
        {"cover": "cover_aff_ms", "paper": "paper_aff_ms",
         "refined": "refined_aff_ms", "dpr": "dpr_aff_ms", "top": "top_aff_ms"},
        "Selected rows R", "Affine latency (ms)",
    )
    plot_coverage_ratios(output_rows, args.output_dir / "coverage_optimality_ratio.png")
    plot_chunk_histograms(output_rows, args.output_dir / "chunk_count_histograms.png")
    print(f"wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
