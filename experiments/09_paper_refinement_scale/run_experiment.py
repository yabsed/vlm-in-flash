#!/usr/bin/env python3
"""Experiment 09: large paired Paper-versus-refinement comparison."""

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
EXPERIMENT_08 = PROJECT_ROOT / "experiments" / "08_greedy_refinement" / "run_experiment.py"


def load_experiment_08():
    spec = importlib.util.spec_from_file_location("experiment_08", EXPERIMENT_08)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 08 from {EXPERIMENT_08}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP8 = load_experiment_08()
BASE = EXP8.BASE
EXP2 = EXP8.EXP2
MATRIX_CONFIGS = EXP8.EXP7.EXP6.MATRIX_CONFIGS
SPATIAL_MODES = EXP8.SPATIAL_MODES
SPATIAL_LABELS = EXP8.SPATIAL_LABELS
PAPER_COLOR = BASE.PLOT_COLORS["greedy"]
MERGE_COLOR = "#F2C14E"
REFINED_COLOR = EXP8.REFINED_COLOR
LOOKUP_COLOR = "#6C5B7B"


def timed_call(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return result, (time.perf_counter_ns() - start) / 1e6


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def lookup_latency(mask: np.ndarray, row_size_kib: float, table) -> float:
    selected = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, selected, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return float(sum(
        table.read_ms(int(end - start) * row_size_kib)
        for start, end in zip(starts, ends)
    ))


def stats(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def cluster_bootstrap_mean(data: pd.DataFrame, metric: str, samples: int,
                           seed: int) -> tuple[float, float]:
    cluster_means = (
        data.groupby(["n", "trial", "target_cv"], sort=False)[metric]
        .mean().to_numpy(dtype=np.float64)
    )
    if len(cluster_means) < 2:
        value = float(cluster_means[0])
        return value, value
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    block = 500
    for start in range(0, samples, block):
        count = min(block, samples - start)
        indices = rng.integers(0, len(cluster_means), size=(count, len(cluster_means)))
        estimates[start:start + count] = cluster_means[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize(data: pd.DataFrame, bootstrap_samples: int, seed: int) -> dict:
    affine_ci = cluster_bootstrap_mean(
        data, "refined_affine_ratio", bootstrap_samples, seed
    )
    lookup_ci = cluster_bootstrap_mean(
        data, "refined_lookup_ratio", bootstrap_samples, seed + 1
    )
    return {
        "cases": int(len(data)),
        "independent_multisets": int(
            len(data[["n", "trial", "target_cv"]].drop_duplicates())
        ),
        "affine_latency_ratio": {
            **stats(data["refined_affine_ratio"]),
            "cluster_bootstrap_ci95_mean": list(affine_ci),
        },
        "affine_saving_pct": stats(data["refined_affine_saving_pct"]),
        "lookup_latency_ratio": {
            **stats(data["refined_lookup_ratio"]),
            "cluster_bootstrap_ci95_mean": list(lookup_ci),
        },
        "lookup_saving_pct": stats(data["refined_lookup_saving_pct"]),
        "affine_strict_win_rate": float(data["refined_affine_strict_win"].mean()),
        "affine_nonworse_rate": float(data["refined_affine_nonworse"].mean()),
        "lookup_strict_win_rate": float(data["refined_lookup_strict_win"].mean()),
        "lookup_nonworse_rate": float(data["refined_lookup_nonworse"].mean()),
        "merge_affine_saving_pct": stats(data["merge_affine_saving_pct"]),
        "paper_chunks": stats(data["paper_chunks"]),
        "refined_chunks": stats(data["refined_chunks"]),
        "chunk_reduction_pct": stats(data["chunk_reduction_pct"]),
        "row_ratio": stats(data["refined_row_ratio"]),
        "importance_overshoot": stats(data["importance_overshoot"]),
        "paper_runtime_ms": stats(data["paper_runtime_ms"]),
        "post_runtime_ms": stats(data["post_runtime_ms"]),
        "total_runtime_ms": stats(data["total_runtime_ms"]),
        "runtime_overhead_pct": stats(data["runtime_overhead_pct"]),
    }


def grouped_summary_rows(data: pd.DataFrame) -> list[dict]:
    specifications = [
        ("N", ["n"]),
        ("N-ordering", ["n", "spatial_mode"]),
        ("N-CV", ["n", "target_cv"]),
        ("N-budget", ["n", "budget_fraction"]),
        ("N-ordering-CV-budget", ["n", "spatial_mode", "target_cv", "budget_fraction"]),
    ]
    rows = []
    for scope, keys in specifications:
        grouper = keys[0] if len(keys) == 1 else keys
        for group_key, group in data.groupby(grouper, sort=True):
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            row = {
                "scope": scope,
                "n": "", "spatial_mode": "", "target_cv": "",
                "budget_fraction": "",
            }
            row.update(dict(zip(keys, values)))
            row.update({
                "cases": len(group),
                "affine_latency_ratio_mean": group.refined_affine_ratio.mean(),
                "affine_latency_ratio_median": group.refined_affine_ratio.median(),
                "affine_saving_pct_mean": group.refined_affine_saving_pct.mean(),
                "lookup_latency_ratio_mean": group.refined_lookup_ratio.mean(),
                "lookup_saving_pct_mean": group.refined_lookup_saving_pct.mean(),
                "affine_strict_win_rate": group.refined_affine_strict_win.mean(),
                "lookup_strict_win_rate": group.refined_lookup_strict_win.mean(),
                "paper_chunks_mean": group.paper_chunks.mean(),
                "refined_chunks_mean": group.refined_chunks.mean(),
                "refined_row_ratio_mean": group.refined_row_ratio.mean(),
                "post_runtime_ms_mean": group.post_runtime_ms.mean(),
                "total_runtime_ms_mean": group.total_runtime_ms.mean(),
            })
            rows.append(row)
    return rows


def plot_overview(data: pd.DataFrame, summaries: dict[int, dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    ns = sorted(summaries)
    labels = [f"{n:,}" for n in ns]
    x = np.arange(len(ns))
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0), constrained_layout=True)

    affine_merge = [100 * data[data.n == n].merge_affine_ratio.mean() for n in ns]
    affine_refined = [100 * summaries[n]["affine_latency_ratio"]["mean"] for n in ns]
    affine_lo = [100 * summaries[n]["affine_latency_ratio"]["cluster_bootstrap_ci95_mean"][0] for n in ns]
    affine_hi = [100 * summaries[n]["affine_latency_ratio"]["cluster_bootstrap_ci95_mean"][1] for n in ns]
    ax = axes[0, 0]
    ax.plot(x, np.full(len(ns), 100.0), color=PAPER_COLOR, marker="o", label="Paper")
    ax.plot(x, affine_merge, color=MERGE_COLOR, marker="s", label="Paper + merge")
    ax.errorbar(x, affine_refined,
                yerr=[np.maximum(0.0, np.asarray(affine_refined)-np.asarray(affine_lo)),
                      np.maximum(0.0, np.asarray(affine_hi)-np.asarray(affine_refined))],
                color=REFINED_COLOR, marker="P", capsize=3,
                label="Paper + merge + trim")
    ax.set_ylabel("Affine latency / Paper latency (%)")
    ax.set_title("A. Affine-model latency")
    ax.set_ylim(bottom=min(affine_refined) - 5, top=102)

    lookup_merge = [100 * data[data.n == n].merge_lookup_ratio.mean() for n in ns]
    lookup_refined = [100 * summaries[n]["lookup_latency_ratio"]["mean"] for n in ns]
    lookup_lo = [100 * summaries[n]["lookup_latency_ratio"]["cluster_bootstrap_ci95_mean"][0] for n in ns]
    lookup_hi = [100 * summaries[n]["lookup_latency_ratio"]["cluster_bootstrap_ci95_mean"][1] for n in ns]
    ax = axes[0, 1]
    ax.plot(x, np.full(len(ns), 100.0), color=PAPER_COLOR, marker="o", label="Paper")
    ax.plot(x, lookup_merge, color=MERGE_COLOR, marker="s", label="Paper + merge")
    ax.errorbar(x, lookup_refined,
                yerr=[np.maximum(0.0, np.asarray(lookup_refined)-np.asarray(lookup_lo)),
                      np.maximum(0.0, np.asarray(lookup_hi)-np.asarray(lookup_refined))],
                color=LOOKUP_COLOR, marker="P", capsize=3,
                label="Paper + merge + trim")
    ax.set_ylabel("Lookup latency / Paper latency (%)")
    ax.set_title("B. Released lookup-table latency")
    ax.set_ylim(bottom=min(lookup_refined) - 2,
                top=max(102, max(lookup_merge) + 2))

    ax = axes[1, 0]
    ax.plot(x, [summaries[n]["paper_chunks"]["mean"] for n in ns],
            color=PAPER_COLOR, marker="o", label="Paper")
    ax.plot(x, [summaries[n]["refined_chunks"]["mean"] for n in ns],
            color=REFINED_COLOR, marker="P", label="Paper + merge + trim")
    ax.set_ylabel("Mean chunk count K")
    ax.set_title("C. Fragmentation")

    ax = axes[1, 1]
    ax.plot(x, [summaries[n]["paper_runtime_ms"]["median"] for n in ns],
            color=PAPER_COLOR, marker="o", label="Paper")
    ax.plot(x, [summaries[n]["post_runtime_ms"]["median"] for n in ns],
            color=REFINED_COLOR, marker="s", linestyle="--", label="Postprocess only")
    ax.plot(x, [summaries[n]["total_runtime_ms"]["median"] for n in ns],
            color=REFINED_COLOR, marker="P", label="Paper + postprocess")
    ax.set_ylabel("Median CPU wall time (ms)")
    ax.set_title("D. Selection runtime")

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Channel count N")
        BASE.polish_axis(ax)
        ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_ecdfs(data: pd.DataFrame, path: Path, metric: str,
               xlabel: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    ns = sorted(data.n.unique())
    colors = {"random": "#577590", "local": "#43AA8B", "hot-cold": "#F8961E"}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, n in zip(axes.flat, ns):
        selected = data[data.n == n]
        for mode in SPATIAL_MODES:
            x = np.sort(100 * selected[selected.spatial_mode == mode][metric].to_numpy())
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, color=colors[mode], label=SPATIAL_LABELS[mode])
        ax.axvline(100, color="#64748B", linestyle="--", linewidth=1.2)
        ax.set_title(f"N={n:,}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative fraction")
        BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(data: pd.DataFrame, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    ns = sorted(data.n.unique())
    cvs = sorted(data.target_cv.unique())
    budgets = sorted(data.budget_fraction.unique())
    cell_means = data.groupby(
        ["n", "spatial_mode", "target_cv", "budget_fraction"]
    ).refined_affine_saving_pct.mean()
    vmax = max(5.0, float(cell_means.max()))
    fig, axes = plt.subplots(len(ns), len(SPATIAL_MODES), figsize=(15.5, 17.0),
                             constrained_layout=True, squeeze=False)
    image = None
    for row_index, n in enumerate(ns):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            selected = data[(data.n == n) & (data.spatial_mode == mode)]
            matrix = np.asarray([
                [selected[(selected.target_cv == cv) &
                          (selected.budget_fraction == budget)]
                 .refined_affine_saving_pct.mean()
                 for budget in budgets]
                for cv in cvs
            ])
            image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
            for i in range(len(cvs)):
                for j in range(len(budgets)):
                    color = "white" if matrix[i, j] > 0.58 * vmax else "#1F2937"
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                            fontsize=7.5, color=color)
            ax.set_xticks(range(len(budgets)), [f"{100*b:g}%" for b in budgets])
            ax.set_yticks(range(len(cvs)), [f"{cv:g}" for cv in cvs])
            ax.set_xlabel("Paper row budget / N")
            ax.set_ylabel("Importance CV")
            ax.set_title(f"N={n:,}, {SPATIAL_LABELS[mode]}")
    fig.colorbar(image, ax=axes, shrink=0.55, label="Mean affine-latency saving (%)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def self_check() -> None:
    EXP8.self_check()
    table = BASE.LatencyTable.load("orin-agx")
    mask = np.asarray([0, 1, 1, 0, 1, 0], dtype=bool)
    expected = table.read_ms(2 * 1.75) + table.read_ms(1.75)
    actual = lookup_latency(mask, 1.75, table)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("lookup-latency self-check failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[item["n"] for item in MATRIX_CONFIGS])
    parser.add_argument("--cv-targets", type=float, nargs="+",
                        default=[1.07, 1.25, 1.44, 2.48, 3.30, 4.55, 9.19])
    parser.add_argument("--budget-fractions", type=float, nargs="+",
                        default=[0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875])
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--dtype-bytes", type=float, default=2.0)
    parser.add_argument("--local-rho", type=float, default=0.95)
    parser.add_argument("--hot-rho", type=float, default=0.75)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.bootstrap_samples < 100:
        raise SystemExit("--trials must be positive and --bootstrap-samples at least 100")
    config_by_n = {int(item["n"]): item for item in MATRIX_CONFIGS}
    unknown = sorted(set(args.n_values) - set(config_by_n))
    if unknown:
        raise SystemExit(f"no real-matrix metadata for N={unknown}")
    configs = [config_by_n[n] for n in sorted(set(args.n_values))]
    cv_targets = sorted(set(float(value) for value in args.cv_targets))
    budget_fractions = sorted(set(float(value) for value in args.budget_fractions))
    if not args.skip_self_check:
        self_check()
        EXP2.self_check()

    table = BASE.LatencyTable.load(args.profile)
    all_rows: list[dict] = []
    per_n_metadata = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        n = int(config["n"])
        row_size_kib = float(config["cols"]) * args.dtype_bytes / 1024.0
        a_ms, c_ms, fit_r2 = BASE.affine_fit(table, row_size_kib)
        params = BASE.ChunkParams(start_kb=config["start_kib"],
                                  jump_cap_kb=config["jump_cap_kib"])
        budgets = [int(round(n * fraction)) for fraction in budget_fractions]
        rng = np.random.default_rng(args.seed + n)
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()
        rows = []
        total_inputs = args.trials * len(cv_targets) * len(SPATIAL_MODES)
        completed = 0
        report_every = max(1, total_inputs // 20)

        for trial in range(args.trials):
            for target_cv in cv_targets:
                base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
                variants = EXP2.spatial_variants(
                    base_values, rng, hotness, args.local_rho, args.hot_rho
                )
                for mode, values in variants.items():
                    values_t = torch.from_numpy(values.astype(np.float32))
                    actual_cv = EXP2.coefficient_of_variation(values)
                    lag1, first_half = EXP2.spatial_stats(values)
                    for fraction, budget in zip(budget_fractions, budgets):
                        paper, paper_ms = timed_call(
                            BASE.select_chunks, values_t, budget, row_size_kib, table,
                            params=params, impl="torch",
                        )
                        paper_mask = paper.mask.cpu().numpy()
                        paper_metrics = EXP8.mask_metrics(paper_mask, values, a_ms, c_ms)
                        paper_lookup = lookup_latency(paper_mask, row_size_kib, table)
                        target = paper_metrics["importance"]

                        (merged_mask, refined_mask), post_ms = timed_call(
                            EXP8.refine_mask, paper_mask, values, target, a_ms, c_ms
                        )
                        merged = EXP8.mask_metrics(merged_mask, values, a_ms, c_ms)
                        refined = EXP8.mask_metrics(refined_mask, values, a_ms, c_ms)
                        merge_lookup = lookup_latency(merged_mask, row_size_kib, table)
                        refined_lookup = lookup_latency(refined_mask, row_size_kib, table)

                        if refined["importance"] < target - 1e-12:
                            raise RuntimeError("refinement violated the Paper coverage target")
                        if refined["aff_ms"] > paper_metrics["aff_ms"] + 1e-12:
                            raise RuntimeError("refinement increased affine latency")

                        affine_ratio = refined["aff_ms"] / paper_metrics["aff_ms"]
                        lookup_ratio = refined_lookup / paper_lookup
                        merge_affine_ratio = merged["aff_ms"] / paper_metrics["aff_ms"]
                        merge_lookup_ratio = merge_lookup / paper_lookup
                        rows.append({
                            "n": n, "model": config["model"], "shape": config["shape"],
                            "trial": trial, "target_cv": target_cv,
                            "actual_cv": actual_cv, "spatial_mode": mode,
                            "lag1_corr": lag1, "first_half_mass": first_half,
                            "budget_fraction": fraction, "budget_rows": budget,
                            "row_size_kib": row_size_kib,
                            "a_ms_per_chunk": a_ms, "c_ms_per_row": c_ms,
                            "profitable_gap_rows": int(math.ceil(a_ms / c_ms) - 1),
                            "paper_importance": paper_metrics["importance"],
                            "paper_rows": paper_metrics["rows"],
                            "paper_chunks": paper_metrics["chunks"],
                            "paper_aff_ms": paper_metrics["aff_ms"],
                            "paper_lookup_ms": paper_lookup,
                            "paper_runtime_ms": paper_ms,
                            "merge_importance": merged["importance"],
                            "merge_rows": merged["rows"],
                            "merge_chunks": merged["chunks"],
                            "merge_aff_ms": merged["aff_ms"],
                            "merge_lookup_ms": merge_lookup,
                            "merge_affine_ratio": merge_affine_ratio,
                            "merge_lookup_ratio": merge_lookup_ratio,
                            "merge_affine_saving_pct": 100.0 * (1.0 - merge_affine_ratio),
                            "merge_lookup_saving_pct": 100.0 * (1.0 - merge_lookup_ratio),
                            "refined_importance": refined["importance"],
                            "refined_rows": refined["rows"],
                            "refined_chunks": refined["chunks"],
                            "refined_aff_ms": refined["aff_ms"],
                            "refined_lookup_ms": refined_lookup,
                            "refined_affine_ratio": affine_ratio,
                            "refined_lookup_ratio": lookup_ratio,
                            "refined_affine_saving_pct": 100.0 * (1.0 - affine_ratio),
                            "refined_lookup_saving_pct": 100.0 * (1.0 - lookup_ratio),
                            "refined_affine_strict_win": affine_ratio < 1.0 - 1e-12,
                            "refined_affine_nonworse": affine_ratio <= 1.0 + 1e-12,
                            "refined_lookup_strict_win": lookup_ratio < 1.0 - 1e-12,
                            "refined_lookup_nonworse": lookup_ratio <= 1.0 + 1e-12,
                            "refined_row_ratio": refined["rows"] / paper_metrics["rows"],
                            "chunk_reduction_pct": 100.0 * (
                                1.0 - refined["chunks"] / paper_metrics["chunks"]
                            ),
                            "importance_overshoot": refined["importance"] - target,
                            "post_runtime_ms": post_ms,
                            "total_runtime_ms": paper_ms + post_ms,
                            "runtime_overhead_pct": 100.0 * post_ms / paper_ms,
                        })

                    completed += 1
                    if completed % report_every == 0 or completed == total_inputs:
                        print(f"N={n}: completed {completed}/{total_inputs} inputs", flush=True)

        result_dir = args.output_dir / f"n_{n}"
        result_dir.mkdir(parents=True, exist_ok=True)
        write_csv(result_dir / "paired_trials.csv", rows)
        frame = pd.DataFrame(rows)
        vlm = frame[frame.target_cv < 9.0].copy()
        stress = frame[frame.target_cv >= 9.0].copy()
        n_summary = summarize(vlm, args.bootstrap_samples, args.seed + n)
        metadata = {
            "format": "experiment-09-paper-refinement-n-v1",
            "n": n, "model": config["model"], "shape": config["shape"],
            "row_size_kib": row_size_kib, "trials_per_cv": args.trials,
            "seed": args.seed + n, "cv_targets": cv_targets,
            "budget_fractions": budget_fractions,
            "affine_fit": {
                "a_ms_per_chunk": a_ms, "c_ms_per_row": c_ms,
                "r_squared": fit_r2,
                "chunk_open_equivalent_rows": a_ms / c_ms,
                "largest_strictly_profitable_gap_rows": int(math.ceil(a_ms / c_ms) - 1),
            },
            "primary_vlm_summary": n_summary,
            "stress_cv_summary": summarize(
                stress, args.bootstrap_samples, args.seed + n + 900_000
            ),
        }
        (result_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
        per_n_metadata.append(metadata)
        all_rows.extend(rows)

    write_csv(args.output_dir / "paired_trials.csv", all_rows)
    all_frame = pd.DataFrame(all_rows)
    vlm = all_frame[all_frame.target_cv < 9.0].copy()
    stress = all_frame[all_frame.target_cv >= 9.0].copy()
    summaries = {
        int(item["n"]): item["primary_vlm_summary"] for item in per_n_metadata
    }
    aggregate_summary = {
        "format": "experiment-09-paper-refinement-scale-v1",
        "trials_per_cv": args.trials,
        "base_seed": args.seed,
        "n_values": [int(item["n"]) for item in per_n_metadata],
        "cv_targets": cv_targets,
        "primary_vlm_cv_targets": [cv for cv in cv_targets if cv < 9.0],
        "spatial_modes": list(SPATIAL_MODES),
        "budget_fractions": budget_fractions,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "independently generated (N, trial, CV) value multiset; orderings and budgets stay clustered",
        "total_paired_cases": len(all_frame),
        "primary_vlm_paired_cases": len(vlm),
        "overall_vlm": summarize(vlm, args.bootstrap_samples, args.seed),
        "overall_stress_cv": summarize(
            stress, args.bootstrap_samples, args.seed + 900_000
        ),
        "per_n": per_n_metadata,
        "latency_models": {
            "primary": "fitted affine aK+cR optimized by the refinement",
            "sensitivity": "released Orin AGX lookup table evaluated after selection",
        },
        "timing_scope": "single-process CPU; Paper and postprocess measured in the same run",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate_summary, indent=2) + "\n"
    )
    write_csv(args.output_dir / "grouped_summary.csv", grouped_summary_rows(vlm))
    plot_overview(vlm, summaries, args.output_dir / "comparison_overview.png")
    plot_ecdfs(
        vlm, args.output_dir / "latency_ratio_ecdf.png",
        "refined_affine_ratio", "Refined affine latency / Paper latency (%)",
    )
    plot_ecdfs(
        vlm, args.output_dir / "lookup_ratio_ecdf.png",
        "refined_lookup_ratio", "Refined lookup latency / Paper latency (%)",
    )
    plot_heatmaps(vlm, args.output_dir / "saving_heatmaps.png")
    print(f"wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
