#!/usr/bin/env python3
"""Experiment 02: CV- and ordering-controlled importance distributions.

The experiment compares Paper greedy, the exact fixed-R DP, and the O(qN)
quantized Pareto coverage solver.  Each base value multiset is calibrated to
an exact target coefficient of variation, then reused under random, locally
clustered, and persistent hot-cold orderings so marginal dispersion and
spatial structure are not confounded.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_EXPERIMENT = PROJECT_ROOT / "experiments" / "01_random_r_bound" / "run_experiment.py"


def load_base_experiment():
    spec = importlib.util.spec_from_file_location("experiment_01", BASE_EXPERIMENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load baseline experiment from {BASE_EXPERIMENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_experiment()
SPATIAL_MODES = ("random", "local", "hot-cold")
SPATIAL_LABELS = {
    "random": "Random order",
    "local": "Locally clustered",
    "hot-cold": "Persistent hot-cold",
}


def coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std() / values.mean())


def exact_cv_lognormal(
    rng: np.random.Generator, n: int, target_cv: float
) -> np.ndarray:
    """Return a positive lognormal-shaped multiset with exact sample CV."""
    if n < 2 or not 0 < target_cv < math.sqrt(n - 1):
        raise ValueError("target CV must lie in (0, sqrt(n-1))")
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std()

    def values_at(sigma: float) -> np.ndarray:
        log_values = sigma * z
        log_values -= log_values.max()
        return np.exp(log_values)

    lower, upper = 0.0, 1.0
    while coefficient_of_variation(values_at(upper)) < target_cv:
        upper *= 2.0
        if upper > 64:
            raise RuntimeError(f"failed to bracket target CV={target_cv}")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if coefficient_of_variation(values_at(middle)) < target_cv:
            lower = middle
        else:
            upper = middle
    values = values_at(upper)
    values /= values.sum()
    if not math.isclose(
        coefficient_of_variation(values), target_cv, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise RuntimeError("CV calibration did not converge")
    return values


def assign_by_score(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Assign one fixed multiset to positions according to ascending scores."""
    output = np.empty_like(values)
    output[np.argsort(scores, kind="stable")] = np.sort(values)
    return output


def spatial_variants(
    values: np.ndarray,
    rng: np.random.Generator,
    hotness: np.ndarray,
    local_rho: float,
    hot_rho: float,
) -> dict[str, np.ndarray]:
    """Reorder an identical multiset under three spatial assumptions."""
    n = len(values)
    random_values = rng.permutation(values)

    innovation = rng.standard_normal(n)
    local_score = np.empty(n, dtype=np.float64)
    local_score[0] = innovation[0]
    scale = math.sqrt(1.0 - local_rho * local_rho)
    for i in range(1, n):
        local_score[i] = local_rho * local_score[i - 1] + scale * innovation[i]
    local_values = assign_by_score(values, local_score)

    hot_noise = rng.standard_normal(n)
    hot_score = hot_rho * hotness + math.sqrt(1.0 - hot_rho * hot_rho) * hot_noise
    hot_values = assign_by_score(values, hot_score)

    return {
        "random": random_values,
        "local": local_values,
        "hot-cold": hot_values,
    }


def spatial_stats(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        lag1 = 0.0
    else:
        lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    first_half_mass = float(values[: len(values) // 2].sum() / values.sum())
    return lag1, first_half_mass


def distribution_stats(values: list[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def summarize(
    rows: list[dict], group_keys: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups):
        members = groups[key]
        item = {name: value for name, value in zip(group_keys, key)}
        item["count"] = len(members)
        for metric in metrics:
            item[metric] = distribution_stats([float(row[metric]) for row in members])
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select(summary: list[dict], **conditions) -> list[dict]:
    return [
        item
        for item in summary
        if all(item[key] == value for key, value in conditions.items())
    ]


def plot_cv_fixed_gain(summary: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    modes = list(SPATIAL_MODES)
    budgets = sorted({float(item["budget_fraction"]) for item in summary})
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(budgets)))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, modes):
        for budget, color in zip(budgets, colors):
            points = sorted(
                select(summary, spatial_mode=mode, budget_fraction=budget),
                key=lambda item: item["target_cv"],
            )
            x = np.asarray([item["target_cv"] for item in points])
            y = np.asarray([item["fixed_gain_pct"]["mean"] for item in points])
            ax.plot(x, y, marker="o", color=color, label=f"R/N={100*budget:g}%")
        ax.axhline(0, color="#455A64", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(sorted({item["target_cv"] for item in summary}))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_title(SPATIAL_LABELS[mode])
        ax.set_xlabel("Target coefficient of variation")
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Exact fixed-R I/L gain over Paper greedy (%)")
    axes[-1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Effect of paper-calibrated CV on the fixed-R optimization gap", fontsize=13)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_matched_latency(condition_summary: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        points = sorted(
            select(condition_summary, spatial_mode=mode),
            key=lambda item: item["target_cv"],
        )
        x = np.asarray([item["target_cv"] for item in points])
        for metric, label, color, marker in (
            (
                "greedy_extra_vs_pareto_pct",
                "Paper greedy",
                BASE.PLOT_COLORS["greedy"],
                "o",
            ),
            (
                "fixed_extra_vs_pareto_pct",
                "Exact fixed-R DP",
                BASE.PLOT_COLORS["fixed"],
                "D",
            ),
        ):
            mean = np.asarray([item[metric]["mean"] for item in points])
            lo = np.asarray([item[metric]["p05"] for item in points])
            hi = np.asarray([item[metric]["p95"] for item in points])
            ax.plot(x, mean, marker=marker, color=color, label=label)
            ax.fill_between(x, lo, hi, alpha=0.12, color=color)
        ax.axhline(
            0,
            color=BASE.PLOT_COLORS["pareto"],
            linestyle=":",
            label="Quantized Pareto reference",
        )
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_title(SPATIAL_LABELS[mode])
        ax.set_xlabel("Target coefficient of variation")
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Extra latency at matched importance (%)")
    axes[-1].legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "Latency relative to Quantized Pareto at each fixed-R method's achieved importance",
        fontsize=13,
    )
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_frontiers(
    fixed_summary: list[dict], frontier_summary: list[dict], path: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    available = sorted({float(item["target_cv"]) for item in fixed_summary})
    desired = (1.25, 3.30, 9.19)
    representative = [min(available, key=lambda value: abs(value - target)) for target in desired]
    representative = list(dict.fromkeys(representative))
    fig, axes = plt.subplots(
        len(representative), 3, figsize=(14.8, 4.0 * len(representative)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row_index, target_cv in enumerate(representative):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            fixed_points = sorted(
                select(fixed_summary, target_cv=target_cv, spatial_mode=mode),
                key=lambda item: item["budget_fraction"],
            )
            for prefix, label, color, marker in (
                ("greedy", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o"),
                ("fixed", "Exact fixed-R DP", BASE.PLOT_COLORS["fixed"], "D"),
            ):
                latency = [item[f"{prefix}_aff_ms"]["mean"] for item in fixed_points]
                importance = [item[f"{prefix}_importance"]["mean"] for item in fixed_points]
                ax.plot(latency, importance, marker=marker, color=color, label=label)
            pareto_points = sorted(
                select(frontier_summary, target_cv=target_cv, spatial_mode=mode),
                key=lambda item: item["coverage_target"],
            )
            ax.plot(
                [item["pareto_aff_ms"]["mean"] for item in pareto_points],
                [item["pareto_achieved"]["mean"] for item in pareto_points],
                marker="P",
                linestyle="-.",
                color=BASE.PLOT_COLORS["pareto"],
                label="Quantized Pareto O(qN)",
            )
            ax.set_xscale("log")
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={target_cv:g}")
            ax.set_xlabel("Affine latency (ms, log scale)")
            ax.set_ylabel("Retained importance")
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def self_check() -> None:
    rng = np.random.default_rng(17)
    hotness = np.linspace(1.0, -1.0, 256)
    hotness = (hotness - hotness.mean()) / hotness.std()
    for target in (1.07, 1.44, 3.30, 4.55, 9.19):
        values = exact_cv_lognormal(rng, 256, target)
        variants = spatial_variants(values, rng, hotness, 0.95, 0.75)
        for variant in variants.values():
            assert math.isclose(
                coefficient_of_variation(variant), target, rel_tol=1e-10, abs_tol=1e-10
            )
            assert np.allclose(np.sort(variant), np.sort(values), rtol=0, atol=0)
    BASE.self_check()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260917)
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
    parser.add_argument("--pareto-q", type=int, default=256)
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
    if cv_targets[0] <= 0 or cv_targets[-1] >= math.sqrt(args.n - 1):
        raise SystemExit("every target CV must lie in (0, sqrt(n-1))")
    budgets = sorted(set(args.budget_rows))
    if not budgets or budgets[0] < 1 or budgets[-1] > args.n:
        raise SystemExit("every row budget must lie in [1, n]")
    coverage_targets = sorted(set(args.coverage_targets))
    if not coverage_targets or coverage_targets[0] <= 0 or coverage_targets[-1] > 1:
        raise SystemExit("every coverage target must lie in (0, 1]")
    if args.pareto_q < 2:
        raise SystemExit("--pareto-q must be at least 2")
    if not 0 <= args.local_rho < 1 or not 0 <= args.hot_rho < 1:
        raise SystemExit("ordering correlations must lie in [0, 1)")
    if not args.skip_self_check:
        self_check()

    table = BASE.LatencyTable.load(args.profile)
    a_ms, c_ms_per_row, fit_r2 = BASE.affine_fit(table, args.row_size_kib)
    params = BASE.ChunkParams(start_kb=args.start_kib, jump_cap_kb=args.jump_cap_kib)
    rng = np.random.default_rng(args.seed)
    hotness = np.linspace(1.0, -1.0, args.n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    fixed_rows: list[dict] = []
    frontier_rows: list[dict] = []
    condition_rows: list[dict] = []

    total_multisets = args.trials * len(cv_targets)
    completed = 0
    for trial in range(args.trials):
        for target_cv in cv_targets:
            base_values = exact_cv_lognormal(rng, args.n, target_cv)
            variants = spatial_variants(
                base_values, rng, hotness, args.local_rho, args.hot_rho
            )
            for spatial_mode, values in variants.items():
                actual_cv = coefficient_of_variation(values)
                lag1_corr, first_half_mass = spatial_stats(values)
                values_t = torch.from_numpy(values.astype(np.float32))
                pareto_oracle = BASE.QuantizedParetoOracle(
                    values, a_ms, c_ms_per_row, args.pareto_q
                )
                condition_rows.append(
                    {
                        "trial": trial,
                        "target_cv": target_cv,
                        "spatial_mode": spatial_mode,
                        "actual_cv": actual_cv,
                        "lag1_corr": lag1_corr,
                        "first_half_mass": first_half_mass,
                        "pareto_peak_states": pareto_oracle.peak_states,
                    }
                )
                for coverage_target in coverage_targets:
                    pareto = pareto_oracle.solve(coverage_target)
                    frontier_rows.append(
                        {
                            "trial": trial,
                            "target_cv": target_cv,
                            "spatial_mode": spatial_mode,
                            "coverage_target": coverage_target,
                            "pareto_achieved": pareto["importance"],
                            "pareto_overshoot": pareto["importance"] - coverage_target,
                            "pareto_rows": pareto["rows"],
                            "pareto_chunks": pareto["chunks"],
                            "pareto_aff_ms": pareto["aff_ms"],
                        }
                    )
                for budget in budgets:
                    greedy = BASE.select_chunks(
                        values_t,
                        budget,
                        args.row_size_kib,
                        table,
                        params=params,
                        impl="torch",
                    )
                    greedy_mask = greedy.mask.cpu().numpy()
                    greedy_rows_count, greedy_chunks = BASE.mask_stats(greedy_mask)
                    if greedy_rows_count == 0:
                        raise RuntimeError("Paper greedy selected an empty mask")
                    fixed_mask, iterations, residual = BASE.exact_fixed_r_affine(
                        values, greedy_rows_count, a_ms, c_ms_per_row
                    )
                    fixed_rows_count, fixed_chunks = BASE.mask_stats(fixed_mask)
                    greedy_importance = float(values[greedy_mask].sum())
                    fixed_importance = float(values[fixed_mask].sum())
                    greedy_latency = BASE.affine_latency(
                        greedy_mask, a_ms, c_ms_per_row
                    )
                    fixed_latency = BASE.affine_latency(fixed_mask, a_ms, c_ms_per_row)
                    greedy_ratio = greedy_importance / greedy_latency
                    fixed_ratio = fixed_importance / fixed_latency
                    if fixed_ratio + 1e-10 < greedy_ratio:
                        raise RuntimeError("exact fixed-R DP lost to a feasible greedy mask")
                    pareto_at_greedy = pareto_oracle.solve(greedy_importance)
                    pareto_at_fixed = pareto_oracle.solve(fixed_importance)
                    fixed_rows.append(
                        {
                            "trial": trial,
                            "target_cv": target_cv,
                            "spatial_mode": spatial_mode,
                            "actual_cv": actual_cv,
                            "lag1_corr": lag1_corr,
                            "first_half_mass": first_half_mass,
                            "budget_rows": budget,
                            "budget_fraction": budget / args.n,
                            "greedy_rows": greedy_rows_count,
                            "greedy_chunks": greedy_chunks,
                            "greedy_importance": greedy_importance,
                            "greedy_aff_ms": greedy_latency,
                            "greedy_aff_ratio": greedy_ratio,
                            "fixed_rows": fixed_rows_count,
                            "fixed_chunks": fixed_chunks,
                            "fixed_importance": fixed_importance,
                            "fixed_aff_ms": fixed_latency,
                            "fixed_aff_ratio": fixed_ratio,
                            "fixed_gain_pct": 100 * (fixed_ratio / greedy_ratio - 1),
                            "pareto_at_greedy_importance": pareto_at_greedy["importance"],
                            "pareto_at_greedy_aff_ms": pareto_at_greedy["aff_ms"],
                            "greedy_extra_vs_pareto_pct": 100
                            * (greedy_latency / pareto_at_greedy["aff_ms"] - 1),
                            "pareto_at_fixed_importance": pareto_at_fixed["importance"],
                            "pareto_at_fixed_aff_ms": pareto_at_fixed["aff_ms"],
                            "fixed_extra_vs_pareto_pct": 100
                            * (fixed_latency / pareto_at_fixed["aff_ms"] - 1),
                            "dinkelbach_iterations": iterations,
                            "dinkelbach_residual": residual,
                        }
                    )
            completed += 1
            if completed % max(1, total_multisets // 20) == 0:
                print(
                    f"completed {completed}/{total_multisets} CV-calibrated multisets",
                    flush=True,
                )

    fixed_metrics = (
        "actual_cv",
        "lag1_corr",
        "first_half_mass",
        "greedy_chunks",
        "greedy_importance",
        "greedy_aff_ms",
        "greedy_aff_ratio",
        "fixed_chunks",
        "fixed_importance",
        "fixed_aff_ms",
        "fixed_aff_ratio",
        "fixed_gain_pct",
        "greedy_extra_vs_pareto_pct",
        "fixed_extra_vs_pareto_pct",
        "dinkelbach_iterations",
    )
    fixed_summary = summarize(
        fixed_rows, ("target_cv", "spatial_mode", "budget_fraction"), fixed_metrics
    )
    comparison_summary = summarize(
        fixed_rows,
        ("target_cv", "spatial_mode"),
        (
            "actual_cv",
            "lag1_corr",
            "first_half_mass",
            "fixed_gain_pct",
            "greedy_extra_vs_pareto_pct",
            "fixed_extra_vs_pareto_pct",
        ),
    )
    condition_summary = summarize(
        condition_rows,
        ("target_cv", "spatial_mode"),
        ("actual_cv", "lag1_corr", "first_half_mass", "pareto_peak_states"),
    )
    frontier_summary = summarize(
        frontier_rows,
        ("target_cv", "spatial_mode", "coverage_target"),
        (
            "pareto_achieved",
            "pareto_overshoot",
            "pareto_rows",
            "pareto_chunks",
            "pareto_aff_ms",
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = args.output_dir / "fixed_r_trials.csv"
    frontier_path = args.output_dir / "pareto_frontier_trials.csv"
    conditions_path = args.output_dir / "condition_trials.csv"
    summary_path = args.output_dir / "summary.json"
    gain_path = args.output_dir / "cv_fixed_r_gain.png"
    matched_path = args.output_dir / "cv_matched_latency.png"
    frontier_figure_path = args.output_dir / "importance_latency_frontiers.png"
    write_csv(fixed_path, fixed_rows)
    write_csv(frontier_path, frontier_rows)
    write_csv(conditions_path, condition_rows)
    metadata = {
        "format": "experiment-02-cv-sweep-v1",
        "seed": args.seed,
        "trials_per_cv": args.trials,
        "n": args.n,
        "cv_targets": cv_targets,
        "cv_source": "VLM-in-a-Flash Appendix C Table 1; 9.19 is an OPT reference",
        "marginal_family": "sample-CV-calibrated lognormal",
        "spatial_modes": list(SPATIAL_MODES),
        "local_rho": args.local_rho,
        "hot_rho": args.hot_rho,
        "budgets": budgets,
        "coverage_targets": coverage_targets,
        "pareto_q": args.pareto_q,
        "row_size_kib": args.row_size_kib,
        "profile": args.profile,
        "affine_fit": {
            "a_ms_per_chunk": a_ms,
            "c_ms_per_row": c_ms_per_row,
            "r_squared": fit_r2,
        },
        "fixed_summary": fixed_summary,
        "comparison_summary": comparison_summary,
        "condition_summary": condition_summary,
        "frontier_summary": frontier_summary,
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    plot_cv_fixed_gain(fixed_summary, gain_path)
    plot_matched_latency(comparison_summary, matched_path)
    plot_frontiers(fixed_summary, frontier_summary, frontier_figure_path)

    for path in (
        fixed_path,
        frontier_path,
        conditions_path,
        summary_path,
        gain_path,
        matched_path,
        frontier_figure_path,
    ):
        if path.suffix == ".png":
            print(f"wrote {path} and {path.with_suffix('.pdf')}")
        else:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
