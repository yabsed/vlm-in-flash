#!/usr/bin/env python3
"""Experiment 16: saturation-aware global-chain DP versus Paper greedy.

The saved Experiment 13 masks are reused.  Its ``lag`` masks are exact global
solutions of each scalarized two-line objective on a dense lambda grid; for a
coverage target, the cheapest feasible supported point is retained.  This
experiment makes that interpretation explicit, validates the shortfall cost
identity, and plots the requested I-L, I-R, L-R, and run-length comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_13 = (
    PROJECT_ROOT / "experiments" / "13_two_line_lagrangian" / "run_experiment.py"
)
DEFAULT_SOURCE = PROJECT_ROOT / "experiments" / "13_two_line_lagrangian" / "results"
GLOBAL_COLOR = "#E6A700"


def load_experiment_13():
    spec = importlib.util.spec_from_file_location("experiment_13_for_16", EXPERIMENT_13)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 13 from {EXPERIMENT_13}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP13 = load_experiment_13()
BASE = EXP13.BASE
SPATIAL_MODES = EXP13.SPATIAL_MODES
SPATIAL_LABELS = EXP13.SPATIAL_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def summarize(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def configure_plot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    return plt


def run_lengths(encoded: str) -> np.ndarray:
    return EXP13.EXP11.run_lengths(encoded)


def shortfall_cost(lengths: np.ndarray, model: dict) -> float:
    c2 = float(model["c2_ms_per_row"])
    delta = float(model["d_ms_per_excess_row"])
    saturation = float(model["saturation_rows"])
    return float(c2 * lengths.sum() + delta * np.maximum(saturation - lengths, 0).sum())


def capped_run_penalized(values: np.ndarray, multiplier: float, c1: float,
                         c2: float, saturation: int) -> tuple[float, float, float]:
    """Reference O(Ns) binary-chain DP for an integer saturation length.

    State 0 is outside a run; state ``saturation`` represents every active run
    whose length is at least saturation. Ties prefer greater importance and
    then lower latency, matching Experiment 13's optimized interval solver.
    """
    delta = c2 - c1
    opening = delta * saturation + c1
    scores = np.full(saturation + 1, -np.inf, dtype=np.float64)
    importance = np.full(saturation + 1, -np.inf, dtype=np.float64)
    costs = np.full(saturation + 1, np.inf, dtype=np.float64)
    scores[0] = importance[0] = costs[0] = 0.0

    def better(score, kept, cost, old_score, old_kept, old_cost) -> bool:
        if np.isneginf(old_score):
            return not np.isneginf(score)
        if np.isneginf(score):
            return False
        scale = max(1.0, abs(score), abs(old_score))
        tolerance = 1e-13 * scale
        return bool(
            score > old_score + tolerance
            or (
                abs(score - old_score) <= tolerance
                and (
                    kept > old_kept + 1e-14
                    or (
                        abs(kept - old_kept) <= 1e-14
                        and cost < old_cost - 1e-14
                    )
                )
            )
        )

    for value in np.asarray(values, dtype=np.float64):
        next_scores = np.full_like(scores, -np.inf)
        next_importance = np.full_like(importance, -np.inf)
        next_costs = np.full_like(costs, np.inf)

        # Skip the current row and close any active run.
        for state in range(saturation + 1):
            if better(
                scores[state], importance[state], costs[state],
                next_scores[0], next_importance[0], next_costs[0],
            ):
                next_scores[0] = scores[state]
                next_importance[0] = importance[state]
                next_costs[0] = costs[state]

        # Start a new run.
        score = scores[0] + multiplier * value - opening
        kept = importance[0] + value
        cost = costs[0] + opening
        if better(score, kept, cost, next_scores[1], next_importance[1], next_costs[1]):
            next_scores[1], next_importance[1], next_costs[1] = score, kept, cost

        # Extend an unsaturated run, including the transition s-1 -> s.
        for state in range(1, saturation):
            target = state + 1
            score = scores[state] + multiplier * value - c1
            kept = importance[state] + value
            cost = costs[state] + c1
            if better(
                score, kept, cost,
                next_scores[target], next_importance[target], next_costs[target],
            ):
                next_scores[target] = score
                next_importance[target] = kept
                next_costs[target] = cost

        # Extend an already saturated run.
        score = scores[saturation] + multiplier * value - c2
        kept = importance[saturation] + value
        cost = costs[saturation] + c2
        if better(
            score, kept, cost,
            next_scores[saturation], next_importance[saturation], next_costs[saturation],
        ):
            next_scores[saturation] = score
            next_importance[saturation] = kept
            next_costs[saturation] = cost

        scores, importance, costs = next_scores, next_importance, next_costs

    best = 0
    for state in range(1, saturation + 1):
        if better(
            scores[state], importance[state], costs[state],
            scores[best], importance[best], costs[best],
        ):
            best = state
    return float(scores[best]), float(importance[best]), float(costs[best])


def self_check_capped_dp() -> None:
    """Check the explicit O(Ns) recurrence against the O(N) interval solver."""
    rng = np.random.default_rng(1616)
    saturation = 4
    c1 = 0.07
    c2 = 0.13
    a = (c2 - c1) * saturation
    for n in (7, 12, 17):
        for _ in range(3):
            values = rng.lognormal(size=n)
            values /= values.sum()
            for multiplier in (0.0, 0.1, 1.0, 10.0):
                actual = capped_run_penalized(
                    values, multiplier, c1, c2, saturation
                )
                expected = EXP13._solve_lambda_metrics(
                    values, multiplier, a, c1, c2, float(saturation), saturation
                )[:3]
                if not all(
                    math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-11)
                    for left, right in zip(actual, expected)
                ):
                    raise RuntimeError(
                        f"capped-run DP self-check failed: {actual} != {expected}"
                    )


def normalized_trials(source: pd.DataFrame, model: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows: list[dict] = []
    length_rows: list[dict] = []
    policies = ("two_line", "released", "tail_linear", "block_split")

    for row in source.to_dict("records"):
        base = {
            "trial": int(row["trial"]),
            "target_cv": float(row["target_cv"]),
            "spatial_mode": row["spatial_mode"],
            "budget_fraction": float(row["budget_fraction"]),
            "budget_rows": int(row["budget_rows"]),
            "target_importance": float(row["target_importance"]),
        }
        output = dict(base)
        for source_prefix, prefix in (("paper", "paper"), ("lag", "global_chain")):
            lengths = run_lengths(row[f"{source_prefix}_runs"])
            if len(lengths) != int(row[f"{source_prefix}_chunks"]):
                raise RuntimeError(f"run-count mismatch for {source_prefix}")
            if int(lengths.sum()) != int(row[f"{source_prefix}_rows"]):
                raise RuntimeError(f"row-count mismatch for {source_prefix}")
            reconstructed = shortfall_cost(lengths, model)
            stored = float(row[f"{source_prefix}_two_line_ms"])
            if not math.isclose(reconstructed, stored, rel_tol=1e-10, abs_tol=1e-11):
                raise RuntimeError(
                    f"shortfall identity mismatch: reconstructed={reconstructed}, stored={stored}"
                )

            output[f"{prefix}_importance"] = float(row[f"{source_prefix}_importance"])
            output[f"{prefix}_rows"] = int(row[f"{source_prefix}_rows"])
            output[f"{prefix}_chunks"] = int(row[f"{source_prefix}_chunks"])
            output[f"{prefix}_runs"] = row[f"{source_prefix}_runs"]
            output[f"{prefix}_shortfall_rows"] = float(
                np.maximum(float(model["saturation_rows"]) - lengths, 0).sum()
            )
            output[f"{prefix}_median_chunk_rows"] = float(np.median(lengths))
            for policy in policies:
                output[f"{prefix}_{policy}_ms"] = float(
                    row[f"{source_prefix}_{policy}_ms"]
                )

            for length in lengths:
                length_rows.append({
                    **base,
                    "method": prefix,
                    "chunk_rows": int(length),
                    "is_saturated": bool(length >= float(model["saturation_rows"])),
                })

        if output["global_chain_importance"] < output["target_importance"] - 1e-12:
            raise RuntimeError("global-chain supported point missed the Paper target")
        for policy in policies:
            global_ms = output[f"global_chain_{policy}_ms"]
            paper_ms = output[f"paper_{policy}_ms"]
            output[f"global_chain_saving_vs_paper_{policy}_pct"] = 100.0 * (
                1.0 - global_ms / paper_ms
            )
            output[f"global_chain_win_{policy}"] = float(global_ms < paper_ms - 1e-12)
        output["global_chain_importance_overshoot"] = (
            output["global_chain_importance"] - output["target_importance"]
        )
        paired_rows.append(output)

    return pd.DataFrame(paired_rows), pd.DataFrame(length_rows)


def build_summary(source_summary: dict, paired: pd.DataFrame,
                  chunks: pd.DataFrame, args: argparse.Namespace) -> dict:
    policies = ("two_line", "released", "tail_linear", "block_split")
    method_structure = {}
    for method in ("global_chain", "paper"):
        method_chunks = chunks[chunks.method == method]
        method_structure[method] = {
            "rows": summarize(paired[f"{method}_rows"]),
            "chunks_per_solution": summarize(paired[f"{method}_chunks"]),
            "shortfall_rows_per_solution": summarize(
                paired[f"{method}_shortfall_rows"]
            ),
            "chunk_rows": summarize(method_chunks.chunk_rows),
            "saturated_chunk_fraction": float(method_chunks.is_saturated.mean()),
        }

    return {
        "format": "experiment-16-saturation-global-chain-v1",
        "source_experiment": "13_two_line_lagrangian",
        "source_results": str(args.source_results.resolve()),
        "source_sha256": {
            name: file_sha256(args.source_results / name)
            for name in ("summary.json", "paper_matched_trials.csv")
        },
        "n": int(source_summary["n"]),
        "q": int(source_summary["q"]),
        "cases": {
            "spatial_inputs": int(
                paired[["trial", "target_cv", "spatial_mode"]].drop_duplicates().shape[0]
            ),
            "paired_targets": int(len(paired)),
            "global_chain_chunks": int((chunks.method == "global_chain").sum()),
            "paper_chunks": int((chunks.method == "paper").sum()),
        },
        "comparison": (
            "For every Paper-greedy achieved-importance target, select the minimum-cost "
            "feasible mask among q exact scalarized binary-chain optima. The chain result "
            "is globally optimal for its lambda, but is a supported-point heuristic for "
            "the constrained coverage problem."
        ),
        "two_line_identity": (
            "L(M)=c2*R(M)+(c2-c1)*sum_C max(s-|C|,0); validated for every saved mask"
        ),
        "solver_validation": (
            "The explicit O(Ns) capped-run recurrence was checked against Experiment "
            "13's exact O(N) interval solver on randomized small inputs."
        ),
        "two_line_model": source_summary["two_line_model"],
        "global_chain_importance_overshoot": summarize(
            paired.global_chain_importance_overshoot
        ),
        "global_chain_saving_vs_paper_pct": {
            policy: {
                **summarize(paired[f"global_chain_saving_vs_paper_{policy}_pct"]),
                "strict_win_rate": float(
                    paired[f"global_chain_win_{policy}"].mean()
                ),
            }
            for policy in policies
        },
        "solution_structure": method_structure,
        "by_spatial_mode_two_line_saving_pct": {
            mode: summarize(
                group.global_chain_saving_vs_paper_two_line_pct
            )
            for mode, group in paired.groupby("spatial_mode", sort=True)
        },
        "by_cv_two_line_saving_pct": {
            str(cv): summarize(group.global_chain_saving_vs_paper_two_line_pct)
            for cv, group in paired.groupby("target_cv", sort=True)
        },
        "by_budget_two_line_saving_pct": {
            str(budget): summarize(group.global_chain_saving_vs_paper_two_line_pct)
            for budget, group in paired.groupby("budget_fraction", sort=True)
        },
    }


def plot_method_grid(frame: pd.DataFrame, path: Path, x_fields: dict,
                     y_fields: dict, xlabel: str, ylabel: str,
                     x_log: bool = False) -> None:
    plt = configure_plot()
    display_cvs = (1.25, 3.30, 4.55)
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 14.5), constrained_layout=True)
    specs = (
        ("global_chain", "Global-chain DP (supported)", GLOBAL_COLOR, "P", "-"),
        ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "--"),
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
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_chunk_histogram(chunks: pd.DataFrame, saturation_rows: float,
                         path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=True,
                             constrained_layout=True)
    specs = (
        ("global_chain", "Global-chain DP (supported)", GLOBAL_COLOR),
        ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
    )
    maximum = int(chunks.chunk_rows.max())
    edges = np.unique(np.r_[
        0.5,
        np.rint(np.geomspace(1, maximum + 1, 45)) + 0.5,
    ])
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = chunks[chunks.spatial_mode == mode]
        for method, label, color in specs:
            values = selected[selected.method == method].chunk_rows.to_numpy()
            ax.hist(
                values, bins=edges, weights=np.ones(len(values)) / len(values),
                histtype="step", linewidth=2.0, color=color, label=label,
            )
        ax.axvline(
            saturation_rows, color="#334155", linewidth=1.2, linestyle=":",
            label=f"s = {saturation_rows:.1f} rows",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Chunk length (rows, log scale)")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Fraction of chunks")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_plots(paired: pd.DataFrame, chunks: pd.DataFrame,
                 model: dict, output_dir: Path) -> None:
    plot_method_grid(
        paired, output_dir / "importance_latency.png",
        {
            "global_chain": "global_chain_two_line_ms",
            "paper": "paper_two_line_ms",
        },
        {
            "global_chain": "global_chain_importance",
            "paper": "paper_importance",
        },
        "Two-line latency L (ms, log scale)", "Retained importance I", True,
    )
    plot_method_grid(
        paired, output_dir / "importance_rows.png",
        {"global_chain": "global_chain_rows", "paper": "paper_rows"},
        {
            "global_chain": "global_chain_importance",
            "paper": "paper_importance",
        },
        "Selected rows R", "Retained importance I",
    )
    plot_method_grid(
        paired, output_dir / "latency_rows.png",
        {"global_chain": "global_chain_rows", "paper": "paper_rows"},
        {
            "global_chain": "global_chain_two_line_ms",
            "paper": "paper_two_line_ms",
        },
        "Selected rows R", "Two-line latency L (ms)",
    )
    plot_chunk_histogram(
        chunks, float(model["saturation_rows"]), output_dir / "chunk_length_histogram.png"
    )


def main() -> None:
    args = parse_args()
    summary_path = args.source_results / "summary.json"
    trials_path = args.source_results / "paper_matched_trials.csv"
    for path in (summary_path, trials_path):
        if not path.is_file():
            raise SystemExit(f"missing source result: {path}")

    source_summary = json.loads(summary_path.read_text())
    if source_summary.get("format") != "experiment-13-two-line-lagrangian-v1":
        raise SystemExit("--source-results is not an Experiment 13 result directory")
    self_check_capped_dp()
    source = pd.read_csv(trials_path)
    paired, chunks = normalized_trials(source, source_summary["two_line_model"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_trials.csv", index=False)
    chunks.to_csv(args.output_dir / "chunk_lengths.csv", index=False)
    summary = build_summary(source_summary, paired, chunks, args)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_plots(paired, chunks, source_summary["two_line_model"], args.output_dir)
    print(json.dumps(summary["global_chain_saving_vs_paper_pct"], indent=2))
    print(f"wrote Experiment 16 results to {args.output_dir}")


if __name__ == "__main__":
    main()
