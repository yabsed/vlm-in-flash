#!/usr/bin/env python3
"""Experiment 17: Paper and supported importance-latency curves across N."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_16 = (
    PROJECT_ROOT / "experiments" / "16_saturation_global_chain" / "run_experiment.py"
)
DEFAULT_N_VALUES = (256, 512, 1024, 2048, 4096, 4864, 8192)
DEFAULT_BUDGETS = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
DEFAULT_TARGETS = DEFAULT_BUDGETS + (0.99,)
PAPER_COLOR = "#D97706"
SUPPORTED_COLOR = "#2A9D8F"


def load_experiment_16():
    spec = importlib.util.spec_from_file_location("experiment_16_for_17", EXPERIMENT_16)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 16 from {EXPERIMENT_16}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP16 = load_experiment_16()
EXP13 = EXP16.EXP13
EXP2 = EXP13.EXP2
BASE = EXP16.BASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cv", type=float, default=3.30)
    parser.add_argument(
        "--paper-budget-fractions", nargs="+", type=float, default=DEFAULT_BUDGETS
    )
    parser.add_argument(
        "--supported-targets", nargs="+", type=float, default=DEFAULT_TARGETS
    )
    parser.add_argument("--row-size-kib", type=float, default=1.75)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--start-kib", type=float, default=12.0)
    parser.add_argument("--jump-cap-kib", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=20261701)
    parser.add_argument(
        "--models", nargs="+", choices=("two_line", "lookup"),
        default=("two_line", "lookup"),
    )
    parser.add_argument(
        "--paper-impl", choices=("native", "torch", "auto"), default="native"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[float], list[float]]:
    n_values = sorted(set(args.n_values))
    budgets = sorted(set(args.paper_budget_fractions))
    targets = sorted(set(args.supported_targets))
    if not n_values or n_values[0] < 2:
        raise SystemExit("--n-values must contain integers >= 2")
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    if args.cv <= 0 or any(args.cv >= math.sqrt(n - 1) for n in n_values):
        raise SystemExit("--cv must lie in (0, sqrt(N-1)) for every N")
    if not budgets or budgets[0] <= 0 or budgets[-1] > 1:
        raise SystemExit("Paper budget fractions must lie in (0, 1]")
    if not targets or targets[0] <= 0 or targets[-1] > 1:
        raise SystemExit("Supported targets must lie in (0, 1]")
    if args.row_size_kib <= 0:
        raise SystemExit("--row-size-kib must be positive")
    return n_values, budgets, targets


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def two_line_continuous_ms(rows: float, model: dict) -> float:
    if rows <= float(model["saturation_rows"]):
        return float(model["a_ms"] + model["c1_ms_per_row"] * rows)
    return float(model["c2_ms_per_row"] * rows)


def two_line_table(model: dict, row_size_kib: float, maximum_kib: int):
    return BASE.LatencyTable({
        kib: two_line_continuous_ms(kib / row_size_kib, model)
        for kib in range(1, maximum_kib + 1)
    }, meta={"model": "two_line"})


def run_lengths(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.int8)
    changes = np.diff(np.r_[0, binary, 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return ends - starts


def mask_metrics(mask: np.ndarray, values: np.ndarray, model_name: str,
                 two_line_model: dict, lookup_table, row_size_kib: float) -> dict:
    mask = np.asarray(mask, dtype=bool)
    lengths = run_lengths(mask)
    if model_name == "two_line":
        latency = float(sum(
            two_line_continuous_ms(float(length), two_line_model)
            for length in lengths
        ))
    else:
        latency = float(sum(
            lookup_table.read_ms(float(length) * row_size_kib)
            for length in lengths
        ))
    return {
        "importance": float(values[mask].sum()),
        "latency_ms": latency,
        "rows": int(mask.sum()),
        "chunks": int(len(lengths)),
    }


def supported_oracle(model_name: str, values: np.ndarray,
                     two_line_model: dict, lookup_model: dict):
    if model_name == "two_line":
        return EXP16.ExactSupportedOracle(values, two_line_model)
    return EXP16.ExactLookupSupportedOracle(values, lookup_model)


def node_metrics(model_name: str, node: dict) -> dict:
    latency_field = "two_line_ms" if model_name == "two_line" else "released_ms"
    return {
        "importance": float(node["importance"]),
        "latency_ms": float(node[latency_field]),
        "rows": int(node["rows"]),
        "chunks": int(node["chunks"]),
    }


def warm_up(two_line_model: dict, lookup_model: dict, paper_tables: dict,
            params, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(1700)
    values = rng.lognormal(size=64).astype(np.float64)
    values /= values.sum()
    values_t = torch.from_numpy(values.astype(np.float32))
    for table in paper_tables.values():
        BASE.select_chunks(
            values_t, 48, args.row_size_kib, table,
            params=params, impl=args.paper_impl,
        )
    EXP13._solve_lambda_metrics(
        values, 1.0, two_line_model["a_ms"], two_line_model["c1_ms_per_row"],
        two_line_model["c2_ms_per_row"], two_line_model["saturation_rows"],
        two_line_model["short_max_rows"],
    )
    EXP16._solve_lookup_lambda_metrics(
        values, 1.0, lookup_model["chunk_costs"], lookup_model["tail_ms_per_row"]
    )
    EXP16.ExactSupportedOracle(values, two_line_model)
    EXP16.ExactLookupSupportedOracle(values, lookup_model)


def collect(args: argparse.Namespace, n_values: list[int], budgets: list[float],
            targets: list[float], two_line_model: dict, lookup_model: dict,
            paper_tables: dict, params):
    outputs = {
        model: {"curve": [], "frontier": [], "matched": [], "input": []}
        for model in args.models
    }
    rng = np.random.default_rng(args.seed)
    for n in n_values:
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()
        for trial in range(args.trials):
            multiset = EXP2.exact_cv_lognormal(rng, n, args.cv)
            variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
            for spatial_mode, values in variants.items():
                values = np.asarray(values, dtype=np.float64)
                values_t = torch.from_numpy(values.astype(np.float32))
                total_importance = float(values.sum())
                base = {
                    "n": n,
                    "trial": trial,
                    "spatial_mode": spatial_mode,
                    "cv": args.cv,
                }
                for model_name in args.models:
                    result = outputs[model_name]
                    oracle = supported_oracle(
                        model_name, values, two_line_model, lookup_model
                    )
                    if oracle.solve_calls != 2 * len(oracle.nodes) - 1:
                        raise RuntimeError("adaptive solve count differs from 2H-1")
                    result["input"].append({
                        **base,
                        "supported_points": len(oracle.nodes),
                        "scalarized_solve_calls": oracle.solve_calls,
                    })
                    for node_index, node in enumerate(oracle.nodes):
                        metrics = node_metrics(model_name, node)
                        result["frontier"].append({
                            **base,
                            "node_index": node_index,
                            "lambda": float(node["lambda"]),
                            "importance_fraction": metrics["importance"]
                            / total_importance,
                            "latency_ms": metrics["latency_ms"],
                            "rows": metrics["rows"],
                            "chunks": metrics["chunks"],
                        })
                    for target_fraction in targets:
                        node = oracle.candidate(target_fraction * total_importance)
                        metrics = node_metrics(model_name, node)
                        result["curve"].append({
                            **base,
                            "method": "supported",
                            "control": target_fraction,
                            "importance_fraction": metrics["importance"]
                            / total_importance,
                            "latency_ms": metrics["latency_ms"],
                            "rows": metrics["rows"],
                            "chunks": metrics["chunks"],
                        })
                    for budget_fraction in budgets:
                        budget_rows = max(1, min(n, int(round(n * budget_fraction))))
                        paper = BASE.select_chunks(
                            values_t, budget_rows, args.row_size_kib,
                            paper_tables[model_name], params=params,
                            impl=args.paper_impl,
                        )
                        paper_metrics = mask_metrics(
                            paper.mask.cpu().numpy(), values, model_name,
                            two_line_model, paper_tables["lookup"],
                            args.row_size_kib,
                        )
                        result["curve"].append({
                            **base,
                            "method": "paper",
                            "control": budget_fraction,
                            "importance_fraction": paper_metrics["importance"]
                            / total_importance,
                            "latency_ms": paper_metrics["latency_ms"],
                            "rows": paper_metrics["rows"],
                            "chunks": paper_metrics["chunks"],
                        })
                        supported = node_metrics(
                            model_name, oracle.candidate(paper_metrics["importance"])
                        )
                        if supported["importance"] < paper_metrics["importance"] - 1e-10:
                            raise RuntimeError("supported point missed Paper importance")
                        result["matched"].append({
                            **base,
                            "paper_budget_fraction": budget_fraction,
                            "paper_importance_fraction": paper_metrics["importance"]
                            / total_importance,
                            "paper_latency_ms": paper_metrics["latency_ms"],
                            "supported_importance_fraction": supported["importance"]
                            / total_importance,
                            "supported_latency_ms": supported["latency_ms"],
                            "supported_saving_pct": 100.0 * (
                                1.0 - supported["latency_ms"]
                                / paper_metrics["latency_ms"]
                            ),
                            "supported_strict_win": (
                                supported["latency_ms"]
                                < paper_metrics["latency_ms"] - 1e-12
                            ),
                        })
                print(
                    f"N={n:5d} trial={trial + 1}/{args.trials} "
                    f"mode={spatial_mode:8s}",
                    flush=True,
                )
    return outputs


def build_summary(model_name: str, matched: pd.DataFrame,
                  inputs: pd.DataFrame, args: argparse.Namespace,
                  n_values: list[int], two_line_model: dict,
                  lookup_model: dict) -> dict:
    by_n = {}
    for n in n_values:
        selected = matched[matched.n == n]
        input_selected = inputs[inputs.n == n]
        by_n[str(n)] = {
            "supported_saving_pct": distribution(selected.supported_saving_pct),
            "supported_strict_win_rate": float(selected.supported_strict_win.mean()),
            "supported_points": distribution(input_selected.supported_points),
            "scalarized_solve_calls": distribution(
                input_selected.scalarized_solve_calls
            ),
        }
    parameters = (
        two_line_model if model_name == "two_line" else {
            "max_kib": lookup_model["max_kib"],
            "cap_rows": lookup_model["cap_rows"],
            "tail_ms_per_row": lookup_model["tail_ms_per_row"],
        }
    )
    return {
        "format": "experiment-17-importance-latency-v1",
        "latency_model": model_name,
        "n_values": n_values,
        "trials_per_n": args.trials * len(EXP2.SPATIAL_MODES),
        "cv": args.cv,
        "spatial_modes": list(EXP2.SPATIAL_MODES),
        "paper_budget_fractions": sorted(set(args.paper_budget_fractions)),
        "supported_targets": sorted(set(args.supported_targets)),
        "row_size_kib": args.row_size_kib,
        "profile": args.profile,
        "paper_impl": args.paper_impl,
        "paper_params_kib": {
            "start": args.start_kib,
            "step": args.start_kib,
            "jump_cap": args.jump_cap_kib,
        },
        "latency_parameters": parameters,
        "by_n": by_n,
    }


def configure_plot():
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    return plt


def plot_importance_latency(curve: pd.DataFrame, model_name: str,
                            output_dir: Path) -> None:
    plt = configure_plot()
    n_values = sorted(int(value) for value in curve.n.unique())
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.4), constrained_layout=True)
    for ax, n in zip(axes.flat, n_values):
        selected = curve[curve.n == n]
        for method, label, color, marker, linestyle in (
            ("paper", "Paper greedy", PAPER_COLOR, "o", "--"),
            ("supported", "Supported", SUPPORTED_COLOR, "X", "-"),
        ):
            data = selected[selected.method == method]
            grouped = data.groupby("control", sort=True)
            latency = grouped.latency_ms.mean().to_numpy()
            importance = grouped.importance_fraction.mean().to_numpy()
            latency_lo = grouped.latency_ms.quantile(0.05).to_numpy()
            latency_hi = grouped.latency_ms.quantile(0.95).to_numpy()
            latency = np.r_[0.0, latency]
            importance = np.r_[0.0, importance]
            latency_lo = np.r_[0.0, latency_lo]
            latency_hi = np.r_[0.0, latency_hi]
            ax.plot(latency, importance, color=color, marker=marker, markersize=4.5,
                    linestyle=linestyle, linewidth=1.9, label=label)
            ax.fill_betweenx(
                importance, latency_lo, latency_hi, color=color, alpha=0.10,
            )
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.01)
        ax.set_xlabel("Predicted I/O latency (ms)")
        ax.set_ylabel("Retained importance")
        ax.set_title(f"N={n:,}")
        BASE.polish_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unused_axes = list(axes.flat[len(n_values):])
    for ax in unused_axes:
        ax.set_axis_off()
    if unused_axes:
        unused_axes[0].legend(
            handles, labels, loc="center", frameon=False, fontsize=12,
        )
    else:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{model_name.replace('_', ' ').title()} importance-latency curves")
    fig.savefig(output_dir / "importance_latency.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / "importance_latency.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_examples(curve: pd.DataFrame, frontier: pd.DataFrame,
                  model_name: str, output_dir: Path) -> None:
    plt = configure_plot()
    n_values = sorted(int(value) for value in curve.n.unique())
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.4), constrained_layout=True)
    for ax, n in zip(axes.flat, n_values):
        paper = curve[
            (curve.n == n) & (curve.trial == 0)
            & (curve.spatial_mode == "random") & (curve.method == "paper")
        ].sort_values("control")
        supported = frontier[
            (frontier.n == n) & (frontier.trial == 0)
            & (frontier.spatial_mode == "random")
        ].sort_values("node_index")
        ax.plot(
            np.r_[0.0, paper.latency_ms],
            np.r_[0.0, paper.importance_fraction],
            color=PAPER_COLOR, marker="o", markersize=4.5,
            linestyle="--", linewidth=1.8, label="Paper greedy",
        )
        ax.plot(
            supported.latency_ms, supported.importance_fraction,
            color=SUPPORTED_COLOR, linewidth=2.0, label="Full supported frontier",
        )
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.01)
        ax.set_xlabel("Predicted I/O latency (ms)")
        ax.set_ylabel("Retained importance")
        ax.set_title(f"N={n:,}")
        BASE.polish_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unused_axes = list(axes.flat[len(n_values):])
    for ax in unused_axes:
        ax.set_axis_off()
    if unused_axes:
        unused_axes[0].legend(
            handles, labels, loc="center", frameon=False, fontsize=12,
        )
    else:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} example frontiers "
        "(trial 0, random order)"
    )
    fig.savefig(
        output_dir / "importance_latency_examples.png", dpi=190,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "importance_latency_examples.pdf", bbox_inches="tight"
    )
    plt.close(fig)


def plot_saving(matched: pd.DataFrame, model_name: str,
                output_dir: Path) -> None:
    plt = configure_plot()
    grouped = matched.groupby("n", sort=True).supported_saving_pct
    n = np.asarray(sorted(int(value) for value in matched.n.unique()))
    median = grouped.median().reindex(n).to_numpy()
    lo = grouped.quantile(0.05).reindex(n).to_numpy()
    hi = grouped.quantile(0.95).reindex(n).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(n, median, color=SUPPORTED_COLOR, marker="X", linewidth=2.0)
    ax.fill_between(n, lo, hi, color=SUPPORTED_COLOR, alpha=0.14)
    ax.axhline(0.0, color="#374151", linewidth=1.0, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Selectable rows N")
    ax.set_ylabel("Supported latency saving vs Paper (%)")
    ax.set_title(f"{model_name.replace('_', ' ').title()} Paper-matched saving")
    BASE.polish_axis(ax)
    fig.savefig(output_dir / "paper_matched_saving.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / "paper_matched_saving.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze_model(model_name: str, output_dir: Path, args: argparse.Namespace,
                  n_values: list[int], two_line_model: dict,
                  lookup_model: dict) -> dict:
    curve = pd.read_csv(output_dir / "curve_trials.csv")
    frontier = pd.read_csv(output_dir / "frontier_nodes.csv")
    matched = pd.read_csv(output_dir / "paper_matched_trials.csv")
    inputs = pd.read_csv(output_dir / "input_trials.csv")
    summary = build_summary(
        model_name, matched, inputs, args, n_values,
        two_line_model, lookup_model,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_importance_latency(curve, model_name, output_dir)
    plot_examples(curve, frontier, model_name, output_dir)
    plot_saving(matched, model_name, output_dir)
    return summary


def write_report(summaries: dict[str, dict]) -> None:
    lines = [
        "# Experiment 17 보고서: N별 importance-latency 곡선",
        "",
        "Paper greedy와 complete supported frontier를 두 latency model에서 각각 비교했다.",
        "모든 latency는 실제 I/O 측정값이 아니라 해당 모델의 predicted I/O latency다.",
        "",
    ]
    for model_name in ("two_line", "lookup"):
        if model_name not in summaries:
            continue
        summary = summaries[model_name]
        lines.extend([
            f"## {model_name.replace('_', ' ').title()} latency model",
            "",
            "| N | Supported points H | Solves P | 평균 절감 | 중앙값 절감 | 승률 |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for n in summary["n_values"]:
            row = summary["by_n"][str(n)]
            lines.append(
                f"| {n:,} | {row['supported_points']['mean']:.1f} "
                f"| {row['scalarized_solve_calls']['mean']:.1f} "
                f"| {row['supported_saving_pct']['mean']:.2f}% "
                f"| {row['supported_saving_pct']['median']:.2f}% "
                f"| {100.0 * row['supported_strict_win_rate']:.1f}% |"
            )
        lines.extend([
            "",
            f"![Importance-latency curves](results/{model_name}/importance_latency.png)",
            "",
            f"![Complete example frontiers](results/{model_name}/importance_latency_examples.png)",
            "",
            f"![Paper-matched saving](results/{model_name}/paper_matched_saving.png)",
            "",
        ])
    lines.extend([
        "## 해석",
        "",
        "Supported 곡선은 동일 importance에서 Paper보다 왼쪽에 있을수록 좋다. "
        "Paper-matched 표와 saving 그림은 각 Paper point가 달성한 importance를 "
        "target으로 supported point를 다시 선택해 계산했으므로, 서로 다른 "
        "importance를 직접 비교하는 오류를 피한다.",
        "",
    ])
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    n_values, budgets, targets = validate_args(args)
    lookup_table = BASE.LatencyTable.load(args.profile)
    two_line_model = EXP13.fit_continuous_two_line(
        lookup_table, args.row_size_kib, args.saturation_kib
    )
    lookup_model = EXP16.lookup_cost_model(lookup_table, args.row_size_kib)
    paper_tables = {
        "two_line": two_line_table(
            two_line_model, args.row_size_kib, int(lookup_table.max_kb)
        ),
        "lookup": lookup_table,
    }
    params = BASE.ChunkParams(
        start_kb=args.start_kib, jump_cap_kb=args.jump_cap_kib
    )

    if not args.analyze_only:
        if not args.skip_self_check:
            EXP16.self_check_supported_oracle()
            EXP16.self_check_lookup_solver()
        warm_up(two_line_model, lookup_model, paper_tables, params, args)
        outputs = collect(
            args, n_values, budgets, targets, two_line_model,
            lookup_model, paper_tables, params,
        )
        for model_name, tables in outputs.items():
            output_dir = args.output_dir / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            write_csv(output_dir / "curve_trials.csv", tables["curve"])
            write_csv(output_dir / "frontier_nodes.csv", tables["frontier"])
            write_csv(output_dir / "paper_matched_trials.csv", tables["matched"])
            write_csv(output_dir / "input_trials.csv", tables["input"])

    summaries = {}
    for model_name in args.models:
        output_dir = args.output_dir / model_name
        summaries[model_name] = analyze_model(
            model_name, output_dir, args, n_values,
            two_line_model, lookup_model,
        )
    write_report(summaries)
    print(json.dumps({
        model: {
            n: {
                "mean_saving_pct": values["supported_saving_pct"]["mean"],
                "win_rate": values["supported_strict_win_rate"],
            }
            for n, values in summary["by_n"].items()
        }
        for model, summary in summaries.items()
    }, indent=2))
    print(f"wrote Experiment 17 curves to {args.output_dir}")


if __name__ == "__main__":
    main()
