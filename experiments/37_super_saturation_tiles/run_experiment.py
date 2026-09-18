#!/usr/bin/env python3
"""Experiment 37: test Cell-C tiles longer than the saturation length s."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import re
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXP36_PATH = (
    PROJECT_ROOT / "experiments" / "36_cell_parameter_sweep"
    / "run_experiment.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP36 = _load_module("experiment_36_for_37", EXP36_PATH)
EXP35 = EXP36.EXP35
EXP32 = EXP36.EXP32
EXP29 = EXP36.EXP29
MODEL_SPECS = EXP36.MODEL_SPECS
PROMPTS = EXP36.PROMPTS
LOCAL_PROFILE = EXP36.LOCAL_PROFILE

CELL_COUNTS = (1, 2, 4, 8)
LENGTH_FACTORS = (1, 2, 3, 4)
TILE_METHODS = tuple(
    f"c{cell}_l{factor}"
    for cell in CELL_COUNTS for factor in LENGTH_FACTORS
)
METHODS = ("paper", *TILE_METHODS)
METHOD_LABELS = {"paper": "Paper"} | {
    f"c{cell}_l{factor}": f"C{cell} · {factor}s"
    for cell in CELL_COUNTS for factor in LENGTH_FACTORS
}
FACTOR_COLORS = {
    1: "#0F766E", 2: "#2563EB", 3: "#7C3AED", 4: "#DC2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument(
        "--budgets", type=float, nargs="+",
        default=[value / 100.0 for value in range(10, 100, 5)],
    )
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--layer-samples", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--selector-repetitions", type=int, default=2)
    parser.add_argument("--io-repetitions", type=int, default=2)
    parser.add_argument("--io-warmup", type=int, default=1)
    parser.add_argument("--gemm-repetitions", type=int, default=5)
    parser.add_argument("--gemm-warmup", type=int, default=2)
    parser.add_argument("--io-threads", type=int, default=6)
    parser.add_argument("--io-max-read-kib", type=int, default=768)
    parser.add_argument("--io-blob", type=Path)
    parser.add_argument("--profile", type=Path, default=LOCAL_PROFILE)
    parser.add_argument("--saturation-kib", type=float, default=240.0)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.analyze_only and not torch.cuda.is_available():
        raise SystemExit("Experiment 37 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if args.budgets != sorted(set(args.budgets)) or any(
        not 0.0 < value < 1.0 for value in args.budgets
    ):
        raise SystemExit("--budgets must be unique, increasing, and inside (0, 1)")
    if any(value < 1 for value in (
        args.layer_samples, args.selector_repetitions, args.io_repetitions,
        args.gemm_repetitions,
    )):
        raise SystemExit("sample and repetition counts must be positive")


def parse_method(method: str) -> tuple[int, int]:
    match = re.fullmatch(r"c(\d+)_l(\d+)", method)
    if match is None:
        raise ValueError(f"unknown super-tile method: {method}")
    return int(match.group(1)), int(match.group(2))


class SuperTileSelector(EXP36.CellSweepSelector):
    def select(self, method: str, importance: torch.Tensor, row_budget: int,
               d: int) -> tuple[torch.Tensor, dict]:
        if method == "paper":
            return EXP32.Selector.select(self, method, importance, row_budget, d)
        if method == "tile8_ceil":
            method = "c8_l1"
        cell_count, length_factor = parse_method(method)
        n = int(importance.numel())
        context = self.context(n, d)
        normalized = importance / importance.sum().clamp_min(1e-20)
        values = normalized.detach().to("cpu").numpy().astype(np.float64)
        model = context["model"]
        cell_rows = max(
            1, int(math.ceil(float(model["saturation_rows"]) / cell_count))
        )
        cells_per_tile = cell_count * length_factor
        row_kib = float(model["saturation_kib"]) / float(model["saturation_rows"])
        run_costs = EXP29.run_costs_for(self.lookup_table, n, row_kib)
        try:
            result = EXP36._select_cell_kernel(
                values, int(row_budget), int(cell_rows), int(cells_per_tile),
                run_costs,
            )
            mask_np, candidates, tile_rows, strategy, repair, chunks, ratio, valid = result
            if not valid or int(mask_np.sum()) != int(row_budget):
                raise RuntimeError("super-s tile selector failed to return exact R")
            metadata = {
                "scalarized_calls": 1,
                "outer_iterations": 1,
                "frontier_candidates": int(candidates),
                "eligible_candidates": int(chunks),
                "trim_work_deletions": int(repair if strategy == 2 else 0),
                "minimum_overfill": int(tile_rows),
                "repair_deletions": int(repair if strategy == 2 else 0),
                "repair_additions": int(repair if strategy == 1 else 0),
                "returned_top_r": False,
                "cell_count": cell_count,
                "length_factor": length_factor,
                "cell_rows": int(cell_rows),
                "tile_rows": int(tile_rows),
                "tile_strategy": (
                    "contiguous" if strategy == 0 else
                    "floor_expand" if strategy == 1 else "ceil_trim"
                ),
                "lookup_efficiency": float(ratio),
                "fallback_used": False,
                "error": "",
            }
        except Exception as exception:
            mask_np = EXP29.EXP24.top_r_mask(values, int(row_budget))
            metadata = {
                "cell_count": cell_count,
                "length_factor": length_factor,
                "fallback_used": True,
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(limit=4),
            }
        mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(
            importance.device
        )
        return mask, metadata


# Reconfigure the reusable real-model hooks and end-to-end runner.
EXP35.METHODS = METHODS
EXP35.METHOD_LABELS = METHOD_LABELS
EXP35.benchmark_selector = EXP36.benchmark_selector
EXP36.METHODS = METHODS
EXP36.METHOD_LABELS = METHOD_LABELS


def _best(curve: pd.DataFrame, error_column: str, ceiling: float,
          time_column: str):
    allowed = curve[curve[error_column] <= ceiling]
    if not len(allowed):
        return None
    return allowed.sort_values(time_column).iloc[0]


def method_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame.method.str.extract(r"c(\d+)_l(\d+)")
    frame = frame.copy()
    frame["cell_count"] = pd.to_numeric(parsed[0], errors="coerce")
    frame["length_factor"] = pd.to_numeric(parsed[1], errors="coerce")
    return frame


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    holdout = candidates[candidates.split == "holdout"]
    fixed = holdout.groupby(
        ["method", "method_label", "budget_index", "budget_fraction"],
        as_index=False,
    ).agg(
        cases=("case_key", "size"),
        actual_total_ms=("actual_total_ms", "mean"),
        selector_ms=("selector_median_ms", "mean"),
        nonselector_ms=("nonselector_total_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        importance_retention=("importance_retention", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        row_match_rate=("row_match", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    fixed = method_metadata(fixed)
    fixed.to_csv(output / "fixed_frontiers.csv", index=False)
    tiles = fixed[fixed.method.isin(TILE_METHODS)]
    base_s = tiles[tiles.length_factor == 1]
    super_s = tiles[tiles.length_factor > 1]

    ceilings = [0.50, 0.45, 0.42, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
    measured_rows = []
    for ceiling in ceilings:
        for family, curve in (
            ("paper", fixed[fixed.method == "paper"]),
            ("best_s", base_s),
            ("best_gt_s", super_s),
            ("best_all", tiles),
        ):
            best = _best(curve, "relative_l2_error", ceiling, "actual_total_ms")
            measured_rows.append({
                "error_ceiling": ceiling,
                "family": family,
                "feasible": best is not None,
                "method": best.method if best is not None else "",
                "method_label": best.method_label if best is not None else "",
                "cell_count": best.cell_count if best is not None else float("nan"),
                "length_factor": (
                    best.length_factor if best is not None else float("nan")
                ),
                "budget_fraction": (
                    best.budget_fraction if best is not None else float("nan")
                ),
                "actual_total_ms": (
                    best.actual_total_ms if best is not None else float("nan")
                ),
                "achieved_error": (
                    best.relative_l2_error if best is not None else float("nan")
                ),
            })
    measured = pd.DataFrame(measured_rows)
    measured.to_csv(output / "quality_constrained_length_comparison.csv", index=False)

    interpolation_rows = []
    for ceiling in ceilings:
        for method in TILE_METHODS:
            cell_count, length_factor = parse_method(method)
            curve = fixed[fixed.method == method]
            interpolation_rows.append({
                "error_ceiling": ceiling,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "cell_count": cell_count,
                "length_factor": length_factor,
                "interpolated_total_ms": EXP36._interpolate_time(
                    curve, "relative_l2_error", "actual_total_ms", ceiling
                ),
            })
    interpolated = pd.DataFrame(interpolation_rows)
    interpolated.to_csv(output / "interpolated_tile_curves.csv", index=False)
    family_rows = []
    for ceiling, group in interpolated.groupby("error_ceiling"):
        for family, subset in (
            ("best_s", group[group.length_factor == 1]),
            ("best_gt_s", group[group.length_factor > 1]),
            ("best_all", group),
        ):
            valid = subset.dropna(subset=["interpolated_total_ms"])
            if len(valid):
                best = valid.sort_values("interpolated_total_ms").iloc[0]
                family_rows.append({
                    "error_ceiling": ceiling,
                    "family": family,
                    "feasible": True,
                    **best.to_dict(),
                })
            else:
                family_rows.append({
                    "error_ceiling": ceiling,
                    "family": family,
                    "feasible": False,
                    "method": "",
                    "method_label": "",
                    "cell_count": float("nan"),
                    "length_factor": float("nan"),
                    "interpolated_total_ms": float("nan"),
                })
    interpolation_comparison = pd.DataFrame(family_rows)
    interpolation_comparison.to_csv(
        output / "interpolated_length_comparison.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), sharex=True, sharey=True)
    for ax, cell_count in zip(axes.flat, CELL_COUNTS):
        paper = fixed[fixed.method == "paper"].sort_values("budget_fraction")
        ax.plot(
            paper.actual_total_ms, paper.relative_l2_error,
            color="#D97706", marker="s", linestyle="--", linewidth=1.6,
            markersize=3.5, label="Paper",
        )
        for factor in LENGTH_FACTORS:
            method = f"c{cell_count}_l{factor}"
            curve = fixed[fixed.method == method].sort_values("budget_fraction")
            ax.plot(
                curve.actual_total_ms, curve.relative_l2_error,
                color=FACTOR_COLORS[factor], marker="o",
                linewidth=2.2 if factor == 1 else 1.5,
                markersize=4 if factor == 1 else 3.2,
                label=f"L={factor}s",
            )
        ax.set_title(f"Placement resolution C={cell_count}")
        ax.grid(alpha=0.24)
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0, 0].invert_yaxis()
    for ax in axes[-1]:
        ax.set_xlabel("Measured projection-path latency (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Projection relative L2 error (lower is better)")
    fig.tight_layout()
    fig.savefig(output / "super_s_frontiers.png", dpi=200)
    fig.savefig(output / "super_s_frontiers.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True)
    for column, fraction in enumerate((0.50, 0.70, 0.90)):
        for cell_count in CELL_COUNTS:
            group = tiles[
                np.isclose(tiles.budget_fraction, fraction)
                & (tiles.cell_count == cell_count)
            ].sort_values("length_factor")
            axes[0, column].plot(
                group.length_factor, group.actual_total_ms, marker="o",
                label=f"C={cell_count}",
            )
            axes[1, column].plot(
                group.length_factor, group.relative_l2_error, marker="o",
                label=f"C={cell_count}",
            )
        axes[0, column].set_title(f"R={100*fraction:.0f}%")
        axes[0, column].grid(alpha=0.24)
        axes[1, column].grid(alpha=0.24)
        axes[1, column].invert_yaxis()
        axes[1, column].set_xlabel("Tile length / saturation length (L/s)")
        axes[0, column].legend(fontsize=7)
    axes[0, 0].set_ylabel("Projection-path latency (ms)")
    axes[1, 0].set_ylabel("Projection relative L2 error")
    fig.tight_layout()
    fig.savefig(output / "super_s_sensitivity.png", dpi=200)
    fig.savefig(output / "super_s_sensitivity.pdf")
    plt.close(fig)

    components = tiles.groupby("length_factor", as_index=False).agg(
        selector_ms=("selector_ms", "mean"),
        nonselector_ms=("nonselector_ms", "mean"),
        actual_total_ms=("actual_total_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
    )
    components.to_csv(output / "length_component_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(components))
    colors = [FACTOR_COLORS[int(value)] for value in components.length_factor]
    ax.bar(
        x, components.nonselector_ms, color=colors, alpha=0.48,
        edgecolor=colors, label="Non-selector",
    )
    ax.bar(
        x, components.selector_ms, bottom=components.nonselector_ms,
        color=colors, edgecolor="white", hatch="///", label="Selector",
    )
    for index, total in enumerate(components.actual_total_ms):
        ax.text(index, total + 0.006, f"{total:.3f}", ha="center", fontsize=8)
    ax.set_xticks(
        x, [f"L={int(value)}s" for value in components.length_factor]
    )
    ax.set_ylabel("Mean measured projection-path latency (ms)")
    ax.set_title("Tile-length latency decomposition")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "length_component_breakdown.png", dpi=200)
    fig.savefig(output / "length_component_breakdown.pdf")
    plt.close(fig)

    e2e_path = output / "end_to_end.csv"
    e2e_summary = pd.DataFrame()
    e2e_comparison = pd.DataFrame()
    if e2e_path.is_file():
        e2e = pd.read_csv(e2e_path)
        e2e_summary = e2e.groupby(
            ["method", "method_label", "budget_index", "budget_fraction"],
            as_index=False,
        ).agg(
            cases=("prompt", "size"),
            dense_to_sparse_kl=("dense_to_sparse_kl", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            nll_delta=("nll_delta", "mean"),
            logit_relative_l2_error=("logit_relative_l2_error", "mean"),
            selected_fraction=("selected_fraction_mean", "mean"),
            fallback_calls=("fallback_calls", "sum"),
        )
        e2e_summary = method_metadata(e2e_summary)
        latency_lookup = {
            (row.method, int(row.budget_index)): row.actual_total_ms
            for row in fixed.itertuples(index=False)
        }
        e2e_summary["projection_path_ms"] = [
            latency_lookup[(row.method, int(row.budget_index))]
            for row in e2e_summary.itertuples(index=False)
        ]
        e2e_summary.to_csv(output / "end_to_end_summary.csv", index=False)
        rows = []
        for ceiling in (12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0):
            for family, curve in (
                ("paper", e2e_summary[e2e_summary.method == "paper"]),
                ("best_s", e2e_summary[e2e_summary.length_factor == 1]),
                ("best_gt_s", e2e_summary[e2e_summary.length_factor > 1]),
            ):
                best = _best(
                    curve, "dense_to_sparse_kl", ceiling,
                    "projection_path_ms",
                )
                rows.append({
                    "kl_ceiling": ceiling,
                    "family": family,
                    "feasible": best is not None,
                    "method": best.method if best is not None else "",
                    "method_label": best.method_label if best is not None else "",
                    "budget_fraction": (
                        best.budget_fraction if best is not None else float("nan")
                    ),
                    "projection_path_ms": (
                        best.projection_path_ms if best is not None else float("nan")
                    ),
                    "achieved_kl": (
                        best.dense_to_sparse_kl if best is not None else float("nan")
                    ),
                })
        e2e_comparison = pd.DataFrame(rows)
        e2e_comparison.to_csv(
            output / "e2e_length_comparison.csv", index=False
        )

        fig, axes = plt.subplots(
            2, 2, figsize=(12.2, 9.0), sharex=True, sharey=True
        )
        for ax, cell_count in zip(axes.flat, CELL_COUNTS):
            paper = e2e_summary[e2e_summary.method == "paper"].sort_values(
                "budget_fraction"
            )
            ax.plot(
                paper.projection_path_ms, paper.dense_to_sparse_kl,
                color="#D97706", marker="s", linestyle="--", linewidth=1.6,
                markersize=3.5, label="Paper",
            )
            for factor in LENGTH_FACTORS:
                method = f"c{cell_count}_l{factor}"
                curve = e2e_summary[e2e_summary.method == method].sort_values(
                    "budget_fraction"
                )
                ax.plot(
                    curve.projection_path_ms, curve.dense_to_sparse_kl,
                    color=FACTOR_COLORS[factor], marker="o",
                    linewidth=2.2 if factor == 1 else 1.5,
                    markersize=4 if factor == 1 else 3.2,
                    label=f"L={factor}s",
                )
            ax.set_title(f"Placement resolution C={cell_count}")
            ax.grid(alpha=0.24)
            ax.legend(fontsize=7.5, loc="lower right")
        axes[0, 0].invert_yaxis()
        for ax in axes[-1]:
            ax.set_xlabel("Measured mean projection-path latency (ms)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Dense→sparse logit KL (lower is better)")
        fig.tight_layout()
        fig.savefig(output / "end_to_end_super_s_frontiers.png", dpi=200)
        fig.savefig(output / "end_to_end_super_s_frontiers.pdf")
        plt.close(fig)

    metadata = json.loads((output / "metadata.json").read_text())
    interp_all = interpolation_comparison[
        (interpolation_comparison.family == "best_all")
        & interpolation_comparison.feasible
    ]
    factor_counts = interp_all.length_factor.astype(int).value_counts().to_dict()
    e2e_wins = 0
    e2e_ceiling_count = 0
    if len(e2e_comparison):
        for _, group in e2e_comparison.groupby("kl_ceiling"):
            indexed = group.set_index("family")
            base, longer = indexed.loc["best_s"], indexed.loc["best_gt_s"]
            if base.feasible and longer.feasible:
                e2e_ceiling_count += 1
                e2e_wins += int(
                    longer.projection_path_ms < base.projection_path_ms
                )
    summary = {
        "format": "experiment-37-super-saturation-tiles-v1",
        "candidate_cases": int(len(candidates)),
        "cell_counts": list(CELL_COUNTS),
        "length_factors": list(LENGTH_FACTORS),
        "all_direct": bool((candidates.direct_rate == 1.0).all()),
        "all_deterministic": bool(candidates.deterministic.all()),
        "fallback_cases": int(candidates.fallback_used.sum()),
        "interpolated_best_length_factor_counts": factor_counts,
        "super_s_wins": int((interp_all.length_factor > 1).sum()),
        "quality_ceilings": int(len(interp_all)),
        "e2e_super_s_wins": e2e_wins,
        "e2e_quality_ceilings": e2e_ceiling_count,
        "e2e_cases": int(pd.read_csv(e2e_path).shape[0]) if e2e_path.is_file() else 0,
        "gpu": metadata["gpu"],
        "flash": metadata["flash"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        args, measured, interpolation_comparison, e2e_comparison,
        components, summary
    )


def write_report(args, measured, interpolated, e2e, components, summary):
    path = args.report_output or HERE / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 37 보고서: saturation보다 긴 tile", "",
        "Experiment 36의 시작점 해상도 C와 tile 길이를 분리했다. "
        "`C={1,2,4,8}`에서 시작점 간격은 계속 약 `s/C`이고, tile 길이만 "
        "`L={s,2s,3s,4s}`로 늘렸다. 모든 조합은 R=10%..95%에서 실제 측정했다.", "",
        "## 동일 error 보간 비교", "",
        "보간은 5% R grid 빈틈을 제거하기 위한 진단이며 실측점 자체는 아니다.", "",
        "| error | best L=s | best L>s | L>s gain | winner |",
        "|---:|---:|---:|---:|---:|",
    ]
    for ceiling in sorted(interpolated.error_ceiling.unique(), reverse=True):
        group = interpolated[interpolated.error_ceiling == ceiling].set_index("family")
        base, longer, winner = group.loc["best_s"], group.loc["best_gt_s"], group.loc["best_all"]
        if not (base.feasible and longer.feasible and winner.feasible):
            lines.append(f"| {ceiling:.2f} | — | — | — | — |")
            continue
        gain = 100.0 * (
            base.interpolated_total_ms - longer.interpolated_total_ms
        ) / base.interpolated_total_ms
        lines.append(
            f"| {ceiling:.2f} | {base.method_label} "
            f"{base.interpolated_total_ms:.3f} ms "
            f"| {longer.method_label} {longer.interpolated_total_ms:.3f} ms "
            f"| {gain:+.1f}% | {winner.method_label} |"
        )
    lines.extend([
        "", "## 실측 grid 비교", "",
        "| error | Paper | best L=s | best L>s | R(L>s) | error(L>s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for ceiling in sorted(measured.error_ceiling.unique(), reverse=True):
        group = measured[measured.error_ceiling == ceiling].set_index("family")
        paper, base, longer = group.loc["paper"], group.loc["best_s"], group.loc["best_gt_s"]
        lines.append(
            f"| {ceiling:.2f} | {paper.actual_total_ms:.3f} ms "
            f"| {base.method_label} {base.actual_total_ms:.3f} ms "
            f"| {longer.method_label} {longer.actual_total_ms:.3f} ms "
            f"| {100*longer.budget_fraction:.0f}% | {longer.achieved_error:.4f} |"
        )
    lines.extend([
        "", "## Selector / non-selector 분해", "",
        "| tile length | selector | non-selector | total | projection error |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in components.itertuples(index=False):
        lines.append(
            f"| {int(row.length_factor)}s | {row.selector_ms:.4f} ms "
            f"| {row.nonselector_ms:.4f} ms | {row.actual_total_ms:.4f} ms "
            f"| {row.relative_l2_error:.4f} |"
        )
    if len(e2e):
        lines.extend([
            "", "## End-to-end logit KL", "",
            "| KL ceiling | Paper | best L=s | best L>s |",
            "|---:|---:|---:|---:|",
        ])
        for ceiling in sorted(e2e.kl_ceiling.unique(), reverse=True):
            group = e2e[e2e.kl_ceiling == ceiling].set_index("family")
            values = []
            for family in ("paper", "best_s", "best_gt_s"):
                row = group.loc[family]
                values.append(
                    f"{row.method_label} {row.projection_path_ms:.3f} ms"
                    if row.feasible else "—"
                )
            lines.append(
                f"| {ceiling:g} | {values[0]} | {values[1]} | {values[2]} |"
            )
    factor_text = ", ".join(
        f"L={factor}s: {count}" for factor, count in
        sorted(summary["interpolated_best_length_factor_counts"].items())
    )
    lines.extend([
        "", "## 판정", "",
        f"동일-error 보간 winner의 길이 분포는 {factor_text}이다. "
        f"L>s가 이긴 ceiling은 `{summary['super_s_wins']}`/"
        f"`{summary['quality_ceilings']}`개다. End-to-end logit KL의 discrete "
        f"value에서는 L>s가 `{summary['e2e_super_s_wins']}`/"
        f"`{summary['e2e_quality_ceilings']}`개 ceiling에서 빨랐다. 전자는 국소 "
        "projection 평균이고 후자는 비선형 누적 오차이므로 두 결론을 구분해야 한다.", "",
        "## 측정 범위", "",
        f"- projection 후보 `{summary['candidate_cases']}`, end-to-end "
        f"`{summary['e2e_cases']}`개.",
        f"- O_DIRECT fallback `{summary['fallback_cases']}`건.",
        "- latency는 selector + O_DIRECT/upload + activation gather + compact GEMM이다.",
        "- error 축은 작은 값이 위에 오도록 뒤집었다.", "",
        "![Super-s frontiers](results_laptop/super_s_frontiers.png)", "",
        "![Super-s sensitivity](results_laptop/super_s_sensitivity.png)", "",
        "![Length component breakdown](results_laptop/length_component_breakdown.png)", "",
        "![End-to-end super-s frontiers](results_laptop/end_to_end_super_s_frontiers.png)", "",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return
    prompts = list(PROMPTS[:args.prompt_limit] if args.prompt_limit else PROMPTS)
    lookup = EXP32.BASE.LatencyTable.load(args.profile)
    selector = SuperTileSelector(lookup, args.saturation_kib)
    vlmflash = EXP32.EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")
    from transformers import AutoConfig
    max_required = 0
    for key in args.models:
        config = AutoConfig.from_pretrained(
            EXP32.cached_snapshot(MODEL_SPECS[key]["repo"]), local_files_only=True
        )
        max_required = max(
            max_required,
            int(config.intermediate_size) * int(config.hidden_size) * 2,
        )
    if args.io_blob.stat().st_size < max_required:
        raise SystemExit(f"I/O blob needs at least {max_required} bytes")

    args.quality_targets = [0.5]
    all_candidates, all_selectors, all_io, all_gemm = [], [], [], []
    all_e2e, models = [], []
    for model_key in args.models:
        print(f"loading and measuring {MODEL_SPECS[model_key]['label']}", flush=True)
        result = EXP35.run_model(
            model_key, selector, native_reader, prompts, args, vlmflash
        )
        (
            model, tokenizer, candidates, selectors, io_rows, gemm_rows,
            _features, metadata,
        ) = result
        all_candidates.extend(candidates)
        all_selectors.extend(selectors)
        all_io.extend(io_rows)
        all_gemm.extend(gemm_rows)
        models.append(metadata)
        pd.DataFrame(all_candidates).to_csv(
            args.output_dir / "candidates.partial.csv", index=False
        )
        if not args.skip_end_to_end and any(
            prompt["split"] == "holdout" for prompt in prompts
        ):
            all_e2e.extend(EXP36.run_end_to_end(
                model, tokenizer, model_key, selector, prompts, args, vlmflash
            ))
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(all_candidates).to_csv(
        args.output_dir / "candidates.csv", index=False
    )
    pd.DataFrame(all_selectors).to_csv(
        args.output_dir / "selector_samples.csv", index=False
    )
    pd.DataFrame(all_io).to_csv(
        args.output_dir / "io_aggregates.csv", index=False
    )
    pd.DataFrame(all_gemm).to_csv(
        args.output_dir / "gemm_aggregates.csv", index=False
    )
    if all_e2e:
        pd.DataFrame(all_e2e).to_csv(
            args.output_dir / "end_to_end.csv", index=False
        )
    partial = args.output_dir / "candidates.partial.csv"
    if partial.exists():
        partial.unlink()

    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-37-super-saturation-tiles-v1",
        "models": models,
        "prompts": [
            {key: value for key, value in prompt.items() if key != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "cell_counts": list(CELL_COUNTS),
        "length_factors": list(LENGTH_FACTORS),
        "budgets": args.budgets,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions,
        "io_warmup": args.io_warmup,
        "gemm_repetitions": args.gemm_repetitions,
        "gemm_warmup": args.gemm_warmup,
        "io_threads": args.io_threads,
        "io_max_read_kib": args.io_max_read_kib,
        "profile": str(args.profile),
        "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "flash": profile_meta.get("flash", "unknown"),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    analyze(args)


if __name__ == "__main__":
    main()
