#!/usr/bin/env python3
"""Experiment 40: objective choices on one theoretical I-L plane.

This is a CPU-only replay.  It excludes selector runtime, GPU upload, gather,
and GEMM, and evaluates masks only with the Experiment 16 latency models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXP16_PATH = (
    PROJECT_ROOT / "experiments" / "16_saturation_global_chain"
    / "run_experiment.py"
)
EXP36_PATH = (
    PROJECT_ROOT / "experiments" / "36_cell_parameter_sweep"
    / "run_experiment.py"
)
DEFAULT_SOURCE = (
    PROJECT_ROOT / "experiments" / "16_saturation_global_chain" / "results"
)
DEFAULT_INPUT_SUMMARY = (
    PROJECT_ROOT / "experiments" / "10_exact_lookup_recheck" / "results"
    / "summary.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP16 = _load_module("experiment_16_for_40", EXP16_PATH)
EXP36 = _load_module("experiment_36_for_40", EXP36_PATH)

METHODS = ("paper", "cell1_fixed", "cell1_bound", "exact_coverage")
METHOD_LABELS = {
    "paper": "Paper fixed-R",
    "cell1_fixed": "Cell-1 fixed-R",
    "cell1_bound": "Cell-1 I-bound",
    "exact_coverage": "Exact global I-bound",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "cell1_fixed": "#0F766E",
    "cell1_bound": "#7C3AED",
    "exact_coverage": "#2563EB",
}
METHOD_MARKERS = {
    "paper": "s", "cell1_fixed": "o", "cell1_bound": "D",
    "exact_coverage": "*",
}
METHOD_STYLES = {
    "paper": "--", "cell1_fixed": ":", "cell1_bound": "-.",
    "exact_coverage": "-",
}
SOLVERS = (
    "paper", "quant", "global_supported", "lookup_supported",
    "exact_coverage",
)
SOLVER_LABELS = {
    "paper": "Paper greedy",
    "quant": "Quant q=131,072",
    "global_supported": "Two-line supported",
    "lookup_supported": "Lookup supported",
    "exact_coverage": "Exact global coverage",
}
SOLVER_COLORS = {
    "paper": "#D97706", "quant": "#E6A700",
    "global_supported": "#2A9D8F", "lookup_supported": "#6D28D9",
    "exact_coverage": "#2563EB",
}
SOLVER_MARKERS = {
    "paper": "s", "quant": "P", "global_supported": "X",
    "lookup_supported": "D", "exact_coverage": "*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--r-step-pct", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--report-output", type=Path, default=HERE / "report.md")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def configure_plot() -> None:
    EXP16.BASE.configure_plot_style(plt)


def two_line_run_costs(n: int, model: dict) -> np.ndarray:
    lengths = np.arange(n + 1, dtype=np.float64)
    c2 = float(model["c2_ms_per_row"])
    delta = float(model["d_ms_per_excess_row"])
    saturation = float(model["saturation_rows"])
    costs = c2 * lengths + delta * np.maximum(saturation - lengths, 0.0)
    costs[0] = 0.0
    return costs


def cell1_metrics(values: np.ndarray, rows: int, cell_rows: int,
                  run_costs: np.ndarray, model: dict, affine, row_size_kib,
                  policies) -> dict:
    result = EXP36._select_cell_kernel(
        np.asarray(values, dtype=np.float64), int(rows), int(cell_rows), 1,
        run_costs,
    )
    mask, candidates, tile_rows, strategy, repair, chunks, ratio, valid = result
    if not valid or int(mask.sum()) != int(rows):
        raise RuntimeError(f"Cell-1 failed at R={rows}")
    metrics = EXP16.EXP13.mask_metrics(
        np.asarray(mask, dtype=bool), values, model, affine, row_size_kib,
        policies,
    )
    return {
        **metrics,
        "candidate_count": int(candidates),
        "tile_rows": int(tile_rows),
        "repair_rows": int(repair),
        "strategy": int(strategy),
        "internal_ratio": float(ratio),
        "kernel_chunks": int(chunks),
    }


def build_cell1_dense(source: pd.DataFrame, values_by_input: dict,
                      input_summary: dict, exp16_summary: dict,
                      r_step_pct: float) -> pd.DataFrame:
    n = int(exp16_summary["n"])
    model = exp16_summary["optimization_surrogate"]
    table = EXP16.BASE.LatencyTable.load(input_summary["profile"])
    row_size_kib = float(input_summary["row_size_kib"])
    affine_a, affine_c, _ = EXP16.BASE.affine_fit(table, row_size_kib)
    affine = (affine_a, affine_c)
    policies = EXP16.EXP13.EXP11.LatencyPolicies(table, row_size_kib, 128)
    run_costs = two_line_run_costs(n, model)
    cell_rows = max(1, int(math.ceil(float(model["saturation_rows"]))))

    fractions = np.arange(r_step_pct, 100.0 + 0.5 * r_step_pct, r_step_pct)
    grid_rows = {
        max(1, min(n, int(round(n * fraction / 100.0))))
        for fraction in fractions
    }
    grid_rows.add(n)
    rows_out: list[dict] = []
    groups = source.groupby(["trial", "target_cv", "spatial_mode"], sort=True)
    for index, ((trial, target_cv, mode), group) in enumerate(groups, 1):
        key = (int(trial), round(float(target_cv), 10), mode)
        values = values_by_input[key]
        requested_rows = grid_rows | set(group.budget_rows.astype(int))
        for rows in sorted(requested_rows):
            metrics = cell1_metrics(
                values, rows, cell_rows, run_costs, model, affine,
                row_size_kib, policies,
            )
            rows_out.append({
                "trial": int(trial), "target_cv": float(target_cv),
                "spatial_mode": mode, "selected_rows": int(metrics["rows"]),
                "selected_fraction": float(metrics["rows"]) / n,
                "importance": float(metrics["importance"]),
                "two_line_ms": float(metrics["two_line_ms"]),
                "released_ms": float(metrics["released_ms"]),
                "chunks": int(metrics["chunks"]),
                "cell_rows": cell_rows,
                "tile_rows": int(metrics["tile_rows"]),
                "repair_rows": int(metrics["repair_rows"]),
                "strategy": int(metrics["strategy"]),
            })
        print(f"Cell-1 theoretical curve {index}/{len(groups)}", flush=True)
    return pd.DataFrame(rows_out)


def _source_point(row, method: str, prefix: str, n: int) -> dict:
    selected_rows = int(round(float(getattr(row, f"{prefix}_rows"))))
    return {
        "trial": int(row.trial), "target_cv": float(row.target_cv),
        "spatial_mode": row.spatial_mode,
        "budget_fraction": float(row.budget_fraction),
        "budget_rows": int(row.budget_rows),
        "target_importance": float(row.target_importance),
        "method": method, "method_label": METHOD_LABELS[method],
        "selected_rows": selected_rows,
        "selected_fraction": selected_rows / n,
        "importance": float(getattr(row, f"{prefix}_importance")),
        "two_line_ms": float(getattr(row, f"{prefix}_two_line_ms")),
        "released_ms": float(getattr(row, f"{prefix}_released_ms")),
        "chunks": int(round(float(getattr(row, f"{prefix}_chunks")))),
    }


def build_points(source: pd.DataFrame, dense: pd.DataFrame, n: int) -> pd.DataFrame:
    dense_groups = {
        key: group.sort_values("selected_rows")
        for key, group in dense.groupby(
            ["trial", "target_cv", "spatial_mode"], sort=False
        )
    }
    output: list[dict] = []
    for row in source.itertuples(index=False):
        key = (int(row.trial), float(row.target_cv), row.spatial_mode)
        curve = dense_groups[key]
        fixed = curve[curve.selected_rows == int(row.budget_rows)]
        if len(fixed) != 1:
            raise RuntimeError(f"missing fixed-R Cell-1 point for {key}")
        feasible = curve[curve.importance >= float(row.target_importance) - 1e-12]
        if not len(feasible):
            raise RuntimeError(f"Cell-1 curve cannot meet target for {key}")
        bound = feasible.sort_values(
            ["two_line_ms", "importance", "selected_rows"],
            ascending=[True, True, True],
        ).iloc[0]
        output.append(_source_point(row, "paper", "paper", n))
        output.append(_source_point(
            row, "exact_coverage", "exact_coverage", n
        ))
        for method, selected in (("cell1_fixed", fixed.iloc[0]),
                                 ("cell1_bound", bound)):
            output.append({
                "trial": int(row.trial), "target_cv": float(row.target_cv),
                "spatial_mode": row.spatial_mode,
                "budget_fraction": float(row.budget_fraction),
                "budget_rows": int(row.budget_rows),
                "target_importance": float(row.target_importance),
                "method": method, "method_label": METHOD_LABELS[method],
                "selected_rows": int(selected.selected_rows),
                "selected_fraction": float(selected.selected_fraction),
                "importance": float(selected.importance),
                "two_line_ms": float(selected.two_line_ms),
                "released_ms": float(selected.released_ms),
                "chunks": int(selected.chunks),
            })
    points = pd.DataFrame(output)
    points["meets_target"] = points.importance >= points.target_importance - 1e-12
    return points


def build_solver_points(source: pd.DataFrame, n: int) -> pd.DataFrame:
    output = []
    for row in source.itertuples(index=False):
        for method in SOLVERS:
            selected_rows = int(round(float(getattr(row, f"{method}_rows"))))
            output.append({
                "trial": int(row.trial), "target_cv": float(row.target_cv),
                "spatial_mode": row.spatial_mode,
                "budget_fraction": float(row.budget_fraction),
                "target_importance": float(row.target_importance),
                "method": method, "method_label": SOLVER_LABELS[method],
                "selected_rows": selected_rows,
                "selected_fraction": selected_rows / n,
                "importance": float(getattr(row, f"{method}_importance")),
                "two_line_ms": float(getattr(row, f"{method}_two_line_ms")),
                "released_ms": float(getattr(row, f"{method}_released_ms")),
                "chunks": int(round(float(getattr(row, f"{method}_chunks")))),
                "meets_target": True,
            })
    return pd.DataFrame(output)


def aggregate_points(points: pd.DataFrame) -> pd.DataFrame:
    return points.groupby(
        ["method", "method_label", "budget_fraction"], as_index=False,
        sort=False,
    ).agg(
        cases=("importance", "size"),
        target_importance=("target_importance", "mean"),
        importance=("importance", "mean"),
        two_line_ms=("two_line_ms", "mean"),
        released_ms=("released_ms", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        chunks=("chunks", "mean"),
        feasible_rate=("meets_target", "mean"),
    )


def comparison_table(points: pd.DataFrame) -> pd.DataFrame:
    keyed = points.set_index([
        "trial", "target_cv", "spatial_mode", "budget_fraction", "method"
    ])
    rows = []
    keys = ["trial", "target_cv", "spatial_mode", "budget_fraction"]
    for key, _group in points.groupby(keys, sort=False):
        method_rows = keyed.loc[key]
        paper = method_rows.loc["paper"]
        fixed = method_rows.loc["cell1_fixed"]
        bound = method_rows.loc["cell1_bound"]
        exact = method_rows.loc["exact_coverage"]
        rows.append({
            **dict(zip(keys, key)),
            "target_importance": float(paper.target_importance),
            "paper_r": float(paper.selected_fraction),
            "cell1_fixed_r": float(fixed.selected_fraction),
            "cell1_bound_r": float(bound.selected_fraction),
            "exact_r": float(exact.selected_fraction),
            "cell1_fixed_importance_change_pct": 100.0 * (
                float(fixed.importance) / float(paper.importance) - 1.0
            ),
            "cell1_fixed_two_line_saving_pct": 100.0 * (
                1.0 - float(fixed.two_line_ms) / float(paper.two_line_ms)
            ),
            "cell1_bound_two_line_saving_pct": 100.0 * (
                1.0 - float(bound.two_line_ms) / float(paper.two_line_ms)
            ),
            "exact_two_line_saving_pct": 100.0 * (
                1.0 - float(exact.two_line_ms) / float(paper.two_line_ms)
            ),
            "exact_headroom_vs_cell1_bound_pct": 100.0 * (
                1.0 - float(exact.two_line_ms) / float(bound.two_line_ms)
            ),
            "cell1_bound_released_saving_pct": 100.0 * (
                1.0 - float(bound.released_ms) / float(paper.released_ms)
            ),
            "exact_released_saving_pct": 100.0 * (
                1.0 - float(exact.released_ms) / float(paper.released_ms)
            ),
        })
    return pd.DataFrame(rows)


def plot_shared(frame: pd.DataFrame, latency: str, path: Path,
                title: str, annotate: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    offsets = {
        "paper": (0, 8), "cell1_fixed": (0, -14),
        "cell1_bound": (0, 8), "exact_coverage": (0, -14),
    }
    for method in METHODS:
        group = frame[frame.method == method].sort_values("budget_fraction")
        ax.plot(
            group[latency], group.importance,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linestyle=METHOD_STYLES[method], linewidth=2.0,
            markersize=7 if method == "exact_coverage" else 5,
            label=METHOD_LABELS[method],
        )
        if annotate:
            for point in group.itertuples(index=False):
                ax.annotate(
                    f"{100 * point.selected_fraction:.1f}%",
                    (getattr(point, latency), point.importance),
                    textcoords="offset points", xytext=offsets[method],
                    ha="center", fontsize=7, color=METHOD_COLORS[method],
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                          "ec": "none", "alpha": 0.76},
                )
    ax.set_xlabel(
        "Two-line latency L(M) (ms)" if latency == "two_line_ms"
        else "Released lookup latency L(M) (ms)"
    )
    ax.set_ylabel("Retained importance I(M)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_annotated_small_multiples(frame: pd.DataFrame, latency: str,
                                   path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), sharex=True, sharey=True)
    for ax, active in zip(axes.flat, METHODS):
        for method in METHODS:
            group = frame[frame.method == method].sort_values("budget_fraction")
            active_line = method == active
            ax.plot(
                group[latency], group.importance,
                color=METHOD_COLORS[method] if active_line else "#CBD5E1",
                marker=METHOD_MARKERS[method] if active_line else None,
                linestyle=METHOD_STYLES[method] if active_line else "-",
                linewidth=2.4 if active_line else 1.0,
                markersize=7 if active_line else 0,
                alpha=1.0 if active_line else 0.7,
                zorder=3 if active_line else 1,
            )
        group = frame[frame.method == active].sort_values("budget_fraction")
        for point in group.itertuples(index=False):
            ax.annotate(
                f"{100 * point.selected_fraction:.1f}%",
                (getattr(point, latency), point.importance),
                textcoords="offset points", xytext=(0, 7), ha="center",
                fontsize=8, color=METHOD_COLORS[active],
                bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                      "ec": "none", "alpha": 0.82},
            )
        ax.set_title(METHOD_LABELS[active], color=METHOD_COLORS[active])
        ax.grid(alpha=0.22)
    for ax in axes[-1, :]:
        ax.set_xlabel(
            "Two-line latency L(M) (ms)" if latency == "two_line_ms"
            else "Released lookup latency L(M) (ms)"
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Retained importance I(M)")
    fig.suptitle(title + "\nlabels = actual selected rows R/N", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_solver_maps(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    for method in SOLVERS:
        group = frame[frame.method == method].sort_values("budget_fraction")
        ax.plot(
            group.two_line_ms, group.importance,
            color=SOLVER_COLORS[method], marker=SOLVER_MARKERS[method],
            linewidth=2.0, markersize=7 if method == "exact_coverage" else 5,
            label=SOLVER_LABELS[method],
        )
    ax.set_xlabel("Two-line latency L(M) (ms)")
    ax.set_ylabel("Retained importance I(M)")
    ax.set_title("Experiment 16 solver family on the shared I–L plane")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "all_solver_two_line_map.png", dpi=210)
    fig.savefig(output / "all_solver_two_line_map.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 14.3), sharex=True, sharey=True)
    for ax, active in zip(axes.flat, SOLVERS):
        for method in SOLVERS:
            group = frame[frame.method == method].sort_values("budget_fraction")
            highlighted = method == active
            ax.plot(
                group.two_line_ms, group.importance,
                color=SOLVER_COLORS[method] if highlighted else "#CBD5E1",
                marker=SOLVER_MARKERS[method] if highlighted else None,
                linewidth=2.4 if highlighted else 1.0,
                markersize=7 if highlighted else 0,
                alpha=1.0 if highlighted else 0.7,
                zorder=3 if highlighted else 1,
            )
        group = frame[frame.method == active].sort_values("budget_fraction")
        for point in group.itertuples(index=False):
            ax.annotate(
                f"{100 * point.selected_fraction:.1f}%",
                (point.two_line_ms, point.importance),
                textcoords="offset points", xytext=(0, 7), ha="center",
                fontsize=8, color=SOLVER_COLORS[active],
                bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                      "ec": "none", "alpha": 0.82},
            )
        ax.set_title(SOLVER_LABELS[active], color=SOLVER_COLORS[active])
        ax.grid(alpha=0.22)
    axes.flat[-1].axis("off")
    for ax in axes[-1, :1]:
        ax.set_xlabel("Two-line latency L(M) (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Retained importance I(M)")
    fig.suptitle(
        "Earlier theoretical solvers\nlabels = actual selected rows R/N",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output / "annotated_solver_map.png", dpi=210)
    fig.savefig(output / "annotated_solver_map.pdf")
    plt.close(fig)


def select_representative(points: pd.DataFrame, comparison: pd.DataFrame) -> tuple:
    middle = comparison[np.isclose(comparison.budget_fraction, 0.5)].copy()
    median = float(middle.exact_headroom_vs_cell1_bound_pct.median())
    row = middle.iloc[(middle.exact_headroom_vs_cell1_bound_pct - median).abs().argmin()]
    return int(row.trial), float(row.target_cv), row.spatial_mode


def write_report(args, points: pd.DataFrame, aggregate: pd.DataFrame,
                 comparison: pd.DataFrame, solver_points: pd.DataFrame,
                 metadata: dict) -> None:
    by_budget = comparison.groupby("budget_fraction", as_index=False).agg(
        paper_r=("paper_r", "mean"),
        fixed_r=("cell1_fixed_r", "mean"),
        fixed_importance_change=("cell1_fixed_importance_change_pct", "mean"),
        fixed_saving=("cell1_fixed_two_line_saving_pct", "mean"),
        bound_r=("cell1_bound_r", "mean"),
        bound_saving=("cell1_bound_two_line_saving_pct", "mean"),
        exact_r=("exact_r", "mean"),
        exact_saving=("exact_two_line_saving_pct", "mean"),
        headroom=("exact_headroom_vs_cell1_bound_pct", "mean"),
    )
    fixed_quality = float(comparison.cell1_fixed_importance_change_pct.mean())
    fixed_saving = float(comparison.cell1_fixed_two_line_saving_pct.mean())
    bound_saving = float(comparison.cell1_bound_two_line_saving_pct.mean())
    exact_saving = float(comparison.exact_two_line_saving_pct.mean())
    headroom = float(comparison.exact_headroom_vs_cell1_bound_pct.mean())
    bound_r = float(100 * comparison.cell1_bound_r.mean())
    exact_r = float(100 * comparison.exact_r.mean())
    fixed_feasible = float(
        points[points.method == "cell1_fixed"].meets_target.mean()
    )
    solver_index = solver_points.set_index([
        "trial", "target_cv", "spatial_mode", "budget_fraction", "method"
    ])
    solver_headroom = []
    quant_same = []
    for key, _group in solver_points.groupby(
        ["trial", "target_cv", "spatial_mode", "budget_fraction"], sort=False
    ):
        rows = solver_index.loc[key]
        exact = rows.loc["exact_coverage"]
        supported = rows.loc["global_supported"]
        quant = rows.loc["quant"]
        solver_headroom.append(100.0 * (
            1.0 - float(exact.two_line_ms) / float(supported.two_line_ms)
        ))
        quant_same.append(
            math.isclose(float(quant.two_line_ms), float(supported.two_line_ms),
                         rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(float(quant.importance), float(supported.importance),
                             rel_tol=1e-12, abs_tol=1e-12)
            and int(quant.selected_rows) == int(supported.selected_rows)
        )

    lines = [
        "# Experiment 40 보고서: 이론 I–L objective map", "",
        "GPU, selector 실행시간, upload, gather, GEMM을 모두 제거하고 동일한 "
        "importance vector와 latency model 위에서 mask만 비교했다. 주 목적은 "
        "`I(M) >= Q`에서 `L(M)`을 최소화하는 문제다.", "",
        "## 점과 % 라벨의 의미", "",
        "그래프의 x는 mask의 예측 I/O latency, y는 실제 retained importance다. "
        "**각 점 위의 `%`는 speedup이나 importance target이 아니라 그 점이 실제로 "
        "선택한 행 비율 `R(M)/N`이다.** 모든 방법은 같은 축과 evaluator를 사용한다.", "",
        "- Paper fixed-R: 논문 greedy에 R을 입력한다.",
        "- Cell-1 fixed-R: saturation 길이 한 칸 격자에서 같은 R을 채운다.",
        f"- Cell-1 I-bound: Cell-1의 `{metadata['r_step_pct']:g}%` R sweep 중 "
        "`I>=Q`인 최소 two-line latency 점을 고른다.",
        "- Exact global I-bound: 모든 binary mask를 대상으로 two-line "
        "`min L subject to I>=Q`를 푼 Experiment 16의 전역해다.", "",
        "## 전체 378개 target 결과", "",
        f"같은 nominal R에서 Cell-1 fixed-R은 Paper보다 importance가 평균 "
        f"`{fixed_quality:+.2f}%`, two-line latency가 `{fixed_saving:+.2f}%` "
        f"절감됐다. 다만 Paper importance target을 직접 만족한 비율은 "
        f"`{100 * fixed_feasible:.1f}%`뿐이다.", "",
        f"목적을 I 하한으로 바꾸면 Cell-1은 평균 R `{bound_r:.1f}%`를 선택했고 "
        f"Paper 대비 two-line latency를 `{bound_saving:.2f}%` 줄였다. 전역해는 "
        f"평균 R `{exact_r:.1f}%`, Paper 대비 `{exact_saving:.2f}%` 절감이다. "
        f"Cell-1 I-bound에서 전역해까지 남은 latency headroom은 평균 "
        f"`{headroom:.2f}%`다.", "",
        "## 앞선 전역 solver들과의 연결", "",
        f"Experiment 16의 dense Quant와 complete two-line supported 점은 이 "
        f"저장 지표 기준 `{100*np.mean(quant_same):.1f}%`가 일치한다. 그러나 "
        f"supported 점에서 unsupported point까지 포함하는 Exact coverage로 가면 "
        f"two-line latency가 추가로 평균 `{np.mean(solver_headroom):.2f}%` 줄어든다. "
        "따라서 이론 곡선에서 `supported frontier = 전역 coverage frontier`는 아니다.", "",
        "## target별 평균", "",
        "| Paper nominal R | Paper actual R | Cell-1 fixed: I 변화 | fixed 절감 | "
        "Cell-1 I-bound R | bound 절감 | Exact R | Exact 절감 | 남은 headroom |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_budget.itertuples(index=False):
        lines.append(
            f"| {100*row.budget_fraction:.1f}% | {100*row.paper_r:.1f}% "
            f"| {row.fixed_importance_change:+.2f}% | {row.fixed_saving:+.2f}% "
            f"| {100*row.bound_r:.1f}% | {row.bound_saving:+.2f}% "
            f"| {100*row.exact_r:.1f}% | {row.exact_saving:+.2f}% "
            f"| {row.headroom:.2f}% |"
        )
    lines.extend([
        "", "## 해석", "",
        "이 그림은 목적함수가 달라도 결과 mask는 같은 I–L 공간의 점이라는 "
        "관점을 그대로 보여준다. R 고정은 곡선의 x축 위치를 미리 정하는 "
        "전략이고, I 하한은 필요한 품질에 도달하는 가장 왼쪽 점을 고르는 "
        "전략이다. Exact와 Cell-1 I-bound의 차이는 그 다음 단계인 **허용 mask "
        "공간의 차이**다.", "",
        "Exact global은 two-line 모델에서만 전역 최적이다. released lookup "
        "그래프는 같은 mask를 다른 evaluator로 재평가한 sensitivity 결과이며 "
        f"그 축에서는 최적성 인증이 없다. Cell-1 I-bound도 "
        f"{metadata['r_step_pct']:g}% R grid 내부의 "
        "최적점이지 모든 R 또는 모든 mask의 전역해가 아니다.", "",
        "## 범위", "",
        f"- 입력 54개 × target 7개 = `{len(comparison)}`개 paired target.",
        f"- N=`{metadata['n']}`, CV 6개, ordering 3개, trial/CV 3개.",
        "- selector 계산시간은 0으로 둔다. 이 실험은 online algorithm "
        "runtime 비교가 아니라 mask geometry와 objective 비교다.",
        "- importance는 synthetic activation surrogate이며 downstream 품질이 아니다.",
        "", "![Shared two-line map](results/shared_two_line_map.png)", "",
        "![Annotated two-line map](results/annotated_two_line_map.png)", "",
        "![Representative map](results/representative_two_line_map.png)", "",
        "![Earlier solver map](results/annotated_solver_map.png)", "",
        "![Earlier solver overlay](results/all_solver_two_line_map.png)", "",
        "![Released sensitivity](results/shared_released_map.png)", "",
    ])
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines))


def analyze(args: argparse.Namespace, metadata: dict) -> None:
    output = args.output_dir
    points = pd.read_csv(output / "points.csv")
    solver_points = pd.read_csv(output / "solver_points.csv")
    dense = pd.read_csv(output / "cell1_dense.csv")
    aggregate = aggregate_points(points)
    solver_aggregate = aggregate_points(solver_points)
    comparison = comparison_table(points)
    aggregate.to_csv(output / "aggregate_points.csv", index=False)
    solver_aggregate.to_csv(output / "solver_aggregate_points.csv", index=False)
    comparison.to_csv(output / "paired_comparison.csv", index=False)

    configure_plot()
    plot_shared(
        aggregate, "two_line_ms", output / "shared_two_line_map.png",
        "One theoretical importance–latency plane (macro mean)",
    )
    plot_annotated_small_multiples(
        aggregate, "two_line_ms", output / "annotated_two_line_map.png",
        "Same points, different objectives (macro mean)",
    )
    plot_shared(
        aggregate, "released_ms", output / "shared_released_map.png",
        "Same masks under the released lookup evaluator", annotate=False,
    )
    plot_solver_maps(solver_aggregate, output)
    representative = select_representative(points, comparison)
    selected = points[
        (points.trial == representative[0])
        & np.isclose(points.target_cv, representative[1])
        & (points.spatial_mode == representative[2])
    ]
    plot_shared(
        selected, "two_line_ms", output / "representative_two_line_map.png",
        f"Single shared curve: trial {representative[0]}, "
        f"CV={representative[1]:g}, {representative[2]}",
    )
    metadata["representative"] = {
        "trial": representative[0], "target_cv": representative[1],
        "spatial_mode": representative[2],
    }
    metadata["outputs"] = {
        "dense_cell1_points": int(len(dense)),
        "objective_points": int(len(points)),
        "earlier_solver_points": int(len(solver_points)),
        "paired_targets": int(len(comparison)),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(
        args, points, aggregate, comparison, solver_points, metadata
    )


def main() -> None:
    args = parse_args()
    if not 0.1 <= args.r_step_pct <= 10.0:
        raise SystemExit("--r-step-pct must be in [0.1, 10]")
    source_path = args.source_results / "paired_trials.csv"
    summary_path = args.source_results / "summary.json"
    for path in (source_path, summary_path, args.input_summary):
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")
    exp16_summary = json.loads(summary_path.read_text())
    input_summary = json.loads(args.input_summary.read_text())
    if exp16_summary.get("format") != "experiment-16-exact-coverage-v4":
        raise SystemExit("source is not Experiment 16 v4")
    if input_summary.get("format") != "experiment-10-exact-lookup-recheck-v1":
        raise SystemExit("input summary is not Experiment 10 v1")
    metadata = {
        "format": "experiment-40-theoretical-objective-map-v1",
        "n": int(exp16_summary["n"]),
        "r_step_pct": float(args.r_step_pct),
        "gpu_used": False,
        "selector_runtime_included": False,
        "primary_objective": "min two-line L(M) subject to I(M) >= Q",
        "cell1_geometry": (
            "one ceil(s)-row placement cell/tile with exact-R endpoint repair; "
            "repair utility uses the two-line run-cost model"
        ),
        "source": {
            "paired_trials": str(source_path.resolve()),
            "paired_trials_sha256": file_sha256(source_path),
            "experiment_16_summary": str(summary_path.resolve()),
            "experiment_16_summary_sha256": file_sha256(summary_path),
            "input_summary": str(args.input_summary.resolve()),
            "input_summary_sha256": file_sha256(args.input_summary),
        },
        "optimization_surrogate": exp16_summary["optimization_surrogate"],
        "released_evaluator": exp16_summary["primary_latency_evaluator"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        source = pd.read_csv(source_path)
        values = EXP16.EXP13.generated_inputs(input_summary)
        dense = build_cell1_dense(
            source, values, input_summary, exp16_summary, args.r_step_pct
        )
        points = build_points(source, dense, int(exp16_summary["n"]))
        solver_points = build_solver_points(source, int(exp16_summary["n"]))
        dense.to_csv(args.output_dir / "cell1_dense.csv", index=False)
        points.to_csv(args.output_dir / "points.csv", index=False)
        solver_points.to_csv(args.output_dir / "solver_points.csv", index=False)
    analyze(args, metadata)
    print(f"wrote Experiment 40 to {args.output_dir}")


if __name__ == "__main__":
    main()
