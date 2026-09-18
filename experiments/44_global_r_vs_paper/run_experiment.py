#!/usr/bin/env python3
"""Experiment 44: overlay global-R Cell-1 strategies with Paper."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_INPUT = (
    PROJECT_ROOT / "experiments" / "43_adaptive_shifted_cell1"
    / "results_laptop_safe"
)

COLORS = {
    "paper": "#D97706",
    "cell1": "#0F766E",
    "importance_bound": "#2563EB",
    "error_oracle": "#DC2626",
}
OBJECTIVE_LABELS = {
    "importance_bound": "Cell-1 global importance-bound",
    "error_oracle": "Cell-1 global error-oracle",
}
COMPONENTS = (
    "score_median_ms", "selector_median_ms", "read_wall_median_ms",
    "gather_median_ms", "gemm_median_ms",
)
COMMON_ERRORS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_laptop")
    parser.add_argument("--report-output", type=Path,
                        default=HERE / "report_laptop.md")
    return parser.parse_args()


def _frontier(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    return (
        frame[frame.method == method]
        .sort_values("relative_l2_error")
        .drop_duplicates("relative_l2_error")
    )


def _interpolate(curve: pd.DataFrame, error: float) -> float:
    errors = curve.relative_l2_error.to_numpy(dtype=float)
    latency = curve.actual_total_ms.to_numpy(dtype=float)
    if not len(errors) or error < errors[0] or error > errors[-1]:
        return float("nan")
    return float(np.interp(error, errors, latency))


def _gain(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate) or baseline <= 0:
        return float("nan")
    return 100.0 * (baseline - candidate) / baseline


def build_comparison(fixed: pd.DataFrame,
                     allocations: pd.DataFrame) -> pd.DataFrame:
    paper = _frontier(fixed, "paper")
    cell1 = _frontier(fixed, "cell1_x2")
    rows = []
    selected = allocations[
        (allocations.method == "cell1_x2")
        & allocations.objective.isin(OBJECTIVE_LABELS)
    ]
    if "coverage_rate" in selected:
        selected = selected[np.isclose(selected.coverage_rate, 1.0)]
    for row in selected.itertuples(index=False):
        paper_latency = _interpolate(paper, row.mean_error)
        fixed_cell1_latency = _interpolate(cell1, row.mean_error)
        rows.append({
            "objective": row.objective,
            "objective_label": OBJECTIVE_LABELS[row.objective],
            "paper_target_r": row.paper_budget_fraction,
            "resulting_r": row.mean_selected_fraction,
            "actual_error": row.mean_error,
            "global_cell1_ms": row.mean_time_ms,
            "calibration_predicted_ms": row.mean_decision_time_ms,
            "paper_same_error_ms": paper_latency,
            "fixed_cell1_same_error_ms": fixed_cell1_latency,
            "gain_vs_paper_pct": _gain(paper_latency, row.mean_time_ms),
            "gain_vs_fixed_cell1_pct": _gain(
                fixed_cell1_latency, row.mean_time_ms
            ),
            "paper_interpolation_feasible": np.isfinite(paper_latency),
            "cell1_interpolation_feasible": np.isfinite(fixed_cell1_latency),
            "workloads": int(row.workloads),
            "total_workloads": int(row.total_workloads),
            **{component: getattr(row, component) for component in COMPONENTS},
        })
    return pd.DataFrame(rows).sort_values(["objective", "actual_error"])


def plot_overlay(output: Path, fixed: pd.DataFrame,
                 comparison: pd.DataFrame) -> None:
    paper = fixed[fixed.method == "paper"].sort_values("budget_fraction")
    cell1 = fixed[fixed.method == "cell1_x2"].sort_values("budget_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.9), sharex=True, sharey=True)
    panels = (
        ("importance_bound", "Measurable importance lower bound"),
        ("error_oracle", "Measured projection-error oracle"),
    )
    for ax, (objective, title) in zip(axes, panels):
        global_curve = comparison[
            comparison.objective == objective
        ].sort_values("resulting_r")
        ax.plot(
            paper.actual_total_ms, paper.relative_l2_error,
            color=COLORS["paper"], marker="s", linewidth=2.0,
            markersize=4, label="Paper fixed R",
        )
        ax.plot(
            cell1.actual_total_ms, cell1.relative_l2_error,
            color=COLORS["cell1"], marker="o", linewidth=2.0,
            markersize=4, label="Cell-1 X² fixed R",
        )
        ax.plot(
            global_curve.global_cell1_ms, global_curve.actual_error,
            color=COLORS[objective], marker="P" if objective == "importance_bound" else "*",
            linewidth=2.4, markersize=6 if objective == "importance_bound" else 8,
            label=OBJECTIVE_LABELS[objective],
        )
        for point_index, row in enumerate(global_curve.itertuples(index=False)):
            ax.annotate(
                f"{100 * row.resulting_r:.0f}%",
                (row.global_cell1_ms, row.actual_error),
                xytext=(4, 4 if point_index % 2 == 0 else -10),
                textcoords="offset points", fontsize=6.2,
                color=COLORS[objective],
            )
        # Label a sparse subset of Paper points so nominal R is explicit but
        # the global strategy's resulting-R labels remain readable.
        for row in paper.itertuples(index=False):
            percent = int(round(100 * row.budget_fraction))
            if percent % 20 == 10:
                ax.annotate(
                    f"P {percent}%", (row.actual_total_ms, row.relative_l2_error),
                    xytext=(4, -10), textcoords="offset points", fontsize=6,
                    color=COLORS["paper"],
                )
        ax.set_title(title)
        ax.set_xlabel("Measured actual total per projection (ms)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Projection relative L2 error (lower is better)")
    axes[0].invert_yaxis()
    fig.suptitle("Paper vs Cell-1 with globally allocated R", fontsize=16)
    fig.tight_layout()
    fig.savefig(output / "paper_vs_global_r_frontiers.png", dpi=220)
    fig.savefig(output / "paper_vs_global_r_frontiers.pdf")
    plt.close(fig)


def plot_gains(output: Path, comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, objective in zip(axes, ("importance_bound", "error_oracle")):
        group = comparison[comparison.objective == objective].sort_values(
            "actual_error"
        )
        ax.plot(
            group.actual_error, group.gain_vs_paper_pct,
            color=COLORS[objective], marker="o", linewidth=2.2,
            label="Gain vs Paper",
        )
        ax.plot(
            group.actual_error, group.gain_vs_fixed_cell1_pct,
            color=COLORS["cell1"], marker="s", linestyle="--",
            linewidth=1.8, label="Gain vs fixed-R Cell-1",
        )
        ax.axhline(0.0, color="#6B7280", linewidth=1)
        ax.set_xlabel("Achieved projection relative L2 error")
        ax.set_title(OBJECTIVE_LABELS[objective])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Same-error measured latency gain (%)")
    fig.tight_layout()
    fig.savefig(output / "paper_same_error_gain.png", dpi=220)
    fig.savefig(output / "paper_same_error_gain.pdf")
    plt.close(fig)


def _interpolate_column(curve: pd.DataFrame, x_column: str, y_column: str,
                        x_value: float) -> float:
    points = (
        curve[[x_column, y_column]].dropna()
        .groupby(x_column, as_index=False)[y_column].mean()
        .sort_values(x_column)
    )
    x = points[x_column].to_numpy(dtype=float)
    y = points[y_column].to_numpy(dtype=float)
    if not len(x) or x_value < x[0] or x_value > x[-1]:
        return float("nan")
    return float(np.interp(x_value, x, y))


def component_breakdown(fixed: pd.DataFrame,
                        comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = {
        "paper": (fixed[fixed.method == "paper"], "relative_l2_error"),
        "cell1_x2": (fixed[fixed.method == "cell1_x2"], "relative_l2_error"),
        "importance_bound": (
            comparison[comparison.objective == "importance_bound"],
            "actual_error",
        ),
        "error_oracle": (
            comparison[comparison.objective == "error_oracle"],
            "actual_error",
        ),
    }
    labels = {
        "paper": "Paper fixed R",
        "cell1_x2": "Cell-1 X² fixed R",
        "importance_bound": "Cell-1 global I-bound",
        "error_oracle": "Cell-1 global error-oracle",
    }
    point_rows = []
    for error in COMMON_ERRORS:
        for strategy, (curve, x_column) in strategies.items():
            row = {
                "error_ceiling": error, "strategy": strategy,
                "strategy_label": labels[strategy],
            }
            for component in COMPONENTS:
                row[component] = _interpolate_column(
                    curve, x_column, component, error
                )
            row["component_total_ms"] = sum(row[c] for c in COMPONENTS)
            point_rows.append(row)
    points = pd.DataFrame(point_rows)
    complete_errors = points.dropna(subset=list(COMPONENTS)).groupby(
        "error_ceiling"
    ).strategy.nunique()
    complete_errors = complete_errors[complete_errors == len(strategies)].index
    common = points[points.error_ceiling.isin(complete_errors)].copy()
    means = common.groupby(
        ["strategy", "strategy_label"], as_index=False
    )[list(COMPONENTS) + ["component_total_ms"]].mean()
    order = {strategy: index for index, strategy in enumerate(strategies)}
    means["order"] = means.strategy.map(order)
    means = means.sort_values("order")
    return common, means


def plot_components(output: Path, means: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    x = np.arange(len(means))
    bottoms = np.zeros(len(means), dtype=float)
    labels = {
        "score_median_ms": "Score construction",
        "selector_median_ms": "Structured selector",
        "read_wall_median_ms": "SSD read/upload wall",
        "gather_median_ms": "Activation gather",
        "gemm_median_ms": "Compact GEMM",
    }
    colors = {
        "score_median_ms": "#DB2777",
        "selector_median_ms": "#7C3AED",
        "read_wall_median_ms": "#2563EB",
        "gather_median_ms": "#D97706",
        "gemm_median_ms": "#0F766E",
    }
    for component in COMPONENTS:
        values = means[component].to_numpy(dtype=float)
        ax.bar(
            x, values, bottom=bottoms, color=colors[component],
            label=labels[component],
        )
        bottoms += values
    for index, total in enumerate(bottoms):
        ax.text(index, total + 0.012, f"{total:.3f} ms", ha="center", fontsize=9)
    ax.set_xticks(x, means.strategy_label, rotation=7)
    ax.set_ylabel("Mean measured latency at common equal-error points (ms)")
    ax.set_title("Paper vs global-R Cell-1: latency decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "paper_global_component_breakdown.png", dpi=220)
    fig.savefig(output / "paper_global_component_breakdown.pdf")
    plt.close(fig)


def _fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}{suffix}"


def write_report(args, comparison: pd.DataFrame, component_means: pd.DataFrame,
                 metadata: dict) -> None:
    lines = [
        "# Experiment 44: Paper vs globally allocated Cell-1 R", "",
        "Experiment 43에서 실측한 동일한 holdout mask와 actual-total latency를 ",
        "사용해 Paper, fixed-R Cell-1, global-R Cell-1을 같은 projection-error ",
        "축에 겹쳤다. 따라서 global point의 error에서 Paper와 fixed Cell-1 ",
        "latency를 보간해 동일-error gain을 계산할 수 있다.", "",
    ]
    for objective in ("importance_bound", "error_oracle"):
        group = comparison[comparison.objective == objective]
        valid = group.dropna(subset=[
            "gain_vs_paper_pct", "gain_vs_fixed_cell1_pct"
        ])
        lines.extend([
            f"## {OBJECTIVE_LABELS[objective]}", "",
            f"공통 error 구간 평균 gain은 Paper 대비 "
            f"**{valid.gain_vs_paper_pct.mean():.2f}%**, fixed-R Cell-1 대비 "
            f"**{valid.gain_vs_fixed_cell1_pct.mean():.2f}%**다.", "",
            "| Paper target R | resulting R | actual error | global Cell-1 | Paper at same error | gain vs Paper | gain vs fixed Cell-1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        shown = group[group.paper_target_r.isin((0.30, 0.50, 0.70, 0.90))]
        for row in shown.itertuples(index=False):
            lines.append(
                f"| {100*row.paper_target_r:.0f}% "
                f"| {100*row.resulting_r:.1f}% | {row.actual_error:.4f} "
                f"| {row.global_cell1_ms:.4f} ms "
                f"| {_fmt(row.paper_same_error_ms, 4, ' ms')} "
                f"| {_fmt(row.gain_vs_paper_pct, 2, '%')} "
                f"| {_fmt(row.gain_vs_fixed_cell1_pct, 2, '%')} |"
            )
        lines.append("")

    importance = comparison[
        comparison.objective == "importance_bound"
    ].dropna(subset=["gain_vs_paper_pct", "gain_vs_fixed_cell1_pct"])
    oracle = comparison[
        comparison.objective == "error_oracle"
    ].dropna(subset=["gain_vs_paper_pct", "gain_vs_fixed_cell1_pct"])
    lines.extend([
        "## 동일-error 구성요소 평균", "",
        f"모든 네 전략이 겹치는 `{len(COMMON_ERRORS)}`개 error 지점"
        "(0.15..0.50)에서 각 구성요소를 보간한 뒤 평균했다.", "",
        "| strategy | score | selector | SSD/upload | gather | GEMM | total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in component_means.itertuples(index=False):
        lines.append(
            f"| {row.strategy_label} | {row.score_median_ms:.4f} "
            f"| {row.selector_median_ms:.4f} | {row.read_wall_median_ms:.4f} "
            f"| {row.gather_median_ms:.4f} | {row.gemm_median_ms:.4f} "
            f"| {row.component_total_ms:.4f} |"
        )
    indexed_components = component_means.set_index("strategy")
    fixed_components = indexed_components.loc["cell1_x2"]
    importance_components = indexed_components.loc["importance_bound"]
    oracle_components = indexed_components.loc["error_oracle"]
    importance_delta = (
        fixed_components.component_total_ms
        - importance_components.component_total_ms
    )
    oracle_delta = (
        fixed_components.component_total_ms
        - oracle_components.component_total_ms
    )
    importance_ssd_delta = (
        fixed_components.read_wall_median_ms
        - importance_components.read_wall_median_ms
    )
    oracle_ssd_delta = (
        fixed_components.read_wall_median_ms
        - oracle_components.read_wall_median_ms
    )
    lines.extend(["",
        f"fixed-R Cell-1에서 importance-bound로 줄어든 `{importance_delta:.4f} ms` "
        f"중 SSD/read-upload 감소가 `{importance_ssd_delta:.4f} ms`"
        f"(`{100*importance_ssd_delta/importance_delta:.1f}%`)다. error-oracle의 "
        f"총 감소 `{oracle_delta:.4f} ms` 중 SSD 감소는 "
        f"`{oracle_ssd_delta:.4f} ms`"
        f"(`{100*oracle_ssd_delta/oracle_delta:.1f}%`)다. 따라서 큰 꺾임은 "
        "selector timing이나 그래프 보간이 아니라, 전역 R 배분이 비싼 "
        "projection의 실제 읽기량을 줄인 데서 나온다.", "",
        "## 판정", "",
        "Paper와 겹쳐도 결론은 유지된다. **현재 노트북에서 가장 큰 여지는 ",
        "mask geometry가 아니라 projection별 R 배분이다.** importance-bound ",
        f"전략의 Paper 대비 평균 gain은 `{importance.gain_vs_paper_pct.mean():.2f}%`, "
        f"error-oracle 상한은 `{oracle.gain_vs_paper_pct.mean():.2f}%`다.", "",
        "다만 importance-bound도 한 workload의 sampled projection importance를 ",
        "모두 본 뒤 배분한 post-hoc 상한이다. decision cost에는 holdout timing을 ",
        "쓰지 않고 calibration module-median latency만 사용했지만, 실제 배포에는 ",
        "순차 layer에서 남은 quality budget을 관리하는 causal controller가 필요하다.",
        "error-oracle은 dense reference error를 사용하므로 배포할 수 없다.", "",
        "평균과 그래프에는 전체 9개 workload에서 feasible한 target만 포함했다. ",
        "importance-bound의 Paper 95% target은 1/9 workload에서만 feasible하여 ",
        "제외했다.", "",
        "## 데이터 범위", "",
        f"- source format: `{metadata.get('format', 'unknown')}`.",
        f"- 모델 `{len(metadata.get('models', []))}`개; R=10%..95%, 5% 간격.",
        "- latency는 score + selector + O_DIRECT/upload + activation gather + compact GEMM.",
        "- 그래프의 global curve 위 %는 고정 입력 R이 아니라 allocation 결과의 평균 R.",
        "", f"![Overlay]({args.output_dir.name}/paper_vs_global_r_frontiers.png)",
        "", f"![Components]({args.output_dir.name}/paper_global_component_breakdown.png)",
        "", f"![Gain]({args.output_dir.name}/paper_same_error_gain.png)", "",
    ])
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    fixed_path = args.input_dir / "fixed_frontiers.csv"
    allocation_path = args.input_dir / "global_quality_summary.csv"
    metadata_path = args.input_dir / "metadata.json"
    for path in (fixed_path, allocation_path, metadata_path):
        if not path.is_file():
            raise SystemExit(f"missing Experiment 43 input: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(fixed_path)
    allocations = pd.read_csv(allocation_path)
    metadata = json.loads(metadata_path.read_text())
    comparison = build_comparison(fixed, allocations)
    comparison.to_csv(args.output_dir / "same_error_comparison.csv", index=False)
    component_points, component_means = component_breakdown(fixed, comparison)
    component_points.to_csv(
        args.output_dir / "same_error_component_points.csv", index=False
    )
    component_means.to_csv(
        args.output_dir / "same_error_component_means.csv", index=False
    )
    jointly_comparable = comparison.dropna(subset=[
        "gain_vs_paper_pct", "gain_vs_fixed_cell1_pct"
    ])
    summary = jointly_comparable.groupby(
        ["objective", "objective_label"], as_index=False
    ).agg(
        comparable_points=("gain_vs_paper_pct", "count"),
        mean_gain_vs_paper_pct=("gain_vs_paper_pct", "mean"),
        min_gain_vs_paper_pct=("gain_vs_paper_pct", "min"),
        max_gain_vs_paper_pct=("gain_vs_paper_pct", "max"),
        mean_gain_vs_fixed_cell1_pct=("gain_vs_fixed_cell1_pct", "mean"),
        mean_resulting_r=("resulting_r", "mean"),
    )
    summary.to_csv(args.output_dir / "aggregate_summary.csv", index=False)
    output_metadata = {
        "format": "experiment-44-global-r-vs-paper-v1",
        "source": str(args.input_dir),
        "source_format": metadata.get("format"),
        "models": metadata.get("models", []),
        "comparison": (
            "Paper and fixed Cell-1 latency are linearly interpolated at each "
            "global-allocation point's measured mean projection error"
        ),
        "objectives": list(OBJECTIVE_LABELS),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n"
    )
    plot_overlay(args.output_dir, fixed, comparison)
    plot_gains(args.output_dir, comparison)
    plot_components(args.output_dir, component_means)
    write_report(args, comparison, component_means, metadata)


if __name__ == "__main__":
    main()
