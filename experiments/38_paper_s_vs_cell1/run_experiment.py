#!/usr/bin/env python3
"""Experiment 38: Paper vs fixed-s Paper vs Cell-1 on the laptop."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
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
EXP37_PATH = (
    PROJECT_ROOT / "experiments" / "37_super_saturation_tiles"
    / "run_experiment.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP37 = _load_module("experiment_37_for_38", EXP37_PATH)
EXP36 = EXP37.EXP36
EXP35 = EXP37.EXP35
EXP32 = EXP37.EXP32
MODEL_SPECS = EXP37.MODEL_SPECS
PROMPTS = EXP37.PROMPTS
LOCAL_PROFILE = EXP37.LOCAL_PROFILE

METHODS = ("paper", "paper_s", "c1_l1")
METHOD_LABELS = {
    "paper": "Paper",
    "paper_s": "Paper-s",
    "c1_l1": "Cell-1",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "paper_s": "#2563EB",
    "c1_l1": "#0F766E",
}
METHOD_MARKERS = {"paper": "s", "paper_s": "^", "c1_l1": "o"}
ERROR_CEILINGS = (0.50, 0.45, 0.42, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15)
KL_CEILINGS = (12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0)
COMPONENTS = (
    "selector_median_ms",
    "read_wall_median_ms",
    "gather_median_ms",
    "gemm_median_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
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
        raise SystemExit("Experiment 38 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if args.budgets != sorted(set(args.budgets)) or any(
        not 0.0 < value < 1.0 for value in args.budgets
    ):
        raise SystemExit("--budgets must be unique, increasing, and inside (0, 1)")
    counts = (
        args.layer_samples, args.selector_repetitions, args.io_repetitions,
        args.gemm_repetitions,
    )
    if any(value < 1 for value in counts):
        raise SystemExit("sample and repetition counts must be positive")


class PaperSSelector(EXP37.SuperTileSelector):
    """Paper Algorithm 1 with exactly one candidate length: saturation s."""

    def select(self, method: str, importance: torch.Tensor, row_budget: int,
               d: int) -> tuple[torch.Tensor, dict]:
        if method != "paper_s":
            return super().select(method, importance, row_budget, d)

        n = int(importance.numel())
        context = self.context(n, d)
        model = context["model"]
        row_kib = float(model["saturation_kib"]) / float(model["saturation_rows"])
        saturation_rows = max(1, int(math.ceil(float(model["saturation_rows"]))))
        _, _, _, jump_cap_kib = context["params"].resolve(self.lookup_table)

        # The half-row offsets make the row quantization robust to floating-point
        # roundoff while leaving exactly one integer window length in the sweep.
        params = EXP32.BASE.ChunkParams(
            start_kb=(saturation_rows + 0.25) * row_kib,
            end_kb=(saturation_rows + 1.25) * row_kib,
            step_kb=row_kib,
            jump_cap_kb=float(jump_cap_kib),
        )
        normalized = importance / importance.sum().clamp_min(1e-20)
        result = EXP32.BASE.select_chunks(
            normalized, int(row_budget), row_kib, self.lookup_table,
            params=params, impl="native",
        )
        return result.mask.to(device=importance.device, dtype=torch.bool), {
            "paper_s_rows": saturation_rows,
            "paper_s_kib": saturation_rows * row_kib,
            "paper_s_jump_cap_kib": float(jump_cap_kib),
            "paper_estimated_ms": float(result.est_cost_ms),
            "paper_selected_rows": int(result.num_selected),
            "fallback_used": False,
            "error": "",
        }


# Reconfigure the reusable real-model measurement hooks and E2E runner.
EXP35.METHODS = METHODS
EXP35.METHOD_LABELS = METHOD_LABELS
EXP35.benchmark_selector = EXP36.benchmark_selector
EXP36.METHODS = METHODS
EXP36.METHOD_LABELS = METHOD_LABELS


def _interpolate(curve: pd.DataFrame, column: str, error: float) -> float:
    points = (
        curve[["relative_l2_error", column]]
        .dropna()
        .groupby("relative_l2_error", as_index=False)[column].min()
        .sort_values("relative_l2_error")
    )
    errors = points.relative_l2_error.to_numpy(dtype=float)
    values = points[column].to_numpy(dtype=float)
    if not len(errors) or error < errors[0] or error > errors[-1]:
        return float("nan")
    return float(np.interp(error, errors, values))


def _best_at_ceiling(curve: pd.DataFrame, error_column: str, ceiling: float,
                     time_column: str):
    allowed = curve[curve[error_column] <= ceiling]
    if not len(allowed):
        return None
    return allowed.sort_values(time_column).iloc[0]


def _pct_gain(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate) or baseline == 0:
        return float("nan")
    return 100.0 * (baseline - candidate) / baseline


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    holdout = candidates[candidates.split == "holdout"].copy()
    aggregate_columns = {
        "cases": ("case_key", "size"),
        "actual_total_ms": ("actual_total_ms", "mean"),
        "selector_median_ms": ("selector_median_ms", "mean"),
        "read_wall_median_ms": ("read_wall_median_ms", "mean"),
        "io_median_ms": ("io_median_ms", "mean"),
        "upload_median_ms": ("upload_median_ms", "mean"),
        "gather_median_ms": ("gather_median_ms", "mean"),
        "gemm_median_ms": ("gemm_median_ms", "mean"),
        "relative_l2_error": ("relative_l2_error", "mean"),
        "cosine_error": ("cosine_error", "mean"),
        "importance_retention": ("importance_retention", "mean"),
        "selected_fraction": ("selected_fraction", "mean"),
        "row_match_rate": ("row_match", "mean"),
        "chunks": ("chunks", "mean"),
        "fallback_rate": ("fallback_used", "mean"),
    }
    fixed = holdout.groupby(
        ["method", "method_label", "budget_index", "budget_fraction"],
        as_index=False,
    ).agg(**aggregate_columns)
    fixed.to_csv(output / "fixed_frontiers.csv", index=False)

    measured_rows = []
    for ceiling in ERROR_CEILINGS:
        for method in METHODS:
            row = _best_at_ceiling(
                fixed[fixed.method == method], "relative_l2_error", ceiling,
                "actual_total_ms",
            )
            measured_rows.append({
                "error_ceiling": ceiling,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "feasible": row is not None,
                "budget_fraction": (
                    float(row.budget_fraction) if row is not None else float("nan")
                ),
                "actual_total_ms": (
                    float(row.actual_total_ms) if row is not None else float("nan")
                ),
                "achieved_error": (
                    float(row.relative_l2_error) if row is not None else float("nan")
                ),
                "selected_fraction": (
                    float(row.selected_fraction) if row is not None else float("nan")
                ),
                "row_match_rate": (
                    float(row.row_match_rate) if row is not None else float("nan")
                ),
            })
    measured = pd.DataFrame(measured_rows)
    measured.to_csv(output / "quality_constrained_measured.csv", index=False)

    interpolation_rows = []
    columns = (
        "actual_total_ms", *COMPONENTS, "io_median_ms", "upload_median_ms",
        "selected_fraction", "row_match_rate", "chunks",
    )
    for ceiling in ERROR_CEILINGS:
        for method in METHODS:
            curve = fixed[fixed.method == method]
            row = {
                "error_ceiling": ceiling,
                "method": method,
                "method_label": METHOD_LABELS[method],
            }
            for column in columns:
                row[column] = _interpolate(curve, column, ceiling)
            interpolation_rows.append(row)
    interpolated = pd.DataFrame(interpolation_rows)
    interpolated.to_csv(output / "same_error_components.csv", index=False)

    comparison_rows = []
    for ceiling in ERROR_CEILINGS:
        group = interpolated[interpolated.error_ceiling == ceiling].set_index("method")
        paper = group.loc["paper"]
        row = {"error_ceiling": ceiling}
        for method in METHODS:
            item = group.loc[method]
            row[f"{method}_total_ms"] = item.actual_total_ms
            row[f"{method}_gain_vs_paper_pct"] = _pct_gain(
                paper.actual_total_ms, item.actual_total_ms
            )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "same_error_comparison.csv", index=False)

    e2e_summary = pd.DataFrame()
    e2e_comparison = pd.DataFrame()
    e2e_path = output / "end_to_end.csv"
    if e2e_path.is_file():
        e2e = pd.read_csv(e2e_path)
        e2e_summary = e2e.groupby(
            ["method", "method_label", "budget_index", "budget_fraction"],
            as_index=False,
        ).agg(
            dense_to_sparse_kl_mean=("dense_to_sparse_kl", "mean"),
            top1_agreement_mean=("top1_agreement", "mean"),
            nll_delta_mean=("nll_delta", "mean"),
            selected_fraction_mean=("selected_fraction_mean", "mean"),
            fallback_calls=("fallback_calls", "sum"),
        )
        latency = fixed.set_index(["method", "budget_index"]).actual_total_ms
        e2e_summary["actual_total_ms"] = [
            float(latency.loc[(row.method, int(row.budget_index))])
            for row in e2e_summary.itertuples(index=False)
        ]
        e2e_summary.to_csv(output / "end_to_end_summary.csv", index=False)

        rows = []
        for ceiling in KL_CEILINGS:
            for method in METHODS:
                best = _best_at_ceiling(
                    e2e_summary[e2e_summary.method == method],
                    "dense_to_sparse_kl_mean", ceiling, "actual_total_ms",
                )
                rows.append({
                    "kl_ceiling": ceiling,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "feasible": best is not None,
                    "actual_total_ms": (
                        float(best.actual_total_ms) if best is not None else float("nan")
                    ),
                    "achieved_kl": (
                        float(best.dense_to_sparse_kl_mean)
                        if best is not None else float("nan")
                    ),
                    "budget_fraction": (
                        float(best.budget_fraction) if best is not None else float("nan")
                    ),
                })
        e2e_comparison = pd.DataFrame(rows)
        e2e_comparison.to_csv(output / "end_to_end_comparison.csv", index=False)

    _plot_frontiers(output, fixed, interpolated, e2e_summary)
    write_report(args, candidates, fixed, interpolated, comparison, e2e_comparison)


def _plot_frontiers(output: Path, fixed: pd.DataFrame,
                    interpolated: pd.DataFrame, e2e_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    for method in METHODS:
        group = fixed[fixed.method == method].sort_values("budget_fraction")
        ax.plot(
            group.actual_total_ms, group.relative_l2_error,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linewidth=1.8, markersize=4, label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Measured projection-path latency (ms)")
    ax.set_ylabel("Projection relative L2 error")
    ax.set_title("Quality-latency frontier")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    if len(e2e_summary):
        for method in METHODS:
            group = e2e_summary[e2e_summary.method == method].sort_values(
                "budget_fraction"
            )
            ax.plot(
                group.actual_total_ms, group.dense_to_sparse_kl_mean,
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                linewidth=1.8, markersize=4, label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Measured projection-path latency (ms)")
        ax.set_ylabel("End-to-end logit KL")
        ax.set_title("End-to-end quality frontier")
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.legend()
    else:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "paper_s_frontiers.png", dpi=200)
    fig.savefig(output / "paper_s_frontiers.pdf")
    plt.close(fig)

    common = interpolated.dropna(subset=["actual_total_ms"])
    counts = common.groupby("error_ceiling").method.nunique()
    common_errors = counts[counts == len(METHODS)].index
    common = common[common.error_ceiling.isin(common_errors)]
    component_means = common.groupby("method", as_index=False)[list(COMPONENTS)].mean()
    component_means["order"] = component_means.method.map(
        {method: index for index, method in enumerate(METHODS)}
    )
    component_means = component_means.sort_values("order")
    component_means.to_csv(output / "same_error_component_means.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(component_means))
    bottoms = np.zeros(len(component_means))
    labels = {
        "selector_median_ms": "Selector",
        "read_wall_median_ms": "SSD read/upload wall",
        "gather_median_ms": "Activation gather",
        "gemm_median_ms": "Compact GEMM",
    }
    colors = {
        "selector_median_ms": "#7C3AED",
        "read_wall_median_ms": "#2563EB",
        "gather_median_ms": "#D97706",
        "gemm_median_ms": "#0F766E",
    }
    for column in COMPONENTS:
        values = component_means[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottoms, label=labels[column], color=colors[column])
        bottoms += values
    for index, total in enumerate(bottoms):
        ax.text(index, total + 0.015, f"{total:.3f} ms", ha="center", fontsize=9)
    ax.set_xticks(x, [METHOD_LABELS[item] for item in component_means.method])
    ax.set_ylabel("Mean latency at common equal-error points (ms)")
    ax.set_title("Where the time goes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "same_error_component_breakdown.png", dpi=200)
    fig.savefig(output / "same_error_component_breakdown.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for method in METHODS:
        group = fixed[fixed.method == method].sort_values("budget_fraction")
        axes[0].plot(
            100 * group.budget_fraction, 100 * group.selected_fraction,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            100 * group.budget_fraction, group.chunks,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].plot([10, 95], [10, 95], color="#6B7280", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Requested R (%)")
    axes[0].set_ylabel("Actually selected rows (%)")
    axes[0].set_title("Whole-window underfill")
    axes[1].set_xlabel("Requested R (%)")
    axes[1].set_ylabel("Mean contiguous chunks")
    axes[1].set_title("Read fragmentation")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "paper_s_structure.png", dpi=200)
    fig.savefig(output / "paper_s_structure.pdf")
    plt.close(fig)


def _fmt(value: float, suffix: str = "") -> str:
    return "—" if not np.isfinite(value) else f"{value:.3f}{suffix}"


def write_report(args: argparse.Namespace, candidates: pd.DataFrame,
                 fixed: pd.DataFrame, interpolated: pd.DataFrame,
                 comparison: pd.DataFrame, e2e: pd.DataFrame) -> None:
    lines = [
        "# Experiment 38 보고서: Paper vs Paper-s vs Cell-1", "",
        "`Paper-s`는 논문 Algorithm 1의 utility score, greedy non-overlap 선택, "
        "projection별 jump cap, whole-window budget 규칙을 유지하고 후보 window "
        "길이만 노트북 saturation 길이 `s` 하나로 고정한다. `Cell-1`도 기본 "
        "tile 길이는 `s`지만 시작점 격자가 `s`이고 floor-expand/ceil-trim으로 "
        "exact-R을 반환한다.", "",
        "## 동일 projection error 보간", "",
        "5% R grid 사이를 선형 보간한 진단값이다. 실측점 자체는 아니다.", "",
        "| error | Paper | Paper-s | vs Paper | Cell-1 | vs Paper |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.error_ceiling:.2f} | {_fmt(row.paper_total_ms, ' ms')} "
            f"| {_fmt(row.paper_s_total_ms, ' ms')} "
            f"| {_fmt(row.paper_s_gain_vs_paper_pct, '%')} "
            f"| {_fmt(row.c1_l1_total_ms, ' ms')} "
            f"| {_fmt(row.c1_l1_gain_vs_paper_pct, '%')} |"
        )

    common = interpolated.dropna(subset=["actual_total_ms"])
    counts = common.groupby("error_ceiling").method.nunique()
    common_errors = counts[counts == len(METHODS)].index
    common = common[common.error_ceiling.isin(common_errors)]
    means = common.groupby("method")[
        ["actual_total_ms", *COMPONENTS, "io_median_ms", "upload_median_ms",
         "selected_fraction", "row_match_rate", "chunks"]
    ].mean()
    lines.extend([
        "", "## 동일-error 구성요소 평균", "",
        "모든 세 방법이 보간 가능한 error ceiling만 평균했다.", "",
        "| method | selector | SSD/upload wall | gather | GEMM | total | chunks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        if method not in means.index:
            continue
        row = means.loc[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row.selector_median_ms:.4f} ms "
            f"| {row.read_wall_median_ms:.4f} ms "
            f"| {row.gather_median_ms:.4f} ms "
            f"| {row.gemm_median_ms:.4f} ms "
            f"| {row.actual_total_ms:.4f} ms | {row.chunks:.2f} |"
        )

    lines.extend([
        "", "## 요청 R 충족", "",
        "| method | mean selected/requested | exact-R case rate |",
        "|---|---:|---:|",
    ])
    requested = fixed.budget_fraction.clip(lower=1e-12)
    ratios = fixed.assign(fill_ratio=fixed.selected_fraction / requested).groupby("method").agg(
        fill_ratio=("fill_ratio", "mean"), row_match_rate=("row_match_rate", "mean")
    )
    for method in METHODS:
        row = ratios.loc[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {100*row.fill_ratio:.2f}% "
            f"| {100*row.row_match_rate:.2f}% |"
        )

    if len(e2e):
        lines.extend([
            "", "## End-to-end logit KL", "",
            "여기서 end-to-end는 모든 decoder projection을 동시에 sparsify한 "
            "forward의 logit 품질이고, 표의 시간은 projection-path 평균이다.", "",
            "| KL ceiling | Paper | Paper-s | Cell-1 |",
            "|---:|---:|---:|---:|",
        ])
        for ceiling in KL_CEILINGS:
            group = e2e[e2e.kl_ceiling == ceiling].set_index("method")
            cells = []
            for method in METHODS:
                row = group.loc[method]
                cells.append(
                    "—" if not row.feasible else
                    f"{row.actual_total_ms:.3f} ms (KL {row.achieved_kl:.3f})"
                )
            lines.append(f"| {ceiling:g} | " + " | ".join(cells) + " |")

    paper_mean = means.loc["paper", "actual_total_ms"] if "paper" in means.index else np.nan
    paper_s_mean = means.loc["paper_s", "actual_total_ms"] if "paper_s" in means.index else np.nan
    cell_mean = means.loc["c1_l1", "actual_total_ms"] if "c1_l1" in means.index else np.nan
    paper_selector = means.loc["paper", "selector_median_ms"] if "paper" in means.index else np.nan
    paper_s_selector = means.loc["paper_s", "selector_median_ms"] if "paper_s" in means.index else np.nan
    cell_selector = means.loc["c1_l1", "selector_median_ms"] if "c1_l1" in means.index else np.nan
    paper_s_error_floor = float(
        fixed[fixed.method == "paper_s"].relative_l2_error.min()
    )
    paper_s_empty = int(
        ((candidates.method == "paper_s") & (candidates.selected_rows == 0)).sum()
    )
    nonempty_direct_failures = int(
        ((candidates.selected_rows > 0) & (candidates.direct_rate < 1.0)).sum()
    )
    lines.extend([
        "", "## 판정", "",
        f"공통 동일-error 구간 평균에서 Paper-s의 Paper 대비 변화는 "
        f"`{_fmt(_pct_gain(paper_mean, paper_s_mean), '%')}`, Cell-1의 Paper 대비 "
        f"변화는 `{_fmt(_pct_gain(paper_mean, cell_mean), '%')}`다. selector만 보면 "
        f"Paper-s는 `{_fmt(_pct_gain(paper_selector, paper_s_selector), '%')}`, "
        f"Cell-1은 `{_fmt(_pct_gain(paper_selector, cell_selector), '%')}` 빠르다.", "",
        f"그러나 strict Paper-s는 평균 projection error를 `{paper_s_error_floor:.4f}` "
        f"아래로 내리지 못했다. 후보 길이 `s`가 R 또는 projection 전체 길이보다 "
        f"크면 원 Paper의 whole-window 규칙상 아무것도 선택하지 못하기 때문이다. "
        f"전체 Paper-s case 중 `{paper_s_empty}`개가 empty selection이다. 따라서 "
        "`s` 고정만으로 얻는 이득은 공통 저·중품질 구간의 일부 selector 개선이며, "
        "Cell-1의 나머지 이득은 coarse start grid와 exact-R repair에서 나온다.",
        "", "## 측정 범위", "",
        f"- projection 후보 `{len(candidates)}`개.",
        f"- non-empty read의 O_DIRECT fallback `{nonempty_direct_failures}`건.",
        f"- Paper-s empty selection `{paper_s_empty}`건(빈 read의 direct flag는 "
        "O_DIRECT 실패로 세지 않았다).",
        "- 모델 3개, prompt 6개, 모델당 표본 layer 3개, decoder projection 7종.",
        "- R=10%..95%를 5% 간격으로 측정했다.",
        "- latency는 selector + native O_DIRECT/GPU-upload wall + activation gather "
        "+ compact GEMM이며 전체 LLM wall-clock은 아니다.",
        "", "![Frontiers](results_laptop/paper_s_frontiers.png)", "",
        "![Component breakdown](results_laptop/same_error_component_breakdown.png)", "",
        "![Structure](results_laptop/paper_s_structure.png)", "",
    ])
    report = args.report_output or HERE / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return

    prompts = list(PROMPTS[:args.prompt_limit] if args.prompt_limit else PROMPTS)
    lookup = EXP32.BASE.LatencyTable.load(args.profile)
    selector = PaperSSelector(lookup, args.saturation_kib)
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

    pd.DataFrame(all_candidates).to_csv(args.output_dir / "candidates.csv", index=False)
    pd.DataFrame(all_selectors).to_csv(
        args.output_dir / "selector_samples.csv", index=False
    )
    pd.DataFrame(all_io).to_csv(args.output_dir / "io_aggregates.csv", index=False)
    pd.DataFrame(all_gemm).to_csv(args.output_dir / "gemm_aggregates.csv", index=False)
    if all_e2e:
        pd.DataFrame(all_e2e).to_csv(args.output_dir / "end_to_end.csv", index=False)
    partial = args.output_dir / "candidates.partial.csv"
    if partial.exists():
        partial.unlink()

    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-38-paper-s-vs-cell1-v1",
        "definition": {
            "paper": "released multi-length Algorithm 1",
            "paper_s": (
                "Algorithm 1 with one candidate length ceil(s/row_size), original "
                "per-shape jump cap, greedy overlap rule, and whole-window budget"
            ),
            "cell1": (
                "one saturation-sized cell/tile with exact-R floor-expand or ceil-trim"
            ),
        },
        "models": models,
        "prompts": [
            {key: value for key, value in prompt.items() if key != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "methods": list(METHODS),
        "budgets": args.budgets,
        "saturation_kib": args.saturation_kib,
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
