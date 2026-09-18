#!/usr/bin/env python3
"""Experiment 45: causal local Cell-1 importance-bound selector."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP39 = _load_module(
    "experiment_39_for_45",
    PROJECT_ROOT / "experiments" / "39_weight_aware_cell1" / "run_experiment.py",
)
EXP37 = EXP39.EXP37
EXP36 = EXP39.EXP36
EXP32 = EXP39.EXP32
EXP29 = EXP36.EXP29
MODEL_SPECS = EXP39.MODEL_SPECS
PROMPTS = EXP39.PROMPTS
LOCAL_PROFILE = EXP39.LOCAL_PROFILE
njit = EXP29.njit

METHODS = ("paper", "cell1_fixed", "cell1_tau")
METHOD_LABELS = {
    "paper": "Paper fixed R |x|",
    "cell1_fixed": "Cell-1 X² fixed R",
    "cell1_tau": "Cell-1-τ X² local I-bound",
}
METHOD_COLORS = {
    "paper": "#D97706", "cell1_fixed": "#0F766E", "cell1_tau": "#2563EB",
}
METHOD_MARKERS = {"paper": "s", "cell1_fixed": "o", "cell1_tau": "P"}
METHOD_SCORE = {"paper": "abs", "cell1_fixed": "x2", "cell1_tau": "x2"}
METHOD_INTERNAL = {
    "paper": "paper", "cell1_fixed": "c1_l1", "cell1_tau": "cell1_tau",
}
COMPONENTS = EXP39.COMPONENTS
ERROR_CEILINGS = EXP39.ERROR_CEILINGS
DEFAULT_CONTROLS = tuple(
    [value / 100.0 for value in range(10, 100, 5)] + [0.97, 0.98, 0.99]
)

EXP39.HERE = HERE
EXP39.METHODS = METHODS
EXP39.METHOD_LABELS = METHOD_LABELS
EXP39.METHOD_COLORS = METHOD_COLORS
EXP39.METHOD_MARKERS = METHOD_MARKERS
EXP39.METHOD_SCORE = METHOD_SCORE
EXP39.METHOD_INTERNAL = METHOD_INTERNAL


@njit(cache=False)
def _trim_to_importance(mask, values, threshold, run_costs):
    """Remove feasible run endpoints until no one-row removal preserves I."""
    output = mask.copy()
    n = len(output)
    left = np.empty(n, dtype=np.int64)
    right = np.empty(n, dtype=np.int64)
    run_count = EXP29._extract_runs(output, left, right)
    importance, cost, _ = EXP29._mask_metrics_lookup(output, values, run_costs)
    removals = 0
    while run_count > 0:
        best_run = -1
        best_index = -1
        best_cost = cost
        best_loss = np.inf
        for run in range(run_count):
            length = right[run] - left[run] + 1
            candidate_cost = cost - run_costs[length] + run_costs[length - 1]
            for side in range(2):
                index = left[run] if side == 0 else right[run]
                if side == 1 and index == left[run]:
                    continue
                loss = values[index]
                if importance - loss + 1e-15 < threshold:
                    continue
                # Minimize predicted read cost first. If the lookup plateau is
                # tied, remove the least importance so later removals remain feasible.
                if candidate_cost < best_cost - 1e-15 or (
                    abs(candidate_cost - best_cost) <= 1e-15
                    and loss < best_loss - 1e-15
                ):
                    best_run = run
                    best_index = index
                    best_cost = candidate_cost
                    best_loss = loss
        if best_index < 0:
            break
        output[best_index] = False
        importance -= best_loss
        cost = best_cost
        if left[best_run] == right[best_run]:
            for other in range(best_run, run_count - 1):
                left[other] = left[other + 1]
                right[other] = right[other + 1]
            run_count -= 1
        elif best_index == left[best_run]:
            left[best_run] += 1
        else:
            right[best_run] -= 1
        removals += 1
    _, final_cost, chunks = EXP29._mask_metrics_lookup(output, values, run_costs)
    selected = 0
    for value in output:
        selected += int(value)
    return output, importance, final_cost, chunks, selected, removals


@njit(cache=False)
def _select_local_importance_bound(values, threshold, cell_rows, run_costs):
    """Fixed-origin s-cell cover followed by importance-preserving trim."""
    n = len(values)
    cell_count = (n + cell_rows - 1) // cell_rows
    weights = np.empty(cell_count, dtype=np.float64)
    starts = np.empty(cell_count, dtype=np.int64)
    ends = np.empty(cell_count, dtype=np.int64)
    for cell in range(cell_count):
        start = cell * cell_rows
        end = min(n, start + cell_rows)
        starts[cell] = start
        ends[cell] = end
        weight = 0.0
        for index in range(start, end):
            weight += values[index]
        weights[cell] = weight
    order = np.argsort(weights)
    mask = np.zeros(n, dtype=np.bool_)
    retained = 0.0
    selected_cells = 0
    for rank in range(cell_count - 1, -1, -1):
        cell = order[rank]
        for index in range(starts[cell], ends[cell]):
            mask[index] = True
        retained += weights[cell]
        selected_cells += 1
        if retained + 1e-15 >= threshold:
            break
    if retained + 1e-12 < threshold:
        return mask, retained, 0.0, 0, int(mask.sum()), selected_cells, 0, False
    result = _trim_to_importance(mask, values, threshold, run_costs)
    trimmed, retained, cost, chunks, selected, removals = result
    valid = retained + 1e-12 >= threshold and selected > 0
    return (
        trimmed, retained, cost, chunks, selected, selected_cells, removals, valid,
    )


class LocalImportanceSelector(EXP37.SuperTileSelector):
    def select(self, method: str, importance: torch.Tensor, row_budget: int,
               d: int) -> tuple[torch.Tensor, dict]:
        if method != "cell1_tau":
            return super().select(method, importance, row_budget, d)
        n = int(importance.numel())
        threshold = min(max(float(row_budget) / n, 1e-8), 1.0)
        context = self.context(n, d)
        normalized = importance / importance.sum().clamp_min(1e-20)
        values = normalized.detach().to("cpu").numpy().astype(np.float64)
        model = context["model"]
        cell_rows = max(1, int(math.ceil(float(model["saturation_rows"]))))
        row_kib = float(model["saturation_kib"]) / float(model["saturation_rows"])
        run_costs = EXP29.run_costs_for(self.lookup_table, n, row_kib).copy()
        # Physical read latency should be monotone. Removing lookup noise keeps
        # the local trim objective from preferring a longer read by accident.
        run_costs = np.maximum.accumulate(run_costs)
        try:
            result = _select_local_importance_bound(
                values, threshold, cell_rows, run_costs
            )
            (
                mask_np, retained, predicted_read, chunks, selected,
                selected_cells, removals, valid,
            ) = result
            if not valid or retained + 1e-9 < threshold:
                raise RuntimeError("local importance lower bound was not met")
            metadata = {
                "quality_control": "x2_importance_retention",
                "importance_threshold": threshold,
                "achieved_importance": float(retained),
                "cell_rows": int(cell_rows),
                "selected_cells_before_trim": int(selected_cells),
                "trimmed_endpoint_rows": int(removals),
                "selected_rows": int(selected),
                "eligible_candidates": int(chunks),
                "predicted_read_ms": float(predicted_read),
                "paper_parameter_source": context["paper_parameter_source"],
                "fallback_used": False,
                "error": "",
            }
        except Exception as exception:
            mask_np = EXP29.EXP24.top_r_mask(values, max(1, int(row_budget)))
            metadata = {
                "importance_threshold": threshold,
                "fallback_used": True,
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(limit=4),
            }
        mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(
            importance.device
        )
        return mask, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=["qwen05"])
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=list(DEFAULT_CONTROLS),
                        help="fixed R for baselines; local importance tau for Cell-1-tau")
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--layer-samples", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--score-repetitions", type=int, default=3)
    parser.add_argument("--score-warmup", type=int, default=1)
    parser.add_argument("--selector-repetitions", type=int, default=3)
    parser.add_argument("--io-repetitions", type=int, default=2)
    parser.add_argument("--io-warmup", type=int, default=1)
    parser.add_argument("--gemm-repetitions", type=int, default=3)
    parser.add_argument("--gemm-warmup", type=int, default=1)
    parser.add_argument("--io-threads", type=int, default=2)
    parser.add_argument("--io-max-read-kib", type=int, default=768)
    parser.add_argument("--io-blob", type=Path)
    parser.add_argument("--profile", type=Path, default=LOCAL_PROFILE)
    parser.add_argument("--saturation-kib", type=float, default=240.0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.55)
    parser.add_argument("--norm-chunk-rows", type=int, default=128)
    parser.add_argument("--projection-throttle-ms", type=float, default=25.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--process-nice", type=int, default=10)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_laptop")
    parser.add_argument("--report-output", type=Path,
                        default=HERE / "report_laptop.md")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.analyze_only and not torch.cuda.is_available():
        raise SystemExit("Experiment 45 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if not args.analyze_only and len(args.models) != 1:
        raise SystemExit("run one model per process and merge afterwards")
    if args.budgets != sorted(set(args.budgets)) or any(
        not 0.0 < value < 1.0 for value in args.budgets
    ):
        raise SystemExit("controls must be unique, increasing, and inside (0,1)")
    if any(value < 1 for value in (
        args.layer_samples, args.score_repetitions, args.selector_repetitions,
        args.io_repetitions, args.gemm_repetitions, args.cpu_threads,
    )):
        raise SystemExit("sample and thread counts must be positive")


def _interpolate(curve: pd.DataFrame, column: str, error: float) -> float:
    return EXP39._interpolate(curve, column, error)


def _gain(baseline: float, candidate: float) -> float:
    return EXP39._pct_gain(baseline, candidate)


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    holdout = candidates[candidates.split == "holdout"].copy()
    holdout["control_kind"] = np.where(
        holdout.method == "cell1_tau", "importance_tau", "fixed_r"
    )
    fixed = holdout.groupby(
        ["method", "method_label", "score_kind", "control_kind",
         "budget_index", "budget_fraction"], as_index=False,
    ).agg(
        cases=("case_key", "size"), actual_total_ms=("actual_total_ms", "mean"),
        score_median_ms=("score_median_ms", "mean"),
        selector_median_ms=("selector_median_ms", "mean"),
        read_wall_median_ms=("read_wall_median_ms", "mean"),
        gather_median_ms=("gather_median_ms", "mean"),
        gemm_median_ms=("gemm_median_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        importance_retention=("importance_retention", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        selected_fraction_std=("selected_fraction", "std"),
        chunks=("chunks", "mean"), fallback_rate=("fallback_used", "mean"),
    )
    fixed.rename(columns={"budget_fraction": "control_value"}, inplace=True)
    fixed.to_csv(output / "frontiers.csv", index=False)

    rows = []
    columns = ("actual_total_ms", *COMPONENTS, "selected_fraction", "chunks")
    for ceiling in ERROR_CEILINGS:
        for method in METHODS:
            curve = fixed[fixed.method == method]
            row = {
                "error_ceiling": ceiling, "method": method,
                "method_label": METHOD_LABELS[method],
            }
            for column in columns:
                row[column] = _interpolate(curve, column, ceiling)
            rows.append(row)
    interpolated = pd.DataFrame(rows)
    interpolated.to_csv(output / "same_error_components.csv", index=False)

    comparison_rows = []
    for ceiling, group in interpolated.groupby("error_ceiling"):
        indexed = group.set_index("method")
        row = {"error_ceiling": ceiling}
        for method in METHODS:
            value = indexed.loc[method, "actual_total_ms"]
            row[f"{method}_total_ms"] = value
            row[f"{method}_gain_vs_paper_pct"] = _gain(
                indexed.loc["paper", "actual_total_ms"], value
            )
            row[f"{method}_gain_vs_fixed_cell1_pct"] = _gain(
                indexed.loc["cell1_fixed", "actual_total_ms"], value
            )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "same_error_comparison.csv", index=False)

    per_model_rows = []
    for model, frame in holdout.groupby("model", sort=True):
        model_fixed = frame.groupby(
            ["method", "budget_index", "budget_fraction"], as_index=False
        ).agg(
            actual_total_ms=("actual_total_ms", "mean"),
            relative_l2_error=("relative_l2_error", "mean"),
        )
        for ceiling in ERROR_CEILINGS:
            values = {
                method: _interpolate(
                    model_fixed[model_fixed.method == method],
                    "actual_total_ms", ceiling,
                )
                for method in METHODS
            }
            per_model_rows.append({
                "model": model, "error_ceiling": ceiling,
                "paper_ms": values["paper"], "cell1_fixed_ms": values["cell1_fixed"],
                "cell1_tau_ms": values["cell1_tau"],
                "tau_gain_vs_paper_pct": _gain(values["paper"], values["cell1_tau"]),
                "tau_gain_vs_fixed_pct": _gain(
                    values["cell1_fixed"], values["cell1_tau"]
                ),
            })
    per_model = pd.DataFrame(per_model_rows)
    per_model.to_csv(output / "per_model_same_error.csv", index=False)

    _plot(output, fixed, interpolated)
    _write_report(args, candidates, fixed, interpolated, comparison, per_model)


def _plot(output: Path, fixed: pd.DataFrame, interpolated: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.1))
    for method in METHODS:
        group = fixed[fixed.method == method].sort_values("control_value")
        ax.plot(
            group.actual_total_ms, group.relative_l2_error,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linewidth=2.1, markersize=4.5, label=METHOD_LABELS[method],
        )
        if method == "cell1_tau":
            for index, row in enumerate(group.itertuples(index=False)):
                if index % 2 == 0 or row.control_value >= 0.97:
                    ax.annotate(
                        f"τ {100*row.control_value:.0f}%",
                        (row.actual_total_ms, row.relative_l2_error),
                        xytext=(4, 4 if index % 4 == 0 else -10),
                        textcoords="offset points", fontsize=6.2,
                        color=METHOD_COLORS[method],
                    )
    ax.set_xlabel("Measured actual total per projection (ms)")
    ax.set_ylabel("Projection relative L2 error (lower is better)")
    ax.set_title("Causal local importance-bound frontier")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "local_importance_frontier.png", dpi=220)
    fig.savefig(output / "local_importance_frontier.pdf")
    plt.close(fig)

    common = interpolated.dropna(subset=["actual_total_ms"])
    counts = common.groupby("error_ceiling").method.nunique()
    common = common[common.error_ceiling.isin(counts[counts == len(METHODS)].index)]
    means = common.groupby("method", as_index=False)[list(COMPONENTS)].mean()
    means["order"] = means.method.map({method: index for index, method in enumerate(METHODS)})
    means = means.sort_values("order")
    means["component_total_ms"] = means[list(COMPONENTS)].sum(axis=1)
    means.to_csv(output / "same_error_component_means.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    x = np.arange(len(means))
    bottoms = np.zeros(len(means))
    labels = {
        "score_median_ms": "Score construction",
        "selector_median_ms": "Structured selector",
        "read_wall_median_ms": "SSD read/upload wall",
        "gather_median_ms": "Activation gather",
        "gemm_median_ms": "Compact GEMM",
    }
    colors = {
        "score_median_ms": "#DB2777", "selector_median_ms": "#7C3AED",
        "read_wall_median_ms": "#2563EB", "gather_median_ms": "#D97706",
        "gemm_median_ms": "#0F766E",
    }
    for component in COMPONENTS:
        values = means[component].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottoms, color=colors[component], label=labels[component])
        bottoms += values
    for index, total in enumerate(bottoms):
        ax.text(index, total + 0.01, f"{total:.3f} ms", ha="center", fontsize=9)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in means.method], rotation=6)
    ax.set_ylabel("Mean latency at common equal-error points (ms)")
    ax.set_title("Local importance-bound latency decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "local_importance_components.png", dpi=220)
    fig.savefig(output / "local_importance_components.pdf")
    plt.close(fig)

    tau = fixed[fixed.method == "cell1_tau"].sort_values("control_value")
    fig, ax = plt.subplots(figsize=(8.3, 5.4))
    ax.plot(
        tau.control_value, tau.selected_fraction, color=METHOD_COLORS["cell1_tau"],
        marker="P", linewidth=2.1, label="Resulting mean R",
    )
    ax.fill_between(
        tau.control_value,
        np.maximum(0.0, tau.selected_fraction - tau.selected_fraction_std),
        np.minimum(1.0, tau.selected_fraction + tau.selected_fraction_std),
        color=METHOD_COLORS["cell1_tau"], alpha=0.15, label="±1 std across calls",
    )
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", label="R=τ reference")
    ax.set_xlim(0.08, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Importance lower bound τ")
    ax.set_ylabel("Resulting selected fraction R")
    ax.set_title("R is an output, not a control")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "tau_to_resulting_r.png", dpi=220)
    fig.savefig(output / "tau_to_resulting_r.pdf")
    plt.close(fig)


def _fmt(value: float, digits: int = 3, suffix: str = "") -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}{suffix}"


def _write_report(args, candidates: pd.DataFrame, fixed: pd.DataFrame,
                  interpolated: pd.DataFrame, comparison: pd.DataFrame,
                  per_model: pd.DataFrame) -> None:
    lines = [
        "# Experiment 45: causal local importance-bound Cell-1", "",
        "Global R allocation과 미래 projection 정보는 사용하지 않는다. 각 projection ",
        "호출에서 현재 X² importance만 계산하고 `I(S) >= τ`를 만족하는 fixed-origin ",
        "s-cell mask 중 lookup read latency가 작은 mask를 선택한 뒤 run endpoint를 ",
        "importance 하한이 깨지지 않는 범위에서 trim한다. R은 입력이 아니라 결과다.",
        "", "## 동일 projection error", "",
        "| error | Paper | Cell-1 fixed R | Cell-1-τ | gain vs Paper | gain vs fixed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.error_ceiling:.2f} | {_fmt(row.paper_total_ms, suffix=' ms')} "
            f"| {_fmt(row.cell1_fixed_total_ms, suffix=' ms')} "
            f"| {_fmt(row.cell1_tau_total_ms, suffix=' ms')} "
            f"| {_fmt(row.cell1_tau_gain_vs_paper_pct, 2, '%')} "
            f"| {_fmt(row.cell1_tau_gain_vs_fixed_cell1_pct, 2, '%')} |"
        )
    common = comparison.dropna(subset=[
        "cell1_tau_gain_vs_paper_pct", "cell1_tau_gain_vs_fixed_cell1_pct"
    ])
    means = pd.read_csv(args.output_dir / "same_error_component_means.csv")
    lines.extend([
        "", "## 동일-error 구성요소 평균", "",
        "| method | score | selector | SSD/upload | gather | GEMM | total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in means.itertuples(index=False):
        lines.append(
            f"| {METHOD_LABELS[row.method]} | {row.score_median_ms:.4f} "
            f"| {row.selector_median_ms:.4f} | {row.read_wall_median_ms:.4f} "
            f"| {row.gather_median_ms:.4f} | {row.gemm_median_ms:.4f} "
            f"| {row.component_total_ms:.4f} |"
        )
    model_means = per_model.groupby("model")[[
        "tau_gain_vs_paper_pct", "tau_gain_vs_fixed_pct"
    ]].mean()
    lines.extend([
        "", "## 모델별 동일-error 평균 gain", "",
        "| model | vs Paper | vs fixed-R Cell-1 |", "|---|---:|---:|",
    ])
    for model, row in model_means.iterrows():
        lines.append(
            f"| {model} | {_fmt(row.tau_gain_vs_paper_pct, 2, '%')} "
            f"| {_fmt(row.tau_gain_vs_fixed_pct, 2, '%')} |"
        )
    tau_rows = candidates[candidates.method == "cell1_tau"]
    effective_tau = tau_rows.row_budget / tau_rows.n
    guarantee_failures = int((
        tau_rows.importance_retention + 1e-7 < effective_tau
    ).sum())
    lines.extend([
        "", "## 판정", "",
        f"공통 error 구간에서 Cell-1-τ의 평균 gain은 Paper 대비 "
        f"`{common.cell1_tau_gain_vs_paper_pct.mean():.2f}%`, fixed-R Cell-1 대비 "
        f"`{common.cell1_tau_gain_vs_fixed_cell1_pct.mean():.2f}%`다.", "",
        "이 결과는 global allocation 상한과 다르다. 각 호출은 현재 activation만 ",
        "사용하므로 causal하고, 하나의 τ만 quality knob로 사용한다. actual error는 ",
        "평가에만 사용하고 selector에는 입력하지 않았다.", "",
        "## 측정 범위", "",
        f"- 모델 `{candidates.model.nunique()}`개, 후보 `{len(candidates)}`개.",
        f"- fallback `{int(candidates.fallback_used.sum())}`건; importance 하한 위반 "
        f"`{guarantee_failures}`건.",
        "- control 10%..95% 5% 간격 + 97/98/99%; baseline은 fixed R, Cell-1-τ는 τ.",
        "- actual total은 score + selector + O_DIRECT/upload + gather + compact GEMM.",
        "", f"![Frontier]({args.output_dir.name}/local_importance_frontier.png)",
        "", f"![Components]({args.output_dir.name}/local_importance_components.png)",
        "", f"![Tau to R]({args.output_dir.name}/tau_to_resulting_r.png)", "",
    ])
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines))


def _synthetic_self_check(selector: LocalImportanceSelector) -> None:
    rng = np.random.default_rng(45)
    for n in (127, 509, 2048):
        score = torch.from_numpy(rng.lognormal(size=n).astype(np.float32))
        normalized = score.numpy() / score.numpy().sum()
        for threshold in (0.10, 0.50, 0.90, 0.99):
            rows = max(1, min(n - 1, int(round(n * threshold))))
            first, metadata = selector.select("cell1_tau", score, rows, 2048)
            second, _ = selector.select("cell1_tau", score, rows, 2048)
            achieved = float(normalized[first.numpy()].sum())
            effective = rows / n
            if (
                achieved + 1e-6 < effective
                or not torch.equal(first, second)
                or metadata.get("fallback_used")
            ):
                raise RuntimeError(
                    f"self-check failed n={n} tau={threshold} achieved={achieved} "
                    f"metadata={metadata}"
                )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return
    if args.process_nice:
        os.nice(args.process_nice)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)

    prompts = list(PROMPTS[:args.prompt_limit] if args.prompt_limit else PROMPTS)
    lookup = EXP32.BASE.LatencyTable.load(args.profile)
    selector = LocalImportanceSelector(lookup, args.saturation_kib)
    _synthetic_self_check(selector)
    vlmflash = EXP32.EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")

    from transformers import AutoConfig
    model_key = args.models[0]
    config = AutoConfig.from_pretrained(
        EXP32.cached_snapshot(MODEL_SPECS[model_key]["repo"]), local_files_only=True
    )
    required = int(config.intermediate_size) * int(config.hidden_size) * 2
    if args.io_blob.stat().st_size < required:
        raise SystemExit(f"I/O blob needs at least {required} bytes")

    print(f"loading and measuring {MODEL_SPECS[model_key]['label']}", flush=True)
    result = EXP39.run_model(model_key, selector, native_reader, prompts, args, vlmflash)
    (
        model, tokenizer, weight_norms, candidates, selectors, scores,
        io_rows, gemm_rows, model_metadata,
    ) = result
    pd.DataFrame(candidates).to_csv(args.output_dir / "candidates.csv", index=False)
    pd.DataFrame(selectors).to_csv(args.output_dir / "selector_samples.csv", index=False)
    pd.DataFrame(scores).to_csv(args.output_dir / "score_samples.csv", index=False)
    pd.DataFrame(io_rows).to_csv(args.output_dir / "io_aggregates.csv", index=False)
    pd.DataFrame(gemm_rows).to_csv(args.output_dir / "gemm_aggregates.csv", index=False)
    if not args.skip_end_to_end and any(p["split"] == "holdout" for p in prompts):
        e2e = EXP39.run_end_to_end(
            model, tokenizer, model_key, weight_norms, selector, prompts, args, vlmflash
        )
        pd.DataFrame(e2e).to_csv(args.output_dir / "end_to_end.csv", index=False)
    del model, tokenizer, weight_norms
    gc.collect()
    torch.cuda.empty_cache()

    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-45-local-importance-bound-v1",
        "models": [model_metadata],
        "prompts": [
            {key: value for key, value in prompt.items() if key != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "methods": list(METHODS), "controls": args.budgets,
        "control_semantics": {
            "paper": "fixed selected fraction R",
            "cell1_fixed": "fixed selected fraction R",
            "cell1_tau": "minimum retained X2 importance tau; R is output",
        },
        "causal": True,
        "future_projection_information": False,
        "actual_error_used_by_selector": False,
        "saturation_kib": args.saturation_kib,
        "score_repetitions": args.score_repetitions,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions,
        "gemm_repetitions": args.gemm_repetitions,
        "profile": str(args.profile), "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    analyze(args)


if __name__ == "__main__":
    main()
