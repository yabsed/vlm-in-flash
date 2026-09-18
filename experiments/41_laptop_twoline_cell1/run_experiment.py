#!/usr/bin/env python3
"""Experiment 41: laptop Cell-1 with lookup versus a fitted two-line cost."""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import platform
import traceback
from pathlib import Path
from types import SimpleNamespace

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
    "experiment_39_for_41",
    PROJECT_ROOT / "experiments" / "39_weight_aware_cell1" / "run_experiment.py",
)
EXP13 = _load_module(
    "experiment_13_for_41",
    PROJECT_ROOT / "experiments" / "13_two_line_lagrangian" / "run_experiment.py",
)
EXP37 = EXP39.EXP37
EXP36 = EXP39.EXP36
EXP32 = EXP39.EXP32

METHODS = (
    "paper",
    "cell1_abs_lookup", "cell1_abs_twoline",
    "cell1_x2_lookup", "cell1_x2_twoline",
)
METHOD_LABELS = {
    "paper": "Paper |x|",
    "cell1_abs_lookup": "Cell-1 |x| lookup",
    "cell1_abs_twoline": "Cell-1 |x| 2-line",
    "cell1_x2_lookup": "Cell-1 X² lookup",
    "cell1_x2_twoline": "Cell-1 X² 2-line",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "cell1_abs_lookup": "#0F766E",
    "cell1_abs_twoline": "#14B8A6",
    "cell1_x2_lookup": "#1D4ED8",
    "cell1_x2_twoline": "#60A5FA",
}
METHOD_MARKERS = {
    "paper": "s",
    "cell1_abs_lookup": "o", "cell1_abs_twoline": "D",
    "cell1_x2_lookup": "^", "cell1_x2_twoline": "v",
}
METHOD_SCORE = {
    "paper": "abs",
    "cell1_abs_lookup": "abs", "cell1_abs_twoline": "abs",
    "cell1_x2_lookup": "x2", "cell1_x2_twoline": "x2",
}
METHOD_INTERNAL = {
    "paper": "paper",
    "cell1_abs_lookup": "c1_l1_cached_lookup",
    "cell1_abs_twoline": "c1_l1_twoline",
    "cell1_x2_lookup": "c1_l1_cached_lookup",
    "cell1_x2_twoline": "c1_l1_twoline",
}
PAIRS = {
    "abs": ("cell1_abs_lookup", "cell1_abs_twoline"),
    "x2": ("cell1_x2_lookup", "cell1_x2_twoline"),
}
ERROR_CEILINGS = EXP39.ERROR_CEILINGS
KL_CEILINGS = EXP39.KL_CEILINGS
COMPONENTS = EXP39.COMPONENTS

# Experiment 39 resolves these globals at call time. Reusing its measurement
# path guarantees that only the selector's cost model changes.
EXP39.HERE = HERE
EXP39.METHODS = METHODS
EXP39.METHOD_LABELS = METHOD_LABELS
EXP39.METHOD_COLORS = METHOD_COLORS
EXP39.METHOD_MARKERS = METHOD_MARKERS
EXP39.METHOD_SCORE = METHOD_SCORE
EXP39.METHOD_INTERNAL = METHOD_INTERNAL


class TwoLineCellSelector(EXP37.SuperTileSelector):
    """Cell-1 selector with either empirical lookup or a fitted two-line cost."""

    def __init__(self, lookup_table, saturation_kib: float):
        super().__init__(lookup_table, saturation_kib)
        self.saturation_kib = float(saturation_kib)
        self._cost_cache: dict[tuple[int, float], tuple[np.ndarray, dict]] = {}
        self._lookup_cost_cache: dict[tuple[int, float], np.ndarray] = {}

    def _costs(self, n: int, row_kib: float) -> tuple[np.ndarray, dict]:
        key = (int(n), round(float(row_kib), 12))
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached
        model = EXP13.fit_continuous_two_line(
            self.lookup_table, float(row_kib), self.saturation_kib
        )
        rows = np.arange(n + 1, dtype=np.float64)
        costs = np.where(
            rows <= model["saturation_rows"],
            model["a_ms"] + model["c1_ms_per_row"] * rows,
            model["c2_ms_per_row"] * rows,
        )
        costs[0] = 0.0
        self._cost_cache[key] = (costs, model)
        return costs, model

    def _lookup_costs(self, n: int, row_kib: float) -> np.ndarray:
        key = (int(n), round(float(row_kib), 12))
        cached = self._lookup_cost_cache.get(key)
        if cached is not None:
            return cached
        costs = np.zeros(n + 1, dtype=np.float64)
        for length in range(1, n + 1):
            costs[length] = self.lookup_table.read_ms(length * row_kib)
        self._lookup_cost_cache[key] = costs
        return costs

    def select(self, method: str, importance: torch.Tensor, row_budget: int,
               d: int) -> tuple[torch.Tensor, dict]:
        if method not in ("c1_l1_cached_lookup", "c1_l1_twoline"):
            return super().select(method, importance, row_budget, d)
        n = int(importance.numel())
        context = self.context(n, d)
        normalized = importance / importance.sum().clamp_min(1e-20)
        values = normalized.detach().to("cpu").numpy().astype(np.float64)
        row_kib = 2.0 * float(d) / 1024.0
        _, model = self._costs(n, row_kib)
        if method == "c1_l1_cached_lookup":
            run_costs = self._lookup_costs(n, row_kib)
            cost_model = "lookup"
        else:
            run_costs, _ = self._costs(n, row_kib)
            cost_model = "two_line"
        cell_rows = max(1, int(math.ceil(float(model["saturation_rows"]))))
        try:
            result = EXP36._select_cell_kernel(
                values, int(row_budget), int(cell_rows), 1, run_costs
            )
            mask_np, candidates, tile_rows, strategy, repair, chunks, ratio, valid = result
            if not valid or int(mask_np.sum()) != int(row_budget):
                raise RuntimeError("two-line Cell-1 failed to return exact R")
            metadata = {
                "scalarized_calls": 1,
                "outer_iterations": 1,
                "frontier_candidates": int(candidates),
                "eligible_candidates": int(chunks),
                "minimum_overfill": int(tile_rows),
                "repair_deletions": int(repair if strategy == 2 else 0),
                "repair_additions": int(repair if strategy == 1 else 0),
                "returned_top_r": False,
                "cell_count": 1,
                "length_factor": 1,
                "cell_rows": int(cell_rows),
                "tile_rows": int(tile_rows),
                "tile_strategy": (
                    "contiguous" if strategy == 0 else
                    "floor_expand" if strategy == 1 else "ceil_trim"
                ),
                "lookup_efficiency": float(ratio),
                "cost_model": cost_model,
                "two_line_a_ms": model["a_ms"],
                "two_line_c1_ms_per_row": model["c1_ms_per_row"],
                "two_line_c2_ms_per_row": model["c2_ms_per_row"],
                "two_line_r_squared": model["r_squared"],
                "two_line_mape_pct": model["mape_pct"],
                "paper_parameter_source": context["paper_parameter_source"],
                "fallback_used": False,
                "error": "",
            }
        except Exception as exception:
            mask_np = EXP36.EXP29.EXP24.top_r_mask(values, int(row_budget))
            metadata = {
                "cost_model": cost_model,
                "fallback_used": True,
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(limit=4),
            }
        mask = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.bool_)).to(
            importance.device
        )
        return mask, metadata


def _interpolate(curve: pd.DataFrame, column: str, error: float) -> float:
    return EXP39._interpolate(curve, column, error)


def _pct_gain(baseline: float, candidate: float) -> float:
    return EXP39._pct_gain(baseline, candidate)


def _fmt(value: float, digits: int = 3, suffix: str = "") -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}{suffix}"


def analyze(args) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    holdout = candidates[candidates.split == "holdout"].copy()
    fixed = holdout.groupby(
        ["method", "method_label", "score_kind", "budget_index", "budget_fraction"],
        as_index=False,
    ).agg(
        cases=("case_key", "size"),
        actual_total_ms=("actual_total_ms", "mean"),
        legacy_total_ms=("legacy_total_ms", "mean"),
        score_median_ms=("score_median_ms", "mean"),
        selector_median_ms=("selector_median_ms", "mean"),
        selector_total_ms=("selector_total_ms", "mean"),
        read_wall_median_ms=("read_wall_median_ms", "mean"),
        io_median_ms=("io_median_ms", "mean"),
        upload_median_ms=("upload_median_ms", "mean"),
        gather_median_ms=("gather_median_ms", "mean"),
        gemm_median_ms=("gemm_median_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        importance_retention=("importance_retention", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        chunks=("chunks", "mean"),
        row_match_rate=("row_match", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    fixed.to_csv(output / "fixed_frontiers.csv", index=False)

    columns = (
        "actual_total_ms", "legacy_total_ms", *COMPONENTS,
        "selector_total_ms", "io_median_ms", "upload_median_ms",
        "selected_fraction", "chunks", "row_match_rate",
    )
    interpolated_rows = []
    for ceiling in ERROR_CEILINGS:
        for method in METHODS:
            curve = fixed[fixed.method == method]
            row = {"error_ceiling": ceiling, "method": method,
                   "method_label": METHOD_LABELS[method]}
            for column in columns:
                row[column] = _interpolate(curve, column, ceiling)
            interpolated_rows.append(row)
    interpolated = pd.DataFrame(interpolated_rows)
    interpolated.to_csv(output / "same_error_components.csv", index=False)

    same_error_rows = []
    for ceiling in ERROR_CEILINGS:
        indexed = interpolated[interpolated.error_ceiling == ceiling].set_index("method")
        for score_kind, (lookup, twoline) in PAIRS.items():
            base = float(indexed.loc[lookup, "actual_total_ms"])
            candidate = float(indexed.loc[twoline, "actual_total_ms"])
            same_error_rows.append({
                "error_ceiling": ceiling, "score_kind": score_kind,
                "lookup_total_ms": base, "twoline_total_ms": candidate,
                "twoline_gain_pct": _pct_gain(base, candidate),
                "selector_delta_ms": float(indexed.loc[twoline, "selector_median_ms"])
                - float(indexed.loc[lookup, "selector_median_ms"]),
                "read_wall_delta_ms": float(indexed.loc[twoline, "read_wall_median_ms"])
                - float(indexed.loc[lookup, "read_wall_median_ms"]),
                "chunk_delta": float(indexed.loc[twoline, "chunks"])
                - float(indexed.loc[lookup, "chunks"]),
            })
    same_error = pd.DataFrame(same_error_rows)
    same_error.to_csv(output / "same_error_comparison.csv", index=False)

    same_r_rows = []
    for score_kind, (lookup, twoline) in PAIRS.items():
        left = holdout[holdout.method == lookup]
        right = holdout[holdout.method == twoline]
        keys = ["case_key", "budget_index", "budget_fraction"]
        joined = left.merge(right, on=keys, suffixes=("_lookup", "_twoline"))
        for row in joined.itertuples(index=False):
            same_r_rows.append({
                "score_kind": score_kind,
                "case_key": row.case_key, "budget_index": row.budget_index,
                "budget_fraction": row.budget_fraction,
                "same_mask": row.mask_sha256_lookup == row.mask_sha256_twoline,
                "error_delta": row.relative_l2_error_twoline
                - row.relative_l2_error_lookup,
                "actual_total_delta_ms": row.actual_total_ms_twoline
                - row.actual_total_ms_lookup,
                "selector_delta_ms": row.selector_median_ms_twoline
                - row.selector_median_ms_lookup,
                "read_wall_delta_ms": row.read_wall_median_ms_twoline
                - row.read_wall_median_ms_lookup,
                "chunk_delta": row.chunks_twoline - row.chunks_lookup,
            })
    same_r = pd.DataFrame(same_r_rows)
    same_r.to_csv(output / "same_r_pairwise.csv", index=False)
    same_r_summary = same_r.groupby("score_kind", as_index=False).agg(
        cases=("same_mask", "size"), same_mask_rate=("same_mask", "mean"),
        error_delta_mean=("error_delta", "mean"),
        error_delta_abs_mean=("error_delta", lambda x: np.abs(x).mean()),
        actual_total_delta_ms=("actual_total_delta_ms", "mean"),
        selector_delta_ms=("selector_delta_ms", "mean"),
        read_wall_delta_ms=("read_wall_delta_ms", "mean"),
        chunk_delta=("chunk_delta", "mean"),
    )
    same_r_summary.to_csv(output / "same_r_summary.csv", index=False)

    e2e_summary = pd.DataFrame()
    e2e_pairwise = pd.DataFrame()
    e2e_path = output / "end_to_end.csv"
    if e2e_path.is_file():
        e2e = pd.read_csv(e2e_path)
        e2e_summary = e2e.groupby(
            ["method", "method_label", "score_kind", "budget_index", "budget_fraction"],
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
        pair_rows = []
        for score_kind, (lookup_method, twoline_method) in PAIRS.items():
            left = e2e_summary[e2e_summary.method == lookup_method]
            right = e2e_summary[e2e_summary.method == twoline_method]
            joined = left.merge(
                right, on=["budget_index", "budget_fraction"],
                suffixes=("_lookup", "_twoline"),
            )
            for row in joined.itertuples(index=False):
                pair_rows.append({
                    "score_kind": score_kind,
                    "budget_index": row.budget_index,
                    "budget_fraction": row.budget_fraction,
                    "lookup_kl": row.dense_to_sparse_kl_mean_lookup,
                    "twoline_kl": row.dense_to_sparse_kl_mean_twoline,
                    "kl_delta": row.dense_to_sparse_kl_mean_twoline
                    - row.dense_to_sparse_kl_mean_lookup,
                    "actual_total_delta_ms": row.actual_total_ms_twoline
                    - row.actual_total_ms_lookup,
                })
        e2e_pairwise = pd.DataFrame(pair_rows)
        e2e_pairwise.to_csv(output / "end_to_end_pairwise.csv", index=False)

    profile = EXP32.BASE.LatencyTable.load(Path(args.profile))
    fit = EXP13.fit_continuous_two_line(profile, 1.0, float(args.saturation_kib))
    _plot(output, profile, fit, fixed, interpolated, same_error)
    write_report(
        args, candidates, fixed, same_error, same_r_summary, fit,
        e2e_summary, e2e_pairwise,
    )


def _plot(output: Path, profile, fit: dict, fixed: pd.DataFrame,
          interpolated: pd.DataFrame, same_error: pd.DataFrame) -> None:
    items = sorted(profile.as_dict().items())
    x = np.asarray([item[0] for item in items], dtype=float)
    y = np.asarray([item[1] for item in items], dtype=float)
    x_model = np.linspace(1.0, max(float(x.max()), 2.0 * fit["saturation_kib"]), 600)
    predicted = np.where(
        x_model <= fit["saturation_kib"],
        fit["a_ms"] + fit["c1_ms_per_row"] * x_model,
        fit["c2_ms_per_row"] * x_model,
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(x, y, color="#94A3B8", linewidth=1.1, label="Laptop lookup")
    before = x_model <= fit["saturation_kib"]
    ax.plot(x_model[before], predicted[before], color="#DC2626", linewidth=2.0,
            label="Fitted pre-s branch")
    ax.plot(x_model[~before], predicted[~before], color="#DC2626", linewidth=2.0,
            linestyle="--", label="Post-s linear extrapolation")
    ax.axvline(fit["saturation_kib"], color="#475569", linestyle="--", linewidth=1)
    ax.text(fit["saturation_kib"] + 5, predicted.max() * 0.12,
            "s = 240 KiB", fontsize=9)
    ax.set_xlabel("Contiguous read size (KiB)")
    ax.set_ylabel("Read latency (ms)")
    ax.set_title(
        f"Laptop SSD profile fit (R²={fit['r_squared']:.4f}, "
        f"MAPE={fit['mape_pct']:.2f}%)"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "twoline_fit.png", dpi=200)
    fig.savefig(output / "twoline_fit.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), sharey=True)
    panels = (
        ("ABS score", "cell1_abs_lookup", "cell1_abs_twoline"),
        ("X² score", "cell1_x2_lookup", "cell1_x2_twoline"),
    )
    for ax, (title, lookup_method, twoline_method) in zip(axes, panels):
        for method in ("paper", lookup_method, twoline_method):
            group = fixed[fixed.method == method].sort_values("budget_fraction")
            ax.plot(
                group.actual_total_ms, group.relative_l2_error,
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                linewidth=1.6, markersize=4, label=METHOD_LABELS[method],
            )
            if method in ("paper", twoline_method):
                for index, row in enumerate(group.itertuples(index=False)):
                    if index % 2 == 0:
                        ax.annotate(
                            f"{100 * row.budget_fraction:.0f}%",
                            (row.actual_total_ms, row.relative_l2_error),
                            xytext=(3, 3), textcoords="offset points", fontsize=6,
                            color=METHOD_COLORS[method],
                        )
        ax.set_xlabel("Measured actual total (ms)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Projection relative L2 error")
    fig.suptitle("Lookup versus 2-line Cell-1 on the laptop")
    fig.tight_layout()
    fig.savefig(output / "lookup_vs_twoline_frontier.png", dpi=200)
    fig.savefig(output / "lookup_vs_twoline_frontier.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for score_kind, color, marker in (("abs", "#0F766E", "o"),
                                       ("x2", "#2563EB", "^")):
        group = same_error[same_error.score_kind == score_kind].sort_values(
            "error_ceiling"
        )
        ax.plot(group.error_ceiling, group.twoline_gain_pct,
                color=color, marker=marker, label=score_kind.upper())
    ax.axhline(0.0, color="#64748B", linewidth=1)
    ax.set_xlabel("Projection error ceiling")
    ax.set_ylabel("2-line latency gain over lookup (%)")
    ax.set_title("Does smoothing the SSD cost improve the frontier?")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "twoline_gain.png", dpi=200)
    fig.savefig(output / "twoline_gain.pdf")
    plt.close(fig)

    common = interpolated.dropna(subset=["actual_total_ms"])
    counts = common.groupby("error_ceiling").method.nunique()
    common = common[common.error_ceiling.isin(counts[counts == len(METHODS)].index)]
    means = common.groupby("method", as_index=False)[list(COMPONENTS)].mean()
    means["order"] = means.method.map({m: i for i, m in enumerate(METHODS)})
    means = means.sort_values("order")
    means.to_csv(output / "same_error_component_means.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x_pos = np.arange(len(means))
    bottoms = np.zeros(len(means))
    labels = {
        "score_median_ms": "Score", "selector_median_ms": "Selector",
        "read_wall_median_ms": "SSD/upload", "gather_median_ms": "Gather",
        "gemm_median_ms": "GEMM",
    }
    colors = {
        "score_median_ms": "#DB2777", "selector_median_ms": "#7C3AED",
        "read_wall_median_ms": "#2563EB", "gather_median_ms": "#D97706",
        "gemm_median_ms": "#0F766E",
    }
    for column in COMPONENTS:
        values = means[column].to_numpy(dtype=float)
        ax.bar(x_pos, values, bottom=bottoms, color=colors[column], label=labels[column])
        bottoms += values
    ax.set_xticks(x_pos, [METHOD_LABELS[m] for m in means.method], rotation=12)
    ax.set_ylabel("Mean latency at common equal-error points (ms)")
    ax.set_title("Equal-error latency decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "twoline_components.png", dpi=200)
    fig.savefig(output / "twoline_components.pdf")
    plt.close(fig)


def write_report(args, candidates: pd.DataFrame, fixed: pd.DataFrame,
                 same_error: pd.DataFrame, same_r: pd.DataFrame,
                 fit: dict, e2e: pd.DataFrame,
                 e2e_pairwise: pd.DataFrame) -> None:
    lines = [
        "# Experiment 41: 노트북 Cell-1 lookup vs 2-line", "",
        "실험 39의 실제 노트북 측정 경로를 그대로 사용하고, Cell-1 selector가 "
        "청크 비용을 평가할 때만 SSD lookup table과 아래 2-line 근사를 교체했다.", "",
        "```text",
        "T(r) = a + c1*r   (r <= s)",
        "     = c2*r       (r > s)",
        "```", "",
        f"노트북 SSD profile fit은 `a={fit['a_ms']:.6f} ms`, "
        f"`c1={fit['c1_ms_per_row']:.9f} ms/KiB`, "
        f"`c2={fit['c2_ms_per_row']:.9f} ms/KiB`, "
        f"`s={fit['saturation_kib']:.0f} KiB`이다. "
        f"적합도는 `R²={fit['r_squared']:.5f}`, "
        f"`MAPE={fit['mape_pct']:.2f}%`다.", "",
        "저장된 laptop lookup은 profiler가 throughput saturation에서 trim했기 "
        "때문에 1–240 KiB만 포함한다. 따라서 pre-s branch는 240개 실측점에 "
        "적합했지만, post-s의 `c2*r`은 continuity가 정하는 선형 외삽이다. "
        "기존 lookup도 240 KiB 밖에서는 마지막 점을 크기에 비례해 외삽한다.", "",
        "## 동일 projection error", "",
        "양수 gain은 2-line이 lookup보다 빠르다는 뜻이다. R=5% grid 사이를 "
        "선형 보간했으므로 작은 차이는 보간 오차로 해석해야 한다.", "",
        "| error | score | lookup | 2-line | gain | selector Δ | SSD Δ | chunk Δ |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in same_error.itertuples(index=False):
        lines.append(
            f"| {row.error_ceiling:.2f} | {row.score_kind.upper()} "
            f"| {_fmt(row.lookup_total_ms, suffix=' ms')} "
            f"| {_fmt(row.twoline_total_ms, suffix=' ms')} "
            f"| {_fmt(row.twoline_gain_pct, suffix='%')} "
            f"| {_fmt(row.selector_delta_ms, 4, ' ms')} "
            f"| {_fmt(row.read_wall_delta_ms, 4, ' ms')} "
            f"| {_fmt(row.chunk_delta, 2)} |"
        )

    lines.extend([
        "", "## 같은 R에서 마스크가 얼마나 바뀌었나", "",
        "| score | cases | same mask | mean error Δ | mean |error Δ| | total Δ | selector Δ | SSD Δ | chunk Δ |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in same_r.itertuples(index=False):
        lines.append(
            f"| {row.score_kind.upper()} | {int(row.cases)} "
            f"| {100 * row.same_mask_rate:.2f}% "
            f"| {row.error_delta_mean:+.6f} | {row.error_delta_abs_mean:.6f} "
            f"| {row.actual_total_delta_ms:+.4f} ms "
            f"| {row.selector_delta_ms:+.4f} ms "
            f"| {row.read_wall_delta_ms:+.4f} ms "
            f"| {row.chunk_delta:+.3f} |"
        )

    if len(e2e_pairwise):
        lines.extend([
            "", "## End-to-end logit KL (같은 R)", "",
            "음수 KL Δ는 2-line의 logit error가 더 작다는 뜻이다. KL은 R에 "
            "대해 단조롭지 않고 holdout이 모델당 3개뿐이므로 방향성 진단이다.", "",
            "| score | R points | 2-line better | mean KL Δ | median KL Δ | mean latency Δ |",
            "|:---:|---:|---:|---:|---:|---:|",
        ])
        for score_kind, group in e2e_pairwise.groupby("score_kind"):
            lines.append(
                f"| {score_kind.upper()} | {len(group)} "
                f"| {int((group.kl_delta < 0).sum())}/{len(group)} "
                f"| {group.kl_delta.mean():+.4f} "
                f"| {group.kl_delta.median():+.4f} "
                f"| {group.actual_total_delta_ms.mean():+.4f} ms |"
            )

    valid = same_error.dropna(subset=["twoline_gain_pct"])
    pair_means = valid.groupby("score_kind").twoline_gain_pct.mean()
    best_method_counts = {}
    for ceiling in ERROR_CEILINGS:
        group = same_error[same_error.error_ceiling == ceiling]
        for row in group.itertuples(index=False):
            winner = "2-line" if row.twoline_total_ms < row.lookup_total_ms else "lookup"
            key = f"{row.score_kind}:{winner}"
            best_method_counts[key] = best_method_counts.get(key, 0) + 1
    same_mask_mean = float(same_r.same_mask_rate.mean())
    lines.extend([
        "", "## 판정", "",
        "**공정하게 캐시를 맞춘 뒤에는 2-line이 lookup frontier를 의미 있게 "
        "줄이지 못했다. 두 곡선은 측정 노이즈 범위에서 사실상 같다.**", "",
        f"2-line의 동일-error 평균 gain은 ABS `{pair_means.get('abs', np.nan):+.3f}%`, "
        f"X² `{pair_means.get('x2', np.nan):+.3f}%`이며, ceiling 승수는 "
        f"`{best_method_counts}`다. 같은 R의 마스크 일치율은 두 score 평균 "
        f"`{100 * same_mask_mean:.2f}%`다.", "",
        "2-line은 실제 측정치를 대체하는 latency 예측기가 아니라 **Cell-1의 "
        "repair 결정을 위한 목적함수**로만 사용했다. 최종 x축은 두 방법 모두 "
        "동일하게 실제 O_DIRECT SSD read/upload, activation gather, compact GEMM을 "
        "재측정한 `actual_total_ms`다. 따라서 결과 차이는 비용모델이 선택한 "
        "마스크 차이이며, 2-line 예측값을 성능값으로 그린 것이 아니다.", "",
        "두 경로는 같은 로컬 cost-array cache와 같은 Cell-1 kernel을 사용한다. "
        "따라서 lookup table을 매 호출마다 hash하는 구현 비용은 비교에서 제거했다.", "",
        "Cell-1의 본체는 동일 길이 s-cell을 고르므로 두 비용모델의 차이는 주로 "
        "exact-R을 맞추는 floor-expand 대 ceil-trim에서 발생한다. 따라서 높은 "
        "마스크 일치율과 작은 frontier 차이가 나와도 버그가 아니라 구조적으로 "
        "예상되는 결과다.", "",
        "## 측정 범위", "",
        f"- projection 후보 `{len(candidates)}`개, holdout aggregate point `{len(fixed)}`개.",
        f"- Cell-1 selector fallback "
        f"`{int(candidates[candidates.method != 'paper'].fallback_used.sum())}`건, "
        f"Cell-1 row mismatch "
        f"`{int((~candidates[candidates.method != 'paper'].row_match.astype(bool)).sum())}`건. "
        "Paper는 원 논문의 8-row chunk granularity 때문에 exact-R 대상이 아니다.",
        "- score 생성 + selector + 실제 SSD/upload + gather + compact GEMM을 total에 포함했다.",
        "- 모델별 별도 프로세스, CUDA allocator 55%, CPU/O_DIRECT thread 2개, "
        "projection 사이 25 ms 양보를 사용했다.", "",
        f"![SSD fit]({args.output_dir.name}/twoline_fit.png)", "",
        f"![Frontier]({args.output_dir.name}/lookup_vs_twoline_frontier.png)", "",
        f"![Gain]({args.output_dir.name}/twoline_gain.png)", "",
        f"![Components]({args.output_dir.name}/twoline_components.png)", "",
    ])
    if len(e2e):
        lines.insert(-8, f"- end-to-end aggregate point `{len(e2e)}`개도 저장했다.")
    report = args.report_output or HERE / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main() -> None:
    args = EXP39.parse_args()
    EXP39.validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return

    if args.process_nice:
        os.nice(args.process_nice)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)

    prompts = list(EXP39.PROMPTS[:args.prompt_limit] if args.prompt_limit else EXP39.PROMPTS)
    lookup = EXP32.BASE.LatencyTable.load(args.profile)
    selector = TwoLineCellSelector(lookup, args.saturation_kib)
    vlmflash = EXP32.EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")

    from transformers import AutoConfig
    spec = EXP39.MODEL_SPECS[args.models[0]]
    config = AutoConfig.from_pretrained(
        EXP32.cached_snapshot(spec["repo"]), local_files_only=True
    )
    max_required = int(config.intermediate_size) * int(config.hidden_size) * 2
    if args.io_blob.stat().st_size < max_required:
        raise SystemExit(f"I/O blob needs at least {max_required} bytes")

    model_key = args.models[0]
    print(f"loading and measuring {spec['label']}", flush=True)
    result = EXP39.run_model(
        model_key, selector, native_reader, prompts, args, vlmflash
    )
    (
        model, tokenizer, weight_norms, candidates, selectors, scores,
        io_rows, gemm_rows, model_metadata,
    ) = result
    pd.DataFrame(candidates).to_csv(
        args.output_dir / "candidates.partial.csv", index=False
    )
    e2e_rows = []
    if not args.skip_end_to_end and any(p["split"] == "holdout" for p in prompts):
        e2e_rows = EXP39.run_end_to_end(
            model, tokenizer, model_key, weight_norms, selector,
            prompts, args, vlmflash,
        )
    del model, tokenizer, weight_norms
    gc.collect()
    torch.cuda.empty_cache()

    pd.DataFrame(candidates).to_csv(args.output_dir / "candidates.csv", index=False)
    pd.DataFrame(selectors).to_csv(args.output_dir / "selector_samples.csv", index=False)
    pd.DataFrame(scores).to_csv(args.output_dir / "score_samples.csv", index=False)
    pd.DataFrame(io_rows).to_csv(args.output_dir / "io_aggregates.csv", index=False)
    pd.DataFrame(gemm_rows).to_csv(args.output_dir / "gemm_aggregates.csv", index=False)
    if e2e_rows:
        pd.DataFrame(e2e_rows).to_csv(args.output_dir / "end_to_end.csv", index=False)
    partial = args.output_dir / "candidates.partial.csv"
    if partial.exists():
        partial.unlink()

    fit_kib = EXP13.fit_continuous_two_line(lookup, 1.0, args.saturation_kib)
    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-41-laptop-twoline-cell1-v1",
        "controlled_change": (
            "Cell-1 run-cost oracle only: empirical laptop lookup versus "
            "continuous two-line fit; actual latency is measured for both"
        ),
        "two_line_kib": fit_kib,
        "models": [model_metadata],
        "prompts": [
            {key: value for key, value in prompt.items() if key != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "methods": list(METHODS), "budgets": args.budgets,
        "saturation_kib": args.saturation_kib,
        "score_repetitions": args.score_repetitions,
        "score_warmup": args.score_warmup,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions, "io_warmup": args.io_warmup,
        "gemm_repetitions": args.gemm_repetitions,
        "gemm_warmup": args.gemm_warmup,
        "io_threads": args.io_threads, "io_max_read_kib": args.io_max_read_kib,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "cpu_threads": args.cpu_threads, "process_nice": args.process_nice,
        "norm_chunk_rows": args.norm_chunk_rows,
        "projection_throttle_ms": args.projection_throttle_ms,
        "profile": str(args.profile), "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "flash": profile_meta.get("flash", "unknown"),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    analyze(args)


if __name__ == "__main__":
    main()
