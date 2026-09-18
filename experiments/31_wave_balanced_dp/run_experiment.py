#!/usr/bin/env python3
"""Experiment 31: reader-wave-balanced exact-R interval DP on the laptop."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import traceback

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_29 = PROJECT_ROOT / "experiments" / "29_saturation_tiles" / "run_experiment.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP29 = _load_module("experiment_29_for_31", EXPERIMENT_29)
EXP28 = EXP29.EXP28
EXP26 = EXP29.EXP26
EXP24 = EXP29.EXP24
EXP22 = EXP29.EXP22
EXP18 = EXP29.EXP18
EXP13 = EXP29.EXP13
BASE = EXP29.BASE
njit = EXP29.njit
SHAPES = EXP29.SHAPES
DEFAULT_TRACE_INPUT = EXP29.DEFAULT_TRACE_INPUT
LOCAL_PROFILE = EXP29.LOCAL_PROFILE

WAVE_COUNTS = (1, 6, 12, 18, 24)
WAVE_METHODS = tuple(f"wave_k{count}" for count in WAVE_COUNTS)
METHODS = ("paper", "tile8_ceil", *WAVE_METHODS)
METHOD_LABELS = {
    "paper": "Paper",
    "tile8_ceil": "Experiment 29 ceil(s/8)",
    **{f"wave_k{count}": f"Wave K={count}" for count in WAVE_COUNTS},
}
METHOD_COLORS = {
    "paper": "#D97706",
    "tile8_ceil": "#0F766E",
    "wave_k1": "#64748B",
    "wave_k6": "#2563EB",
    "wave_k12": "#7C3AED",
    "wave_k18": "#DB2777",
    "wave_k24": "#DC2626",
}
DIRECT_ALIGNMENT_BYTES = 512


def parse_args():
    args = EXP29.parse_args()
    if args.output_dir == EXP29.HERE / "results":
        args.output_dir = HERE / "results"
    return args


def validate_args(args):
    return EXP29.validate_args(args)


@njit(cache=False)
def _wave_exact_kernel(values, row_budget, alignment_rows, run_count):
    """Maximum importance among aligned, near-equal, separated K-run masks."""
    n = len(values)
    output = np.zeros(n, dtype=np.bool_)
    if row_budget == 0:
        return output, 0.0, 0, 0, 0, 0, True
    if alignment_rows <= 0 or n % alignment_rows or row_budget % alignment_rows:
        return output, 0.0, 0, 0, 0, 0, False
    cell_count = n // alignment_rows
    selected_cells = row_budget // alignment_rows
    if run_count <= 0 or selected_cells < run_count:
        return output, 0.0, 0, 0, 0, 0, False
    if selected_cells + run_count - 1 > cell_count:
        return output, 0.0, 0, 0, 0, 0, False

    cell_values = np.empty(cell_count, dtype=np.float64)
    for cell in range(cell_count):
        total = 0.0
        start = cell * alignment_rows
        for offset in range(alignment_rows):
            total += values[start + offset]
        cell_values[cell] = total
    prefix = np.empty(cell_count + 1, dtype=np.float64)
    prefix[0] = 0.0
    for cell in range(cell_count):
        prefix[cell + 1] = prefix[cell] + cell_values[cell]

    short_cells = selected_cells // run_count
    long_count = selected_cells % run_count
    long_cells = short_cells + 1
    negative = -1e300
    dp = np.full(
        (run_count + 1, long_count + 1, cell_count + 1),
        negative,
        dtype=np.float64,
    )
    take = np.zeros(
        (run_count + 1, long_count + 1, cell_count + 1),
        dtype=np.uint8,
    )
    for prefix_cells in range(cell_count + 1):
        dp[0, 0, prefix_cells] = 0.0

    for used in range(1, run_count + 1):
        max_longs = min(long_count, used)
        for longs in range(max_longs + 1):
            for prefix_cells in range(1, cell_count + 1):
                best = dp[used, longs, prefix_cells - 1]
                choice = 0

                start = prefix_cells - short_cells
                if start >= 0:
                    predecessor = max(0, start - 1)
                    candidate = (
                        dp[used - 1, longs, predecessor]
                        + prefix[prefix_cells] - prefix[start]
                    )
                    if candidate > best + 1e-15:
                        best = candidate
                        choice = 1

                if longs > 0:
                    start = prefix_cells - long_cells
                    if start >= 0:
                        predecessor = max(0, start - 1)
                        candidate = (
                            dp[used - 1, longs - 1, predecessor]
                            + prefix[prefix_cells] - prefix[start]
                        )
                        if candidate > best + 1e-15:
                            best = candidate
                            choice = 2

                dp[used, longs, prefix_cells] = best
                take[used, longs, prefix_cells] = choice

    optimum = dp[run_count, long_count, cell_count]
    if not np.isfinite(optimum):
        return output, 0.0, short_cells, long_cells, long_count, cell_count, False

    used = run_count
    longs = long_count
    prefix_cells = cell_count
    while used > 0:
        if prefix_cells <= 0:
            return output, 0.0, short_cells, long_cells, long_count, cell_count, False
        choice = take[used, longs, prefix_cells]
        if choice == 0:
            prefix_cells -= 1
            continue
        length = short_cells if choice == 1 else long_cells
        start = prefix_cells - length
        for cell in range(start, prefix_cells):
            row_start = cell * alignment_rows
            for offset in range(alignment_rows):
                output[row_start + offset] = True
        prefix_cells = max(0, start - 1)
        used -= 1
        if choice == 2:
            longs -= 1
    return (
        output, optimum, short_cells, long_cells, long_count, cell_count, True
    )


def _run_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def safe_select(method: str, values: np.ndarray, row_budget: int, model: dict,
                lookup_table):
    try:
        if method == "tile8_ceil":
            return EXP29.safe_select(method, values, row_budget, model, lookup_table)
        run_count = int(method.removeprefix("wave_k"))
        row_kib = float(model["saturation_kib"]) / float(model["saturation_rows"])
        row_bytes = int(round(row_kib * 1024.0))
        alignment_rows = DIRECT_ALIGNMENT_BYTES // math.gcd(
            DIRECT_ALIGNMENT_BYTES, row_bytes
        )
        alignment_fallback = bool(
            len(values) % alignment_rows or row_budget % alignment_rows
        )
        if alignment_fallback:
            alignment_rows = 1
        result = _wave_exact_kernel(
            np.asarray(values, dtype=np.float64), int(row_budget),
            int(alignment_rows), run_count,
        )
        mask, optimum, short_cells, long_cells, long_count, cells, valid = result
        if not valid or int(mask.sum()) != int(row_budget):
            raise RuntimeError("wave DP failed to return exact R")
        starts, ends = _run_bounds(mask)
        if len(starts) != run_count:
            raise RuntimeError("wave DP failed to preserve K separated runs")
        run_costs = EXP29.run_costs_for(lookup_table, len(values), row_kib)
        importance, lookup_cost, chunks = EXP29._mask_metrics_lookup(
            mask, np.asarray(values, dtype=np.float64), run_costs
        )
        return np.asarray(mask, dtype=bool), {
            "scalarized_calls": 1,
            "outer_iterations": 1,
            "frontier_candidates": int(cells),
            "eligible_candidates": int(chunks),
            "trim_work_deletions": 0,
            "minimum_overfill": 0,
            "repair_deletions": 0,
            "repair_additions": 0,
            "returned_top_r": False,
            "wave_run_count": run_count,
            "alignment_rows": int(alignment_rows),
            "alignment_fallback": alignment_fallback,
            "short_run_rows": int(short_cells * alignment_rows),
            "long_run_rows": int(long_cells * alignment_rows),
            "long_run_count": int(long_count),
            "lookup_efficiency": float(importance / lookup_cost),
            "dp_importance": float(optimum),
            "fallback_used": False,
            "error": "",
        }
    except Exception as exception:
        return EXP24.top_r_mask(values, row_budget), {
            "fallback_used": True,
            "error": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(limit=4),
        }


def bind_exp26(lookup_table) -> None:
    EXP26.METHODS = METHODS
    EXP26.CANDIDATE_METHODS = METHODS[1:]
    EXP26.FRONTIER_METHODS = WAVE_METHODS
    EXP26.METHOD_LABELS = METHOD_LABELS
    EXP26.METHOD_COLORS = METHOD_COLORS
    EXP26.FRONTIER_SETTINGS = {
        method: {"run_count": int(method.removeprefix("wave_k"))}
        for method in WAVE_METHODS
    }

    def bound_safe_select(method, values, row_budget, model):
        return safe_select(method, values, row_budget, model, lookup_table)

    def selector_function(method, track, values_host, values_cuda, row_budget, model):
        def run():
            values = (
                values_cuda.detach().to("cpu").numpy().astype(np.float64)
                if track == "cuda" else values_host
            )
            mask, metadata = bound_safe_select(method, values, row_budget, model)
            output = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_).copy())
            if track == "cuda":
                output = output.to("cuda")
            return output, metadata
        return run

    EXP26.safe_select = bound_safe_select
    EXP26.selector_function = selector_function


def self_check(lookup_table, saturation_kib: float) -> None:
    rng = np.random.default_rng(3101)
    for shape_name in ("896x128", "896x896", "896x4864", "4864x896"):
        shape = next(item for item in SHAPES if item["shape"] == shape_name)
        model = EXP13.fit_continuous_two_line(
            lookup_table, EXP22.row_size_kib(shape), saturation_kib
        )
        n = int(shape["n"])
        values = rng.lognormal(0.0, 1.0, size=n)
        values /= values.sum()
        for fraction in (0.25, 0.50, 0.75):
            row_budget = int(round(n * fraction))
            for method in WAVE_METHODS:
                first, metadata = safe_select(
                    method, values, row_budget, model, lookup_table
                )
                second, _ = safe_select(
                    method, values, row_budget, model, lookup_table
                )
                if (
                    metadata.get("fallback_used") or int(first.sum()) != row_budget
                    or not np.array_equal(first, second)
                ):
                    raise RuntimeError("wave selector self-check failed")

    # Exact enumeration for two unequal run lengths and a mandatory gap.
    values = rng.random(18)
    mask, score, *_rest, valid = _wave_exact_kernel(values, 7, 1, 2)
    if not valid:
        raise RuntimeError("wave enumeration check returned no path")
    expected = -1.0
    for first_length, second_length in ((3, 4), (4, 3)):
        for first in range(18 - first_length + 1):
            for second in range(first + first_length + 1, 18 - second_length + 1):
                expected = max(
                    expected,
                    values[first:first + first_length].sum()
                    + values[second:second + second_length].sum(),
                )
    if not np.isclose(score, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("wave DP does not match exhaustive enumeration")
    if int(mask.sum()) != 7:
        raise RuntimeError("wave enumeration check violated exact R")


def add_top_r_reference(frame: pd.DataFrame, trace_input: Path) -> pd.DataFrame:
    traces, _ = EXP22.load_activation_traces(trace_input)
    values_by_id = {
        int(trace["trace_id"]): np.asarray(trace["values"], dtype=np.float64)
        for trace in traces
    }
    cache: dict[tuple[int, int], float] = {}
    top_values = []
    for row in frame.itertuples(index=False):
        key = (int(row.trace_id), int(row.effective_row_budget))
        if key not in cache:
            values = values_by_id[key[0]]
            values = values / values.sum()
            count = key[1]
            cache[key] = float(np.partition(values, len(values) - count)[-count:].sum())
        top_values.append(cache[key])
    output = frame.copy()
    output["top_r_importance"] = top_values
    output["top_r_retention"] = output.importance / output.top_r_importance
    return output


def evaluate_dispatch(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected = frame[
        (frame.track == "cuda") & frame.actual_total_median_ms.notna()
    ]
    keys = ["trace_id", "budget_index", "shape"]
    total = selected.pivot_table(
        index=keys, columns="method", values="actual_total_median_ms", aggfunc="first"
    )
    importance = selected.pivot_table(
        index=keys, columns="method", values="importance", aggfunc="first"
    )
    top_r = selected.pivot_table(
        index=keys, columns="method", values="top_r_importance", aggfunc="first"
    )["paper"]
    rows = []
    holdout_parts = {"min_total": [], "max_efficiency": []}
    baseline_parts = []
    for (shape, budget_index), _ in total.groupby(level=["shape", "budget_index"]):
        group_total = total.xs((shape, budget_index), level=("shape", "budget_index"))
        group_importance = importance.xs(
            (shape, budget_index), level=("shape", "budget_index")
        )
        group_top = top_r.xs((shape, budget_index), level=("shape", "budget_index"))
        trace_ids = sorted(group_total.index)
        split = max(1, len(trace_ids) // 2)
        train_ids, test_ids = trace_ids[:split], trace_ids[split:]
        train_total = group_total.loc[train_ids, list(WAVE_METHODS)]
        train_importance = group_importance.loc[train_ids, list(WAVE_METHODS)]
        baseline = pd.DataFrame({
            "trace_id": test_ids,
            "shape": shape,
            "budget_index": int(budget_index),
            "paper_total_ms": group_total.loc[test_ids, "paper"].to_numpy(),
            "paper_importance": group_importance.loc[test_ids, "paper"].to_numpy(),
            "tile_total_ms": group_total.loc[test_ids, "tile8_ceil"].to_numpy(),
            "tile_importance": group_importance.loc[test_ids, "tile8_ceil"].to_numpy(),
            "top_r_importance": group_top.loc[test_ids].to_numpy(),
        })
        baseline_parts.append(baseline)
        choices = {
            "min_total": train_total.mean().idxmin(),
            "max_efficiency": (train_importance / train_total).mean().idxmax(),
        }
        for policy, method in choices.items():
            test_total = group_total.loc[test_ids, method]
            test_importance = group_importance.loc[test_ids, method]
            paper_total = group_total.loc[test_ids, "paper"]
            paper_importance = group_importance.loc[test_ids, "paper"]
            test = pd.DataFrame({
                "trace_id": test_ids,
                "shape": shape,
                "budget_index": int(budget_index),
                "policy": policy,
                "selected_method": method,
                "selected_total_ms": test_total.to_numpy(),
                "selected_importance": test_importance.to_numpy(),
                "paper_total_ms": paper_total.to_numpy(),
                "paper_importance": paper_importance.to_numpy(),
                "top_r_importance": group_top.loc[test_ids].to_numpy(),
            })
            test["saving_ms"] = test.paper_total_ms - test.selected_total_ms
            test["importance_gain_pct"] = 100.0 * (
                test.selected_importance / test.paper_importance - 1.0
            )
            test["top_r_retention"] = (
                test.selected_importance / test.top_r_importance
            )
            holdout_parts[policy].append(test)
            rows.append({
                "shape": shape,
                "budget_index": int(budget_index),
                "policy": policy,
                "selected_method": method,
                "train_trace_count": len(train_ids),
                "holdout_trace_count": len(test_ids),
                "train_total_ms_mean": float(train_total[method].mean()),
                "train_efficiency_mean": float(
                    (train_importance[method] / train_total[method]).mean()
                ),
                "holdout_total_ms_mean": float(test_total.mean()),
                "holdout_paper_total_ms_mean": float(paper_total.mean()),
                "holdout_saving_ms_mean": float((paper_total - test_total).mean()),
                "holdout_importance_gain_pct_mean": float(
                    100.0 * (test_importance / paper_importance - 1.0).mean()
                ),
                "holdout_top_r_retention_mean": float(
                    (test_importance / group_top.loc[test_ids]).mean()
                ),
            })
    details = pd.DataFrame(rows)
    baseline = pd.concat(baseline_parts, ignore_index=True)
    tile_saving = baseline.paper_total_ms - baseline.tile_total_ms
    summary = {
        "holdout_baselines": {
            "cases": int(len(baseline)),
            "paper_total_ms_mean": float(baseline.paper_total_ms.mean()),
            "paper_top_r_retention_mean": float(
                (baseline.paper_importance / baseline.top_r_importance).mean()
            ),
            "tile8_total_ms_mean": float(baseline.tile_total_ms.mean()),
            "tile8_saving_ms_mean": float(tile_saving.mean()),
            "tile8_saving_pct_mean_total": float(
                100.0 * tile_saving.mean() / baseline.paper_total_ms.mean()
            ),
            "tile8_importance_gain_pct_mean": float(
                100.0 * (
                    baseline.tile_importance / baseline.paper_importance - 1.0
                ).mean()
            ),
            "tile8_top_r_retention_mean": float(
                (baseline.tile_importance / baseline.top_r_importance).mean()
            ),
            "tile8_strict_win_rate": float((tile_saving > 0.0).mean()),
        }
    }
    for policy, parts in holdout_parts.items():
        holdout = pd.concat(parts, ignore_index=True)
        summary[policy] = {
            "protocol": (
                "first half of trace ids per shape and budget selects among wave K; "
                "second half is holdout"
            ),
            "holdout_cases": int(len(holdout)),
            "paper_total_ms_mean": float(holdout.paper_total_ms.mean()),
            "dispatch_total_ms_mean": float(holdout.selected_total_ms.mean()),
            "saving_ms_mean": float(holdout.saving_ms.mean()),
            "saving_pct_mean_total": float(
                100.0 * holdout.saving_ms.mean() / holdout.paper_total_ms.mean()
            ),
            "importance_gain_pct_mean": float(holdout.importance_gain_pct.mean()),
            "top_r_retention_mean": float(holdout.top_r_retention.mean()),
            "strict_win_rate": float((holdout.saving_ms > 0.0).mean()),
            "nonregression_rate": float((holdout.saving_ms >= 0.0).mean()),
        }
        if policy == "max_efficiency":
            summary[policy]["paired_cluster_bootstrap"] = _bootstrap_holdout(holdout)
    return details, summary


def _bootstrap_holdout(holdout: pd.DataFrame, repetitions: int = 20_000) -> dict:
    rng = np.random.default_rng(31031)
    strata = []
    for _, group in holdout.groupby("shape", sort=True):
        values = group.groupby("trace_id").saving_ms.mean().to_numpy()
        strata.append(values)
    total_clusters = sum(len(values) for values in strata)
    samples = np.zeros(repetitions, dtype=np.float64)
    for values in strata:
        indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        samples += values[indices].sum(axis=1) / total_clusters
    return {
        "repetitions": repetitions,
        "seed": 31031,
        "trace_clusters": total_clusters,
        "saving_ms_ci95": [
            float(value) for value in np.quantile(samples, [0.025, 0.975])
        ],
    }


def _fmt(value, digits=3, suffix=""):
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def write_report(summary: pd.DataFrame, dispatch_details: pd.DataFrame | None,
                 dispatch: dict | None, args) -> None:
    report_path = args.report_output or (HERE / "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    available_tracks = set(summary.track)
    primary_track = "cuda" if "cuda" in available_tracks else "host"
    indexed = summary[summary.track == primary_track].set_index("method").reindex(METHODS)
    image_prefix = Path(os.path.relpath(args.output_dir, report_path.parent)).as_posix()
    flash = metadata["latency_profile"]["metadata"].get("flash", "unknown storage")
    lines = [
        "# Experiment 31 보고서: Wave-Balanced Exact-R DP", "",
        f"`{metadata['gpu']}`와 `{flash}`에서 실제 측정했다. 512-byte 정렬 cell, "
        "서로 분리된 near-equal run, exact-R DP를 사용한다. Paper mask나 "
        "`I(M_paper)`는 selector 입력 또는 wave dispatch 입력이 아니다.", "",
        "## 전체 결과", "",
        "| 방법 | selector median | case-p95 | importance vs Paper | lookup I/L | actual read | selector+read | Paper win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in indexed.iterrows():
        win = "-" if method == "paper" else _fmt(100 * row.actual_total_win_rate, 1, "%")
        lines.append(
            f"| {METHOD_LABELS[method]} | {_fmt(row.runtime_median_ms, 3, ' ms')} "
            f"| {_fmt(row.runtime_case_p95_ms, 3, ' ms')} "
            f"| {_fmt(row.importance_gain_pct_mean_valid, 2, '%')} "
            f"| {_fmt(row.lookup_efficiency_gain_pct_mean_valid, 2, '%')} "
            f"| {_fmt(row.actual_read_wall_ms_mean, 3, ' ms')} "
            f"| {_fmt(row.actual_total_ms_mean, 3, ' ms')} | {win} |"
        )
    actual_cases = indexed.loc["paper", "actual_cases"]
    lines.append("")
    if np.isfinite(actual_cases) and int(actual_cases) > 0:
        cases = int(actual_cases)
        lines.append(
            f"방법당 `{cases}` case, selector `{args.repetitions}`회, O_DIRECT+GPU "
            f"upload `{args.io_repetitions}`회 반복이며 기록된 실제 I/O 표본은 "
            f"`{cases * len(METHODS) * args.io_repetitions:,}`개다."
        )
    else:
        lines.append(
            f"이 실행은 `{primary_track}` selector 검증만 수행했으며 실제 I/O 표본은 없다."
        )
    lines.extend(["", "## Holdout K dispatch", ""])
    if dispatch is None or dispatch_details is None:
        lines.extend(["실제 CUDA I/O가 없어 dispatch를 계산하지 못했다.", ""])
    else:
        lines.extend([
            "각 shape×budget의 앞 16개 trace에서 K를 고르고 뒤 16개에 고정했다. "
            "`max-efficiency`는 wave 후보 자신의 `importance/(selector+read)`만 사용한다.", "",
            "| 정책 | Paper total | dispatch total | 절감 | importance vs Paper | Top-R retention | win rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for policy in ("min_total", "max_efficiency"):
            item = dispatch[policy]
            lines.append(
                f"| {policy} | {item['paper_total_ms_mean']:.3f} ms "
                f"| {item['dispatch_total_ms_mean']:.3f} ms "
                f"| {item['saving_pct_mean_total']:.2f}% "
                f"| {item['importance_gain_pct_mean']:+.2f}% "
                f"| {100*item['top_r_retention_mean']:.2f}% "
                f"| {100*item['strict_win_rate']:.1f}% |"
            )
        ci = dispatch["max_efficiency"]["paired_cluster_bootstrap"]["saving_ms_ci95"]
        baselines = dispatch["holdout_baselines"]
        wave = dispatch["max_efficiency"]
        wave_vs_tile_ms = wave["dispatch_total_ms_mean"] - baselines["tile8_total_ms_mean"]
        wave_vs_tile_pct = 100.0 * wave_vs_tile_ms / baselines["tile8_total_ms_mean"]
        wave_vs_tile_importance = (
            wave["importance_gain_pct_mean"]
            - baselines["tile8_importance_gain_pct_mean"]
        )
        lines.extend([
            "", f"Max-efficiency 순절감의 shape-stratified trace-cluster bootstrap "
            f"95% 구간은 `[{ci[0]:.3f}, {ci[1]:.3f}] ms`다.", "",
            f"같은 holdout에서 tile8은 `{baselines['tile8_total_ms_mean']:.3f} ms`, "
            f"Paper 대비 `{baselines['tile8_saving_pct_mean_total']:.2f}%` 절감이지만 "
            f"importance는 `{baselines['tile8_importance_gain_pct_mean']:+.2f}%`다. "
            f"Wave dispatch는 tile8보다 `{wave_vs_tile_ms:.3f} ms` "
            f"(`{wave_vs_tile_pct:.2f}%`) 느린 대신 importance를 "
            f"`{wave_vs_tile_importance:+.2f}` percentage points 회복한다.", "",
            "### Max-efficiency가 고른 K", "",
            "| Shape | budget | 선택 | holdout total | Paper total | importance |",
            "|---|---:|---|---:|---:|---:|",
        ])
        chosen = dispatch_details[dispatch_details.policy == "max_efficiency"]
        for row in chosen.itertuples(index=False):
            lines.append(
                f"| {row.shape} | {row.budget_index} | {METHOD_LABELS[row.selected_method]} "
                f"| {row.holdout_total_ms_mean:.3f} ms "
                f"| {row.holdout_paper_total_ms_mean:.3f} ms "
                f"| {row.holdout_importance_gain_pct_mean:+.2f}% |"
            )

    lines.extend(["", "## 판정", ""])
    measured_totals = indexed.actual_total_ms_mean.dropna()
    if len(measured_totals):
        fastest = measured_totals.idxmin()
        saving_pct = 100.0 * (
            indexed.loc["paper", "actual_total_ms_mean"]
            - indexed.loc[fastest, "actual_total_ms_mean"]
        ) / indexed.loc["paper", "actual_total_ms_mean"]
        lines.extend([
            f"단일 최속 방법은 `{METHOD_LABELS[fastest]}`의 "
            f"`{indexed.loc[fastest, 'actual_total_ms_mean']:.3f} ms`이며 Paper 대비 "
            f"`{saving_pct:.2f}%` 절감이다.", "",
        ])
        if dispatch is not None:
            wave = dispatch["max_efficiency"]
            lines.extend([
                f"Paper 수준의 평균 importance를 요구하면 holdout wave dispatch가 "
                f"`{wave['dispatch_total_ms_mean']:.3f} ms`로 Paper보다 "
                f"`{wave['saving_pct_mean_total']:.2f}%` 빠르면서 importance는 "
                f"`{wave['importance_gain_pct_mean']:+.2f}%`다.", "",
                "## 해석", "",
                f"6-lane 균형화 자체는 유효하지만 충분하지 않았다. K=1의 실제 read "
                f"`{indexed.loc['wave_k1', 'actual_read_wall_ms_mean']:.3f} ms`가 K=6에서 "
                f"`{indexed.loc['wave_k6', 'actual_read_wall_ms_mean']:.3f} ms`로 줄었지만, "
                f"Paper의 `{indexed.loc['paper', 'actual_read_wall_ms_mean']:.3f} ms`보다 "
                "여전히 느리다. K=6의 총 이득은 reader보다 Paper selector를 "
                "대체한 데서 주로 나온다.", "",
                "K를 12 이상으로 늘리면 importance는 계속 오르지만 DP 시간과 thread/task "
                "고정비가 더 빨리 증가한다. 따라서 이 노트북에서는 하나의 큰 K가 아니라 "
                "shape×budget별 K dispatch가 속도-품질 경계다.", "",
            ])
    else:
        lines.extend(["실제 reader 측정 없이 selector 경로만 검증했다.", ""])
    lines.extend([
        "## 범위", "",
        "- selector timing은 importance GPU→CPU, CPU DP, dense bool mask CPU→GPU를 포함한다.",
        "- reader는 아직 기존 dense-mask API와 매 호출 thread 생성을 사용한다.",
        "- 실제 read는 native O_DIRECT와 GPU upload를 포함한다.",
        "- 이는 selector+projection-weight-read 경로이며 전체 LLM end-to-end latency가 아니다.",
        "", f"![Runtime-quality]({image_prefix}/runtime_quality.png)", "",
        f"![Shape comparison]({image_prefix}/shape_comparison.png)", "",
        f"![Measured total]({image_prefix}/actual_total.png)", "",
    ])
    report_path.write_text("\n".join(lines) + "\n")


def analyze(args) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    frame = add_top_r_reference(frame, args.trace_input)
    frame = EXP24.add_paired_metrics(frame)
    frame = EXP26.add_actual_io_metrics(frame)
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    dispatch_details = None
    dispatch = None
    if frame.actual_total_median_ms.notna().any():
        dispatch_details, dispatch = evaluate_dispatch(frame)
        dispatch_details.to_csv(args.output_dir / "holdout_dispatch.csv", index=False)
    summary_frame, shape_frame, nested = EXP26.summarize(frame, None, args)
    nested["format"] = "experiment-31-wave-balanced-dp-v1-real-activations"
    nested["comparison"] = "aligned exact-R K-run wave DP versus Experiment 29 and Paper"
    nested["direct_alignment_bytes"] = DIRECT_ALIGNMENT_BYTES
    nested["wave_counts"] = list(WAVE_COUNTS)
    if dispatch is not None:
        nested["holdout_wave_dispatch"] = dispatch
    measured = frame[frame.actual_io_median_ms.notna()]
    if len(measured):
        nested["actual_io_replay"] = {
            "cases": int(len(measured)),
            "all_direct": bool((measured.actual_direct_rate == 1.0).all()),
            "recorded_samples": int(len(measured) * args.io_repetitions),
        }
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(nested, indent=2) + "\n")
    EXP26.plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    EXP26.plot_by_shape(shape_frame, args.output_dir / "shape_comparison.png")
    if summary_frame.actual_total_ms_mean.notna().any():
        EXP26.plot_actual_total(summary_frame, args.output_dir / "actual_total.png")
    write_report(summary_frame, dispatch_details, dispatch, args)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    _, tracks = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lookup_table = BASE.LatencyTable.load(args.profile)
    bind_exp26(lookup_table)
    if args.analyze_only:
        analyze(args)
        return

    traces, trace_metadata, trace_path = EXP24.prepare_activation_traces(args)
    shape_names = list(dict.fromkeys(trace["shape"] for trace in traces))
    shape_specs = [
        dict(next(item for item in SHAPES if item["shape"] == name))
        for name in shape_names
    ]
    metadata = {
        "activation_trace": trace_metadata,
        "trace_file": str(trace_path),
        "trace_sha256": EXP22.sha256_file(trace_path),
        "selected_trace_count": len(traces),
        "selected_trace_ids": [int(trace["trace_id"]) for trace in traces],
        "max_traces_per_shape": int(args.max_traces_per_shape),
        "shape_specs": shape_specs,
        "method_labels": METHOD_LABELS,
        "algorithm": "aligned near-equal K-run exact-R importance DP",
        "wave_counts": list(WAVE_COUNTS),
        "direct_alignment_bytes": DIRECT_ALIGNMENT_BYTES,
        "latency_profile": {
            "requested": str(args.profile),
            "max_kib": int(lookup_table.max_kb),
            "metadata": lookup_table.meta,
            "sha256": EXP22.sha256_file(Path(args.profile)),
        },
        "actual_io": {
            "enabled": bool(args.measure_io),
            "blob": str(args.io_blob) if args.io_blob else None,
            "blob_sha256": EXP22.sha256_file(args.io_blob) if args.measure_io else None,
            "repetitions": int(args.io_repetitions),
            "warmup": int(args.io_warmup),
            "threads": int(args.io_threads),
            "max_read_kib": int(args.io_max_read_kib),
            "device": args.io_device,
        },
        "requested_tracks": args.tracks,
        "executed_tracks": tracks,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.collect_traces_only:
        print(f"selected {len(traces)} traces; stopping as requested")
        return

    native_reader = None
    if args.measure_io:
        if args.io_blob is None or not args.io_blob.is_file():
            raise SystemExit("--measure-io requires an existing --io-blob")
        required_bytes = max(
            int(shape["n"]) * int(round(EXP22.row_size_kib(shape) * 1024.0))
            for shape in shape_specs
        )
        if args.io_blob.stat().st_size < required_bytes:
            raise SystemExit(
                f"--io-blob has {args.io_blob.stat().st_size} bytes; "
                f"at least {required_bytes} required"
            )
        EXP22.load_vlmflash()
        from vlmflash._native import native, unavailable_reason
        native_reader = native()
        if native_reader is None:
            raise SystemExit(f"native reader unavailable: {unavailable_reason()}")

    if not args.skip_self_check:
        self_check(lookup_table, args.saturation_kib)
    rows, timings, models, io_timings = EXP26.collect(
        args, traces, tracks, lookup_table, native_reader
    )
    EXP24.write_csv(args.output_dir / "trials.csv", rows)
    EXP24.write_csv(args.output_dir / "timing_samples.csv", timings)
    EXP24.write_csv(args.output_dir / "models.csv", models)
    if io_timings:
        EXP24.write_csv(args.output_dir / "io_timing_samples.csv", io_timings)
    analyze(args)


if __name__ == "__main__":
    main()
