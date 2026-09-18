#!/usr/bin/env python3
"""Experiment 29: fixed eight-cell saturation tiles on the laptop."""

from __future__ import annotations

import argparse
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
EXPERIMENT_28 = PROJECT_ROOT / "experiments" / "28_exact_corridor" / "run_experiment.py"
LOCAL_PROFILE = (
    PROJECT_ROOT / "experiments" / "26_frontier_adaptive_trim" / "results_laptop"
    / "laptop_sn850x_profile.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP28 = _load_module("experiment_28_for_29", EXPERIMENT_28)
EXP26 = EXP28.EXP26
EXP24 = EXP28.EXP24
EXP22 = EXP28.EXP22
EXP18 = EXP28.EXP18
EXP13 = EXP28.EXP13
BASE = EXP28.BASE
njit = EXP28.njit
SHAPES = EXP28.SHAPES
DEFAULT_TRACE_INPUT = EXP28.DEFAULT_TRACE_INPUT

METHODS = (
    "paper",
    "top_r",
    "tile8_floor",
    "tile8_round",
    "tile8_ceil",
)
TILE_METHODS = METHODS[2:]
METHOD_MODES = {"tile8_floor": 0, "tile8_round": 1, "tile8_ceil": 2}
METHOD_LABELS = {
    "paper": "Paper",
    "top_r": "Top-R",
    "tile8_floor": "8-cell tile, floor(s/8)",
    "tile8_round": "8-cell tile, round(s/8)",
    "tile8_ceil": "8-cell tile, ceil(s/8)",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "top_r": "#64748B",
    "tile8_floor": "#7C3AED",
    "tile8_round": "#2563EB",
    "tile8_ceil": "#0F766E",
}

# The throwaway Python-reference probe performed immediately before creating
# this experiment.  It used all 384 paired Experiment-28 cases, selected the
# better of floor-expand and ceil-trim, and evaluated the laptop lookup table.
INTERNAL_PROBE = {
    "cases": 384,
    "description": "Python reference, cell_rows=ceil(s/8), model-side only",
    "overall": {
        "lookup_efficiency_gain_pct_mean": 4.331352,
        "lookup_efficiency_win_rate": 0.744792,
        "importance_gain_pct_mean": -1.811752,
        "lookup_saving_pct_mean": 5.703157,
        "chunks_mean": 3.158854,
    },
    "by_shape": {
        "896x896": {"gain_pct": 6.991116, "saving_pct": 9.528241,
                     "importance_pct": -3.473693, "win_rate": 0.9375},
        "896x128": {"gain_pct": -1.195170, "saving_pct": 0.0,
                     "importance_pct": -1.195170, "win_rate": 0.041667},
        "896x4864": {"gain_pct": 4.735130, "saving_pct": 6.320381,
                      "importance_pct": -1.915070, "win_rate": 1.0},
        "4864x896": {"gain_pct": 6.794333, "saving_pct": 6.964005,
                      "importance_pct": -0.663074, "win_rate": 1.0},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=None)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--model", default=EXP22.DEFAULT_MODEL)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trace-input", type=Path, default=DEFAULT_TRACE_INPUT)
    parser.add_argument("--capture-traces", action="store_true")
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--max-traces-per-shape", type=int, default=32)
    parser.add_argument("--collect-traces-only", action="store_true")
    parser.add_argument(
        "--row-budget-fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument(
        "--tracks", nargs="+", choices=("host", "cuda"), default=["host", "cuda"]
    )
    parser.add_argument("--profile", default=str(LOCAL_PROFILE))
    parser.add_argument("--saturation-kib", type=float, default=240.0)
    parser.add_argument("--deadline-ms", type=float, default=2.0)
    parser.add_argument("--paper-impl", choices=("auto", "native", "torch"), default="native")
    parser.add_argument("--oracle-n", type=int, default=18)
    parser.add_argument("--oracle-trials", type=int, default=10)
    parser.add_argument("--oracle-cv", type=float, default=3.30)
    parser.add_argument("--seed", type=int, default=20262601)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--measure-io", action="store_true")
    parser.add_argument("--io-blob", type=Path)
    parser.add_argument("--io-repetitions", type=int, default=30)
    parser.add_argument("--io-warmup", type=int, default=3)
    parser.add_argument("--io-threads", type=int, default=6)
    parser.add_argument("--io-max-read-kib", type=int, default=768)
    parser.add_argument("--io-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    shapes, tracks = EXP26.validate_args(args)
    if not Path(args.profile).is_file():
        raise SystemExit(f"laptop latency profile does not exist: {args.profile}")
    return shapes, tracks


@njit(cache=False, inline="always")
def _mask_metrics_lookup(mask, values, run_costs):
    importance = 0.0
    cost = 0.0
    chunks = 0
    run = 0
    for index in range(len(mask)):
        if mask[index]:
            importance += values[index]
            run += 1
        elif run:
            cost += run_costs[run]
            chunks += 1
            run = 0
    if run:
        cost += run_costs[run]
        chunks += 1
    return importance, cost, chunks


@njit(cache=False)
def _best_k_tiles(values, tile_rows, cell_rows, tile_count):
    n = len(values)
    output = np.zeros(n, dtype=np.bool_)
    if tile_count == 0:
        return output, 0, True
    if tile_rows > n:
        return output, 0, False
    candidate_count = (n - tile_rows) // cell_rows + 1
    if candidate_count <= 0:
        return output, 0, False
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
    weights = np.empty(candidate_count, dtype=np.float64)
    for candidate in range(candidate_count):
        start = candidate * cell_rows
        weights[candidate] = prefix[start + tile_rows] - prefix[start]

    negative = -1e300
    dp = np.full((tile_count + 1, candidate_count + 1), negative, dtype=np.float64)
    take = np.zeros((tile_count + 1, candidate_count + 1), dtype=np.uint8)
    for candidate_prefix in range(candidate_count + 1):
        dp[0, candidate_prefix] = 0.0
    for count in range(1, tile_count + 1):
        for candidate_prefix in range(1, candidate_count + 1):
            candidate = candidate_prefix - 1
            skip_score = dp[count, candidate_prefix - 1]
            predecessor = candidate_prefix - 8
            if predecessor < 0:
                predecessor = 0
            take_score = dp[count - 1, predecessor] + weights[candidate]
            if take_score > skip_score:
                dp[count, candidate_prefix] = take_score
                take[count, candidate_prefix] = 1
            else:
                dp[count, candidate_prefix] = skip_score
    if not np.isfinite(dp[tile_count, candidate_count]):
        return output, candidate_count, False

    count = tile_count
    candidate_prefix = candidate_count
    while count > 0:
        if candidate_prefix <= 0:
            return np.zeros(n, dtype=np.bool_), candidate_count, False
        if take[count, candidate_prefix]:
            candidate = candidate_prefix - 1
            start = candidate * cell_rows
            for index in range(start, start + tile_rows):
                output[index] = True
            count -= 1
            candidate_prefix -= 8
            if candidate_prefix < 0:
                candidate_prefix = 0
        else:
            candidate_prefix -= 1
    return output, candidate_count, True


@njit(cache=False)
def _best_contiguous(values, row_budget, stride):
    n = len(values)
    output = np.zeros(n, dtype=np.bool_)
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
    best_start = 0
    best_value = -1.0
    start = 0
    while start + row_budget <= n:
        candidate = prefix[start + row_budget] - prefix[start]
        if candidate > best_value:
            best_value = candidate
            best_start = start
        start += stride
    for index in range(best_start, best_start + row_budget):
        output[index] = True
    return output


@njit(cache=False)
def _extract_runs(mask, left, right):
    count = 0
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        left[count] = index
        while index + 1 < len(mask) and mask[index + 1]:
            index += 1
        right[count] = index
        count += 1
        index += 1
    return count


@njit(cache=False)
def _expand_endpoints(mask, values, row_budget, run_costs):
    output = mask.copy()
    n = len(output)
    left = np.empty(n, dtype=np.int64)
    right = np.empty(n, dtype=np.int64)
    run_count = _extract_runs(output, left, right)
    importance, cost, _ = _mask_metrics_lookup(output, values, run_costs)
    selected = 0
    for value in output:
        selected += int(value)
    additions = 0
    while selected < row_budget:
        best_ratio = -np.inf
        best_index = -1
        best_left_run = -1
        best_right_run = -1
        for run_index in range(run_count):
            for side in range(2):
                index = left[run_index] - 1 if side == 0 else right[run_index] + 1
                if index < 0 or index >= n or output[index]:
                    continue
                left_run = -1
                right_run = -1
                for other in range(run_count):
                    if right[other] == index - 1:
                        left_run = other
                    if left[other] == index + 1:
                        right_run = other
                left_length = 0 if left_run < 0 else right[left_run] - left[left_run] + 1
                right_length = 0 if right_run < 0 else right[right_run] - left[right_run] + 1
                candidate_cost = (
                    cost - run_costs[left_length] - run_costs[right_length]
                    + run_costs[left_length + 1 + right_length]
                )
                candidate_ratio = (importance + values[index]) / candidate_cost
                if candidate_ratio > best_ratio + 1e-15 or (
                    abs(candidate_ratio - best_ratio) <= 1e-15
                    and (best_index < 0 or index < best_index)
                ):
                    best_ratio = candidate_ratio
                    best_index = index
                    best_left_run = left_run
                    best_right_run = right_run
        if best_index < 0:
            return output, additions, False
        output[best_index] = True
        importance += values[best_index]
        left_length = (
            0 if best_left_run < 0
            else right[best_left_run] - left[best_left_run] + 1
        )
        right_length = (
            0 if best_right_run < 0
            else right[best_right_run] - left[best_right_run] + 1
        )
        cost = (
            cost - run_costs[left_length] - run_costs[right_length]
            + run_costs[left_length + 1 + right_length]
        )
        if best_left_run >= 0 and best_right_run >= 0:
            right[best_left_run] = right[best_right_run]
            for other in range(best_right_run, run_count - 1):
                left[other] = left[other + 1]
                right[other] = right[other + 1]
            run_count -= 1
        elif best_left_run >= 0:
            right[best_left_run] += 1
        elif best_right_run >= 0:
            left[best_right_run] -= 1
        else:
            return output, additions, False
        selected += 1
        additions += 1
    return output, additions, True


@njit(cache=False)
def _trim_endpoints(mask, values, row_budget, run_costs):
    output = mask.copy()
    n = len(output)
    left = np.empty(n, dtype=np.int64)
    right = np.empty(n, dtype=np.int64)
    run_count = _extract_runs(output, left, right)
    importance, cost, _ = _mask_metrics_lookup(output, values, run_costs)
    selected = 0
    for value in output:
        selected += int(value)
    deletions = 0
    while selected > row_budget:
        best_ratio = -np.inf
        best_index = -1
        best_run = -1
        best_cost = cost
        for run_index in range(run_count):
            length = right[run_index] - left[run_index] + 1
            candidate_cost = cost - run_costs[length] + run_costs[length - 1]
            for side in range(2):
                index = left[run_index] if side == 0 else right[run_index]
                if side == 1 and index == left[run_index]:
                    continue
                candidate_ratio = (importance - values[index]) / candidate_cost
                if candidate_ratio > best_ratio + 1e-15 or (
                    abs(candidate_ratio - best_ratio) <= 1e-15
                    and (best_index < 0 or index < best_index)
                ):
                    best_ratio = candidate_ratio
                    best_index = index
                    best_run = run_index
                    best_cost = candidate_cost
        if best_index < 0:
            return output, deletions, False
        output[best_index] = False
        importance -= values[best_index]
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
        selected -= 1
        deletions += 1
    return output, deletions, True


@njit(cache=False)
def _select_tile8_kernel(values, row_budget, cell_rows, run_costs):
    n = len(values)
    tile_rows = 8 * cell_rows
    if tile_rows > n or row_budget < tile_rows:
        mask = _best_contiguous(values, row_budget, cell_rows)
        importance, cost, chunks = _mask_metrics_lookup(mask, values, run_costs)
        return mask, 0, tile_rows, 0, 0, chunks, importance / cost, True

    floor_count = row_budget // tile_rows
    floor_mask, candidate_count, valid = _best_k_tiles(
        values, tile_rows, cell_rows, floor_count
    )
    if not valid:
        return np.zeros(n, dtype=np.bool_), candidate_count, tile_rows, 0, 0, 0, 0.0, False
    floor_mask, additions, valid = _expand_endpoints(
        floor_mask, values, row_budget, run_costs
    )
    if not valid:
        return floor_mask, candidate_count, tile_rows, 0, additions, 0, 0.0, False
    floor_importance, floor_cost, floor_chunks = _mask_metrics_lookup(
        floor_mask, values, run_costs
    )
    best_mask = floor_mask
    best_ratio = floor_importance / floor_cost
    strategy = 1
    repair = additions
    chunks = floor_chunks

    ceil_count = (row_budget + tile_rows - 1) // tile_rows
    if ceil_count != floor_count and ceil_count * tile_rows <= n:
        ceil_mask, _, ceil_valid = _best_k_tiles(
            values, tile_rows, cell_rows, ceil_count
        )
        if ceil_valid:
            ceil_mask, deletions, ceil_valid = _trim_endpoints(
                ceil_mask, values, row_budget, run_costs
            )
            if ceil_valid:
                ceil_importance, ceil_cost, ceil_chunks = _mask_metrics_lookup(
                    ceil_mask, values, run_costs
                )
                ceil_ratio = ceil_importance / ceil_cost
                if ceil_ratio > best_ratio + 1e-15:
                    best_mask = ceil_mask
                    best_ratio = ceil_ratio
                    strategy = 2
                    repair = deletions
                    chunks = ceil_chunks
    return (
        best_mask, candidate_count, tile_rows, strategy, repair, chunks,
        best_ratio, True,
    )


_RUN_COST_CACHE: dict[tuple, np.ndarray] = {}


def run_costs_for(lookup_table, n: int, row_kib: float) -> np.ndarray:
    table_items = tuple(sorted(lookup_table.as_dict().items()))
    key = (n, round(row_kib, 12), hash(table_items))
    cached = _RUN_COST_CACHE.get(key)
    if cached is not None:
        return cached
    costs = np.zeros(n + 1, dtype=np.float64)
    for length in range(1, n + 1):
        costs[length] = lookup_table.read_ms(length * row_kib)
    _RUN_COST_CACHE[key] = costs
    return costs


def cell_rows_for(model: dict, mode: int) -> int:
    raw = float(model["saturation_rows"]) / 8.0
    if mode == 0:
        return max(1, int(math.floor(raw)))
    if mode == 1:
        return max(1, int(math.floor(raw + 0.5)))
    return max(1, int(math.ceil(raw)))


def safe_select(method: str, values: np.ndarray, row_budget: int, model: dict,
                lookup_table):
    try:
        if method == "top_r":
            return EXP24.top_r_mask(values, row_budget), {
                "fallback_used": False, "error": "",
            }
        mode = METHOD_MODES[method]
        cell_rows = cell_rows_for(model, mode)
        row_kib = float(model["saturation_kib"]) / float(model["saturation_rows"])
        run_costs = run_costs_for(lookup_table, len(values), row_kib)
        mask, candidates, tile_rows, strategy, repair, chunks, ratio, valid = (
            _select_tile8_kernel(
                np.asarray(values, dtype=np.float64), int(row_budget),
                int(cell_rows), run_costs,
            )
        )
        if not valid or int(mask.sum()) != int(row_budget):
            raise RuntimeError("tile selector failed to return exact R")
        return np.asarray(mask, dtype=bool), {
            "scalarized_calls": 1,
            "outer_iterations": 1,
            "frontier_candidates": int(candidates),
            "eligible_candidates": int(chunks),
            "trim_work_deletions": int(repair if strategy == 2 else 0),
            "minimum_overfill": int(tile_rows),
            "repair_deletions": int(repair if strategy == 2 else 0),
            "repair_additions": int(repair if strategy == 1 else 0),
            "returned_top_r": False,
            "cell_rows": int(cell_rows),
            "tile_rows": int(tile_rows),
            "tile_strategy": ("contiguous" if strategy == 0 else
                              "floor_expand" if strategy == 1 else "ceil_trim"),
            "lookup_efficiency": float(ratio),
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
    EXP26.CANDIDATE_METHODS = TILE_METHODS
    EXP26.FRONTIER_METHODS = TILE_METHODS
    EXP26.METHOD_LABELS = METHOD_LABELS
    EXP26.METHOD_COLORS = METHOD_COLORS
    EXP26.FRONTIER_SETTINGS = {
        method: {"cell_rounding": method.removeprefix("tile8_")}
        for method in TILE_METHODS
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
    rng = np.random.default_rng(2901)
    shape = next(item for item in SHAPES if item["shape"] == "896x896")
    row_kib = EXP22.row_size_kib(shape)
    model = EXP13.fit_continuous_two_line(lookup_table, row_kib, saturation_kib)
    for n in (37, 97, 257):
        values = rng.lognormal(0.0, 1.1, size=n)
        values /= values.sum()
        for row_budget in (max(1, n // 4), n // 2, 3 * n // 4):
            for method in TILE_METHODS:
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
                    raise RuntimeError("tile selector self-check failed")

    # Verify the fixed-length interval recurrence against enumeration.
    values = rng.random(24)
    tile_rows, cell_rows, count = 8, 1, 2
    mask, _, valid = _best_k_tiles(values, tile_rows, cell_rows, count)
    if not valid:
        raise RuntimeError("tile recurrence self-check returned no path")
    expected = -1.0
    for first in range(0, 24 - tile_rows + 1):
        for second in range(first + tile_rows, 24 - tile_rows + 1):
            expected = max(
                expected,
                values[first:first + tile_rows].sum()
                + values[second:second + tile_rows].sum(),
            )
    if not np.isclose(values[mask].sum(), expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("tile recurrence does not match enumeration")


def evaluate_holdout_dispatch(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected = frame[
        (frame.track == "cuda") & frame.actual_total_median_ms.notna()
    ]
    keys = ["trace_id", "budget_index", "track"]
    wide = selected.pivot_table(
        index=keys + ["shape"], columns="method",
        values="actual_total_median_ms", aggfunc="first",
    ).reset_index()
    rows = []
    holdout_parts = []
    for shape, group in wide.groupby("shape", sort=True):
        trace_ids = sorted(group.trace_id.unique())
        split = max(1, len(trace_ids) // 2)
        train = group[group.trace_id.isin(set(trace_ids[:split]))]
        test = group[group.trace_id.isin(set(trace_ids[split:]))].copy()
        train_means = {method: float(train[method].mean()) for method in ("paper", *TILE_METHODS)}
        chosen = min(train_means, key=train_means.get)
        test["selected_method"] = chosen
        test["dispatch_total_ms"] = test[chosen]
        test["dispatch_saving_ms"] = test.paper - test.dispatch_total_ms
        holdout_parts.append(test)
        rows.append({
            "shape": shape, "selected_method": chosen,
            "train_trace_count": split,
            "holdout_trace_count": len(trace_ids) - split,
            "train_paper_total_ms_mean": train_means["paper"],
            "train_selected_total_ms_mean": train_means[chosen],
            "holdout_cases": int(len(test)),
            "holdout_paper_total_ms_mean": float(test.paper.mean()),
            "holdout_dispatch_total_ms_mean": float(test.dispatch_total_ms.mean()),
            "holdout_saving_ms_mean": float(test.dispatch_saving_ms.mean()),
            "holdout_strict_win_rate": float((test.dispatch_saving_ms > 0.0).mean()),
            "holdout_nonregression_rate": float((test.dispatch_saving_ms >= 0.0).mean()),
        })
    details = pd.DataFrame(rows)
    holdout = pd.concat(holdout_parts, ignore_index=True)
    summary = {
        "protocol": "first half of trace ids per shape selects Paper/floor/round/ceil; second half is holdout",
        "holdout_cases": int(len(holdout)),
        "paper_total_ms_mean": float(holdout.paper.mean()),
        "dispatch_total_ms_mean": float(holdout.dispatch_total_ms.mean()),
        "saving_ms_mean": float(holdout.dispatch_saving_ms.mean()),
        "saving_pct_mean_total": float(
            100.0 * holdout.dispatch_saving_ms.mean() / holdout.paper.mean()
        ),
        "strict_win_rate": float((holdout.dispatch_saving_ms > 0.0).mean()),
        "nonregression_rate": float((holdout.dispatch_saving_ms >= 0.0).mean()),
        "selected_methods": dict(zip(details["shape"], details.selected_method)),
    }
    return details, summary


def paired_cluster_bootstrap(frame: pd.DataFrame, repetitions: int = 20_000) -> dict:
    """Bootstrap paired savings by trace, stratified by matrix shape.

    All three row budgets from a trace stay in the same resampled cluster.  Shape
    stratification keeps the benchmark's four-shape mixture fixed.
    """
    selected = frame[
        (frame.track == "cuda") & frame.actual_total_median_ms.notna()
    ].copy()
    output = {
        "protocol": (
            "20,000 paired cluster bootstrap resamples; trace_id is the cluster, "
            "all budgets stay together, and shape counts stay fixed"
        ),
        "repetitions": int(repetitions),
        "seed": 29029,
        "methods": {},
    }
    rng = np.random.default_rng(output["seed"])
    for method in TILE_METHODS:
        rows = selected[selected.method == method].copy()
        rows["selector_saving_ms"] = (
            rows.paper_actual_selector_median_ms - rows.runtime_median_ms
        )
        method_summary = {
            "cases": int(len(rows)),
            "trace_clusters": int(rows.trace_id.nunique()),
        }
        for metric in (
            "selector_saving_ms", "actual_read_saving_ms", "actual_total_saving_ms"
        ):
            strata = []
            for _, group in rows.groupby("shape", sort=True):
                values = group.groupby("trace_id")[metric].mean().to_numpy()
                strata.append(values)
            cluster_count = sum(len(values) for values in strata)
            bootstrap = np.zeros(repetitions, dtype=np.float64)
            for values in strata:
                indices = rng.integers(
                    0, len(values), size=(repetitions, len(values))
                )
                bootstrap += values[indices].sum(axis=1) / cluster_count
            method_summary[f"{metric}_mean"] = float(rows[metric].mean())
            method_summary[f"{metric}_ci95"] = [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ]
        output["methods"][method] = method_summary
    return output


def _fmt(value, digits=3, suffix=""):
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def write_report(summary: pd.DataFrame, shape_summary: pd.DataFrame,
                 oracle: pd.DataFrame | None, dispatch_details: pd.DataFrame | None,
                 dispatch: dict | None, bootstrap: dict | None,
                 args: argparse.Namespace) -> None:
    report_path = args.report_output or (HERE / "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    image_prefix = Path(os.path.relpath(args.output_dir, report_path.parent)).as_posix()
    metadata_path = args.output_dir / "metadata.json"
    recorded = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    gpu_name = recorded.get("gpu", "unknown GPU")
    flash_name = (
        recorded.get("latency_profile", {}).get("metadata", {}).get(
            "flash", "unknown storage"
        )
    )
    primary = "cuda" if "cuda" in set(summary.track) else "host"
    indexed = summary[summary.track == primary].set_index("method").reindex(METHODS)
    lines = [
        "# Experiment 29 보고서: Saturation 8-Cell Tiles", "",
        "`s/8`행을 한 cell로 양자화하고 길이 8 cell인 sliding chunk만 고려한다. "
        "고정 길이 interval의 exact-K DP와 exact-R endpoint 보정만 사용하며, Paper mask나 "
        "`I(M_paper)`는 selector 입력이 아니다.", "",
        f"측정 환경은 `{gpu_name}`와 `{flash_name}`이며, 로컬 포화점은 "
        f"`s={args.saturation_kib:g} KiB`다.", "", "## 실험 생성 직전 internal probe", "",
        f"Experiment 28의 {INTERNAL_PROBE['cases']}개 paired case에 Python reference를 "
        "적용했다. 이는 selector 시간과 실제 I/O를 재지 않은 모델 측 사전 결과이며, "
        "Experiment 29의 정식 결과와 분리한다.", "",
        "| Shape | lookup I/L | lookup 절감 | importance | win rate |", "|---|---:|---:|---:|---:|",
    ]
    for shape, row in INTERNAL_PROBE["by_shape"].items():
        lines.append(
            f"| {shape} | {row['gain_pct']:+.2f}% | {row['saving_pct']:+.2f}% "
            f"| {row['importance_pct']:+.2f}% | {100*row['win_rate']:.1f}% |"
        )
    probe = INTERNAL_PROBE["overall"]
    lines.extend([
        "", f"전체 평균은 lookup I/L `{probe['lookup_efficiency_gain_pct_mean']:+.2f}%`, "
        f"lookup latency `{probe['lookup_saving_pct_mean']:.2f}%` 절감이었다.", "",
        "## 노트북 정식 결과", "",
        "| 방법 | selector median | case-p95 | lookup I/L | importance | 실제 read wall | selector+read | Paper보다 빠른 case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method, row in indexed.iterrows():
        win = "-" if method == "paper" else _fmt(100 * row.get("actual_total_win_rate", math.nan), 1, "%")
        lines.append(
            f"| {METHOD_LABELS[method]} | {_fmt(row.runtime_median_ms, 3, ' ms')} "
            f"| {_fmt(row.runtime_case_p95_ms, 3, ' ms')} "
            f"| {_fmt(row.lookup_efficiency_gain_pct_mean_valid, 2, '%')} "
            f"| {_fmt(row.importance_gain_pct_mean_valid, 2, '%')} "
            f"| {_fmt(row.get('actual_read_wall_ms_mean', math.nan), 3, ' ms')} "
            f"| {_fmt(row.get('actual_total_ms_mean', math.nan), 3, ' ms')} | {win} |"
        )

    measured_cases = int(indexed.loc["paper", "actual_cases"])
    lines.extend([
        "", f"실제 activation은 `{measured_cases}` case/방법이고, selector와 I/O를 각각 "
        f"case당 `{args.repetitions}`회와 `{args.io_repetitions}`회 반복했다. "
        f"실제 I/O 기록은 총 `{measured_cases * len(METHODS) * args.io_repetitions:,}`개이며 "
        f"O_DIRECT 비율은 `{100 * indexed.loc['paper', 'actual_direct_rate']:.1f}%`다.",
    ])

    if bootstrap is not None:
        fastest_tile = indexed.loc[list(TILE_METHODS)].actual_total_ms_mean.idxmin()
        paired = bootstrap["methods"][fastest_tile]
        total_ci = paired["actual_total_saving_ms_ci95"]
        selector_saving = paired["selector_saving_ms_mean"]
        read_saving = paired["actual_read_saving_ms_mean"]
        lines.extend([
            "", "### Paired 시간 분해", "",
            f"최선인 `{METHOD_LABELS[fastest_tile]}`는 case 평균으로 selector에서 "
            f"`{selector_saving:+.3f} ms`를 절감하고 실제 read에서는 "
            f"`{read_saving:+.3f} ms`를 절감했다(음수는 tile이 더 느리다는 뜻). "
            f"따라서 순절감은 `{paired['actual_total_saving_ms_mean']:+.3f} ms`이고, "
            f"shape-stratified trace-cluster bootstrap 95% 구간은 "
            f"`[{total_ci[0]:.3f}, {total_ci[1]:.3f}] ms`다.",
        ])

    lines.extend(["", "### Shape별 best tile", "",
                  "| Shape | best tile | lookup I/L | 실제 read | 총시간 | Paper 총시간 |", "|---|---|---:|---:|---:|---:|"])
    primary_shapes = shape_summary[shape_summary.track == primary]
    for shape in primary_shapes[primary_shapes.method == "paper"]["shape"]:
        group = primary_shapes[primary_shapes["shape"] == shape].set_index("method")
        tiles = group.reindex(TILE_METHODS)
        if tiles.actual_total_ms_mean.notna().any():
            best = tiles.actual_total_ms_mean.idxmin()
        else:
            best = tiles.lookup_efficiency_gain_pct_mean_valid.idxmax()
        lines.append(
            f"| {shape} | {METHOD_LABELS[best]} "
            f"| {group.loc[best, 'lookup_efficiency_gain_pct_mean_valid']:+.2f}% "
            f"| {_fmt(group.loc[best, 'actual_read_wall_ms_mean'], 3, ' ms')} "
            f"| {_fmt(group.loc[best, 'actual_total_ms_mean'], 3, ' ms')} "
            f"| {_fmt(group.loc['paper', 'actual_total_ms_mean'], 3, ' ms')} |"
        )

    lines.extend(["", "## Holdout shape dispatch", ""])
    if dispatch is None or dispatch_details is None or dispatch_details.empty:
        lines.extend(["실제 CUDA 총시간이 없어 dispatch를 계산하지 못했다.", ""])
    else:
        lines.extend([
            "각 shape의 앞 절반 trace에서 Paper와 세 tile 방식 중 평균 총시간이 가장 "
            "짧은 방법을 선택하고, 뒤 절반에 고정 적용했다.", "",
            f"Holdout Paper `{dispatch['paper_total_ms_mean']:.3f} ms`, dispatch "
            f"`{dispatch['dispatch_total_ms_mean']:.3f} ms`, 절감 "
            f"`{dispatch['saving_ms_mean']:.3f} ms` (`{dispatch['saving_pct_mean_total']:.2f}%`)다. "
            f"Non-regression rate는 `{100*dispatch['nonregression_rate']:.1f}%`다.", "",
            "| Shape | 선택 | holdout Paper | holdout dispatch | 절감 | strict win |", "|---|---|---:|---:|---:|---:|",
        ])
        for _, row in dispatch_details.iterrows():
            lines.append(
                f"| {row['shape']} | {METHOD_LABELS[row.selected_method]} "
                f"| {row.holdout_paper_total_ms_mean:.3f} ms "
                f"| {row.holdout_dispatch_total_ms_mean:.3f} ms "
                f"| {row.holdout_saving_ms_mean:+.3f} ms "
                f"| {100*row.holdout_strict_win_rate:.1f}% |"
            )

    fastest = indexed.actual_total_ms_mean.idxmin() if indexed.actual_total_ms_mean.notna().any() else None
    lines.extend(["", "## 판정", ""])
    if fastest is not None:
        if fastest == "paper":
            lines.append(
                f"단일 방법의 전체 평균은 Paper `{indexed.loc['paper', 'actual_total_ms_mean']:.3f} ms`가 가장 빨랐다."
            )
        else:
            saving_pct = 100.0 * (
                indexed.loc["paper", "actual_total_ms_mean"]
                - indexed.loc[fastest, "actual_total_ms_mean"]
            ) / indexed.loc["paper", "actual_total_ms_mean"]
            lines.append(
                f"`{METHOD_LABELS[fastest]}`가 전체 평균 `{indexed.loc[fastest, 'actual_total_ms_mean']:.3f} ms`로 "
                f"Paper `{indexed.loc['paper', 'actual_total_ms_mean']:.3f} ms`를 "
                f"`{saving_pct:.2f}%` 이겼다."
            )
    if dispatch is not None and dispatch.get("saving_ms_mean", 0.0) > 0.0:
        lines.append(
            f" Holdout shape dispatch도 Paper를 `{dispatch['saving_pct_mean_total']:.2f}%` 이겼다."
        )
    lines.extend([
        "", "## 범위와 한계", "",
        "- mask 선택은 로컬 SN850X lookup table을 직접 사용한다.",
        "- selector timing은 CUDA importance의 D2H, CPU Numba solve, bool mask H2D를 포함한다.",
        "- 실제 read는 native O_DIRECT와 GPU upload를 반복 측정했다." if args.measure_io else "- 실제 I/O replay를 생략했다.",
        "- `s>N` 또는 `R<8*cell_rows`이면 하나의 exact-R contiguous window로 퇴화한다.",
        "", f"![Runtime-quality]({image_prefix}/runtime_quality.png)", "",
        f"![Shape comparison]({image_prefix}/shape_comparison.png)", "",
    ])
    if args.measure_io:
        lines.extend([f"![Measured total]({image_prefix}/actual_total.png)", ""])
    if oracle is not None and len(oracle):
        lines.extend([f"![Small-N oracle]({image_prefix}/oracle_optimality.png)", ""])
    report_path.write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    frame = EXP24.add_paired_metrics(frame)
    frame = EXP26.add_actual_io_metrics(frame)
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    oracle_path = args.output_dir / "oracle_trials.csv"
    oracle = pd.read_csv(oracle_path) if oracle_path.exists() else None
    dispatch_details = None
    dispatch = None
    bootstrap = None
    if frame.actual_total_median_ms.notna().any():
        dispatch_details, dispatch = evaluate_holdout_dispatch(frame)
        dispatch_details.to_csv(args.output_dir / "holdout_dispatch.csv", index=False)
        bootstrap = paired_cluster_bootstrap(frame)
    summary_frame, shape_frame, nested = EXP26.summarize(frame, oracle, args)
    nested["format"] = "experiment-29-saturation-tiles-v1-real-activations"
    nested["comparison"] = "fixed eight-cell saturation tiles versus Paper"
    nested["internal_probe"] = INTERNAL_PROBE
    if dispatch is not None:
        nested["holdout_shape_dispatch"] = dispatch
    if bootstrap is not None:
        nested["paired_cluster_bootstrap"] = bootstrap
    measured = frame[frame.actual_io_median_ms.notna()]
    if len(measured):
        nested["actual_io_replay"] = {
            "cases": int(len(measured)),
            "all_direct": bool((measured.actual_direct_rate == 1.0).all()),
            "profile_vs_io_pearson_r": float(measured.lookup_ms.corr(measured.actual_io_median_ms)),
            "profile_vs_read_wall_pearson_r": float(
                measured.lookup_ms.corr(measured.actual_read_wall_median_ms)
            ),
        }
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(nested, indent=2) + "\n")
    EXP26.plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    EXP26.plot_by_shape(shape_frame, args.output_dir / "shape_comparison.png")
    if summary_frame.get("actual_total_ms_mean", pd.Series(dtype=float)).notna().any():
        EXP26.plot_actual_total(summary_frame, args.output_dir / "actual_total.png")
    if oracle is not None and len(oracle):
        EXP26.plot_oracle(oracle, args.output_dir / "oracle_optimality.png")
    write_report(
        summary_frame, shape_frame, oracle, dispatch_details, dispatch, bootstrap, args
    )
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
        dict(next(item for item in SHAPES if item["shape"] == name)) for name in shape_names
    ]
    metadata = {
        "activation_trace": trace_metadata,
        "trace_file": str(trace_path),
        "trace_sha256": EXP22.sha256_file(trace_path),
        "selected_trace_count": len(traces),
        "selected_trace_ids": [int(trace["trace_id"]) for trace in traces],
        "max_traces_per_shape": int(args.max_traces_per_shape),
        "shape_specs": shape_specs, "method_labels": METHOD_LABELS,
        "algorithm": "eight-cell fixed saturation tiles; floor-expand and ceil-trim",
        "internal_probe": INTERNAL_PROBE,
        "latency_profile": {
            "requested": str(args.profile), "max_kib": int(lookup_table.max_kb),
            "metadata": lookup_table.meta, "sha256": EXP22.sha256_file(Path(args.profile)),
        },
        "actual_io": {
            "enabled": bool(args.measure_io), "blob": str(args.io_blob) if args.io_blob else None,
            "blob_sha256": EXP22.sha256_file(args.io_blob) if args.measure_io else None,
            "repetitions": int(args.io_repetitions), "warmup": int(args.io_warmup),
            "threads": int(args.io_threads), "max_read_kib": int(args.io_max_read_kib),
            "device": args.io_device,
        },
        "requested_tracks": args.tracks, "executed_tracks": tracks,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
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
                f"--io-blob has {args.io_blob.stat().st_size} bytes; at least {required_bytes} required"
            )
        EXP22.load_vlmflash()
        from vlmflash._native import native, unavailable_reason
        native_reader = native()
        if native_reader is None:
            raise SystemExit(f"native reader unavailable: {unavailable_reason()}")

    if not args.skip_self_check:
        self_check(lookup_table, args.saturation_kib)
    if not args.skip_oracle:
        EXP24.write_csv(
            args.output_dir / "oracle_trials.csv", EXP26.collect_oracle(args, lookup_table)
        )
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
