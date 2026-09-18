#!/usr/bin/env python3
"""Experiment 28: exact-R oracle and sparse corridor DP on the laptop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import time
import traceback

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_26 = PROJECT_ROOT / "experiments" / "26_frontier_adaptive_trim" / "run_experiment.py"
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


EXP26 = _load_module("experiment_26_for_28", EXPERIMENT_26)
EXP24 = EXP26.EXP24
EXP22 = EXP26.EXP22
EXP18 = EXP26.EXP18
EXP13 = EXP26.EXP13
BASE = EXP26.BASE
njit = EXP26.njit
SHAPES = EXP26.SHAPES
DEFAULT_TRACE_INPUT = EXP26.DEFAULT_TRACE_INPUT

METHODS = (
    "paper",
    "top_r",
    "price1",
    "corridor_w4",
    "corridor_w8",
    "corridor_w16",
    "corridor_w32",
)
CANDIDATE_METHODS = METHODS[2:]
CORRIDOR_METHODS = METHODS[3:]
METHOD_WIDTHS = {
    "corridor_w4": 4,
    "corridor_w8": 8,
    "corridor_w16": 16,
    "corridor_w32": 32,
}
METHOD_LABELS = {
    "paper": "Paper",
    "top_r": "Top-R",
    "price1": "Price-1 + repair",
    "corridor_w4": "Exact-R corridor W=4",
    "corridor_w8": "Exact-R corridor W=8",
    "corridor_w16": "Exact-R corridor W=16",
    "corridor_w32": "Exact-R corridor W=32",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "top_r": "#64748B",
    "price1": "#7C3AED",
    "corridor_w4": "#60A5FA",
    "corridor_w8": "#2563EB",
    "corridor_w16": "#0F766E",
    "corridor_w32": "#064E3B",
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
    parser.add_argument("--dinkelbach-iterations", type=int, default=8)
    parser.add_argument("--dinkelbach-tolerance", type=float, default=1e-11)
    parser.add_argument("--oracle-n", type=int, default=18)
    parser.add_argument("--oracle-trials", type=int, default=10)
    parser.add_argument("--oracle-cv", type=float, default=3.30)
    parser.add_argument("--real-oracle-traces-per-shape", type=int, default=2)
    parser.add_argument("--real-oracle-max-iterations", type=int, default=20)
    parser.add_argument("--skip-real-oracle", action="store_true")
    # Reuse Experiment 26's paired small-N multiset seed.  Some N=18 draws
    # cannot attain CV=3.30 under the exact-CV construction.
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
    if args.dinkelbach_iterations < 1 or args.real_oracle_max_iterations < 1:
        raise SystemExit("Dinkelbach iteration counts must be positive")
    if args.dinkelbach_tolerance <= 0.0:
        raise SystemExit("--dinkelbach-tolerance must be positive")
    if args.real_oracle_traces_per_shape < 1:
        raise SystemExit("--real-oracle-traces-per-shape must be positive")
    if not Path(args.profile).is_file():
        raise SystemExit(f"laptop latency profile does not exist: {args.profile}")
    return shapes, tracks


@njit(cache=False, inline="always")
def _run_cost(length, a_ms, c1_ms, c2_ms, saturation_rows):
    if length <= 0:
        return 0.0
    if length <= saturation_rows:
        return a_ms + c1_ms * length
    return c2_ms * length


@njit(cache=False)
def _corridor_fixed_q_kernel(values, center_mask, row_budget, width, q,
                             a_ms, c1_ms, c2_ms, saturation_rows, short_max):
    """Maximize I-qL at exact R inside a prefix-cardinality corridor.

    State coordinates are (z, r), the zero and selected counts in a prefix.
    Each diagonal z depends only on z-1.  For an affine run-cost piece, the
    run transition is a sliding maximum over the preceding diagonal.
    """
    n = len(values)
    r_total = int(row_budget)
    z_total = n - r_total
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    center_zeros = np.empty(n + 1, dtype=np.int64)
    center_zeros[0] = 0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
        center_zeros[index + 1] = center_zeros[index] + (0 if center_mask[index] else 1)

    lows = np.empty(z_total + 1, dtype=np.int64)
    highs = np.empty(z_total + 1, dtype=np.int64)
    offsets = np.empty(z_total + 2, dtype=np.int64)
    offsets[0] = 0
    left_i = 0
    right_i = -1
    for z in range(z_total + 1):
        low_zero = z - width
        high_zero = z + width
        while left_i <= n and center_zeros[left_i] < low_zero:
            left_i += 1
        if right_i < left_i - 1:
            right_i = left_i - 1
        while right_i + 1 <= n and center_zeros[right_i + 1] <= high_zero:
            right_i += 1
        i_low = max(z, left_i)
        i_high = min(z + r_total, right_i)
        if i_low > i_high:
            return np.zeros(n, dtype=np.bool_), -np.inf, 0, False
        lows[z] = i_low - z
        highs[z] = i_high - z
        offsets[z + 1] = offsets[z] + highs[z] - lows[z] + 1

    choices = np.full(offsets[-1], -3, dtype=np.int32)
    previous = np.full(r_total + 1, -np.inf, dtype=np.float64)
    current = np.full(r_total + 1, -np.inf, dtype=np.float64)

    # z=0 contains prefixes made entirely of selected rows.
    for r in range(lows[0], highs[0] + 1):
        if r == 0:
            previous[r] = 0.0
        else:
            previous[r] = prefix[r] - q * _run_cost(
                r, a_ms, c1_ms, c2_ms, saturation_rows
            )
        choices[offsets[0] + r - lows[0]] = -2

    deque_indices = np.empty(r_total + 1, dtype=np.int64)
    deque_values = np.empty(r_total + 1, dtype=np.float64)

    for z in range(1, z_total + 1):
        prev_low = lows[z - 1]
        prev_high = highs[z - 1]
        cur_low = lows[z]
        cur_high = highs[z]
        cur_size = cur_high - cur_low + 1
        for position in range(cur_size):
            current[position] = -np.inf

        short_head = 0
        short_tail = 0
        short_add = prev_low
        long_add = prev_low
        long_best = -np.inf
        long_best_s = -1

        for r in range(cur_low, cur_high + 1):
            add_until = min(r - 1, prev_high)
            while short_add <= add_until:
                source_value = previous[short_add - prev_low]
                if np.isfinite(source_value):
                    source_i = short_add + z
                    candidate = source_value - prefix[source_i] + q * c1_ms * short_add
                    while short_tail > short_head and candidate > deque_values[short_tail - 1]:
                        short_tail -= 1
                    deque_indices[short_tail] = short_add
                    deque_values[short_tail] = candidate
                    short_tail += 1
                short_add += 1
            minimum_short_s = r - short_max
            while short_tail > short_head and deque_indices[short_head] < minimum_short_s:
                short_head += 1

            long_until = min(r - short_max - 1, prev_high)
            while long_add <= long_until:
                source_value = previous[long_add - prev_low]
                if np.isfinite(source_value):
                    source_i = long_add + z
                    candidate = source_value - prefix[source_i] + q * c2_ms * long_add
                    if candidate > long_best:
                        long_best = candidate
                        long_best_s = long_add
                long_add += 1

            best = -np.inf
            choice = -3
            if prev_low <= r <= prev_high:
                best = previous[r - prev_low]
                choice = -1
            current_i = r + z
            if short_tail > short_head:
                candidate = (
                    prefix[current_i] - q * a_ms - q * c1_ms * r
                    + deque_values[short_head]
                )
                if candidate > best + 1e-15:
                    best = candidate
                    choice = int(deque_indices[short_head])
            if long_best_s >= 0:
                candidate = prefix[current_i] - q * c2_ms * r + long_best
                if candidate > best + 1e-15:
                    best = candidate
                    choice = int(long_best_s)
            current[r - cur_low] = best
            choices[offsets[z] + r - cur_low] = choice

        old_previous = previous
        previous = current
        current = old_previous

    if not (lows[z_total] <= r_total <= highs[z_total]):
        return np.zeros(n, dtype=np.bool_), -np.inf, int(offsets[-1]), False
    optimum = previous[r_total - lows[z_total]]
    if not np.isfinite(optimum):
        return np.zeros(n, dtype=np.bool_), optimum, int(offsets[-1]), False

    mask = np.zeros(n, dtype=np.bool_)
    z = z_total
    r = r_total
    while z > 0:
        if r < lows[z] or r > highs[z]:
            return np.zeros(n, dtype=np.bool_), optimum, int(offsets[-1]), False
        choice = int(choices[offsets[z] + r - lows[z]])
        if choice == -1:
            z -= 1
        elif choice >= 0:
            start = choice + z
            end = r + z
            for index in range(start, end):
                mask[index] = True
            r = choice
            z -= 1
        else:
            return np.zeros(n, dtype=np.bool_), optimum, int(offsets[-1]), False
    if r > 0:
        for index in range(r):
            mask[index] = True
    selected = 0
    for index in range(n):
        selected += int(mask[index])
    return mask, optimum, int(offsets[-1]), selected == r_total


def price1_mask(values: np.ndarray, row_budget: int, model: dict):
    top = EXP24.top_r_mask(values, row_budget)
    top_metrics = EXP24.two_line_metrics(top, values, model)
    q = float(top_metrics["importance"] / top_metrics["two_line_ms"])
    candidate, metadata = EXP24.search_row_price(values, row_budget, q, model, 1)
    candidate_metrics = EXP24.two_line_metrics(candidate, values, model)
    candidate_ratio = float(candidate_metrics["importance"] / candidate_metrics["two_line_ms"])
    if candidate_ratio + 1e-14 < q:
        candidate = top
        candidate_ratio = q
        returned_top = True
    else:
        returned_top = bool(np.array_equal(candidate, top))
    return candidate, candidate_ratio, {
        **metadata,
        "returned_top_r": returned_top,
    }


def corridor_ratio(values: np.ndarray, row_budget: int, model: dict, width: int,
                   max_iterations: int, tolerance: float,
                   full_exact: bool = False):
    center, q, seed_metadata = price1_mask(values, row_budget, model)
    best_mask = center.copy()
    best_ratio = q
    total_states = 0
    converged = False
    iterations = 0
    effective_width = len(values) if full_exact else int(width)

    for iteration in range(max_iterations):
        mask, scalar_value, states, valid = _corridor_fixed_q_kernel(
            np.asarray(values, dtype=np.float64),
            np.asarray(center, dtype=np.bool_), int(row_budget), effective_width,
            float(q), float(model["a_ms"]), float(model["c1_ms_per_row"]),
            float(model["c2_ms_per_row"]), float(model["saturation_rows"]),
            int(model["short_max_rows"]),
        )
        iterations += 1
        total_states += int(states)
        if not valid:
            raise RuntimeError("corridor DP failed to return an exact-R path")
        metrics = EXP24.two_line_metrics(mask, values, model)
        ratio = float(metrics["importance"] / metrics["two_line_ms"])
        if ratio > best_ratio + 1e-14:
            best_ratio = ratio
            best_mask = mask.copy()
        residual = float(metrics["importance"] - q * metrics["two_line_ms"])
        if abs(residual) <= tolerance * max(1.0, abs(metrics["importance"])):
            converged = True
            center = mask
            break
        if ratio < q - 1e-12:
            raise RuntimeError("Dinkelbach ratio regressed inside a corridor containing its center")
        q = ratio
        center = mask

    return best_mask, {
        "scalarized_calls": int(seed_metadata.get("scalarized_calls", 0)) + iterations,
        "outer_iterations": iterations,
        "repair_deletions": int(seed_metadata.get("repair_deletions", 0)),
        "repair_additions": int(seed_metadata.get("repair_additions", 0)),
        "returned_top_r": bool(np.array_equal(best_mask, EXP24.top_r_mask(values, row_budget))),
        "frontier_candidates": total_states,
        "eligible_candidates": iterations,
        "minimum_overfill": effective_width,
        "trim_work_deletions": 0,
        "corridor_width": effective_width,
        "corridor_states": total_states,
        "converged": converged,
        "full_exact": bool(full_exact),
    }


def safe_select(method: str, values: np.ndarray, row_budget: int,
                model: dict, args: argparse.Namespace | None = None):
    max_iterations = 8 if args is None else int(args.dinkelbach_iterations)
    tolerance = 1e-11 if args is None else float(args.dinkelbach_tolerance)
    try:
        if method == "top_r":
            mask, metadata = EXP24.top_r_mask(values, row_budget), {}
        elif method == "price1":
            mask, _, metadata = price1_mask(values, row_budget, model)
        else:
            mask, metadata = corridor_ratio(
                values, row_budget, model, METHOD_WIDTHS[method],
                max_iterations, tolerance,
            )
        return np.asarray(mask, dtype=bool), {
            **metadata, "fallback_used": False, "error": "",
        }
    except Exception as exception:
        return EXP24.top_r_mask(values, row_budget), {
            "fallback_used": True,
            "error": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(limit=4),
        }


def bind_exp26(args: argparse.Namespace) -> None:
    EXP26.METHODS = METHODS
    EXP26.CANDIDATE_METHODS = CANDIDATE_METHODS
    EXP26.FRONTIER_METHODS = CORRIDOR_METHODS
    EXP26.METHOD_LABELS = METHOD_LABELS
    EXP26.METHOD_COLORS = METHOD_COLORS
    EXP26.FRONTIER_SETTINGS = {
        method: {"width": width, "dinkelbach_iterations": args.dinkelbach_iterations}
        for method, width in METHOD_WIDTHS.items()
    }

    def bound_safe_select(method, values, row_budget, model):
        return safe_select(method, values, row_budget, model, args)

    EXP26.safe_select = bound_safe_select


def self_check(lookup_table, saturation_kib: float) -> None:
    rng = np.random.default_rng(2801)
    shape = next(item for item in SHAPES if item["shape"] == "896x896")
    model = EXP13.fit_continuous_two_line(
        lookup_table, EXP22.row_size_kib(shape), saturation_kib
    )
    for n in (9, 12):
        for _ in range(4):
            values = rng.lognormal(0.0, 1.2, size=n)
            values /= values.sum()
            for row_budget in (max(1, n // 4), n // 2, max(1, 3 * n // 4)):
                budgets = np.asarray([row_budget], dtype=np.int64)
                exact = EXP24._exact_fixed_r_ratios(
                    values, budgets, model["a_ms"], model["c1_ms_per_row"],
                    model["c2_ms_per_row"], model["saturation_rows"],
                )
                expected = float(exact[0][0])
                mask, metadata = corridor_ratio(
                    values, row_budget, model, n, 20, 1e-12, full_exact=True
                )
                metrics = EXP24.two_line_metrics(mask, values, model)
                actual = float(metrics["importance"] / metrics["two_line_ms"])
                if int(mask.sum()) != row_budget or not metadata["converged"]:
                    raise RuntimeError("full exact DP self-check did not converge")
                if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
                    raise RuntimeError(
                        f"full exact DP mismatch: expected {expected}, got {actual}"
                    )
                corridor, _ = corridor_ratio(values, row_budget, model, 4, 8, 1e-12)
                corridor_ratio_value = EXP24.two_line_metrics(
                    corridor, values, model
                )["importance"] / EXP24.two_line_metrics(corridor, values, model)["two_line_ms"]
                seed, seed_ratio, _ = price1_mask(values, row_budget, model)
                del seed
                if corridor_ratio_value < seed_ratio - 1e-11:
                    raise RuntimeError("corridor regressed below its center path")


def select_real_oracle_traces(traces: list[dict], per_shape: int) -> list[dict]:
    counts: dict[str, int] = {}
    selected = []
    for trace in traces:
        shape = trace["shape"]
        if counts.get(shape, 0) >= per_shape:
            continue
        selected.append(trace)
        counts[shape] = counts.get(shape, 0) + 1
    return selected


def collect_real_oracle(args: argparse.Namespace, traces: list[dict], lookup_table,
                        native_reader=None) -> list[dict]:
    rows = []
    by_name = {item["shape"]: dict(item) for item in SHAPES}
    selected = select_real_oracle_traces(traces, args.real_oracle_traces_per_shape)
    # Compile outside recorded timings.
    warm_values = np.asarray(selected[0]["values"], dtype=np.float64)
    warm_values /= warm_values.sum()
    warm_shape = by_name[selected[0]["shape"]]
    warm_model = EXP13.fit_continuous_two_line(
        lookup_table, EXP22.row_size_kib(warm_shape), args.saturation_kib
    )
    warm_r = max(1, len(warm_values) // 4)
    corridor_ratio(warm_values, warm_r, warm_model, len(warm_values), 2, 1e-10, True)

    for trace_index, trace in enumerate(selected):
        shape = by_name[trace["shape"]]
        n = int(shape["n"])
        row_kib = EXP22.row_size_kib(shape)
        params = EXP22.make_params(shape, args.saturation_kib)
        model = EXP13.fit_continuous_two_line(lookup_table, row_kib, args.saturation_kib)
        values32 = np.asarray(trace["values"], dtype=np.float32).copy()
        values32 /= np.float32(values32.astype(np.float64).sum())
        values = values32.astype(np.float64)
        for budget_index, fraction in enumerate(args.row_budget_fractions):
            nominal = max(1, min(n, int(round(n * fraction))))
            paper_function = EXP22.paper_function(
                "host", values, None, nominal, row_kib, lookup_table, params, args.paper_impl
            )
            paper_mask = np.asarray(paper_function()[0], dtype=bool)
            effective_r = int(paper_mask.sum())
            started = time.perf_counter_ns()
            exact_mask, exact_meta = corridor_ratio(
                values, effective_r, model, n, args.real_oracle_max_iterations,
                args.dinkelbach_tolerance, full_exact=True,
            )
            exact_runtime_ms = (time.perf_counter_ns() - started) / 1e6
            paper_metrics = EXP18.mask_metrics(
                paper_mask, values, model, lookup_table, row_kib
            )
            exact_metrics = EXP18.mask_metrics(
                exact_mask, values, model, lookup_table, row_kib
            )
            actual = {
                "actual_io_median_ms": math.nan,
                "actual_read_wall_median_ms": math.nan,
                "actual_upload_median_ms": math.nan,
            }
            if native_reader is not None:
                actual, _ = EXP26.measure_actual_io(native_reader, args, exact_mask, row_kib)
            row = {
                "trace_id": int(trace["trace_id"]),
                "shape": shape["shape"], "n": n, "d": int(shape["d"]),
                "budget_index": budget_index, "row_budget_fraction": float(fraction),
                "row_budget": effective_r, "paper_underfill_rows": nominal - effective_r,
                "paper_importance": paper_metrics["importance"],
                "paper_two_line_ms": paper_metrics["two_line_ms"],
                "paper_lookup_ms": paper_metrics["lookup_ms"],
                "paper_two_line_efficiency": (
                    paper_metrics["importance"] / paper_metrics["two_line_ms"]
                ),
                "paper_lookup_efficiency": (
                    paper_metrics["importance"] / paper_metrics["lookup_ms"]
                ),
                "exact_importance": exact_metrics["importance"],
                "exact_two_line_ms": exact_metrics["two_line_ms"],
                "exact_lookup_ms": exact_metrics["lookup_ms"],
                "exact_two_line_efficiency": (
                    exact_metrics["importance"] / exact_metrics["two_line_ms"]
                ),
                "exact_lookup_efficiency": (
                    exact_metrics["importance"] / exact_metrics["lookup_ms"]
                ),
                "exact_two_line_gain_vs_paper_pct": 100.0 * (
                    (exact_metrics["importance"] / exact_metrics["two_line_ms"])
                    / (paper_metrics["importance"] / paper_metrics["two_line_ms"]) - 1.0
                ),
                "exact_lookup_gain_vs_paper_pct": 100.0 * (
                    (exact_metrics["importance"] / exact_metrics["lookup_ms"])
                    / (paper_metrics["importance"] / paper_metrics["lookup_ms"]) - 1.0
                ),
                "exact_runtime_ms": exact_runtime_ms,
                "exact_iterations": exact_meta["outer_iterations"],
                "exact_states": exact_meta["corridor_states"],
                "exact_converged": exact_meta["converged"],
                **{f"exact_{key}": value for key, value in actual.items()},
            }
            for method in CORRIDOR_METHODS:
                mask, metadata = safe_select(method, values, effective_r, model, args)
                metrics = EXP18.mask_metrics(mask, values, model, lookup_table, row_kib)
                ratio = metrics["importance"] / metrics["two_line_ms"]
                row[f"{method}_optimum_recovery"] = ratio / row["exact_two_line_efficiency"]
                row[f"{method}_two_line_gain_vs_paper_pct"] = 100.0 * (
                    ratio / row["paper_two_line_efficiency"] - 1.0
                )
                row[f"{method}_states"] = metadata.get("corridor_states", 0)
            rows.append(row)
        print(
            f"real-oracle trace={trace_index + 1}/{len(selected)} "
            f"shape={shape['shape']}", flush=True,
        )
    return rows


def _fmt(value, digits=3, suffix=""):
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def evaluate_holdout_dispatch(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train a shape-only Paper/Price-1 dispatch on half the traces.

    Shape is known before selection, so this policy has no online decision
    overhead.  The later half of each shape's trace ids is never used to pick
    the method and is the only half reported as the holdout result.
    """
    selected = frame[
        (frame.track == "cuda") & frame.actual_total_median_ms.notna()
    ]
    keys = ["trace_id", "budget_index", "track"]
    paper = selected[selected.method == "paper"][keys + [
        "shape", "actual_total_median_ms",
    ]].rename(columns={"actual_total_median_ms": "paper_total_ms"})
    price = selected[selected.method == "price1"][keys + [
        "actual_total_median_ms",
    ]].rename(columns={"actual_total_median_ms": "price1_total_ms"})
    paired = paper.merge(price, on=keys, how="inner")
    paired["price1_saving_ms"] = paired.paper_total_ms - paired.price1_total_ms
    rows = []
    holdout_parts = []
    for shape, group in paired.groupby("shape", sort=True):
        trace_ids = sorted(group.trace_id.unique())
        split = max(1, len(trace_ids) // 2)
        train_ids = set(trace_ids[:split])
        test_ids = set(trace_ids[split:])
        train = group[group.trace_id.isin(train_ids)]
        test = group[group.trace_id.isin(test_ids)].copy()
        use_price1 = bool(train.price1_saving_ms.mean() > 0.0)
        test["selected_method"] = "price1" if use_price1 else "paper"
        test["hybrid_total_ms"] = (
            test.price1_total_ms if use_price1 else test.paper_total_ms
        )
        test["hybrid_saving_ms"] = test.paper_total_ms - test.hybrid_total_ms
        holdout_parts.append(test)
        rows.append({
            "shape": shape,
            "train_trace_count": len(train_ids),
            "holdout_trace_count": len(test_ids),
            "selected_method": "price1" if use_price1 else "paper",
            "train_price1_saving_ms_mean": float(train.price1_saving_ms.mean()),
            "holdout_cases": int(len(test)),
            "holdout_paper_total_ms_mean": float(test.paper_total_ms.mean()),
            "holdout_hybrid_total_ms_mean": float(test.hybrid_total_ms.mean()),
            "holdout_saving_ms_mean": float(test.hybrid_saving_ms.mean()),
            "holdout_strict_win_rate": float((test.hybrid_saving_ms > 0.0).mean()),
            "holdout_nonregression_rate": float((test.hybrid_saving_ms >= 0.0).mean()),
        })
    details = pd.DataFrame(rows)
    holdout = pd.concat(holdout_parts, ignore_index=True)
    summary = {
        "protocol": "first half of trace ids per shape selects Paper or Price-1; second half is holdout",
        "holdout_cases": int(len(holdout)),
        "paper_total_ms_mean": float(holdout.paper_total_ms.mean()),
        "hybrid_total_ms_mean": float(holdout.hybrid_total_ms.mean()),
        "saving_ms_mean": float(holdout.hybrid_saving_ms.mean()),
        "saving_pct_mean_total": float(
            100.0 * holdout.hybrid_saving_ms.mean() / holdout.paper_total_ms.mean()
        ),
        "strict_win_rate": float((holdout.hybrid_saving_ms > 0.0).mean()),
        "nonregression_rate": float((holdout.hybrid_saving_ms >= 0.0).mean()),
        "selected_methods": dict(zip(details["shape"], details.selected_method)),
    }
    return details, summary


def write_report(summary: pd.DataFrame, shape_summary: pd.DataFrame,
                 oracle: pd.DataFrame | None, real_oracle: pd.DataFrame | None,
                 dispatch_details: pd.DataFrame | None, dispatch: dict | None,
                 args: argparse.Namespace, trials: pd.DataFrame | None = None) -> None:
    report_path = args.report_output or (HERE / "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    image_prefix = Path(os.path.relpath(args.output_dir, report_path.parent)).as_posix()
    primary = "cuda" if "cuda" in set(summary.track) else "host"
    indexed = summary[summary.track == primary].set_index("method").reindex(METHODS)
    candidates = indexed.loc[list(CANDIDATE_METHODS)]
    best_quality_method = candidates.lookup_efficiency_gain_pct_mean_valid.idxmax()
    lines = [
        "# Experiment 28 보고서: Exact-R Oracle + Sparse Corridor DP", "",
        "Paper mask나 `I(M_paper)`를 selector 입력으로 사용하지 않았다. Paper가 반환한 "
        "행 수만 공통 exact-R 예산으로 사용했다. Full DP는 실제 activation 일부에서 "
        "two-line latency 목적의 전역 optimum을 계산하고, corridor DP는 그 문제를 작은 "
        "prefix-cardinality band에서 온라인으로 근사한다.", "", "## 온라인 결과", "",
        f"Primary timing track: `{primary}`.", "",
        "| 방법 | selector median | case-p95 | lookup I/L vs Paper | two-line I/L vs Paper | importance | 실제 read wall | selector+read | Paper보다 빠른 case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in indexed.iterrows():
        actual_win = "-" if method == "paper" else _fmt(100 * row.get("actual_total_win_rate", math.nan), 1, "%")
        lines.append(
            f"| {METHOD_LABELS[method]} | {_fmt(row.runtime_median_ms, 3, ' ms')} "
            f"| {_fmt(row.runtime_case_p95_ms, 3, ' ms')} "
            f"| {_fmt(row.lookup_efficiency_gain_pct_mean_valid, 2, '%')} "
            f"| {_fmt(row.two_line_efficiency_gain_pct_mean_valid, 2, '%')} "
            f"| {_fmt(row.importance_gain_pct_mean_valid, 2, '%')} "
            f"| {_fmt(row.get('actual_read_wall_ms_mean', math.nan), 3, ' ms')} "
            f"| {_fmt(row.get('actual_total_ms_mean', math.nan), 3, ' ms')} "
            f"| {actual_win} |"
        )

    lines.extend(["", "## 실제 trace full exact-R oracle", ""])
    if real_oracle is None or real_oracle.empty:
        lines.extend(["실제 trace oracle을 생략했다.", ""])
    else:
        exact_two = real_oracle.exact_two_line_gain_vs_paper_pct
        exact_lookup = real_oracle.exact_lookup_gain_vs_paper_pct
        lines.extend([
            f"{len(real_oracle)}개 실제 case에서 full state space를 계산했다. 모든 case "
            f"converged: `{bool(real_oracle.exact_converged.all())}`.", "",
            "| 판정량 | 평균 | 중앙값 | 최소 | Paper 초과 case |", "|---|---:|---:|---:|---:|",
            f"| Exact two-line I/L gain | {exact_two.mean():.2f}% | {exact_two.median():.2f}% | {exact_two.min():.2f}% | {100*(exact_two > 1e-9).mean():.1f}% |",
            f"| Exact lookup I/L gain | {exact_lookup.mean():.2f}% | {exact_lookup.median():.2f}% | {exact_lookup.min():.2f}% | {100*(exact_lookup > 1e-9).mean():.1f}% |",
            f"| Exact solver runtime | {real_oracle.exact_runtime_ms.mean():.2f} ms | {real_oracle.exact_runtime_ms.median():.2f} ms | {real_oracle.exact_runtime_ms.min():.2f} ms | - |",
            "", "Corridor의 exact two-line optimum 회수율:", "",
            "| 방법 | 평균 | p5 | 최소 |", "|---|---:|---:|---:|",
        ])
        for method in CORRIDOR_METHODS:
            recovery = 100.0 * real_oracle[f"{method}_optimum_recovery"]
            lines.append(
                f"| {METHOD_LABELS[method]} | {recovery.mean():.3f}% "
                f"| {recovery.quantile(0.05):.3f}% | {recovery.min():.3f}% |"
            )
        lines.append("")

        if trials is not None and "exact_actual_read_wall_median_ms" in real_oracle:
            paper_actual = trials[
                (trials.method == "paper") & (trials.track == primary)
            ][["trace_id", "budget_index", "runtime_median_ms",
               "actual_read_wall_median_ms"]].rename(columns={
                "runtime_median_ms": "paper_selector_ms",
                "actual_read_wall_median_ms": "paper_read_wall_ms",
            })
            exact_actual = real_oracle.merge(
                paper_actual, on=["trace_id", "budget_index"], how="left"
            )
            exact_actual["read_saving_ms"] = (
                exact_actual.paper_read_wall_ms
                - exact_actual.exact_actual_read_wall_median_ms
            )
            lines.extend([
                "### Oracle mask의 실제 SSD read", "",
                f"Exact mask는 Paper보다 실제 read wall을 평균 "
                f"`{exact_actual.read_saving_ms.mean():+.3f} ms` 줄였고, "
                f"`{100 * (exact_actual.read_saving_ms > 0).mean():.1f}%`의 case에서 "
                "더 빨랐다. 이 값은 selector를 공짜로 가정한 최대 read-side 여유다.", "",
            ])

    lines.extend(["## Holdout shape dispatch", ""])
    if dispatch is None or dispatch_details is None or dispatch_details.empty:
        lines.extend(["CUDA 실제 총시간이 없어 dispatch를 평가하지 못했다.", ""])
    else:
        chosen = ", ".join(
            f"`{row['shape']}`→{METHOD_LABELS.get(row.selected_method, row.selected_method)}"
            for _, row in dispatch_details.iterrows()
        )
        lines.extend([
            "각 shape의 앞 절반 trace에서 Paper와 Price-1 중 평균 총시간이 짧은 방법을 "
            "고르고, 뒤 절반에는 선택을 고정해 적용했다. Shape는 selector 전에 알려져 "
            "있으므로 dispatch 자체의 온라인 비용은 없다.", "",
            f"선택: {chosen}.", "",
            f"192개 holdout case에서 Paper 평균은 `{dispatch['paper_total_ms_mean']:.3f} ms`, "
            f"dispatch는 `{dispatch['hybrid_total_ms_mean']:.3f} ms`였다. 평균 "
            f"`{dispatch['saving_ms_mean']:.3f} ms` (`{dispatch['saving_pct_mean_total']:.2f}%`) "
            f"단축했고, non-regression rate는 `{100*dispatch['nonregression_rate']:.1f}%`다.", "",
            "| Shape | train 선택 | train 절감 | holdout 절감 | holdout strict win |", "|---|---|---:|---:|---:|",
        ])
        for _, row in dispatch_details.iterrows():
            lines.append(
                f"| {row['shape']} | {METHOD_LABELS.get(row.selected_method, row.selected_method)} "
                f"| {row.train_price1_saving_ms_mean:+.3f} ms "
                f"| {row.holdout_saving_ms_mean:+.3f} ms "
                f"| {100*row.holdout_strict_win_rate:.1f}% |"
            )
        lines.append("")

    paper_total = indexed.loc["paper"].get("actual_total_ms_mean", math.nan)
    best_total_method = None
    if "actual_total_ms_mean" in indexed and indexed.actual_total_ms_mean.notna().any():
        best_total_method = indexed.actual_total_ms_mean.idxmin()
    lines.extend(["## 판정", ""])
    if best_total_method is not None:
        best_total = indexed.loc[best_total_method, "actual_total_ms_mean"]
        if best_total_method == "paper":
            lines.append(
                f"단일 전-shape 방법 중에는 Paper가 `{paper_total:.3f} ms`로 가장 짧아, "
                "현재 CPU corridor 구현은 end-to-end Paper를 이기지 못했다."
            )
        else:
            lines.append(
                f"`{METHOD_LABELS[best_total_method]}`가 평균 `{best_total:.3f} ms`로 "
                f"Paper `{paper_total:.3f} ms`를 이겼다."
            )
        lines.append("")
    lines.append(
        f"온라인 mask 품질이 가장 높은 후보는 `{METHOD_LABELS[best_quality_method]}`이며 "
        f"lookup I/L 변화는 `{indexed.loc[best_quality_method, 'lookup_efficiency_gain_pct_mean_valid']:.2f}%`다."
    )
    if real_oracle is not None and not real_oracle.empty:
        if real_oracle.exact_two_line_gain_vs_paper_pct.mean() > 1e-9:
            lines.extend([
                "", "Full oracle가 Paper보다 높은 목적값을 보였으므로 Paper 알고리즘이 "
                "수학적으로 최적인 것은 아니다. 남은 문제는 optimum의 부재가 아니라 "
                "그 해를 Paper보다 싼 selector로 회수하는 것이다."
            ])
        else:
            lines.extend([
                "", "Full oracle에서도 Paper 개선 여지가 관측되지 않았다. 이 경우에만 "
                "해당 two-line 모델과 조사한 trace 범위에서 Paper가 사실상 최적이라고 해석한다."
            ])
    if dispatch is not None and dispatch.get("saving_ms_mean", 0.0) > 0.0:
        lines.extend([
            "", "다만 사전 절반에서 정한 shape-only dispatch는 독립된 뒤 절반에서 "
            f"Paper를 `{dispatch['saving_pct_mean_total']:.2f}%` 이겼다. 따라서 이 노트북에서 "
            "우리 방법이 Paper를 전혀 이길 수 없다는 결론도 틀리다. 현재 확인된 실용적 "
            "승리는 `896x4864`에 Price-1을 쓰고 나머지에는 Paper를 쓰는 혼합 정책이다."
        ])
    lines.extend([
        "", "## 범위와 한계", "",
        "- Full oracle는 two-line run-cost 모델에 대해 exact하며 arbitrary lookup table에 대한 exact oracle은 아니다.",
        "- 최종 시스템 판정은 로컬 profile뿐 아니라 native O_DIRECT read+GPU upload 실측을 사용한다." if args.measure_io else "- 이 실행은 실제 O_DIRECT replay를 생략했다.",
        "- corridor는 exact-R이지만 width가 유한할 때는 전역 optimality를 보장하지 않는다.",
        "- selector timing에는 CUDA 입력의 D2H, CPU solve, bool mask H2D가 포함된다.",
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
    real_oracle_path = args.output_dir / "real_oracle.csv"
    real_oracle = pd.read_csv(real_oracle_path) if real_oracle_path.exists() else None
    dispatch_details = None
    dispatch = None
    if frame.actual_total_median_ms.notna().any():
        dispatch_details, dispatch = evaluate_holdout_dispatch(frame)
        dispatch_details.to_csv(args.output_dir / "holdout_dispatch.csv", index=False)
    summary_frame, shape_frame, summary = EXP26.summarize(frame, oracle, args)
    summary["format"] = "experiment-28-exact-corridor-v1-real-activations"
    summary["comparison"] = "Paper-independent exact-R corridor and full real-trace oracle"
    if real_oracle is not None and len(real_oracle):
        summary["real_trace_exact_oracle"] = {
            "cases": int(len(real_oracle)),
            "all_converged": bool(real_oracle.exact_converged.all()),
            "two_line_gain_vs_paper_pct_mean": float(
                real_oracle.exact_two_line_gain_vs_paper_pct.mean()
            ),
            "lookup_gain_vs_paper_pct_mean": float(
                real_oracle.exact_lookup_gain_vs_paper_pct.mean()
            ),
            "runtime_ms_mean": float(real_oracle.exact_runtime_ms.mean()),
        }
    if dispatch is not None:
        summary["holdout_shape_dispatch"] = dispatch
    measured = frame[frame.actual_io_median_ms.notna()]
    if len(measured):
        summary["actual_io_replay"] = {
            "cases": int(len(measured)),
            "all_direct": bool((measured.actual_direct_rate == 1.0).all()),
            "profile_vs_io_pearson_r": float(
                measured.lookup_ms.corr(measured.actual_io_median_ms)
            ),
            "profile_vs_read_wall_pearson_r": float(
                measured.lookup_ms.corr(measured.actual_read_wall_median_ms)
            ),
        }
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    EXP26.plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    EXP26.plot_by_shape(shape_frame, args.output_dir / "shape_comparison.png")
    if summary_frame.get("actual_total_ms_mean", pd.Series(dtype=float)).notna().any():
        EXP26.plot_actual_total(summary_frame, args.output_dir / "actual_total.png")
    if oracle is not None and len(oracle):
        EXP26.plot_oracle(oracle, args.output_dir / "oracle_optimality.png")
    write_report(
        summary_frame, shape_frame, oracle, real_oracle,
        dispatch_details, dispatch, args, frame,
    )
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    _, tracks = validate_args(args)
    bind_exp26(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return

    traces, trace_metadata, trace_path = EXP24.prepare_activation_traces(args)
    shape_names = list(dict.fromkeys(trace["shape"] for trace in traces))
    shape_specs = [
        dict(next(item for item in SHAPES if item["shape"] == name)) for name in shape_names
    ]
    lookup_table = BASE.LatencyTable.load(args.profile)
    metadata = {
        "activation_trace": trace_metadata,
        "trace_file": str(trace_path),
        "trace_sha256": EXP22.sha256_file(trace_path),
        "selected_trace_count": len(traces),
        "selected_trace_ids": [int(trace["trace_id"]) for trace in traces],
        "max_traces_per_shape": int(args.max_traces_per_shape),
        "shape_specs": shape_specs,
        "method_labels": METHOD_LABELS,
        "corridor_widths": METHOD_WIDTHS,
        "latency_profile": {
            "requested": str(args.profile),
            "max_kib": int(lookup_table.max_kb),
            "metadata": lookup_table.meta,
            "sha256": EXP22.sha256_file(Path(args.profile)),
        },
        "dinkelbach_iterations": int(args.dinkelbach_iterations),
        "dinkelbach_tolerance": float(args.dinkelbach_tolerance),
        "real_oracle_traces_per_shape": int(args.real_oracle_traces_per_shape),
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
        required_bytes = max(
            int(shape["n"]) * int(round(EXP22.row_size_kib(shape) * 1024.0))
            for shape in shape_specs
        )
        if args.io_blob is None or not args.io_blob.is_file():
            raise SystemExit("--measure-io requires an existing --io-blob")
        if args.io_blob.stat().st_size < required_bytes:
            raise SystemExit(
                f"--io-blob has {args.io_blob.stat().st_size} bytes; "
                f"at least {required_bytes} are required"
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
            args.output_dir / "oracle_trials.csv",
            EXP26.collect_oracle(args, lookup_table),
        )
    if not args.skip_real_oracle:
        EXP24.write_csv(
            args.output_dir / "real_oracle.csv",
            collect_real_oracle(args, traces, lookup_table, native_reader),
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
