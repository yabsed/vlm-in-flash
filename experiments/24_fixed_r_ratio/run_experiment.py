#!/usr/bin/env python3
"""Experiment 24: same-row Paper versus fixed-R two-line ratio selection."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import traceback
import warnings

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_22 = (
    PROJECT_ROOT / "experiments" / "22_predicted_lambda_trim"
    / "run_experiment.py"
)


def load_experiment_22():
    spec = importlib.util.spec_from_file_location("experiment_22_for_24", EXPERIMENT_22)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 22 from {EXPERIMENT_22}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP22 = load_experiment_22()
EXP20 = EXP22.EXP20
EXP18 = EXP22.EXP18
EXP13 = EXP22.EXP13
EXP2 = EXP22.EXP2
BASE = EXP22.BASE
njit = EXP13.njit
SHAPES = EXP22.SHAPES

METHODS = (
    "paper",
    "top_r",
    "fixed_r_mu4",
    "fixed_r_mu8",
    "fixed_r_d2_mu4",
)
FIXED_R_METHODS = METHODS[2:]
METHOD_LABELS = {
    "paper": "Paper",
    "top_r": "Top-R",
    "fixed_r_mu4": "Fixed-R 2L (rho1, mu4)",
    "fixed_r_mu8": "Fixed-R 2L (rho1, mu8)",
    "fixed_r_d2_mu4": "Fixed-R 2L (rho2, mu4)",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "top_r": "#475569",
    "fixed_r_mu4": "#7C3AED",
    "fixed_r_mu8": "#0F766E",
    "fixed_r_d2_mu4": "#064E3B",
}
METHOD_SETTINGS = {
    "fixed_r_mu4": (1, 4),
    "fixed_r_mu8": (1, 8),
    "fixed_r_d2_mu4": (2, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+", default=[item["shape"] for item in SHAPES],
        help="subset of Table-2 shapes, written as NxD",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--cv", type=float, default=3.30)
    parser.add_argument(
        "--row-budget-fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument(
        "--tracks", nargs="+", choices=("host", "cuda"), default=["host", "cuda"]
    )
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--deadline-ms", type=float, default=2.0)
    parser.add_argument("--paper-impl", choices=("auto", "native", "torch"), default="native")
    parser.add_argument("--oracle-n", type=int, default=18)
    parser.add_argument("--oracle-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20262401)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    by_name = {item["shape"]: item for item in SHAPES}
    unknown = sorted(set(args.shapes) - set(by_name))
    if unknown:
        raise SystemExit(f"unknown --shapes: {unknown}; available: {sorted(by_name)}")
    shapes = [dict(by_name[name]) for name in args.shapes]
    if len(shapes) != len({item["shape"] for item in shapes}):
        raise SystemExit("--shapes must not contain duplicates")
    if args.trials < 1 or args.repetitions < 1 or args.oracle_trials < 1:
        raise SystemExit("trial and repetition counts must be positive")
    if not 4 <= args.oracle_n <= 22:
        raise SystemExit("--oracle-n must lie in [4, 22]")
    if not args.row_budget_fractions or any(
        not 0.0 < value <= 1.0 for value in args.row_budget_fractions
    ):
        raise SystemExit("--row-budget-fractions must lie in (0, 1]")
    minimum_n = min([item["n"] for item in shapes] + [args.oracle_n])
    if args.cv <= 0 or args.cv >= math.sqrt(minimum_n - 1):
        raise SystemExit("--cv must lie in (0, sqrt(N-1)) for every N")
    if args.saturation_kib <= 0 or args.deadline_ms <= 0:
        raise SystemExit("saturation and deadline must be positive")
    tracks = list(dict.fromkeys(args.tracks))
    if "cuda" in tracks and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; omitting the cuda-resident round-trip track")
        tracks.remove("cuda")
    if not tracks:
        raise SystemExit("no runnable timing track remains")
    return shapes, tracks


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def top_r_mask(values: np.ndarray, rows: int) -> np.ndarray:
    n = len(values)
    rows = int(rows)
    if not 0 <= rows <= n:
        raise ValueError(f"row count must lie in [0, {n}]")
    mask = np.zeros(n, dtype=bool)
    if rows == n:
        mask[:] = True
    elif rows:
        indices = np.argpartition(values, n - rows)[n - rows:]
        mask[indices] = True
    return mask


def two_line_metrics(mask: np.ndarray, values: np.ndarray, model: dict) -> dict:
    return EXP18.mask_metrics_two_line(np.asarray(mask, dtype=bool), values, model)


@njit(cache=False, inline="always")
def _repair_chunk_ms(length, a_ms, c1_ms, c2_ms, saturation_rows):
    if length <= 0:
        return 0.0
    if length <= saturation_rows:
        return a_ms + c1_ms * length
    return c2_ms * length


@njit(cache=False, inline="always")
def _heap_less(key_a, index_a, key_b, index_b):
    return key_a < key_b or (key_a == key_b and index_a < index_b)


@njit(cache=False, inline="always")
def _heap_push(keys, runs, indices, versions, size, key, run, index, version):
    position = size
    keys[position] = key
    runs[position] = run
    indices[position] = index
    versions[position] = version
    size += 1
    while position > 0:
        parent = (position - 1) // 2
        if not _heap_less(keys[position], indices[position], keys[parent], indices[parent]):
            break
        keys[position], keys[parent] = keys[parent], keys[position]
        runs[position], runs[parent] = runs[parent], runs[position]
        indices[position], indices[parent] = indices[parent], indices[position]
        versions[position], versions[parent] = versions[parent], versions[position]
        position = parent
    return size


@njit(cache=False, inline="always")
def _heap_pop(keys, runs, indices, versions, size):
    key = keys[0]
    run = runs[0]
    index = indices[0]
    version = versions[0]
    size -= 1
    if size > 0:
        keys[0] = keys[size]
        runs[0] = runs[size]
        indices[0] = indices[size]
        versions[0] = versions[size]
        position = 0
        while True:
            left_child = 2 * position + 1
            if left_child >= size:
                break
            right_child = left_child + 1
            child = left_child
            if right_child < size and _heap_less(
                keys[right_child], indices[right_child],
                keys[left_child], indices[left_child],
            ):
                child = right_child
            if not _heap_less(keys[child], indices[child], keys[position], indices[position]):
                break
            keys[position], keys[child] = keys[child], keys[position]
            runs[position], runs[child] = runs[child], runs[position]
            indices[position], indices[child] = indices[child], indices[position]
            versions[position], versions[child] = versions[child], versions[position]
            position = child
    return key, run, index, version, size


@njit(cache=False)
def _remove_endpoints_kernel(mask, values, row_budget, rho, a_ms, c1_ms,
                             c2_ms, saturation_rows):
    output = mask.copy()
    n = len(output)
    selected = 0
    run_count = 0
    previous = False
    for index in range(n):
        if output[index]:
            selected += 1
            if not previous:
                run_count += 1
            previous = True
        else:
            previous = False
    left = np.empty(run_count, dtype=np.int64)
    right = np.empty(run_count, dtype=np.int64)
    versions_by_run = np.zeros(run_count, dtype=np.int64)
    run = -1
    previous = False
    for index in range(n):
        if output[index] and not previous:
            run += 1
            left[run] = index
        if output[index]:
            right[run] = index
        previous = output[index]

    # Two offers initially and after every deletion; stale entries remain.
    capacity = max(1, 2 * run_count + 2 * (selected - row_budget) + 4)
    heap_keys = np.empty(capacity, dtype=np.float64)
    heap_runs = np.empty(capacity, dtype=np.int64)
    heap_indices = np.empty(capacity, dtype=np.int64)
    heap_versions = np.empty(capacity, dtype=np.int64)
    heap_size = 0

    for current_run in range(run_count):
        length = right[current_run] - left[current_run] + 1
        saving = (
            _repair_chunk_ms(length, a_ms, c1_ms, c2_ms, saturation_rows)
            - _repair_chunk_ms(length - 1, a_ms, c1_ms, c2_ms, saturation_rows)
        )
        left_index = left[current_run]
        heap_size = _heap_push(
            heap_keys, heap_runs, heap_indices, heap_versions, heap_size,
            values[left_index] - rho * saving, current_run, left_index, 0,
        )
        if right[current_run] != left_index:
            right_index = right[current_run]
            heap_size = _heap_push(
                heap_keys, heap_runs, heap_indices, heap_versions, heap_size,
                values[right_index] - rho * saving, current_run, right_index, 0,
            )

    deletions = 0
    while selected > row_budget:
        if heap_size == 0:
            return output, -1
        _, current_run, index, version, heap_size = _heap_pop(
            heap_keys, heap_runs, heap_indices, heap_versions, heap_size
        )
        if version != versions_by_run[current_run] or not output[index]:
            continue
        if index != left[current_run] and index != right[current_run]:
            continue
        output[index] = False
        selected -= 1
        deletions += 1
        if left[current_run] == right[current_run]:
            left[current_run] = 1
            right[current_run] = 0
        elif index == left[current_run]:
            left[current_run] += 1
        else:
            right[current_run] -= 1
        versions_by_run[current_run] += 1
        length = right[current_run] - left[current_run] + 1
        if length <= 0:
            continue
        saving = (
            _repair_chunk_ms(length, a_ms, c1_ms, c2_ms, saturation_rows)
            - _repair_chunk_ms(length - 1, a_ms, c1_ms, c2_ms, saturation_rows)
        )
        left_index = left[current_run]
        heap_size = _heap_push(
            heap_keys, heap_runs, heap_indices, heap_versions, heap_size,
            values[left_index] - rho * saving, current_run, left_index,
            versions_by_run[current_run],
        )
        if right[current_run] != left_index:
            right_index = right[current_run]
            heap_size = _heap_push(
                heap_keys, heap_runs, heap_indices, heap_versions, heap_size,
                values[right_index] - rho * saving, current_run, right_index,
                versions_by_run[current_run],
            )
    return output, deletions


def remove_endpoints_to_r(mask: np.ndarray, values: np.ndarray, row_budget: int,
                          rho: float, model: dict) -> tuple[np.ndarray, dict]:
    """Peel run endpoints with minimum loss in I-rho*L until exactly R remain."""
    if int(np.asarray(mask, dtype=bool).sum()) < row_budget:
        raise ValueError("endpoint removal cannot increase cardinality")
    output, deletions = _remove_endpoints_kernel(
        np.asarray(mask, dtype=np.bool_), values, int(row_budget), float(rho),
        float(model["a_ms"]), float(model["c1_ms_per_row"]),
        float(model["c2_ms_per_row"]), float(model["saturation_rows"]),
    )
    if deletions < 0:
        raise RuntimeError("endpoint heap exhausted before reaching R")
    return np.asarray(output, dtype=bool), {
        "repair_deletions": int(deletions), "repair_additions": 0,
    }


def repair_cardinality(mask: np.ndarray, values: np.ndarray, row_budget: int,
                       rho: float, model: dict) -> tuple[np.ndarray, dict]:
    output = np.asarray(mask, dtype=bool).copy()
    selected = int(output.sum())
    if selected > row_budget:
        return remove_endpoints_to_r(output, values, row_budget, rho, model)
    additions = 0
    if selected < row_budget:
        need = row_budget - selected
        unselected = np.flatnonzero(~output)
        chosen = unselected[np.argpartition(values[unselected], len(unselected) - need)[-need:]]
        output[chosen] = True
        additions = need
    return output, {"repair_deletions": 0, "repair_additions": additions}


def search_row_price(values: np.ndarray, row_budget: int, rho: float,
                     model: dict, max_calls: int) -> tuple[np.ndarray, dict]:
    """Bracket the row price and return the closest over-R supported mask."""
    if rho <= 0.0 or max_calls < 1:
        raise ValueError("rho and max_calls must be positive")
    n = len(values)
    if row_budget == n:
        return np.ones(n, dtype=bool), {
            "scalarized_calls": 0, "mu_exact_hit": True,
            "pre_repair_rows": n, "repair_deletions": 0, "repair_additions": 0,
        }
    eps = np.finfo(np.float64).eps
    singleton_cost = EXP18.two_line_chunk_ms(1, model)
    low = float(values.min() - rho * singleton_cost - 32.0 * eps)
    high = float(values.max() + 32.0 * eps)
    threshold = float(np.partition(values, n - row_budget)[n - row_budget])
    predicted = float(threshold - rho * model["c2_ms_per_row"])
    predicted = min(max(predicted, np.nextafter(low, high)), np.nextafter(high, low))
    best_mu: float | None = None
    best_rows = n + 1
    calls = 0
    exact_hit = False
    multiplier = 1.0 / rho

    for call_index in range(max_calls):
        mu = predicted if call_index == 0 else 0.5 * (low + high)
        adjusted = values - mu
        solved = EXP13._solve_lambda_metrics(
            adjusted, multiplier, model["a_ms"], model["c1_ms_per_row"],
            model["c2_ms_per_row"], model["saturation_rows"],
            model["short_max_rows"],
        )
        calls += 1
        rows = int(solved[3])
        if rows >= row_budget:
            low = mu
            if rows < best_rows:
                best_rows = rows
                best_mu = mu
        else:
            high = mu
        if rows == row_budget:
            exact_hit = True
            break

    if best_mu is None:
        candidate = np.ones(n, dtype=bool)
        best_rows = n
    else:
        adjusted = values - best_mu
        replay = EXP13._solve_lambda_mask(
            adjusted, multiplier, model["a_ms"], model["c1_ms_per_row"],
            model["c2_ms_per_row"], model["saturation_rows"],
            model["short_max_rows"],
        )
        candidate = np.asarray(replay[0], dtype=bool)
        best_rows = int(candidate.sum())
    candidate, repair = repair_cardinality(candidate, values, row_budget, rho, model)
    if int(candidate.sum()) != row_budget:
        raise RuntimeError("row-price repair failed to produce exactly R rows")
    return candidate, {
        "scalarized_calls": calls,
        "mu_exact_hit": exact_hit,
        "pre_repair_rows": best_rows,
        **repair,
    }


def fixed_r_ratio(values: np.ndarray, row_budget: int, model: dict,
                  outer_iterations: int, mu_calls: int) -> tuple[np.ndarray, dict]:
    """Approximate fixed-R Dinkelbach iterations with an O(N) row-price DP."""
    initial = top_r_mask(values, row_budget)
    initial_metrics = two_line_metrics(initial, values, model)
    rho = float(initial_metrics["importance"] / initial_metrics["two_line_ms"])
    best_mask = initial
    best_ratio = rho
    total_calls = 0
    total_deletions = 0
    total_additions = 0
    exact_hits = 0
    pre_repair_rows = row_budget

    for _ in range(outer_iterations):
        candidate, metadata = search_row_price(
            values, row_budget, rho, model, mu_calls
        )
        metrics = two_line_metrics(candidate, values, model)
        candidate_ratio = float(metrics["importance"] / metrics["two_line_ms"])
        total_calls += int(metadata["scalarized_calls"])
        total_deletions += int(metadata["repair_deletions"])
        total_additions += int(metadata["repair_additions"])
        exact_hits += int(metadata["mu_exact_hit"])
        pre_repair_rows = int(metadata["pre_repair_rows"])
        if candidate_ratio > best_ratio + 1e-14:
            best_ratio = candidate_ratio
            best_mask = candidate
        rho = candidate_ratio

    return np.asarray(best_mask, dtype=bool), {
        "scalarized_calls": total_calls,
        "outer_iterations": int(outer_iterations),
        "mu_calls_per_outer": int(mu_calls),
        "mu_exact_hits": exact_hits,
        "pre_repair_rows": pre_repair_rows,
        "repair_deletions": total_deletions,
        "repair_additions": total_additions,
        "returned_top_r": bool(np.array_equal(best_mask, initial)),
    }


def safe_select(method: str, values: np.ndarray, row_budget: int,
                model: dict) -> tuple[np.ndarray, dict]:
    try:
        if method == "top_r":
            return top_r_mask(values, row_budget), {}
        outer, calls = METHOD_SETTINGS[method]
        mask, metadata = fixed_r_ratio(values, row_budget, model, outer, calls)
        metadata = dict(metadata)
        metadata.update({"fallback_used": False, "error": ""})
        return mask, metadata
    except Exception as exception:
        return top_r_mask(values, row_budget), {
            "fallback_used": True,
            "error": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(limit=3),
        }


def selector_function(method: str, track: str, values_host: np.ndarray,
                      values_cuda: torch.Tensor | None, row_budget: int,
                      model: dict):
    def run():
        if track == "cuda":
            values = values_cuda.detach().to("cpu").numpy().astype(np.float64)
        else:
            values = values_host
        mask, metadata = safe_select(method, values, row_budget, model)
        handoff = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_).copy())
        if track == "cuda":
            handoff = handoff.to("cuda")
        return handoff, metadata
    return run


@njit(cache=False)
def _exact_fixed_r_ratios(values, budgets, a_ms, c1_ms, c2_ms, saturation_rows):
    """Exhaustive small-N oracle for all requested cardinalities in one pass."""
    n = len(values)
    best_ratio = np.full(len(budgets), -np.inf, dtype=np.float64)
    best_importance = np.zeros(len(budgets), dtype=np.float64)
    best_cost = np.zeros(len(budgets), dtype=np.float64)
    best_code = np.zeros(len(budgets), dtype=np.int64)
    for code in range(1 << n):
        rows = 0
        importance = 0.0
        cost = 0.0
        run = 0
        for index in range(n):
            if (code >> index) & 1:
                rows += 1
                importance += values[index]
                run += 1
            elif run:
                if run <= saturation_rows:
                    cost += a_ms + c1_ms * run
                else:
                    cost += c2_ms * run
                run = 0
        if run:
            if run <= saturation_rows:
                cost += a_ms + c1_ms * run
            else:
                cost += c2_ms * run
        if cost <= 0.0:
            continue
        ratio = importance / cost
        for budget_index in range(len(budgets)):
            if rows == budgets[budget_index] and ratio > best_ratio[budget_index]:
                best_ratio[budget_index] = ratio
                best_importance[budget_index] = importance
                best_cost[budget_index] = cost
                best_code[budget_index] = code
    return best_ratio, best_importance, best_cost, best_code


def encoded_mask(code: int, n: int) -> np.ndarray:
    return np.asarray([(int(code) >> index) & 1 for index in range(n)], dtype=bool)


def self_check(lookup_table, saturation_kib: float) -> None:
    rng = np.random.default_rng(2401)
    shape = next(item for item in SHAPES if item["shape"] == "4096x14336")
    row_kib = EXP22.row_size_kib(shape)
    model = EXP13.fit_continuous_two_line(lookup_table, row_kib, saturation_kib)
    values = rng.lognormal(size=12)
    values /= values.sum()
    budgets = np.asarray([3, 6, 9], dtype=np.int64)
    exact = _exact_fixed_r_ratios(
        values, budgets, model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"],
    )
    for index, budget in enumerate(budgets):
        mask = encoded_mask(int(exact[3][index]), len(values))
        metrics = two_line_metrics(mask, values, model)
        if (
            int(mask.sum()) != budget
            or not math.isclose(
                metrics["importance"] / metrics["two_line_ms"],
                float(exact[0][index]), rel_tol=1e-11, abs_tol=1e-12,
            )
        ):
            raise RuntimeError("small-N fixed-R exhaustive oracle self-check failed")

    values = rng.lognormal(size=97)
    values /= values.sum()
    for budget in (24, 48, 73):
        top = top_r_mask(values, budget)
        top_ratio = (
            two_line_metrics(top, values, model)["importance"]
            / two_line_metrics(top, values, model)["two_line_ms"]
        )
        for outer, calls in METHOD_SETTINGS.values():
            mask, metadata = fixed_r_ratio(values, budget, model, outer, calls)
            metrics = two_line_metrics(mask, values, model)
            if (
                int(mask.sum()) != budget
                or metrics["importance"] / metrics["two_line_ms"] < top_ratio - 1e-11
                or metadata["scalarized_calls"] > outer * calls
            ):
                raise RuntimeError("fixed-R online selector self-check failed")


def collect_oracle(args: argparse.Namespace, lookup_table) -> list[dict]:
    n = int(args.oracle_n)
    shape = next(item for item in SHAPES if item["shape"] == "4096x14336")
    row_kib = EXP22.row_size_kib(shape)
    model = EXP13.fit_continuous_two_line(
        lookup_table, row_kib, args.saturation_kib
    )
    budgets = np.asarray([
        max(1, min(n, int(round(n * fraction))))
        for fraction in args.row_budget_fractions
    ], dtype=np.int64)
    rng = np.random.default_rng(args.seed + 101)
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    rows: list[dict] = []
    for trial in range(args.oracle_trials):
        multiset = EXP2.exact_cv_lognormal(rng, n, args.cv)
        variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
        for spatial_mode, raw_values in variants.items():
            values32 = np.asarray(raw_values, dtype=np.float32)
            values32 /= np.float32(values32.astype(np.float64).sum())
            values = values32.astype(np.float64)
            exact = _exact_fixed_r_ratios(
                values, budgets, model["a_ms"], model["c1_ms_per_row"],
                model["c2_ms_per_row"], model["saturation_rows"],
            )
            for budget_index, budget in enumerate(budgets):
                exact_ratio = float(exact[0][budget_index])
                for method in METHODS[1:]:
                    mask, metadata = safe_select(method, values, int(budget), model)
                    metrics = two_line_metrics(mask, values, model)
                    ratio = metrics["importance"] / metrics["two_line_ms"]
                    rows.append({
                        "n": n, "trial": trial, "spatial_mode": spatial_mode,
                        "budget_index": budget_index,
                        "row_budget_fraction": float(args.row_budget_fractions[budget_index]),
                        "row_budget": int(budget), "method": method,
                        "method_label": METHOD_LABELS[method],
                        "importance": metrics["importance"],
                        "two_line_ms": metrics["two_line_ms"],
                        "two_line_efficiency": ratio,
                        "exact_importance": float(exact[1][budget_index]),
                        "exact_two_line_ms": float(exact[2][budget_index]),
                        "exact_efficiency": exact_ratio,
                        "optimality_ratio": ratio / exact_ratio,
                        "optimality_gap_pct": 100.0 * (1.0 - ratio / exact_ratio),
                        "row_match": int(mask.sum()) == int(budget),
                        "fallback_used": metadata.get("fallback_used", False),
                        "error": metadata.get("error", ""),
                    })
    return rows


def collect(args: argparse.Namespace, shapes: list[dict], tracks: list[str],
            lookup_table) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    timing_rows: list[dict] = []
    model_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for shape_index, shape in enumerate(shapes):
        n = int(shape["n"])
        row_kib = EXP22.row_size_kib(shape)
        params = EXP22.make_params(shape, args.saturation_kib)
        model = EXP13.fit_continuous_two_line(
            lookup_table, row_kib, args.saturation_kib
        )
        model_rows.append({**shape, "row_size_kib": row_kib, **model})
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()

        for trial in range(args.trials):
            multiset = EXP2.exact_cv_lognormal(rng, n, args.cv)
            variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
            for spatial_mode, raw_values in variants.items():
                values32 = np.asarray(raw_values, dtype=np.float32)
                values32 /= np.float32(values32.astype(np.float64).sum())
                values = values32.astype(np.float64)
                values_cuda = (
                    torch.from_numpy(values32).to("cuda") if "cuda" in tracks else None
                )
                for budget_index, budget_fraction in enumerate(args.row_budget_fractions):
                    nominal_budget = max(1, min(n, int(round(n * budget_fraction))))
                    for track in tracks:
                        pfun = EXP22.paper_function(
                            track, values, values_cuda, nominal_budget, row_kib,
                            lookup_table, params, args.paper_impl,
                        )
                        pfun()
                        paper_mask, paper_meta, paper_timings = EXP22.benchmark(
                            pfun, args.repetitions, bool(torch.cuda.is_available())
                        )
                        effective_r = int(paper_mask.sum())
                        if effective_r <= 0:
                            raise RuntimeError(
                                f"Paper selected no rows for {shape['shape']} at R={nominal_budget}"
                            )
                        paper_metrics = EXP18.mask_metrics(
                            paper_mask, values, model, lookup_table, row_kib
                        )
                        case_base = {
                            "shape": shape["shape"], "shape_index": shape_index,
                            "n": n, "d": int(shape["d"]), "row_size_kib": row_kib,
                            "paper_start_kib": float(shape["start_kib"]),
                            "paper_jump_kib": float(shape["jump_kib"]),
                            "trial": trial, "spatial_mode": spatial_mode,
                            "budget_index": budget_index,
                            "row_budget_fraction": float(budget_fraction),
                            "nominal_row_budget": nominal_budget,
                            "effective_row_budget": effective_r,
                            "paper_underfill_rows": nominal_budget - effective_r,
                            "track": track,
                        }

                        def append_result(method: str, mask: np.ndarray, metadata: dict,
                                          timings: list[float]) -> None:
                            metrics = EXP18.mask_metrics(
                                mask, values, model, lookup_table, row_kib
                            )
                            row_match = metrics["rows"] == effective_r
                            error = metadata.get("error", "")
                            runtime_p95 = float(np.quantile(timings, 0.95))
                            base = {
                                **case_base, "method": method,
                                "method_label": METHOD_LABELS[method],
                            }
                            rows.append({
                                **base, **metrics,
                                "lookup_efficiency": metrics["importance"] / metrics["lookup_ms"],
                                "two_line_efficiency": metrics["importance"] / metrics["two_line_ms"],
                                "row_match": row_match,
                                "valid": row_match and not bool(error),
                                "runtime_median_ms": float(np.median(timings)),
                                "runtime_p95_ms": runtime_p95,
                                "deadline_met": runtime_p95 <= args.deadline_ms,
                                "valid_and_deadline_met": (
                                    row_match and not bool(error)
                                    and runtime_p95 <= args.deadline_ms
                                ),
                                "deterministic": metadata.get("deterministic", False),
                                "scalarized_calls": metadata.get("scalarized_calls", 0),
                                "outer_iterations": metadata.get("outer_iterations", 0),
                                "mu_exact_hits": metadata.get("mu_exact_hits", 0),
                                "pre_repair_rows": metadata.get("pre_repair_rows", effective_r),
                                "repair_deletions": metadata.get("repair_deletions", 0),
                                "repair_additions": metadata.get("repair_additions", 0),
                                "returned_top_r": metadata.get("returned_top_r", False),
                                "fallback_used": metadata.get("fallback_used", False),
                                "error": error,
                            })
                            for repetition, elapsed in enumerate(timings):
                                timing_rows.append({
                                    **base, "repetition": repetition,
                                    "runtime_ms": elapsed,
                                })

                        append_result("paper", paper_mask, paper_meta, paper_timings)
                        for method in METHODS[1:]:
                            function = selector_function(
                                method, track, values, values_cuda, effective_r, model
                            )
                            function()
                            mask, metadata, timings = EXP22.benchmark(
                                function, args.repetitions, track == "cuda"
                            )
                            append_result(method, mask, metadata, timings)

                print(
                    f"shape={shape['shape']} ({shape_index + 1}/{len(shapes)}) "
                    f"trial={trial + 1}/{args.trials} mode={spatial_mode}",
                    flush=True,
                )
    return rows, timing_rows, model_rows


def add_paired_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    derived = [
        "paper_importance", "paper_lookup_ms", "paper_two_line_ms",
        "paper_lookup_efficiency", "paper_two_line_efficiency",
        "paper_runtime_median_ms", "importance_gain_pct", "lookup_saving_pct",
        "two_line_saving_pct", "lookup_efficiency_gain_pct",
        "two_line_efficiency_gain_pct", "lookup_saving_abs_ms",
        "selector_overhead_median_ms", "mixed_net_saving_ms", "mixed_net_win",
    ]
    frame = frame.drop(columns=[column for column in derived if column in frame])
    keys = ["shape", "trial", "spatial_mode", "budget_index", "track"]
    paper = frame[frame.method == "paper"][keys + [
        "importance", "lookup_ms", "two_line_ms", "lookup_efficiency",
        "two_line_efficiency", "runtime_median_ms",
    ]].rename(columns={
        "importance": "paper_importance",
        "lookup_ms": "paper_lookup_ms",
        "two_line_ms": "paper_two_line_ms",
        "lookup_efficiency": "paper_lookup_efficiency",
        "two_line_efficiency": "paper_two_line_efficiency",
        "runtime_median_ms": "paper_runtime_median_ms",
    })
    output = frame.merge(paper, on=keys, how="left")
    output["importance_gain_pct"] = 100.0 * (
        output.importance / output.paper_importance - 1.0
    )
    output["lookup_saving_pct"] = 100.0 * (
        1.0 - output.lookup_ms / output.paper_lookup_ms
    )
    output["two_line_saving_pct"] = 100.0 * (
        1.0 - output.two_line_ms / output.paper_two_line_ms
    )
    output["lookup_efficiency_gain_pct"] = 100.0 * (
        output.lookup_efficiency / output.paper_lookup_efficiency - 1.0
    )
    output["two_line_efficiency_gain_pct"] = 100.0 * (
        output.two_line_efficiency / output.paper_two_line_efficiency - 1.0
    )
    output["lookup_saving_abs_ms"] = output.paper_lookup_ms - output.lookup_ms
    output["selector_overhead_median_ms"] = (
        output.runtime_median_ms - output.paper_runtime_median_ms
    )
    output["mixed_net_saving_ms"] = (
        output.lookup_saving_abs_ms - output.selector_overhead_median_ms
    )
    output["mixed_net_win"] = output.mixed_net_saving_ms > 0.0
    return output


def distribution(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        return {key: None for key in ("mean", "median", "p05", "p95", "min", "max")}
    return {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()), "max": float(x.max()),
    }


def aggregate_group(group: pd.DataFrame) -> dict:
    valid = group[group.valid]
    return {
        "cases": int(len(group)), "valid_cases": int(len(valid)),
        "valid_rate": float(group.valid.mean()),
        "exact_row_match_rate": float(group.row_match.mean()),
        "deadline_pass_rate": float(group.deadline_met.mean()),
        "valid_and_deadline_pass_rate": float(group.valid_and_deadline_met.mean()),
        "deterministic_rate": float(group.deterministic.mean()),
        "fallback_rate": float(group.fallback_used.mean()),
        "error_rate": float(group.error.fillna("").astype(bool).mean()),
        "runtime_median_ms": float(group.runtime_median_ms.median()),
        "runtime_case_p95_ms": float(group.runtime_p95_ms.quantile(0.95)),
        "runtime_worst_case_p95_ms": float(group.runtime_p95_ms.max()),
        "lookup_efficiency_gain_pct_mean_valid": (
            float(valid.lookup_efficiency_gain_pct.mean()) if len(valid) else math.nan
        ),
        "lookup_efficiency_gain_pct_median_valid": (
            float(valid.lookup_efficiency_gain_pct.median()) if len(valid) else math.nan
        ),
        "lookup_efficiency_win_rate_valid": (
            float((valid.lookup_efficiency_gain_pct > 1e-9).mean())
            if len(valid) else math.nan
        ),
        "two_line_efficiency_gain_pct_mean_valid": (
            float(valid.two_line_efficiency_gain_pct.mean()) if len(valid) else math.nan
        ),
        "two_line_efficiency_win_rate_valid": (
            float((valid.two_line_efficiency_gain_pct > 1e-9).mean())
            if len(valid) else math.nan
        ),
        "importance_gain_pct_mean_valid": (
            float(valid.importance_gain_pct.mean()) if len(valid) else math.nan
        ),
        "importance_nonregression_rate_valid": (
            float((valid.importance_gain_pct >= -1e-9).mean())
            if len(valid) else math.nan
        ),
        "lookup_saving_pct_mean_valid": (
            float(valid.lookup_saving_pct.mean()) if len(valid) else math.nan
        ),
        "lookup_saving_pct_weighted_valid": (
            100.0 * (1.0 - float(valid.lookup_ms.sum()) / float(valid.paper_lookup_ms.sum()))
            if len(valid) and float(valid.paper_lookup_ms.sum()) > 0 else math.nan
        ),
        "lookup_nonregression_rate_valid": (
            float((valid.lookup_saving_pct >= -1e-9).mean())
            if len(valid) else math.nan
        ),
        "pareto_nonregression_rate_valid": (
            float(((valid.importance_gain_pct >= -1e-9)
                   & (valid.lookup_saving_pct >= -1e-9)).mean())
            if len(valid) else math.nan
        ),
        "lookup_saving_abs_ms_mean_valid": (
            float(valid.lookup_saving_abs_ms.mean()) if len(valid) else math.nan
        ),
        "selector_overhead_median_ms_median_valid": (
            float(valid.selector_overhead_median_ms.median()) if len(valid) else math.nan
        ),
        "mixed_net_win_rate_valid": (
            float(valid.mixed_net_win.mean()) if len(valid) else math.nan
        ),
        "scalarized_calls_median": float(group.scalarized_calls.median()),
        "repair_deletions_mean": float(group.repair_deletions.mean()),
        "repair_additions_mean": float(group.repair_additions.mean()),
        "returned_top_r_rate": float(group.returned_top_r.mean()),
        "paper_underfill_rows_mean": float(group.paper_underfill_rows.mean()),
    }


def summarize(frame: pd.DataFrame, oracle: pd.DataFrame | None,
              args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary_rows = []
    nested = {}
    for (track, method), group in frame.groupby(["track", "method"], sort=False):
        row = {
            "track": track, "method": method, "method_label": METHOD_LABELS[method],
            **aggregate_group(group),
        }
        summary_rows.append(row)
        nested.setdefault(track, {})[method] = {
            **row,
            "runtime_median_ms_distribution": distribution(group.runtime_median_ms),
            "runtime_p95_ms_distribution": distribution(group.runtime_p95_ms),
            "lookup_efficiency_gain_pct_valid_distribution": distribution(
                group.loc[group.valid, "lookup_efficiency_gain_pct"]
            ),
        }
    shape_rows = []
    for (track, shape, method), group in frame.groupby(
        ["track", "shape", "method"], sort=False
    ):
        first = group.iloc[0]
        shape_rows.append({
            "track": track, "shape": shape, "n": int(first.n), "d": int(first.d),
            "row_size_kib": float(first.row_size_kib), "method": method,
            "method_label": METHOD_LABELS[method], **aggregate_group(group),
        })
    oracle_summary = {}
    if oracle is not None and len(oracle):
        for method, group in oracle.groupby("method", sort=False):
            oracle_summary[method] = {
                "cases": int(len(group)),
                "optimality_ratio_mean": float(group.optimality_ratio.mean()),
                "optimality_ratio_median": float(group.optimality_ratio.median()),
                "optimality_ratio_min": float(group.optimality_ratio.min()),
                "optimality_gap_pct_p95": float(group.optimality_gap_pct.quantile(0.95)),
                "exact_hit_rate": float((group.optimality_gap_pct <= 1e-9).mean()),
                "row_match_rate": float(group.row_match.mean()),
            }
    summary = {
        "format": "experiment-24-fixed-r-ratio-v1",
        "comparison": (
            "Paper mask versus proposed mask at exact paired cardinality "
            "R_eff = rows(Paper mask)"
        ),
        "deadline_ms": float(args.deadline_ms),
        "timing_scope": {
            "host": "warm host float32 importance to CPU bool mask",
            "cuda": (
                "warm CUDA float32 importance to CUDA bool mask; CPU methods include "
                "D2H, float64 conversion, selection/repair, H2D, and synchronization"
            ),
            "excluded": "importance production, actual storage I/O, and model compute",
        },
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": int(torch.get_num_threads()),
        },
        "configuration": {
            "shape_count": int(frame["shape"].nunique()),
            "shapes": list(dict.fromkeys(frame["shape"])),
            "trials": int(args.trials), "repetitions": int(args.repetitions),
            "cv": float(args.cv),
            "row_budget_fractions": list(args.row_budget_fractions),
            "tracks": list(dict.fromkeys(frame["track"])),
            "profile": args.profile, "saturation_kib": float(args.saturation_kib),
            "paper_impl": args.paper_impl,
            "oracle_n": int(args.oracle_n), "oracle_trials": int(args.oracle_trials),
        },
        "tracks": nested,
        "small_n_oracle": oracle_summary,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(shape_rows), summary


def summarize_hybrids(frame: pd.DataFrame, shape_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track in dict.fromkeys(frame.track):
        for method in ("fixed_r_mu4", "fixed_r_d2_mu4"):
            safe_shapes = set(shape_summary[
                (shape_summary.track == track)
                & (shape_summary.method == method)
                & (shape_summary.valid_and_deadline_pass_rate == 1.0)
            ]["shape"])
            chosen = frame[
                (frame.track == track)
                & (
                    ((frame.method == method) & frame["shape"].isin(safe_shapes))
                    | ((frame.method == "paper") & ~frame["shape"].isin(safe_shapes))
                )
            ]
            rows.append({
                "track": track,
                "method": f"hybrid_{method}",
                "method_label": f"Shape hybrid: {METHOD_LABELS[method]}",
                "candidate_method": method,
                "safe_shape_count": len(safe_shapes),
                "safe_shapes": ";".join(sorted(safe_shapes)),
                **aggregate_group(chosen),
            })
    return pd.DataFrame(rows)


def configure_plot():
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_overall(summary: pd.DataFrame, path: Path, deadline_ms: float) -> None:
    plt = configure_plot()
    tracks = list(dict.fromkeys(summary.track))
    fig, axes = plt.subplots(1, len(tracks), figsize=(7.0 * len(tracks), 5.6),
                             squeeze=False, constrained_layout=True)
    for ax, track in zip(axes[0], tracks):
        selected = summary[summary.track == track].set_index("method").reindex(METHODS)
        for method, row in selected.iterrows():
            ax.scatter(
                row.runtime_median_ms, row.lookup_efficiency_gain_pct_mean_valid,
                s=85, color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3,
            )
        ax.axvline(deadline_ms, color="#111827", linestyle="--", linewidth=1.2)
        ax.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Median selector time (ms)")
        ax.set_ylabel("Mean lookup I/L gain vs Paper (%)")
        ax.set_title("Host input / CPU mask" if track == "host" else "CUDA input / CUDA mask")
        BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_by_shape(shape_summary: pd.DataFrame, method: str, path: Path) -> None:
    plt = configure_plot()
    tracks = list(dict.fromkeys(shape_summary.track))
    fig, axes = plt.subplots(
        len(tracks), 2, figsize=(15.0, 5.4 * len(tracks)),
        squeeze=False, constrained_layout=True,
    )
    for row_index, track in enumerate(tracks):
        selected = shape_summary[shape_summary.track == track]
        order = (
            selected[selected.method == "paper"].sort_values(["n", "d"])["shape"].tolist()
        )
        x = np.arange(len(order))
        chosen = selected[selected.method == method].set_index("shape").reindex(order)
        paper = selected[selected.method == "paper"].set_index("shape").reindex(order)
        ax_quality, ax_runtime = axes[row_index]
        ax_quality.bar(
            x, chosen.lookup_efficiency_gain_pct_mean_valid,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
        ax_quality.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
        ax_runtime.plot(
            x, paper.runtime_case_p95_ms, marker="o", color=METHOD_COLORS["paper"],
            label=METHOD_LABELS["paper"],
        )
        ax_runtime.plot(
            x, chosen.runtime_case_p95_ms, marker="o", color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax_runtime.axhline(2.0, color="#111827", linestyle="--", linewidth=1.1)
        for ax in (ax_quality, ax_runtime):
            ax.set_xticks(x, order, rotation=55, ha="right")
            BASE.polish_axis(ax)
        label = "Host" if track == "host" else "CUDA round trip"
        ax_quality.set_ylabel("Mean lookup I/L gain vs Paper (%)")
        ax_quality.set_title(f"{label}: exact same-row efficiency")
        ax_runtime.set_ylabel("95th percentile of case p95 (ms)")
        ax_runtime.set_title(f"{label}: selector scaling")
    handles, labels = axes[-1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_oracle(oracle: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    data = [
        100.0 * oracle.loc[oracle.method == method, "optimality_ratio"].to_numpy()
        for method in METHODS[1:]
    ]
    ax.boxplot(
        data, tick_labels=[METHOD_LABELS[method] for method in METHODS[1:]],
        showfliers=True, flierprops={"markersize": 3.5, "alpha": 0.65},
    )
    ax.axhline(100.0, color="#111827", linestyle="--", linewidth=1.1)
    ax.set_ylabel("Two-line fixed-R optimum recovered (%)")
    ax.tick_params(axis="x", rotation=18)
    BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}%"


def write_report(summary: pd.DataFrame, shape_summary: pd.DataFrame,
                 hybrid_summary: pd.DataFrame, oracle: pd.DataFrame | None,
                 args: argparse.Namespace) -> str:
    primary_track = "cuda" if "cuda" in set(summary.track) else "host"
    primary = summary[summary.track == primary_track].set_index("method")
    winner_method = primary.loc[list(FIXED_R_METHODS)].sort_values(
        "lookup_efficiency_gain_pct_mean_valid", ascending=False
    ).index[0]
    winner = primary.loc[winner_method]
    total_shape_count = int(shape_summary["shape"].nunique())
    lines = [
        "# Experiment 24 보고서: 동일 row 수의 fixed-R 비율 최적화",
        "",
        "각 paired case에서 Paper를 원래 nominal budget으로 실행한 뒤 실제 선택 행 수 "
        "`R_eff = |M_paper|`를 측정했다. Top-R와 모든 proposed mask는 정확히 `R_eff`개를 "
        "선택한다. 따라서 Experiment 22와 달리 실제 row 수가 완전히 같다.",
        "",
        "제안법은 two-line 목적 `I/L`을 최적화한다. 공개 lookup table의 `I/L`, importance, "
        "lookup latency는 별도 지표이며 서로 혼동하지 않는다.",
        "",
        "## 전체 결과", "",
    ]
    for track, track_label in (("host", "Host input -> CPU mask"),
                               ("cuda", "CUDA input -> CUDA mask")):
        selected = summary[summary.track == track]
        if selected.empty:
            continue
        lines.extend([
            f"### {track_label}", "",
            "| 방법 | 중앙 runtime | case-p95 | 유효+2ms | lookup I/L | two-line I/L | importance | lookup 절감 | 혼합 손익분기 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        indexed = selected.set_index("method").reindex(METHODS)
        for method, row in indexed.iterrows():
            lines.append(
                f"| {METHOD_LABELS[method]} | {fmt(row.runtime_median_ms)} ms "
                f"| {fmt(row.runtime_case_p95_ms)} ms "
                f"| {100 * row.valid_and_deadline_pass_rate:.1f}% "
                f"| {fmt_pct(row.lookup_efficiency_gain_pct_mean_valid)} "
                f"| {fmt_pct(row.two_line_efficiency_gain_pct_mean_valid)} "
                f"| {fmt_pct(row.importance_gain_pct_mean_valid)} "
                f"| {fmt_pct(row.lookup_saving_pct_weighted_valid)} "
                f"| {100 * row.mixed_net_win_rate_valid:.1f}% |"
            )
        lines.append("")

    lines.extend(["## Small-N exact oracle", ""])
    if oracle is None or not len(oracle):
        lines.extend(["Oracle 실행을 생략했다.", ""])
    else:
        lines.extend([
            f"`N={args.oracle_n}` exhaustive fixed-R optimum과 비교했다.", "",
            "| 방법 | 평균 optimum 회수 | 최악 회수 | p95 gap | exact hit |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in METHODS[1:]:
            group = oracle[oracle.method == method]
            lines.append(
                f"| {METHOD_LABELS[method]} "
                f"| {100 * group.optimality_ratio.mean():.3f}% "
                f"| {100 * group.optimality_ratio.min():.3f}% "
                f"| {group.optimality_gap_pct.quantile(0.95):.3f}% "
                f"| {100 * (group.optimality_gap_pct <= 1e-9).mean():.1f}% |"
            )
        lines.append("")

    lines.extend([
        "## Shape-adaptive hybrid", "",
        "각 timing track에서 모든 case가 2 ms를 통과한 shape에만 fixed-R selector를 "
        "사용하고 나머지는 Paper로 되돌리는 정적 정책이다.", "",
        "| Track | 후보 | fixed-R shape | deadline | lookup I/L | importance | 가중 lookup 절감 | case-p95 | worst p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in hybrid_summary.iterrows():
        lines.append(
            f"| {row.track} | {METHOD_LABELS[row.candidate_method]} "
            f"| {int(row.safe_shape_count)}/{total_shape_count} "
            f"| {100 * row.valid_and_deadline_pass_rate:.1f}% "
            f"| {fmt_pct(row.lookup_efficiency_gain_pct_mean_valid)} "
            f"| {fmt_pct(row.importance_gain_pct_mean_valid)} "
            f"| {fmt_pct(row.lookup_saving_pct_weighted_valid)} "
            f"| {fmt(row.runtime_case_p95_ms)} ms "
            f"| {fmt(row.runtime_worst_case_p95_ms)} ms |"
        )
    lines.append("")

    winner_shapes = shape_summary[
        (shape_summary.track == primary_track) & (shape_summary.method == winner_method)
    ].sort_values(["n", "d"])
    fully_passing = int((winner_shapes.valid_and_deadline_pass_rate == 1.0).sum())
    practical_hybrid = hybrid_summary[
        (hybrid_summary.track == primary_track)
        & (hybrid_summary.candidate_method == "fixed_r_mu4")
    ].iloc[0]
    lines.extend([
        "## 판정", "",
        f"lookup-table `I/L` 평균이 가장 높은 fixed-R 설정은 `{METHOD_LABELS[winner_method]}`다. "
        f"Paper 대비 lookup `I/L`은 평균 `{winner.lookup_efficiency_gain_pct_mean_valid:.2f}%`, "
        f"two-line `I/L`은 `{winner.two_line_efficiency_gain_pct_mean_valid:.2f}%` 변했다.",
        "",
        f"모든 method case의 정확한 row-match와 deterministic rate는 각각 "
        f"`{100 * winner.exact_row_match_rate:.1f}%`, `{100 * winner.deterministic_rate:.1f}%`다. "
        f"`{primary_track}` track에서 모든 case가 2 ms를 통과한 shape는 "
        f"`{fully_passing}/{len(winner_shapes)}`개다.",
        "",
        f"importance가 Paper보다 낮지 않은 비율은 "
        f"`{100 * winner.importance_nonregression_rate_valid:.1f}%`, lookup도 동시에 나쁘지 않은 "
        f"Pareto 비악화율은 `{100 * winner.pareto_nonregression_rate_valid:.1f}%`다. "
        "따라서 비율 개선만으로 동일 accuracy를 주장할 수 없다.",
        "",
        f"노트북 selector overhead와 Orin lookup 예측을 섞은 진단적 손익분기 통과율은 "
        f"`{100 * winner.mixed_net_win_rate_valid:.1f}%`다. 서로 다른 장치의 시간을 더한 값이므로 "
        "end-to-end 결과가 아니라 Jetson에서 검증할 조건이다.",
        "",
        f"2 ms 우선의 현실적인 선택은 `{METHOD_LABELS['fixed_r_mu4']}` shape hybrid다. "
        f"모든 case의 deadline을 지키면서 lookup `I/L` 평균 "
        f"`{practical_hybrid.lookup_efficiency_gain_pct_mean_valid:.2f}%`, importance 평균 "
        f"`{practical_hybrid.importance_gain_pct_mean_valid:.2f}%`, 가중 lookup 절감 "
        f"`{practical_hybrid.lookup_saving_pct_weighted_valid:.2f}%`를 보였다. 다만 case별 "
        f"importance 비악화율은 `{100 * practical_hybrid.importance_nonregression_rate_valid:.1f}%`라 "
        "동일 accuracy는 여전히 보장되지 않는다.",
        "",
        f"`mu=8`은 small-N oracle 평균을 소폭 높였지만 production-N에서는 `mu=4`보다 "
        f"느리고 lookup `I/L`도 낮았다. `mu=4`도 한 ratio iteration당 평균 "
        f"`{primary.loc['fixed_r_mu4'].repair_deletions_mean:.1f}`개 endpoint를 repair하므로, "
        "다음 최적화 지점은 더 촘촘한 row-price 탐색보다 repair 대상 cardinality를 줄이는 예측이다.",
        "",
        f"## Shape별 {METHOD_LABELS[winner_method]} ({primary_track})", "",
        "| Shape | rows | lookup I/L | importance | lookup 절감 | case-p95 | 유효+2ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in winner_shapes.iterrows():
        lines.append(
            f"| {row['shape']} | {int(row.n):,} "
            f"| {fmt_pct(row.lookup_efficiency_gain_pct_mean_valid)} "
            f"| {fmt_pct(row.importance_gain_pct_mean_valid)} "
            f"| {fmt_pct(row.lookup_saving_pct_weighted_valid)} "
            f"| {fmt(row.runtime_case_p95_ms)} ms "
            f"| {100 * row.valid_and_deadline_pass_rate:.1f}% |"
        )
    lines.extend([
        "", "## 측정 한계", "",
        "- importance는 CV 3.30 synthetic lognormal이며 실제 activation trace가 아니다.",
        "- lookup은 released Orin AGX profile의 예측값이고 실제 NVMe I/O를 실행하지 않았다.",
        "- `cuda` track은 D2H, CPU solve/repair, H2D와 synchronization을 포함하지만 model compute는 제외한다.",
        "- row-price scalarization과 endpoint repair는 unsupported exact-R 해를 건너뛸 수 있으므로 전역 최적 알고리즘이 아니다.",
        "- shape hybrid의 허용 목록은 같은 benchmark에서 사후 선택했으므로 별도 trace에서 재검증해야 한다.",
        "", "![Runtime-quality](results/runtime_quality.png)", "",
        "![Shape comparison](results/shape_comparison.png)", "",
    ])
    if oracle is not None and len(oracle):
        lines.extend(["![Small-N oracle](results/oracle_optimality.png)", ""])
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    return winner_method


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    frame = add_paired_metrics(frame)
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    oracle_path = args.output_dir / "oracle_trials.csv"
    oracle = pd.read_csv(oracle_path) if oracle_path.exists() else None
    summary_frame, shape_frame, summary = summarize(frame, oracle, args)
    hybrid_frame = summarize_hybrids(frame, shape_frame)
    summary["shape_hybrids"] = {
        f"{row.track}:{row.method}": row.to_dict()
        for _, row in hybrid_frame.iterrows()
    }
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    hybrid_frame.to_csv(args.output_dir / "hybrid_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    primary_track = "cuda" if "cuda" in set(summary_frame.track) else "host"
    primary = summary_frame[summary_frame.track == primary_track].set_index("method")
    winner_method = primary.loc[list(FIXED_R_METHODS)].sort_values(
        "lookup_efficiency_gain_pct_mean_valid", ascending=False
    ).index[0]
    plot_by_shape(shape_frame, winner_method, args.output_dir / "shape_comparison.png")
    if oracle is not None and len(oracle):
        plot_oracle(oracle, args.output_dir / "oracle_optimality.png")
    write_report(summary_frame, shape_frame, hybrid_frame, oracle, args)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    shapes, tracks = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lookup_table = BASE.LatencyTable.load(args.profile)
    if not args.analyze_only:
        if not args.skip_self_check:
            self_check(lookup_table, args.saturation_kib)
        if not args.skip_oracle:
            oracle_rows = collect_oracle(args, lookup_table)
            write_csv(args.output_dir / "oracle_trials.csv", oracle_rows)
        rows, timing_rows, model_rows = collect(args, shapes, tracks, lookup_table)
        write_csv(args.output_dir / "trials.csv", rows)
        write_csv(args.output_dir / "timing_samples.csv", timing_rows)
        write_csv(args.output_dir / "models.csv", model_rows)
        metadata = {
            "shape_specs": shapes,
            "method_labels": METHOD_LABELS,
            "method_settings": {
                method: {"rho_iterations": setting[0], "mu_calls": setting[1]}
                for method, setting in METHOD_SETTINGS.items()
            },
            "requested_tracks": args.tracks,
            "executed_tracks": tracks,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    analyze(args)


if __name__ == "__main__":
    main()
