#!/usr/bin/env python3
"""Experiment 13: high-q two-line Lagrangian coverage heuristic.

Fit the saturation-aware two-line cost proposed in Experiment 12 note 2,
solve each penalized lambda problem exactly in O(N), and use a very dense
lambda grid as a coverage heuristic.  Experiment 10 masks and Experiment 11
latency policies are reused for paired comparisons; no O(N^3) oracle is run.

Here q is the number of lambda multipliers.  It is not the number of
importance buckets used by the earlier Quantized-Pareto experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

try:
    from numba import njit, prange
except ImportError as error:  # pragma: no cover
    raise SystemExit("Experiment 13 requires numba") from error


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_11 = PROJECT_ROOT / "experiments" / "11_affine_lookup_gap" / "run_experiment.py"
DEFAULT_SOURCE = PROJECT_ROOT / "experiments" / "10_exact_lookup_recheck" / "results"


def load_experiment_11():
    spec = importlib.util.spec_from_file_location("experiment_11_for_13", EXPERIMENT_11)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 11 from {EXPERIMENT_11}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP11 = load_experiment_11()
EXP10 = EXP11.EXP10
BASE = EXP10.BASE
EXP2 = EXP10.EXP2
CONFIG = EXP10.CONFIG
SPATIAL_MODES = EXP10.SPATIAL_MODES
SPATIAL_LABELS = EXP10.SPATIAL_LABELS
LAGRANGIAN_COLOR = "#E6A700"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def timed_call(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return result, (time.perf_counter_ns() - start) / 1e6


def summarize(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def input_key(row: dict) -> tuple:
    return int(row["trial"]), round(float(row["target_cv"]), 10), row["spatial_mode"]


def fit_continuous_two_line(table, row_size_kib: float,
                            saturation_kib: float) -> dict:
    """Constrained fit: pre-saturation a+c1*r, post-saturation c2*r."""
    pairs = sorted(table.as_dict().items())
    kib = np.asarray([key for key, _ in pairs], dtype=np.float64)
    latency = np.asarray([value for _, value in pairs], dtype=np.float64)
    rows = kib / row_size_kib
    saturation_rows = saturation_kib / row_size_kib
    design = np.empty((len(rows), 2), dtype=np.float64)
    short = rows <= saturation_rows
    design[short, 0] = rows[short] - saturation_rows
    design[short, 1] = saturation_rows
    design[~short, 0] = 0.0
    design[~short, 1] = rows[~short]
    c1, c2 = np.linalg.lstsq(design, latency, rcond=None)[0]
    a = (c2 - c1) * saturation_rows
    predicted = design @ np.asarray([c1, c2])
    residual = float(np.sum((latency - predicted) ** 2))
    total = float(np.sum((latency - latency.mean()) ** 2))
    return {
        "a_ms": float(a),
        "c1_ms_per_row": float(c1),
        "c2_ms_per_row": float(c2),
        "d_ms_per_excess_row": float(c2 - c1),
        "saturation_kib": float(saturation_kib),
        "saturation_rows": float(saturation_rows),
        "short_max_rows": int(math.floor(saturation_rows)),
        "r_squared": float(1.0 - residual / total),
        "mape_pct": float(np.mean(np.abs((latency - predicted) / latency)) * 100.0),
    }


def two_line_chunk_ms(length: int, model: dict) -> float:
    length = int(length)
    if length <= model["saturation_rows"]:
        return model["a_ms"] + model["c1_ms_per_row"] * length
    return model["c2_ms_per_row"] * length


def two_line_mask_ms(encoded: str, model: dict) -> float:
    return float(sum(
        two_line_chunk_ms(int(item.split(":")[1]), model)
        for item in encoded.split(";") if item
    ))


@njit(cache=False, inline="always")
def _better(score, importance, cost, incumbent_score, incumbent_importance,
            incumbent_cost):
    scale = max(1.0, abs(score), abs(incumbent_score))
    tolerance = 1e-13 * scale
    if score > incumbent_score + tolerance:
        return True
    if abs(score - incumbent_score) <= tolerance:
        if importance > incumbent_importance + 1e-14:
            return True
        if abs(importance - incumbent_importance) <= 1e-14:
            return cost < incumbent_cost - 1e-14
    return False


@njit(cache=False)
def _solve_lambda_metrics(values, multiplier, a_ms, c1_ms, c2_ms,
                          saturation_rows, short_max):
    """Exact O(N) penalized solver for one lambda, returning final metrics."""
    n = len(values)
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for i in range(n):
        prefix[i + 1] = prefix[i] + values[i]

    score = np.zeros(n + 1, dtype=np.float64)
    importance = np.zeros(n + 1, dtype=np.float64)
    cost = np.zeros(n + 1, dtype=np.float64)
    rows = np.zeros(n + 1, dtype=np.int32)
    chunks = np.zeros(n + 1, dtype=np.int32)
    excess = np.zeros(n + 1, dtype=np.float64)

    deque_indices = np.empty(n, dtype=np.int32)
    deque_values = np.empty(n, dtype=np.float64)
    deque_importance_base = np.empty(n, dtype=np.float64)
    head = 0
    tail = 0
    long_index = -1
    long_value = -np.inf
    long_importance_base = -np.inf

    for j in range(n):
        i = j
        previous = 0 if i == 0 else i - 1
        h_value = score[previous] - multiplier * prefix[i] + c1_ms * i
        h_importance = importance[previous] - prefix[i]
        while tail > head:
            last = tail - 1
            if deque_values[last] > h_value + 1e-13:
                break
            if abs(deque_values[last] - h_value) <= 1e-13 and (
                deque_importance_base[last] > h_importance + 1e-14
            ):
                break
            tail -= 1
        deque_indices[tail] = i
        deque_values[tail] = h_value
        deque_importance_base[tail] = h_importance
        tail += 1

        minimum_short_start = j - short_max + 1
        while tail > head and deque_indices[head] < minimum_short_start:
            head += 1

        new_long_start = j - short_max
        if new_long_start >= 0:
            i = new_long_start
            previous = 0 if i == 0 else i - 1
            g_value = score[previous] - multiplier * prefix[i] + c2_ms * i
            g_importance = importance[previous] - prefix[i]
            if (g_value > long_value + 1e-13 or (
                abs(g_value - long_value) <= 1e-13
                and g_importance > long_importance_base + 1e-14
            )):
                long_value = g_value
                long_importance_base = g_importance
                long_index = i

        best_score = score[j]
        best_importance = importance[j]
        best_cost = cost[j]
        best_rows = rows[j]
        best_chunks = chunks[j]
        best_excess = excess[j]

        if tail > head:
            i = deque_indices[head]
            previous = 0 if i == 0 else i - 1
            length = j - i + 1
            candidate_importance = importance[previous] + prefix[j + 1] - prefix[i]
            candidate_cost = cost[previous] + a_ms + c1_ms * length
            candidate_score = multiplier * candidate_importance - candidate_cost
            if _better(candidate_score, candidate_importance, candidate_cost,
                       best_score, best_importance, best_cost):
                best_score = candidate_score
                best_importance = candidate_importance
                best_cost = candidate_cost
                best_rows = rows[previous] + length
                best_chunks = chunks[previous] + 1
                best_excess = excess[previous]

        if long_index >= 0:
            i = long_index
            previous = 0 if i == 0 else i - 1
            length = j - i + 1
            candidate_importance = importance[previous] + prefix[j + 1] - prefix[i]
            candidate_cost = cost[previous] + c2_ms * length
            candidate_score = multiplier * candidate_importance - candidate_cost
            if _better(candidate_score, candidate_importance, candidate_cost,
                       best_score, best_importance, best_cost):
                best_score = candidate_score
                best_importance = candidate_importance
                best_cost = candidate_cost
                best_rows = rows[previous] + length
                best_chunks = chunks[previous] + 1
                best_excess = excess[previous] + length - saturation_rows

        score[j + 1] = best_score
        importance[j + 1] = best_importance
        cost[j + 1] = best_cost
        rows[j + 1] = best_rows
        chunks[j + 1] = best_chunks
        excess[j + 1] = best_excess

    return (score[n], importance[n], cost[n], rows[n], chunks[n], excess[n])


@njit(cache=False, parallel=True)
def _lambda_grid_kernel(values, multipliers, a_ms, c1_ms, c2_ms,
                        saturation_rows, short_max):
    q = len(multipliers)
    scores = np.empty(q, dtype=np.float64)
    importance = np.empty(q, dtype=np.float64)
    costs = np.empty(q, dtype=np.float64)
    rows = np.empty(q, dtype=np.int32)
    chunks = np.empty(q, dtype=np.int32)
    excess = np.empty(q, dtype=np.float64)
    for index in prange(q):
        result = _solve_lambda_metrics(
            values, multipliers[index], a_ms, c1_ms, c2_ms,
            saturation_rows, short_max,
        )
        scores[index] = result[0]
        importance[index] = result[1]
        costs[index] = result[2]
        rows[index] = result[3]
        chunks[index] = result[4]
        excess[index] = result[5]
    return scores, importance, costs, rows, chunks, excess


@njit(cache=False)
def _solve_lambda_mask(values, multiplier, a_ms, c1_ms, c2_ms,
                       saturation_rows, short_max):
    """Same exact solver with O(N) parents for one selected multiplier."""
    n = len(values)
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for i in range(n):
        prefix[i + 1] = prefix[i] + values[i]

    score = np.zeros(n + 1, dtype=np.float64)
    importance = np.zeros(n + 1, dtype=np.float64)
    cost = np.zeros(n + 1, dtype=np.float64)
    rows = np.zeros(n + 1, dtype=np.int32)
    chunks = np.zeros(n + 1, dtype=np.int32)
    excess = np.zeros(n + 1, dtype=np.float64)
    parent_kind = np.zeros(n + 1, dtype=np.uint8)
    parent_start = np.full(n + 1, -1, dtype=np.int32)

    deque_indices = np.empty(n, dtype=np.int32)
    deque_values = np.empty(n, dtype=np.float64)
    deque_importance_base = np.empty(n, dtype=np.float64)
    head = 0
    tail = 0
    long_index = -1
    long_value = -np.inf
    long_importance_base = -np.inf

    for j in range(n):
        i = j
        previous = 0 if i == 0 else i - 1
        h_value = score[previous] - multiplier * prefix[i] + c1_ms * i
        h_importance = importance[previous] - prefix[i]
        while tail > head:
            last = tail - 1
            if deque_values[last] > h_value + 1e-13:
                break
            if abs(deque_values[last] - h_value) <= 1e-13 and (
                deque_importance_base[last] > h_importance + 1e-14
            ):
                break
            tail -= 1
        deque_indices[tail] = i
        deque_values[tail] = h_value
        deque_importance_base[tail] = h_importance
        tail += 1
        minimum_short_start = j - short_max + 1
        while tail > head and deque_indices[head] < minimum_short_start:
            head += 1

        new_long_start = j - short_max
        if new_long_start >= 0:
            i = new_long_start
            previous = 0 if i == 0 else i - 1
            g_value = score[previous] - multiplier * prefix[i] + c2_ms * i
            g_importance = importance[previous] - prefix[i]
            if (g_value > long_value + 1e-13 or (
                abs(g_value - long_value) <= 1e-13
                and g_importance > long_importance_base + 1e-14
            )):
                long_value = g_value
                long_importance_base = g_importance
                long_index = i

        best_score = score[j]
        best_importance = importance[j]
        best_cost = cost[j]
        best_rows = rows[j]
        best_chunks = chunks[j]
        best_excess = excess[j]
        best_kind = 0
        best_start = -1

        if tail > head:
            i = deque_indices[head]
            previous = 0 if i == 0 else i - 1
            length = j - i + 1
            candidate_importance = importance[previous] + prefix[j + 1] - prefix[i]
            candidate_cost = cost[previous] + a_ms + c1_ms * length
            candidate_score = multiplier * candidate_importance - candidate_cost
            if _better(candidate_score, candidate_importance, candidate_cost,
                       best_score, best_importance, best_cost):
                best_score = candidate_score
                best_importance = candidate_importance
                best_cost = candidate_cost
                best_rows = rows[previous] + length
                best_chunks = chunks[previous] + 1
                best_excess = excess[previous]
                best_kind = 1
                best_start = i

        if long_index >= 0:
            i = long_index
            previous = 0 if i == 0 else i - 1
            length = j - i + 1
            candidate_importance = importance[previous] + prefix[j + 1] - prefix[i]
            candidate_cost = cost[previous] + c2_ms * length
            candidate_score = multiplier * candidate_importance - candidate_cost
            if _better(candidate_score, candidate_importance, candidate_cost,
                       best_score, best_importance, best_cost):
                best_score = candidate_score
                best_importance = candidate_importance
                best_cost = candidate_cost
                best_rows = rows[previous] + length
                best_chunks = chunks[previous] + 1
                best_excess = excess[previous] + length - saturation_rows
                best_kind = 2
                best_start = i

        score[j + 1] = best_score
        importance[j + 1] = best_importance
        cost[j + 1] = best_cost
        rows[j + 1] = best_rows
        chunks[j + 1] = best_chunks
        excess[j + 1] = best_excess
        parent_kind[j + 1] = best_kind
        parent_start[j + 1] = best_start

    mask = np.zeros(n, dtype=np.bool_)
    position = n
    while position > 0:
        if parent_kind[position] == 0:
            position -= 1
        else:
            start = parent_start[position]
            for index in range(start, position):
                mask[index] = True
            position = 0 if start == 0 else start - 1
    return mask, score[n], importance[n], cost[n], rows[n], chunks[n], excess[n]


def make_lambda_grid(values: np.ndarray, model: dict, q: int) -> np.ndarray:
    if q < 2:
        raise ValueError("q must be at least two")
    full_cost = model["c2_ms_per_row"] * len(values)
    minimum_value = max(float(values.min()), np.finfo(np.float64).tiny)
    first_cost = model["a_ms"] + model["c1_ms_per_row"]
    lambda_min = first_cost / float(values.max()) * 1e-3
    lambda_max = 2.0 * full_cost / minimum_value
    if not (0 < lambda_min < lambda_max and np.isfinite(lambda_max)):
        raise RuntimeError("could not construct a finite lambda grid")
    grid = np.empty(q, dtype=np.float64)
    grid[0] = 0.0
    grid[1:] = np.geomspace(lambda_min, lambda_max, q - 1)
    return grid


class TwoLineLagrangianOracle:
    def __init__(self, values: np.ndarray, model: dict, q: int):
        self.values = np.asarray(values, dtype=np.float64)
        self.model = model
        self.q = int(q)
        self.multipliers = make_lambda_grid(self.values, model, self.q)
        (
            self.scores, self.importance, self.costs, self.rows,
            self.chunks, self.excess,
        ) = _lambda_grid_kernel(
            self.values, self.multipliers,
            model["a_ms"], model["c1_ms_per_row"], model["c2_ms_per_row"],
            model["saturation_rows"], model["short_max_rows"],
        )
        self.full_cost = model["c2_ms_per_row"] * len(self.values)

    def candidate_index(self, bound: float, indices: np.ndarray | None = None) -> int:
        if not 0 < bound <= float(self.values.sum()) + 1e-12:
            raise ValueError("coverage bound must lie in (0, total importance]")
        candidates = np.arange(self.q) if indices is None else indices
        feasible = self.importance[candidates] >= bound - 1e-14
        if not np.any(feasible):
            return -1
        feasible_indices = candidates[feasible]
        candidate_costs = self.costs[feasible_indices]
        minimum = float(candidate_costs.min())
        tied = feasible_indices[np.isclose(candidate_costs, minimum, rtol=1e-12,
                                            atol=1e-14)]
        return int(tied[np.argmax(self.importance[tied])])

    def solve(self, bound: float, indices: np.ndarray | None = None) -> dict:
        index = self.candidate_index(bound, indices)
        if index < 0 or self.full_cost < self.costs[index] - 1e-14:
            mask = np.ones(len(self.values), dtype=bool)
            return {
                "mask": mask,
                "lambda_index": -1,
                "lambda": math.inf,
                "importance": float(self.values.sum()),
                "two_line_ms": float(self.full_cost),
                "rows": len(self.values),
                "chunks": 1,
                "excess_rows": len(self.values) - self.model["saturation_rows"],
            }
        result = _solve_lambda_mask(
            self.values, self.multipliers[index],
            self.model["a_ms"], self.model["c1_ms_per_row"],
            self.model["c2_ms_per_row"], self.model["saturation_rows"],
            self.model["short_max_rows"],
        )
        mask = result[0]
        for actual, expected, label in (
            (result[2], self.importance[index], "importance"),
            (result[3], self.costs[index], "cost"),
            (result[4], self.rows[index], "rows"),
            (result[5], self.chunks[index], "chunks"),
            (result[6], self.excess[index], "excess"),
        ):
            if not math.isclose(float(actual), float(expected), rel_tol=1e-10,
                                abs_tol=1e-11):
                raise RuntimeError(f"lambda mask replay mismatch for {label}")
        return {
            "mask": mask,
            "lambda_index": index,
            "lambda": float(self.multipliers[index]),
            "importance": float(result[2]),
            "two_line_ms": float(result[3]),
            "rows": int(result[4]),
            "chunks": int(result[5]),
            "excess_rows": float(result[6]),
        }


def mask_metrics(mask: np.ndarray, values: np.ndarray, model: dict,
                 affine: tuple[float, float], row_size_kib: float,
                 policies: EXP11.LatencyPolicies) -> dict:
    encoded = EXP10.encode_runs(mask)
    lengths = EXP11.run_lengths(encoded)
    rows, chunks = BASE.mask_stats(mask)
    outside = lengths > policies.cutoff_rows
    a_aff, c_aff = affine
    return {
        "importance": float(values[mask].sum()),
        "rows": rows,
        "chunks": chunks,
        "runs": encoded,
        "run_length_median": float(np.median(lengths)),
        "rows_beyond_profile_fraction": float(lengths[outside].sum() / lengths.sum()),
        "aff_ms": float(a_aff * chunks + c_aff * rows),
        "two_line_ms": two_line_mask_ms(encoded, model),
        "released_ms": policies.mask_ms(encoded, "released"),
        "tail_linear_ms": policies.mask_ms(encoded, "tail_linear"),
        "block_split_ms": policies.mask_ms(encoded, "block_split"),
    }


def solution_fields(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def brute_penalized(values: np.ndarray, multiplier: float,
                    model: dict) -> tuple[float, float, float]:
    best = (-math.inf, -math.inf, math.inf)
    for bits in range(1 << len(values)):
        mask = np.asarray([(bits >> index) & 1 for index in range(len(values))],
                          dtype=bool)
        encoded = EXP10.encode_runs(mask)
        importance = float(values[mask].sum())
        cost = two_line_mask_ms(encoded, model)
        score = multiplier * importance - cost
        key = (score, importance, -cost)
        if key > (best[0], best[1], -best[2]):
            best = score, importance, cost
    return best


def self_check(model: dict) -> None:
    # Use a small breakpoint so randomized tests exercise both branches.
    check_model = dict(model)
    check_model["saturation_rows"] = 3.5
    check_model["short_max_rows"] = 3
    check_model["c1_ms_per_row"] = 0.07
    check_model["c2_ms_per_row"] = 0.13
    check_model["d_ms_per_excess_row"] = 0.06
    check_model["a_ms"] = 0.06 * 3.5
    rng = np.random.default_rng(1313)
    for n in (5, 8, 11):
        for _ in range(4):
            values = rng.lognormal(size=n)
            values /= values.sum()
            for multiplier in (0.0, 0.01, 0.1, 1.0, 10.0):
                expected = brute_penalized(values, multiplier, check_model)
                actual = _solve_lambda_metrics(
                    values, multiplier, check_model["a_ms"],
                    check_model["c1_ms_per_row"], check_model["c2_ms_per_row"],
                    check_model["saturation_rows"], check_model["short_max_rows"],
                )
                for observed, wanted, label in zip(
                    actual[:3], expected, ("score", "importance", "cost")
                ):
                    if not math.isclose(float(observed), float(wanted), rel_tol=1e-9,
                                        abs_tol=1e-10):
                        raise RuntimeError(f"two-line O(N) self-check failed for {label}")
                mask_result = _solve_lambda_mask(
                    values, multiplier, check_model["a_ms"],
                    check_model["c1_ms_per_row"], check_model["c2_ms_per_row"],
                    check_model["saturation_rows"], check_model["short_max_rows"],
                )
                if not math.isclose(mask_result[2], actual[1], rel_tol=1e-10,
                                    abs_tol=1e-11):
                    raise RuntimeError("mask and metric solvers disagree")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--q", type=int, default=131072)
    parser.add_argument("--q-checkpoints", type=int, nargs="+",
                        default=[1024, 4096, 16384, 65536, 131072])
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--tail-fit-start-kib", type=int, default=128)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--plots-only", action="store_true",
        help="reuse the saved Experiment 13 CSV files and only regenerate plots",
    )
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--input-limit", type=int, default=None,
                        help="development-only limit; omitted for the complete experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = ("summary.json", "fixed_r_lookup_trials.csv",
                "coverage_lookup_trials.csv")
    for name in required:
        if not (args.source_results / name).is_file():
            raise SystemExit(f"missing Experiment 10 result: {args.source_results / name}")
    if args.q < 2:
        raise SystemExit("--q must be at least two")
    checkpoints = sorted(set(args.q_checkpoints))
    if checkpoints[-1] != args.q or checkpoints[0] < 2:
        raise SystemExit("--q-checkpoints must end at --q and contain values >= 2")

    source_summary = json.loads((args.source_results / "summary.json").read_text())
    if source_summary.get("format") != "experiment-10-exact-lookup-recheck-v1":
        raise SystemExit("--source-results is not an Experiment 10 result directory")
    if args.profile != source_summary["profile"]:
        raise SystemExit("--profile must match Experiment 10")

    fixed_source = pd.read_csv(args.source_results / "fixed_r_lookup_trials.csv")
    coverage_source = pd.read_csv(args.source_results / "coverage_lookup_trials.csv")
    fixed_groups = {
        key: group.sort_values("budget_fraction")
        for key, group in fixed_source.groupby(["trial", "target_cv", "spatial_mode"])
    }
    coverage_groups = {
        key: group.sort_values("coverage_target")
        for key, group in coverage_source.groupby(["trial", "target_cv", "spatial_mode"])
    }

    n = int(source_summary["n"])
    row_size_kib = float(source_summary["row_size_kib"])
    table = BASE.LatencyTable.load(args.profile)
    affine_a, affine_c, affine_r2 = BASE.affine_fit(table, row_size_kib)
    affine = (affine_a, affine_c)
    model = fit_continuous_two_line(table, row_size_kib, args.saturation_kib)
    policies = EXP11.LatencyPolicies(table, row_size_kib, args.tail_fit_start_kib)

    if args.plots_only:
        result_files = (
            "paper_matched_trials.csv", "coverage_trials.csv",
            "input_trials.csv", "q_convergence.csv",
        )
        for name in result_files:
            if not (args.output_dir / name).is_file():
                raise SystemExit(f"--plots-only is missing {args.output_dir / name}")
        fixed = pd.read_csv(args.output_dir / "paper_matched_trials.csv")
        coverage = pd.read_csv(args.output_dir / "coverage_trials.csv")
        convergence = pd.read_csv(args.output_dir / "q_convergence.csv")
        render_plots(
            args, source_summary, fixed_source, model, affine, row_size_kib,
            policies, fixed, coverage, convergence,
        )
        print(f"regenerated plots from saved CSVs in {args.output_dir}", flush=True)
        return

    if not args.skip_self_check:
        self_check(model)

    # Warm up compiled kernels outside recorded timings.
    warm_values = np.full(16, 1 / 16, dtype=np.float64)
    warm_grid = np.asarray([0.0, 0.1, 1.0, 10.0])
    _lambda_grid_kernel(
        warm_values, warm_grid, model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"],
        model["short_max_rows"],
    )

    seed = int(source_summary["seed"])
    trials = int(source_summary["trials_per_cv"])
    cv_targets = [float(value) for value in source_summary["cv_targets"]]
    primary_cvs = [value for value in cv_targets if value < 9.0]
    rng = np.random.default_rng(seed)
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    fixed_rows: list[dict] = []
    coverage_rows: list[dict] = []
    input_rows: list[dict] = []
    convergence_rows: list[dict] = []
    total_inputs = trials * len(primary_cvs) * len(SPATIAL_MODES)
    completed = 0

    for trial in range(trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
            variants = EXP2.spatial_variants(base_values, rng, hotness, 0.95, 0.75)
            if target_cv >= 9.0:
                continue
            for mode, values in variants.items():
                key = (trial, target_cv, mode)
                source_fixed = fixed_groups[key]
                source_coverage = coverage_groups[key]
                oracle, build_ms = timed_call(TwoLineLagrangianOracle, values, model, args.q)
                unique_solutions = len(set(zip(
                    np.round(oracle.importance, 14), np.round(oracle.costs, 14),
                    oracle.rows, oracle.chunks,
                )))
                input_rows.append({
                    "trial": trial, "target_cv": target_cv, "spatial_mode": mode,
                    "q": args.q, "lambda_min_positive": oracle.multipliers[1],
                    "lambda_max": oracle.multipliers[-1],
                    "unique_supported_solutions": unique_solutions,
                    "build_runtime_ms": build_ms,
                })

                selected_cache: dict[int, dict] = {}

                def solve_target(bound: float) -> dict:
                    index = oracle.candidate_index(bound)
                    cache_key = index
                    if index < 0 or oracle.full_cost < oracle.costs[index] - 1e-14:
                        cache_key = -1
                    if cache_key not in selected_cache:
                        solution = oracle.solve(bound)
                        metrics = mask_metrics(
                            solution["mask"], values, model, affine,
                            row_size_kib, policies,
                        )
                        if metrics["importance"] < bound - 1e-12:
                            raise RuntimeError("two-line solution missed coverage target")
                        if not math.isclose(metrics["two_line_ms"],
                                            solution["two_line_ms"],
                                            rel_tol=1e-10, abs_tol=1e-11):
                            raise RuntimeError("two-line mask latency mismatch")
                        metrics.update({
                            "lambda_index": solution["lambda_index"],
                            "lambda": solution["lambda"],
                            "excess_rows": solution["excess_rows"],
                        })
                        selected_cache[cache_key] = metrics
                    return selected_cache[cache_key]

                checkpoint_indices = {
                    checkpoint: np.unique(np.linspace(
                        0, args.q - 1, checkpoint, dtype=np.int64
                    ))
                    for checkpoint in checkpoints
                }

                for source in source_coverage.to_dict("records"):
                    bound = float(source["target_importance"])
                    lagrangian = solve_target(bound)
                    cover_mask = EXP10.decode_runs(source["cover_runs"], n)
                    affine_cover = mask_metrics(
                        cover_mask, values, model, affine, row_size_kib, policies
                    )
                    row = {
                        "trial": trial, "target_cv": target_cv,
                        "spatial_mode": mode,
                        "coverage_target": float(source["coverage_target"]),
                        "target_importance": bound,
                        **solution_fields("lag", lagrangian),
                        **solution_fields("affine_cover", affine_cover),
                        "lag_vs_affine_cover_two_line_ratio": (
                            lagrangian["two_line_ms"] / affine_cover["two_line_ms"]
                        ),
                    }
                    coverage_rows.append(row)

                for source in source_fixed.to_dict("records"):
                    bound = float(source["paper_importance"])
                    lagrangian = solve_target(bound)
                    paper_mask = EXP10.decode_runs(source["paper_runs"], n)
                    cover_mask = EXP10.decode_runs(source["paper_cover_runs"], n)
                    paper = mask_metrics(
                        paper_mask, values, model, affine, row_size_kib, policies
                    )
                    affine_cover = mask_metrics(
                        cover_mask, values, model, affine, row_size_kib, policies
                    )
                    if not math.isclose(paper["importance"], bound, rel_tol=1e-10,
                                        abs_tol=1e-11):
                        raise RuntimeError("Paper source mask replay mismatch")
                    row = {
                        "trial": trial, "target_cv": target_cv,
                        "spatial_mode": mode,
                        "budget_fraction": float(source["budget_fraction"]),
                        "budget_rows": int(source["budget_rows"]),
                        "paper_runtime_ms": float(source["paper_runtime_ms"]),
                        "target_importance": bound,
                        **solution_fields("paper", paper),
                        **solution_fields("affine_cover", affine_cover),
                        **solution_fields("lag", lagrangian),
                    }
                    for policy in ("aff", "two_line", "released", "tail_linear",
                                   "block_split"):
                        field = f"{policy}_ms"
                        row[f"paper_vs_affine_cover_{policy}_ratio"] = (
                            paper[field] / affine_cover[field]
                        )
                        row[f"paper_vs_lag_{policy}_ratio"] = (
                            paper[field] / lagrangian[field]
                        )
                        row[f"lag_vs_affine_cover_{policy}_ratio"] = (
                            lagrangian[field] / affine_cover[field]
                        )
                        row[f"lag_saving_vs_paper_{policy}_pct"] = 100.0 * (
                            1.0 - lagrangian[field] / paper[field]
                        )
                    fixed_rows.append(row)

                    final_cost = lagrangian["two_line_ms"]
                    for checkpoint, indices in checkpoint_indices.items():
                        candidate = oracle.candidate_index(bound, indices)
                        checkpoint_cost = (
                            oracle.full_cost if candidate < 0
                            else min(oracle.full_cost, float(oracle.costs[candidate]))
                        )
                        convergence_rows.append({
                            "trial": trial, "target_cv": target_cv,
                            "spatial_mode": mode,
                            "budget_fraction": float(source["budget_fraction"]),
                            "q": checkpoint,
                            "two_line_ms": checkpoint_cost,
                            "gap_to_max_q_pct": max(0.0, 100.0 * (
                                checkpoint_cost / final_cost - 1.0
                            )),
                        })

                completed += 1
                print(
                    f"completed {completed}/{total_inputs} inputs "
                    f"({build_ms / 1000:.2f}s, {unique_solutions} supported masks)",
                    flush=True,
                )
                if args.input_limit is not None and completed >= args.input_limit:
                    break
            if args.input_limit is not None and completed >= args.input_limit:
                break
        if args.input_limit is not None and completed >= args.input_limit:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paper_matched_trials.csv", fixed_rows)
    write_csv(args.output_dir / "coverage_trials.csv", coverage_rows)
    write_csv(args.output_dir / "input_trials.csv", input_rows)
    write_csv(args.output_dir / "q_convergence.csv", convergence_rows)

    fixed = pd.DataFrame(fixed_rows)
    coverage = pd.DataFrame(coverage_rows)
    inputs = pd.DataFrame(input_rows)
    convergence = pd.DataFrame(convergence_rows)
    summary = build_summary(
        args, source_summary, model, affine_r2, policies,
        fixed, coverage, inputs, convergence,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_plots(
        args, source_summary, fixed_source, model, affine, row_size_kib,
        policies, fixed, coverage, convergence,
    )
    print(f"wrote results to {args.output_dir}", flush=True)


def generated_inputs(source_summary: dict) -> dict[tuple, np.ndarray]:
    """Regenerate the deterministic inputs without rerunning either solver."""
    n = int(source_summary["n"])
    rng = np.random.default_rng(int(source_summary["seed"]))
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    output = {}
    for trial in range(int(source_summary["trials_per_cv"])):
        for target_cv in (float(value) for value in source_summary["cv_targets"]):
            base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
            variants = EXP2.spatial_variants(base_values, rng, hotness, 0.95, 0.75)
            if target_cv >= 9.0:
                continue
            for mode, values in variants.items():
                output[(trial, round(target_cv, 10), mode)] = values
    return output


def top_r_mask(values: np.ndarray, bound: float) -> np.ndarray:
    """Smallest stable Top-R prefix whose importance reaches ``bound``."""
    order = np.argsort(-values, kind="stable")
    prefix = np.cumsum(values[order])
    count = int(np.searchsorted(prefix, bound - 1e-14, side="left") + 1)
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:min(count, len(values))]] = True
    return mask


def lookup_analysis_frames(source_summary: dict, fixed_source: pd.DataFrame,
                           model: dict, affine: tuple[float, float],
                           row_size_kib: float, policies,
                           fixed: pd.DataFrame,
                           coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add fixed-R and matched-importance Top-R masks to saved Quant results."""
    values_by_input = generated_inputs(source_summary)
    source_by_key = {
        (int(row.trial), round(float(row.target_cv), 10), row.spatial_mode,
         round(float(row.budget_fraction), 10)): row
        for row in fixed_source.itertuples(index=False)
        if float(row.target_cv) < 9.0
    }
    fixed_rows = []
    for row in fixed.to_dict("records"):
        input_id = (
            int(row["trial"]), round(float(row["target_cv"]), 10),
            row["spatial_mode"],
        )
        source_key = (*input_id, round(float(row["budget_fraction"]), 10))
        values = values_by_input[input_id]
        source = source_by_key[source_key]
        top_fixed = mask_metrics(
            EXP10.decode_runs(source.top_runs, len(values)), values, model,
            affine, row_size_kib, policies,
        )
        top_matched = mask_metrics(
            top_r_mask(values, float(row["target_importance"])), values, model,
            affine, row_size_kib, policies,
        )
        if top_matched["importance"] < float(row["target_importance"]) - 1e-12:
            raise RuntimeError("matched Top-R missed the Paper importance target")
        output = dict(row)
        output.update(solution_fields("top_fixed", top_fixed))
        output.update(solution_fields("top_matched", top_matched))
        quant_ms = float(row["lag_released_ms"])
        output.update({
            "quant_lookup_reference_ratio": 1.0,
            "paper_vs_quant_lookup_ratio": float(row["paper_released_ms"]) / quant_ms,
            "top_matched_vs_quant_lookup_ratio": top_matched["released_ms"] / quant_ms,
        })
        fixed_rows.append(output)

    coverage_rows = []
    for row in coverage.to_dict("records"):
        input_id = (
            int(row["trial"]), round(float(row["target_cv"]), 10),
            row["spatial_mode"],
        )
        values = values_by_input[input_id]
        top = mask_metrics(
            top_r_mask(values, float(row["target_importance"])), values, model,
            affine, row_size_kib, policies,
        )
        output = dict(row)
        output.update(solution_fields("top", top))
        coverage_rows.append(output)
    return pd.DataFrame(fixed_rows), pd.DataFrame(coverage_rows)


def lookup_plot_summary(fixed: pd.DataFrame, q: int) -> dict:
    return {
        "format": "experiment-13-released-lookup-plots-v1",
        "latency_evaluator": "released Orin AGX lookup table with endpoint scaling beyond 255 KiB",
        "quant_definition": (
            f"two-line Lagrangian supported-point heuristic with q={q} multipliers; "
            "all masks are re-evaluated under released lookup latency"
        ),
        "coverage_ratio_denominator": (
            "high-q Quant at the same Paper importance target; this is not an exact "
            "lookup-aware coverage optimum"
        ),
        "paper_vs_quant_lookup_ratio": summarize(fixed.paper_vs_quant_lookup_ratio),
        "top_r_matched_vs_quant_lookup_ratio": summarize(
            fixed.top_matched_vs_quant_lookup_ratio
        ),
        "paper_matched_chunk_count": {
            method: summarize(fixed[f"{method}_chunks"])
            for method in ("paper", "lag", "top_matched")
        },
    }


def render_plots(args, source_summary: dict, fixed_source: pd.DataFrame,
                 model: dict, affine: tuple[float, float], row_size_kib: float,
                 policies, fixed: pd.DataFrame, coverage: pd.DataFrame,
                 convergence: pd.DataFrame) -> None:
    two_line_dir = args.output_dir / "two_line"
    lookup_dir = args.output_dir / "lookup"
    two_line_dir.mkdir(parents=True, exist_ok=True)
    lookup_dir.mkdir(parents=True, exist_ok=True)

    plot_run_distributions(fixed, policies, two_line_dir / "run_distributions.png")
    plot_q_convergence(convergence, two_line_dir / "q_convergence.png")

    lookup_fixed, lookup_coverage = lookup_analysis_frames(
        source_summary, fixed_source, model, affine, row_size_kib, policies,
        fixed, coverage,
    )
    lookup_fixed.to_csv(lookup_dir / "paper_matched_trials.csv", index=False)
    lookup_coverage.to_csv(lookup_dir / "coverage_trials.csv", index=False)
    (lookup_dir / "summary.json").write_text(
        json.dumps(lookup_plot_summary(lookup_fixed, args.q), indent=2) + "\n"
    )
    plot_lookup_method_grid(
        lookup_coverage, lookup_fixed,
        lookup_dir / "importance_latency_frontiers.png",
        {"quant": "lag_released_ms", "paper": "paper_released_ms",
         "top": "top_fixed_released_ms"},
        {"quant": "lag_importance", "paper": "paper_importance",
         "top": "top_fixed_importance"},
        "Released lookup latency (ms, log scale)", "Retained importance", True,
    )
    plot_lookup_method_grid(
        lookup_coverage, lookup_fixed, lookup_dir / "r_importance.png",
        {"quant": "lag_rows", "paper": "paper_rows", "top": "top_fixed_rows"},
        {"quant": "lag_importance", "paper": "paper_importance",
         "top": "top_fixed_importance"},
        "Selected rows R", "Retained importance",
    )
    plot_lookup_method_grid(
        lookup_coverage, lookup_fixed, lookup_dir / "r_latency.png",
        {"quant": "lag_rows", "paper": "paper_rows", "top": "top_fixed_rows"},
        {"quant": "lag_released_ms", "paper": "paper_released_ms",
         "top": "top_fixed_released_ms"},
        "Selected rows R", "Released lookup latency (ms)",
    )
    plot_lookup_coverage_ratios(
        lookup_fixed, lookup_dir / "coverage_opt_ratio.png"
    )
    plot_lookup_chunk_counts(lookup_fixed, lookup_dir / "chunk_count.png")


def build_summary(args, source_summary, model, affine_r2, policies,
                  fixed, coverage, inputs, convergence) -> dict:
    def ratio_summary(values) -> dict:
        ratios = np.asarray(values, dtype=np.float64)
        return {
            **summarize(ratios),
            "denominator_strict_win_rate": float(np.mean(ratios > 1.0 + 1e-12)),
            "denominator_nonworse_rate": float(np.mean(ratios >= 1.0 - 1e-12)),
            "numerator_strict_win_rate": float(np.mean(ratios < 1.0 - 1e-12)),
        }

    comparison = {}
    for denominator in ("affine_cover", "lag"):
        comparison[denominator] = {
            policy: ratio_summary(fixed[f"paper_vs_{denominator}_{policy}_ratio"])
            for policy in ("aff", "two_line", "released", "tail_linear", "block_split")
        }
    comparison["lag_vs_affine_cover"] = {
        policy: ratio_summary(fixed[f"lag_vs_affine_cover_{policy}_ratio"])
        for policy in ("aff", "two_line", "released", "tail_linear", "block_split")
    }
    return {
        "format": "experiment-13-two-line-lagrangian-v1",
        "source_experiment": "10_exact_lookup_recheck",
        "source_results": str(args.source_results.resolve()),
        "source_sha256": {
            name: file_sha256(args.source_results / name)
            for name in ("summary.json", "fixed_r_lookup_trials.csv",
                         "coverage_lookup_trials.csv")
        },
        "n": int(source_summary["n"]),
        "q": args.q,
        "q_meaning": "number of lambda multipliers, not importance buckets",
        "coverage_status": (
            "each penalized lambda problem is exact O(N); the constrained "
            "coverage result is a supported-point heuristic"
        ),
        "two_line_model": model,
        "single_affine_r_squared": affine_r2,
        "cases": {
            "inputs": len(inputs), "paper_targets": len(fixed),
            "common_coverage_targets": len(coverage),
        },
        "runtime_ms": summarize(inputs.build_runtime_ms),
        "paper_reference_runtime_ms": summarize(fixed.paper_runtime_ms),
        "runtime_scope": (
            "lambda grid uses Numba parallel CPU build per input; Paper is the "
            "stored single-query CPU timing from Experiment 07"
        ),
        "unique_supported_solutions": summarize(inputs.unique_supported_solutions),
        "paper_ratio_comparison": comparison,
        "lag_saving_vs_paper_pct": {
            policy: summarize(fixed[f"lag_saving_vs_paper_{policy}_pct"])
            for policy in ("aff", "two_line", "released", "tail_linear", "block_split")
        },
        "common_coverage_lag_vs_affine_cover_two_line": ratio_summary(
            coverage.lag_vs_affine_cover_two_line_ratio
        ),
        "solution_structure": {
            method: {
                "chunks": summarize(fixed[f"{method}_chunks"]),
                "run_length_median": summarize(fixed[f"{method}_run_length_median"]),
                "rows_beyond_profile_pct": summarize(
                    100.0 * fixed[f"{method}_rows_beyond_profile_fraction"]
                ),
                "coverage_overshoot": summarize(
                    fixed[f"{method}_importance"] - fixed.target_importance
                ),
            }
            for method in ("paper", "affine_cover", "lag")
        },
        "q_convergence_gap_pct": {
            str(q): summarize(group.gap_to_max_q_pct)
            for q, group in convergence.groupby("q", sort=True)
        },
        "lookup_policy_context": {
            "measured_max_kib": policies.max_kib,
            "measured_max_rows": policies.cutoff_rows,
            "tail_fit_ms_per_row": policies.fitted_tail_ms_per_row,
        },
    }


def plot_lookup_method_grid(coverage: pd.DataFrame, fixed: pd.DataFrame,
                            path: Path, x_fields: dict, y_fields: dict,
                            xlabel: str, ylabel: str,
                            x_log: bool = False) -> None:
    """Plot only Quant, Paper, and Top-R under the released lookup evaluator."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    display_cvs = (1.25, 3.30, 4.55)
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.0), constrained_layout=True)
    for row_index, cv in enumerate(display_cvs):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            cov = coverage[
                np.isclose(coverage.target_cv, cv)
                & (coverage.spatial_mode == mode)
            ]
            fix = fixed[
                np.isclose(fixed.target_cv, cv)
                & (fixed.spatial_mode == mode)
            ]
            for frame, prefix, label, color, marker, linestyle, group_key in (
                (cov, "quant", "Quant (high-q)", LAGRANGIAN_COLOR, "P", "-", "coverage_target"),
                (fix, "paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "-", "budget_fraction"),
                (fix, "top", "Top-R", BASE.PLOT_COLORS["top_r"], "^", ":", "budget_fraction"),
            ):
                grouped = frame.groupby(group_key).agg(
                    x=(x_fields[prefix], "mean"),
                    y=(y_fields[prefix], "mean"),
                ).sort_index()
                ax.plot(grouped.x, grouped.y, color=color, marker=marker,
                        linestyle=linestyle, label=label)
            if x_log:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={cv:g}")
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_lookup_coverage_ratios(fixed: pd.DataFrame, path: Path) -> None:
    """Latency ratios at a common Paper importance target.

    High-q Quant is the reference, not a certified lookup-aware optimum.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True,
                             constrained_layout=True)
    specs = (
        ("quant_lookup_reference_ratio", "Quant (high-q)", LAGRANGIAN_COLOR, "P"),
        ("paper_vs_quant_lookup_ratio", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o"),
        ("top_matched_vs_quant_lookup_ratio", "Top-R", BASE.PLOT_COLORS["top_r"], "^"),
    )
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = fixed[fixed.spatial_mode == mode]
        grouped = selected.groupby("budget_fraction")
        x = np.asarray(sorted(selected.budget_fraction.unique()), dtype=float)
        for field, label, color, marker in specs:
            median = np.asarray([
                grouped.get_group(value)[field].median() for value in x
            ])
            ax.plot(x, median, color=color, marker=marker, label=label)
        ax.axhline(1.0, color="#334155", linewidth=1.1, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("Paper fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Lookup latency / high-q Quant latency (median, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_lookup_chunk_counts(fixed: pd.DataFrame, path: Path) -> None:
    """Chunk-count histograms at the same Paper importance target."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    methods = (
        ("lag_chunks", "Quant (high-q)", LAGRANGIAN_COLOR),
        ("paper_chunks", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
        ("top_matched_chunks", "Top-R", BASE.PLOT_COLORS["top_r"]),
    )
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = fixed[fixed.spatial_mode == mode]
        values_by_method = {
            label: selected[field].to_numpy(dtype=int)
            for field, label, _ in methods
        }
        upper = max(int(values.max()) for values in values_by_method.values())
        edges = np.unique(np.r_[0.5, np.rint(np.geomspace(1, upper + 1, 42)) + 0.5])
        for field, label, color in methods:
            values = values_by_method[label]
            ax.hist(
                values, bins=edges,
                weights=np.ones(len(values), dtype=float) / len(values),
                histtype="step", linewidth=1.9, color=color, label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Chunk count K (log scale)")
        ax.set_ylabel("Fraction of solutions")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_run_distributions(data: pd.DataFrame, policies, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = data[data.spatial_mode == mode]
        for prefix, label, color in (
            ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
            ("affine_cover", "Affine DP-Cover", BASE.PLOT_COLORS["coverage"]),
            ("lag", "Two-line λ-grid", LAGRANGIAN_COLOR),
        ):
            if selected.empty:
                continue
            lengths = np.sort(np.concatenate([
                EXP11.run_lengths(encoded) for encoded in selected[f"{prefix}_runs"]
            ]))
            y = np.arange(1, len(lengths) + 1) / len(lengths)
            ax.step(lengths, y, where="post", color=color, label=label)
        ax.axvline(policies.cutoff_rows, color="#334155", linestyle=":",
                   label="Measured table limit")
        ax.set_xscale("log")
        ax.set_xlabel("Chunk length (rows, log scale)")
        ax.set_ylabel("Fraction of chunks ≤ length")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_q_convergence(data: pd.DataFrame, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    grouped = data.groupby("q").gap_to_max_q_pct
    q = np.asarray(sorted(data.q.unique()), dtype=np.int64)
    mean = np.asarray([grouped.get_group(value).mean() for value in q])
    p95 = np.asarray([grouped.get_group(value).quantile(0.95) for value in q])
    maximum = np.asarray([grouped.get_group(value).max() for value in q])
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    ax.plot(q, mean, marker="o", label="Mean")
    ax.plot(q, p95, marker="s", label="95th percentile")
    ax.plot(q, maximum, marker="^", label="Maximum")
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Lambda-grid size q")
    ax.set_ylabel("Two-line latency gap to q=max (%)")
    ax.set_title("Dense-grid convergence (not an exact-optimum certificate)")
    ax.legend(frameon=False)
    BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
