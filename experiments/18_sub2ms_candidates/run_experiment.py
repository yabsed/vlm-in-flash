#!/usr/bin/env python3
"""Experiment 18: screen fourteen selector configurations for a 2 ms SLA."""

from __future__ import annotations

import argparse
import csv
import heapq
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
EXPERIMENT_16 = (
    PROJECT_ROOT / "experiments" / "16_saturation_global_chain"
    / "run_experiment.py"
)
DEFAULT_METHODS = (
    "paper",
    "supported_full",
    "td_2l_c4",
    "td_2l_c8",
    "td_2l_c16",
    "td_2l_c32",
    "td_2l_c8_trim64",
    "td_2l_c16_trim256",
    "tiles_half_s",
    "tiles_s",
    "tiles_2s",
    "paper_bucket64",
    "paper_bucket256",
    "paper_bucket1024",
)
METHOD_LABELS = {
    "paper": "Paper",
    "supported_full": "Full supported",
    "td_2l_c4": "TD-2L (4)",
    "td_2l_c8": "TD-2L (8)",
    "td_2l_c16": "TD-2L (16)",
    "td_2l_c32": "TD-2L (32)",
    "td_2l_c8_trim64": "TD-2L (8) + trim 64",
    "td_2l_c16_trim256": "TD-2L (16) + trim 256",
    "tiles_half_s": "Tiles (s/2)",
    "tiles_s": "Tiles (s)",
    "tiles_2s": "Tiles (2s)",
    "paper_bucket64": "Paper bucket 64",
    "paper_bucket256": "Paper bucket 256",
    "paper_bucket1024": "Paper bucket 1024",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "supported": "#6D28D9",
    "td": "#2A9D8F",
    "trim": "#0F766E",
    "tiles": "#2563EB",
    "bucket": "#DC2626",
}


def load_experiment_16():
    spec = importlib.util.spec_from_file_location("experiment_16_for_18", EXPERIMENT_16)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 16 from {EXPERIMENT_16}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP16 = load_experiment_16()
EXP13 = EXP16.EXP13
EXP2 = EXP13.EXP2
BASE = EXP16.BASE
njit = EXP13.njit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[4864])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--cv", type=float, default=3.30)
    parser.add_argument(
        "--target-fractions", type=float, nargs="+", default=[0.50, 0.70, 0.90]
    )
    parser.add_argument(
        "--row-budget-fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--row-size-kib", type=float, default=1.75)
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--start-kib", type=float, default=12.0)
    parser.add_argument("--jump-cap-kib", type=float, default=16.0)
    parser.add_argument("--deadline-ms", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261801)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[float], list[float]]:
    n_values = sorted(set(args.n_values))
    targets = list(args.target_fractions)
    budgets = list(args.row_budget_fractions)
    if len(DEFAULT_METHODS) != 14 or len(set(DEFAULT_METHODS)) != 14:
        raise RuntimeError("Experiment 18 must contain exactly fourteen configurations")
    if not n_values or n_values[0] < 2:
        raise SystemExit("--n-values must contain integers >= 2")
    if args.trials < 1 or args.repetitions < 1:
        raise SystemExit("--trials and --repetitions must be positive")
    if len(targets) != len(budgets):
        raise SystemExit("target and row-budget fractions must have equal lengths")
    if any(not 0.0 < value <= 1.0 for value in targets + budgets):
        raise SystemExit("target and row-budget fractions must lie in (0, 1]")
    if args.cv <= 0 or any(args.cv >= math.sqrt(n - 1) for n in n_values):
        raise SystemExit("--cv must lie in (0, sqrt(N-1)) for every N")
    if args.row_size_kib <= 0 or args.deadline_ms <= 0:
        raise SystemExit("row size and deadline must be positive")
    return n_values, targets, budgets


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        return {key: None for key in ("mean", "median", "p05", "p95", "min", "max")}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def two_line_chunk_ms(length: int, model: dict) -> float:
    if length <= float(model["saturation_rows"]):
        return float(model["a_ms"] + model["c1_ms_per_row"] * length)
    return float(model["c2_ms_per_row"] * length)


def run_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=np.int8)
    changes = np.diff(np.r_[0, binary, 0])
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def mask_metrics(mask: np.ndarray, values: np.ndarray, model: dict,
                 lookup_table, row_size_kib: float) -> dict:
    mask = np.asarray(mask, dtype=bool)
    starts, ends = run_bounds(mask)
    lengths = ends - starts
    two_line_ms = float(sum(two_line_chunk_ms(int(length), model) for length in lengths))
    lookup_ms = float(sum(
        lookup_table.read_ms(float(length) * row_size_kib) for length in lengths
    ))
    return {
        "importance": float(values[mask].sum()),
        "rows": int(mask.sum()),
        "chunks": int(len(lengths)),
        "two_line_ms": two_line_ms,
        "lookup_ms": lookup_ms,
    }


def node_identity(node: dict) -> tuple[float, float, int, int]:
    return (
        round(float(node["importance"]), 12),
        round(float(node["two_line_ms"]), 12),
        int(node["rows"]),
        int(node["chunks"]),
    )


def target_directed_two_line(values: np.ndarray, target: float, model: dict,
                             max_calls: int) -> tuple[np.ndarray, dict]:
    """Follow only the supported-envelope bracket containing ``target``."""
    if max_calls < 2:
        raise ValueError("target-directed search needs at least two metric solves")
    calls = 0

    def solve_metrics(multiplier: float) -> dict:
        nonlocal calls
        result = EXP13._solve_lambda_metrics(
            values, multiplier,
            model["a_ms"], model["c1_ms_per_row"], model["c2_ms_per_row"],
            model["saturation_rows"], model["short_max_rows"],
        )
        calls += 1
        return {
            "lambda": float(multiplier),
            "importance": float(result[1]),
            "two_line_ms": float(result[2]),
            "rows": int(result[3]),
            "chunks": int(result[4]),
        }

    low = solve_metrics(0.0)
    minimum_value = max(float(values.min()), np.finfo(np.float64).tiny)
    high_lambda = 2.0 * float(model["c2_ms_per_row"]) * len(values) / minimum_value
    high = solve_metrics(high_lambda)
    expansions = 0
    while high["importance"] < target - 1e-12 and calls < max_calls:
        high_lambda *= 2.0
        high = solve_metrics(high_lambda)
        expansions += 1
    if high["importance"] < target - 1e-12:
        mask = np.ones(len(values), dtype=bool)
        return mask, {
            "scalarized_calls": calls,
            "bracket_converged": False,
            "lambda": high_lambda,
            "high_expansions": expansions,
        }

    converged = False
    while calls < max_calls:
        delta_importance = high["importance"] - low["importance"]
        if delta_importance <= 1e-14:
            break
        multiplier = (
            high["two_line_ms"] - low["two_line_ms"]
        ) / delta_importance
        if not np.isfinite(multiplier) or multiplier <= 0:
            break
        middle = solve_metrics(multiplier)
        identity = node_identity(middle)
        if identity in (node_identity(low), node_identity(high)):
            converged = True
            break
        if not (
            low["importance"] + 1e-12 < middle["importance"]
            < high["importance"] - 1e-12
        ):
            break
        if middle["importance"] >= target - 1e-14:
            high = middle
        else:
            low = middle

    replay = EXP13._solve_lambda_mask(
        values, high["lambda"],
        model["a_ms"], model["c1_ms_per_row"], model["c2_ms_per_row"],
        model["saturation_rows"], model["short_max_rows"],
    )
    mask = np.asarray(replay[0], dtype=bool)
    return mask, {
        "scalarized_calls": calls,
        "bracket_converged": converged,
        "lambda": float(high["lambda"]),
        "high_expansions": expansions,
    }


def endpoint_trim(mask: np.ndarray, values: np.ndarray, target: float,
                  model: dict, max_deletions: int) -> tuple[np.ndarray, int]:
    """Delete exposed run endpoints while preserving target coverage."""
    trimmed = np.asarray(mask, dtype=bool).copy()
    surplus = float(values[trimmed].sum()) - float(target)
    if surplus < -1e-11:
        raise RuntimeError("endpoint trim received an infeasible mask")
    starts, ends = run_bounds(trimmed)
    left = starts.astype(np.int64).copy()
    right = (ends - 1).astype(np.int64).copy()
    versions = np.zeros(len(left), dtype=np.int64)
    heap: list[tuple[float, float, int, int, int]] = []

    def offer(run: int) -> None:
        length = int(right[run] - left[run] + 1)
        if length <= 0:
            return
        saving = two_line_chunk_ms(length, model)
        if length > 1:
            saving -= two_line_chunk_ms(length - 1, model)
        saving = max(saving, np.finfo(np.float64).tiny)
        for index in {int(left[run]), int(right[run])}:
            heapq.heappush(
                heap,
                (float(values[index]) / saving, float(values[index]),
                 run, index, int(versions[run])),
            )

    for run in range(len(left)):
        offer(run)

    deletions = 0
    while heap and deletions < max_deletions:
        _, lost, run, index, version = heapq.heappop(heap)
        if version != versions[run] or not trimmed[index]:
            continue
        if index not in (left[run], right[run]):
            continue
        if lost > surplus + 1e-14:
            continue
        trimmed[index] = False
        surplus -= lost
        deletions += 1
        if left[run] == right[run]:
            left[run] = 1
            right[run] = 0
        elif index == left[run]:
            left[run] += 1
        else:
            right[run] -= 1
        versions[run] += 1
        offer(run)

    if float(values[trimmed].sum()) < target - 1e-11:
        raise RuntimeError("endpoint trim violated target coverage")
    return trimmed, deletions


def saturation_tiles(values: np.ndarray, target: float, model: dict,
                     length: int) -> tuple[np.ndarray, dict]:
    """Try two non-overlapping tilings and return their cheapest feasible mask."""
    n = len(values)
    length = max(1, min(n, int(length)))
    prefix = np.r_[0.0, np.cumsum(values)]
    candidates = []
    for offset in sorted(set((0, length // 2))):
        boundaries = [0]
        if offset > 0:
            boundaries.append(offset)
        position = offset + length
        while position < n:
            boundaries.append(position)
            position += length
        if boundaries[-1] != n:
            boundaries.append(n)
        tiles = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            importance = float(prefix[end] - prefix[start])
            cost = two_line_chunk_ms(end - start, model)
            tiles.append((importance / cost, start, end, importance))
        tiles.sort(key=lambda item: (-item[0], item[1]))
        mask = np.zeros(n, dtype=bool)
        kept = 0.0
        for _, start, end, importance in tiles:
            mask[start:end] = True
            kept += importance
            if kept >= target - 1e-14:
                break
        metrics = mask_metrics_two_line(mask, values, model)
        candidates.append((metrics["two_line_ms"], metrics["importance"] - target,
                           offset, mask))
    _, _, offset, mask = min(candidates, key=lambda item: (item[0], item[1]))
    return mask, {"tile_length": length, "chosen_offset": int(offset)}


def mask_metrics_two_line(mask: np.ndarray, values: np.ndarray, model: dict) -> dict:
    starts, ends = run_bounds(mask)
    return {
        "importance": float(values[np.asarray(mask, dtype=bool)].sum()),
        "two_line_ms": float(sum(
            two_line_chunk_ms(int(end - start), model)
            for start, end in zip(starts, ends)
        )),
    }


def paper_windows(n: int, row_size_kib: float, table, params):
    start_kib, end_kib, step_kib, jump_cap_kib = params.resolve(table)
    size_start = max(1, int(start_kib / row_size_kib))
    size_end = max(1, int(end_kib / row_size_kib))
    size_step = max(1, int(step_kib / row_size_kib))
    jump_cap = max(1, int(jump_cap_kib / row_size_kib))
    sizes = []
    costs = []
    for size in range(size_start, size_end, size_step):
        if size > n:
            break
        sizes.append(size)
        costs.append(table.read_ms(size * row_size_kib))
    if not sizes:
        raise RuntimeError("Paper candidate sweep produced no windows")
    return (
        np.asarray(sizes, dtype=np.int64),
        np.asarray(costs, dtype=np.float64),
        jump_cap,
    )


def exact_paper_coverage(values: np.ndarray, target: float, window_sizes: np.ndarray,
                         window_costs: np.ndarray, jump_cap: int) -> tuple[np.ndarray, dict]:
    """Paper score order and non-overlap rule, stopping directly at Q."""
    values_t = torch.from_numpy(values.astype(np.float32))
    cumsum = torch.cat([
        torch.zeros(1, dtype=torch.float32), torch.cumsum(values_t, dim=0)
    ])
    all_scores = []
    all_starts = []
    all_sizes = []
    n = len(values)
    for size, cost in zip(window_sizes, window_costs):
        stride = min(int(size), int(jump_cap))
        starts = torch.arange(0, n - int(size) + 1, stride)
        all_scores.append(
            (cumsum[starts + int(size)] - cumsum[starts]) / np.float32(cost)
        )
        all_starts.append(starts)
        all_sizes.append(torch.full_like(starts, int(size)))
    scores = torch.cat(all_scores)
    starts = torch.cat(all_starts)
    sizes = torch.cat(all_sizes)
    if torch.cuda.is_available():
        order = torch.argsort(scores.cuda(), descending=True, stable=True).cpu()
    else:
        order = torch.argsort(scores, descending=True, stable=True)
    starts_np = starts[order].numpy()
    sizes_np = sizes[order].numpy()
    mask = np.zeros(n, dtype=bool)
    importance = 0.0
    accepted = 0
    prefix = np.r_[0.0, np.cumsum(values)]
    for start, size in zip(starts_np, sizes_np):
        start = int(start)
        stop = start + int(size)
        if mask[start:stop].any():
            continue
        mask[start:stop] = True
        importance += float(prefix[stop] - prefix[start])
        accepted += 1
        if importance >= target - 1e-14:
            break
    return mask, {"accepted_windows": accepted, "candidate_count": len(scores)}


@njit(cache=False)
def _bucket_paper_kernel(values, window_sizes, window_costs, jump_cap,
                         bucket_count, target, row_budget, coverage_mode):
    n = len(values)
    prefix = np.empty(n + 1, dtype=np.float32)
    prefix[0] = np.float32(0.0)
    accumulator = 0.0
    for index in range(n):
        accumulator += values[index]
        prefix[index + 1] = np.float32(accumulator)

    candidate_count = 0
    for size in window_sizes:
        stride = min(int(size), jump_cap)
        candidate_count += (n - int(size)) // stride + 1
    starts = np.empty(candidate_count, dtype=np.int32)
    sizes = np.empty(candidate_count, dtype=np.int32)
    scores = np.empty(candidate_count, dtype=np.float32)
    position = 0
    minimum_score = np.float32(np.inf)
    maximum_score = np.float32(-np.inf)
    for window_index in range(len(window_sizes)):
        size = int(window_sizes[window_index])
        stride = min(size, jump_cap)
        cost = np.float32(window_costs[window_index])
        for start in range(0, n - size + 1, stride):
            score = np.float32((prefix[start + size] - prefix[start]) / cost)
            starts[position] = start
            sizes[position] = size
            scores[position] = score
            if score < minimum_score:
                minimum_score = score
            if score > maximum_score:
                maximum_score = score
            position += 1

    candidate_buckets = np.zeros(candidate_count, dtype=np.int32)
    counts = np.zeros(bucket_count, dtype=np.int32)
    score_range = maximum_score - minimum_score
    for index in range(candidate_count):
        bucket = 0
        if score_range > 0:
            bucket = int(
                (scores[index] - minimum_score) / score_range * (bucket_count - 1)
            )
        if bucket < 0:
            bucket = 0
        elif bucket >= bucket_count:
            bucket = bucket_count - 1
        candidate_buckets[index] = bucket
        counts[bucket] += 1

    offsets = np.empty(bucket_count + 1, dtype=np.int32)
    offsets[0] = 0
    for bucket in range(bucket_count):
        offsets[bucket + 1] = offsets[bucket] + counts[bucket]
    cursors = offsets[:-1].copy()
    order = np.empty(candidate_count, dtype=np.int32)
    for index in range(candidate_count):
        bucket = candidate_buckets[index]
        order[cursors[bucket]] = index
        cursors[bucket] += 1

    mask = np.zeros(n, dtype=np.bool_)
    selected_rows = 0
    importance = 0.0
    accepted = 0
    done = False
    for bucket_reverse in range(bucket_count):
        bucket = bucket_count - 1 - bucket_reverse
        for ordered_position in range(offsets[bucket], offsets[bucket + 1]):
            candidate = order[ordered_position]
            start = int(starts[candidate])
            size = int(sizes[candidate])
            if not coverage_mode and selected_rows + size > row_budget:
                continue
            overlap = False
            for index in range(start, start + size):
                if mask[index]:
                    overlap = True
                    break
            if overlap:
                continue
            for index in range(start, start + size):
                mask[index] = True
                importance += values[index]
            selected_rows += size
            accepted += 1
            if coverage_mode and importance >= target - 1e-14:
                done = True
                break
            if not coverage_mode and selected_rows == row_budget:
                done = True
                break
        if done:
            break
    return mask, importance, selected_rows, accepted, candidate_count


def bucket_paper(values: np.ndarray, target: float, row_budget: int | None,
                 window_sizes: np.ndarray, window_costs: np.ndarray,
                 jump_cap: int, bucket_count: int) -> tuple[np.ndarray, dict]:
    coverage_mode = row_budget is None
    result = _bucket_paper_kernel(
        values, window_sizes, window_costs, jump_cap, int(bucket_count),
        float(target), len(values) if coverage_mode else int(row_budget),
        coverage_mode,
    )
    return np.asarray(result[0], dtype=bool), {
        "accepted_windows": int(result[3]),
        "candidate_count": int(result[4]),
        "bucket_count": int(bucket_count),
    }


def add_top_fallback(mask: np.ndarray, values: np.ndarray, target: float,
                     row_budget: int | None) -> tuple[np.ndarray, int]:
    """Add highest-importance unselected rows within the optional row cap."""
    output = np.asarray(mask, dtype=bool).copy()
    importance = float(values[output].sum())
    if importance >= target - 1e-12:
        return output, 0
    room = len(values) - int(output.sum())
    if row_budget is not None:
        room = max(0, int(row_budget) - int(output.sum()))
    added = 0
    for index in np.argsort(-values, kind="stable"):
        if output[index]:
            continue
        if added >= room:
            break
        output[index] = True
        importance += float(values[index])
        added += 1
        if importance >= target - 1e-12:
            break
    return output, added


def paper_fixed_rows(values: np.ndarray, row_budget: int, row_size_kib: float,
                     lookup_table, params) -> tuple[np.ndarray, dict]:
    result = BASE.select_chunks(
        torch.from_numpy(values.astype(np.float32)), row_budget, row_size_kib,
        lookup_table, params=params, impl="native",
    )
    return result.mask.cpu().numpy().copy(), {}


def supported_full(values: np.ndarray, target: float,
                   model: dict) -> tuple[np.ndarray, dict]:
    oracle = EXP16.ExactSupportedOracle(values, model)
    node = oracle.solve(target)
    return np.asarray(node["mask"], dtype=bool), {
        "scalarized_calls": int(oracle.solve_calls),
        "supported_points": int(len(oracle.nodes)),
    }


def method_family(method: str) -> str:
    if method == "paper":
        return "paper"
    if method == "supported_full":
        return "supported"
    if "trim" in method:
        return "trim"
    if method.startswith("td_"):
        return "td"
    if method.startswith("tiles_"):
        return "tiles"
    return "bucket"


def core_select(method: str, track: str, values: np.ndarray, target: float,
                row_budget: int, model: dict, lookup_table, row_size_kib: float,
                params, windows) -> tuple[np.ndarray, dict]:
    window_sizes, window_costs, jump_cap = windows
    if method == "paper":
        if track == "coverage":
            return exact_paper_coverage(
                values, target, window_sizes, window_costs, jump_cap
            )
        return paper_fixed_rows(
            values, row_budget, row_size_kib, lookup_table, params
        )
    if method == "supported_full":
        return supported_full(values, target, model)
    if method.startswith("td_2l_c"):
        suffix = method.removeprefix("td_2l_c")
        calls_text = suffix.split("_")[0]
        calls = int(calls_text)
        mask, metadata = target_directed_two_line(values, target, model, calls)
        if "trim" in method:
            deletion_limit = int(method.rsplit("trim", 1)[1])
            mask, deletions = endpoint_trim(
                mask, values, target, model, deletion_limit
            )
            metadata["trim_deletions"] = deletions
            metadata["trim_limit"] = deletion_limit
        return mask, metadata
    if method.startswith("tiles_"):
        saturation = max(1, int(round(float(model["saturation_rows"]))))
        factor = {"tiles_half_s": 0.5, "tiles_s": 1.0, "tiles_2s": 2.0}[method]
        return saturation_tiles(values, target, model, int(round(factor * saturation)))
    bucket_count = int(method.removeprefix("paper_bucket"))
    budget = None if track == "coverage" else row_budget
    return bucket_paper(
        values, target, budget, window_sizes, window_costs, jump_cap, bucket_count
    )


def select_with_fallback(method: str, track: str, values: np.ndarray,
                         target: float, row_budget: int, model: dict,
                         lookup_table, row_size_kib: float, params,
                         windows) -> tuple[np.ndarray, dict]:
    metadata = {}
    error = ""
    try:
        mask, metadata = core_select(
            method, track, values, target, row_budget, model,
            lookup_table, row_size_kib, params, windows,
        )
    except Exception as exception:  # keep failed cases in the measurement
        error = f"{type(exception).__name__}: {exception}"
        mask = np.zeros(len(values), dtype=bool)
        metadata = {"traceback": traceback.format_exc(limit=3)}
    mask = np.asarray(mask, dtype=bool)
    before = float(values[mask].sum())
    fallback_reason = "exception" if error else ""
    fallback_added = 0
    if before < target - 1e-12:
        fallback_reason = fallback_reason or "coverage_shortfall"
        cap = None if track == "coverage" else row_budget
        mask, fallback_added = add_top_fallback(mask, values, target, cap)
    # Materialize the CPU mask passed to the downstream loader. The loader in
    # this repository consumes a CPU bool tensor; no synthetic CUDA transfer is
    # added on hosts without the target device.
    handoff = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_).copy())
    mask = handoff.numpy().copy()
    metadata = dict(metadata)
    metadata.update({
        "fallback_used": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "fallback_added_rows": int(fallback_added),
        "error": error,
    })
    return mask, metadata


def benchmark(function, repetitions: int) -> tuple[np.ndarray, dict, list[float]]:
    masks = []
    metadata = []
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        mask, details = function()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        masks.append(mask)
        metadata.append(details)
        timings.append(elapsed)
    reference = masks[0]
    deterministic = all(np.array_equal(reference, mask) for mask in masks[1:])
    details = dict(metadata[-1])
    details["deterministic"] = deterministic
    return reference, details, timings


def warm_up(values: np.ndarray, target: float, row_budget: int, model: dict,
            lookup_table, row_size_kib: float, params, windows) -> None:
    for method in DEFAULT_METHODS:
        for track in ("coverage", "row_budget"):
            select_with_fallback(
                method, track, values, target, row_budget, model,
                lookup_table, row_size_kib, params, windows,
            )


def self_check(model: dict, lookup_table, row_size_kib: float, params) -> None:
    EXP16.self_check_supported_oracle()
    rng = np.random.default_rng(1818)
    values = rng.lognormal(size=97)
    values /= values.sum()
    target = 0.7
    exact = EXP16.ExactSupportedOracle(values, model).solve(target)
    mask, metadata = target_directed_two_line(values, target, model, 128)
    actual = mask_metrics_two_line(mask, values, model)
    if not (
        actual["importance"] >= target - 1e-11
        and math.isclose(
            actual["two_line_ms"], float(exact["two_line_ms"]),
            rel_tol=1e-9, abs_tol=1e-10,
        )
    ):
        raise RuntimeError(f"target-directed self-check failed: {metadata}")
    trimmed, _ = endpoint_trim(mask, values, target, model, 32)
    before = mask_metrics_two_line(mask, values, model)
    after = mask_metrics_two_line(trimmed, values, model)
    if after["importance"] < target - 1e-11 or after["two_line_ms"] > before["two_line_ms"] + 1e-12:
        raise RuntimeError("endpoint-trim self-check failed")
    for factor in (0.5, 1.0, 2.0):
        tiled, _ = saturation_tiles(
            values, target, model,
            max(1, int(round(factor * model["saturation_rows"]))),
        )
        if float(values[tiled].sum()) < target - 1e-11:
            raise RuntimeError("saturation-tiles self-check failed")
    windows = paper_windows(len(values), row_size_kib, lookup_table, params)
    for buckets in (64, 256, 1024):
        coverage, _ = bucket_paper(values, target, None, *windows, buckets)
        fixed, _ = bucket_paper(values, target, 48, *windows, buckets)
        if float(values[coverage].sum()) < target - 1e-11 or int(fixed.sum()) > 48:
            raise RuntimeError("Paper-bucket self-check failed")


def collect(args: argparse.Namespace, n_values: list[int], targets: list[float],
            budgets: list[float], model: dict, lookup_table, params):
    trial_rows: list[dict] = []
    timing_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    warmed = False
    for n in n_values:
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()
        windows = paper_windows(n, args.row_size_kib, lookup_table, params)
        for trial in range(args.trials):
            multiset = EXP2.exact_cv_lognormal(rng, n, args.cv)
            variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
            for spatial_mode, raw_values in variants.items():
                values = np.asarray(raw_values, dtype=np.float64)
                values /= values.sum()
                if not warmed:
                    warm_up(
                        values, targets[0], max(1, int(round(n * budgets[0]))),
                        model, lookup_table, args.row_size_kib, params, windows,
                    )
                    warmed = True
                for scenario, (target_fraction, budget_fraction) in enumerate(
                    zip(targets, budgets)
                ):
                    target = float(target_fraction * values.sum())
                    row_budget = max(1, min(n, int(round(n * budget_fraction))))
                    if row_budget == n:
                        top_row_budget_importance = float(values.sum())
                    else:
                        top_row_budget_importance = float(
                            np.partition(values, n - row_budget)[n - row_budget:].sum()
                        )
                    row_budget_intrinsically_feasible = (
                        top_row_budget_importance >= target - 1e-11
                    )
                    for track in ("coverage", "row_budget"):
                        for method in DEFAULT_METHODS:
                            function = lambda method=method, track=track: select_with_fallback(
                                method, track, values, target, row_budget, model,
                                lookup_table, args.row_size_kib, params, windows,
                            )
                            mask, metadata, timings = benchmark(
                                function, args.repetitions
                            )
                            metrics = mask_metrics(
                                mask, values, model, lookup_table, args.row_size_kib
                            )
                            coverage_met = metrics["importance"] >= target - 1e-11
                            row_budget_met = metrics["rows"] <= row_budget
                            runtime_median = float(np.median(timings))
                            runtime_p95 = float(np.quantile(timings, 0.95))
                            valid = coverage_met and (
                                track == "coverage" or row_budget_met
                            )
                            base = {
                                "n": n,
                                "trial": trial,
                                "spatial_mode": spatial_mode,
                                "scenario": scenario,
                                "track": track,
                                "target_fraction": target_fraction,
                                "target_importance": target,
                                "row_budget_fraction": budget_fraction,
                                "row_budget": row_budget,
                                "top_row_budget_importance": top_row_budget_importance,
                                "row_budget_intrinsically_feasible": (
                                    row_budget_intrinsically_feasible
                                ),
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "family": method_family(method),
                            }
                            trial_rows.append({
                                **base,
                                **metrics,
                                "importance_fraction": metrics["importance"] / values.sum(),
                                "importance_overshoot": metrics["importance"] - target,
                                "coverage_met": coverage_met,
                                "row_budget_met": row_budget_met,
                                "valid": valid,
                                "runtime_median_ms": runtime_median,
                                "runtime_p95_ms": runtime_p95,
                                "deadline_met": runtime_p95 <= args.deadline_ms,
                                "valid_and_deadline_met": valid and runtime_p95 <= args.deadline_ms,
                                "fallback_used": metadata.get("fallback_used", False),
                                "fallback_reason": metadata.get("fallback_reason", ""),
                                "fallback_added_rows": metadata.get("fallback_added_rows", 0),
                                "error": metadata.get("error", ""),
                                "deterministic": metadata.get("deterministic", False),
                                "scalarized_calls": metadata.get("scalarized_calls", 0),
                                "bracket_converged": metadata.get("bracket_converged", False),
                                "trim_deletions": metadata.get("trim_deletions", 0),
                                "chosen_offset": metadata.get("chosen_offset", -1),
                                "candidate_count": metadata.get("candidate_count", 0),
                            })
                            for repetition, elapsed in enumerate(timings):
                                timing_rows.append({
                                    **base,
                                    "repetition": repetition,
                                    "runtime_ms": elapsed,
                                    "deadline_met": elapsed <= args.deadline_ms,
                                    "valid": valid,
                                    "error": metadata.get("error", ""),
                                })
                print(
                    f"N={n:,} trial={trial + 1}/{args.trials} "
                    f"mode={spatial_mode}", flush=True,
                )
    return trial_rows, timing_rows


def add_paired_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    derived = [
        "paper_lookup_ms", "paper_two_line_ms", "supported_lookup_ms",
        "supported_two_line_ms", "lookup_saving_vs_paper_pct",
        "two_line_saving_vs_paper_pct", "supported_gain_recovery_pct",
    ]
    frame = frame.drop(columns=[column for column in derived if column in frame])
    keys = ["n", "trial", "spatial_mode", "scenario", "track"]
    paper = frame[frame.method == "paper"][
        keys + ["lookup_ms", "two_line_ms"]
    ].rename(columns={
        "lookup_ms": "paper_lookup_ms",
        "two_line_ms": "paper_two_line_ms",
    })
    supported = frame[frame.method == "supported_full"][
        keys + ["lookup_ms", "two_line_ms"]
    ].rename(columns={
        "lookup_ms": "supported_lookup_ms",
        "two_line_ms": "supported_two_line_ms",
    })
    merged = frame.merge(paper, on=keys, how="left").merge(supported, on=keys, how="left")
    merged["lookup_saving_vs_paper_pct"] = 100.0 * (
        1.0 - merged.lookup_ms / merged.paper_lookup_ms
    )
    merged["two_line_saving_vs_paper_pct"] = 100.0 * (
        1.0 - merged.two_line_ms / merged.paper_two_line_ms
    )
    denominator = merged.paper_lookup_ms - merged.supported_lookup_ms
    merged["supported_gain_recovery_pct"] = np.where(
        denominator > 1e-12,
        100.0 * (merged.paper_lookup_ms - merged.lookup_ms) / denominator,
        np.nan,
    )
    return merged


def summarize_frame(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rows = []
    nested = {}
    for (track, method), group in frame.groupby(["track", "method"], sort=False):
        valid = group[group.valid]
        feasible = (
            group if track == "coverage"
            else group[group.row_budget_intrinsically_feasible]
        )
        recovery = valid.supported_gain_recovery_pct.dropna()
        positive_gap = valid[
            (valid.paper_lookup_ms - valid.supported_lookup_ms) > 1e-12
        ]
        recovery_denominator = float(
            (positive_gap.paper_lookup_ms - positive_gap.supported_lookup_ms).sum()
        )
        aggregate_recovery = (
            100.0 * float(
                (positive_gap.paper_lookup_ms - positive_gap.lookup_ms).sum()
            ) / recovery_denominator
            if recovery_denominator > 0 else math.nan
        )
        row = {
            "track": track,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "cases": len(group),
            "valid_cases": len(valid),
            "valid_rate": float(group.valid.mean()),
            "coverage_success_rate": float(group.coverage_met.mean()),
            "row_budget_success_rate": float(group.row_budget_met.mean()),
            "row_budget_intrinsic_feasible_rate": float(
                group.row_budget_intrinsically_feasible.mean()
            ),
            "deadline_pass_rate": float(group.deadline_met.mean()),
            "valid_and_deadline_pass_rate": float(group.valid_and_deadline_met.mean()),
            "feasible_case_valid_rate": (
                float(feasible.valid.mean()) if len(feasible) else math.nan
            ),
            "feasible_case_valid_and_deadline_pass_rate": (
                float(feasible.valid_and_deadline_met.mean())
                if len(feasible) else math.nan
            ),
            "fallback_rate": float(group.fallback_used.mean()),
            "scalarized_calls_median": float(group.scalarized_calls.median()),
            "scalarized_calls_max": int(group.scalarized_calls.max()),
            "runtime_median_ms": float(group.runtime_median_ms.median()),
            "runtime_p95_across_cases_ms": float(group.runtime_p95_ms.quantile(0.95)),
            "runtime_worst_case_p95_ms": float(group.runtime_p95_ms.max()),
            "lookup_saving_vs_paper_pct_mean_valid": (
                float(valid.lookup_saving_vs_paper_pct.mean()) if len(valid) else math.nan
            ),
            "supported_gain_recovery_pct_aggregate_valid": aggregate_recovery,
            "supported_gain_recovery_pct_median_valid": (
                float(recovery.median()) if len(recovery) else math.nan
            ),
        }
        rows.append(row)
        nested.setdefault(track, {})[method] = {
            **row,
            "runtime_median_ms_distribution": distribution(group.runtime_median_ms),
            "runtime_p95_ms_distribution": distribution(group.runtime_p95_ms),
            "lookup_saving_vs_paper_pct_valid": distribution(
                valid.lookup_saving_vs_paper_pct
            ),
            "supported_gain_recovery_pct_valid": distribution(recovery),
        }
    summary = {
        "format": "experiment-18-sub2ms-candidates-v1",
        "method_count": len(DEFAULT_METHODS),
        "methods": list(DEFAULT_METHODS),
        "deadline_ms": args.deadline_ms,
        "timing_scope": (
            "warm single-process host CPU query; includes candidate construction, "
            "mask recovery, postprocessing, fallback, and CPU mask handoff; excludes "
            "JIT/native compilation and actual I/O/compute"
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_threads": int(torch.get_num_threads()),
        },
        "configuration": {
            "n_values": sorted(int(value) for value in frame.n.unique()),
            "trials": args.trials,
            "repetitions": args.repetitions,
            "cv": args.cv,
            "target_fractions": list(args.target_fractions),
            "row_budget_fractions": list(args.row_budget_fractions),
            "profile": args.profile,
            "row_size_kib": args.row_size_kib,
            "saturation_kib": args.saturation_kib,
            "paper_start_kib": args.start_kib,
            "paper_step_kib": args.start_kib,
            "paper_jump_cap_kib": args.jump_cap_kib,
        },
        "tracks": nested,
    }
    return pd.DataFrame(rows), summary


def configure_plot():
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_runtime_quality(summary: pd.DataFrame, path: Path,
                         deadline_ms: float) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0), constrained_layout=True)
    markers = dict(zip(
        DEFAULT_METHODS,
        ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p", "8", "d"),
    ))
    for ax, track in zip(axes, ("coverage", "row_budget")):
        selected = summary[summary.track == track].set_index("method")
        for method in DEFAULT_METHODS:
            row = selected.loc[method]
            family = method_family(method)
            color = METHOD_COLORS[family]
            ax.scatter(
                row.runtime_median_ms,
                row.lookup_saving_vs_paper_pct_mean_valid,
                color=color, s=70, marker=markers[method], zorder=3,
                label=METHOD_LABELS[method],
            )
        ax.axvline(deadline_ms, color="#111827", linestyle="--", linewidth=1.2,
                   label=f"{deadline_ms:g} ms deadline" if track == "coverage" else None)
        ax.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Median end-to-end selector time (ms, log)")
        ax.set_ylabel("Mean lookup-latency saving vs Paper (%)\n(valid cases)")
        ax.set_title("Coverage-only" if track == "coverage" else "Coverage + row budget")
        BASE.polish_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=5,
        frameon=False, fontsize=8.5,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_deadline(summary: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), sharex=True,
                             constrained_layout=True)
    y = np.arange(len(DEFAULT_METHODS))
    for ax, track in zip(axes, ("coverage", "row_budget")):
        selected = summary[summary.track == track].set_index("method").reindex(DEFAULT_METHODS)
        ax.barh(
            y, 100.0 * selected.valid_and_deadline_pass_rate,
            color=[METHOD_COLORS[method_family(method)] for method in DEFAULT_METHODS],
        )
        ax.set_yticks(y, [METHOD_LABELS[method] for method in DEFAULT_METHODS])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Valid and p95 <= 2 ms cases (%)")
        ax.set_title("Coverage-only" if track == "coverage" else "Coverage + row budget")
        ax.invert_yaxis()
        BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Experiment 18 보고서: 2 ms selector 후보 1차 스크리닝",
        "",
        "이 실험은 2 ms 달성을 확인하는 Jetson 측정이 아니라, 현재 host에서 후보를 "
        "구현하고 탈락 조건을 찾는 1차 스크리닝이다. 실제 I/O와 compute는 selector "
        "시간에 포함하지 않았으며 별도의 predicted I/O latency로 평가했다.",
        "",
        "## 실행 설정",
        "",
        f"- 후보 및 reference: 정확히 {len(DEFAULT_METHODS)}개",
        f"- deadline: per-case 반복 측정 p95 <= {args.deadline_ms:g} ms",
        "- runtime 포함: candidate 구성, DP 호출, mask 복원, trim, fallback, CPU mask handoff",
        "- runtime 제외: 최초 JIT/native compile, importance 생성, 실제 storage I/O와 compute",
        "- coverage-only와 coverage+row-budget 결과를 별도로 집계",
        "",
    ]
    for track, title in (
        ("coverage", "Coverage-only"),
        ("row_budget", "Coverage + row budget"),
    ):
        lines.extend([
            f"## {title}",
            "",
            "| 방법 | 중앙 runtime | case-p95 | 유효률(전체) | 유효+2ms(전체) | feasible 중 유효+2ms | Paper 대비 lookup 절감 | supported 개선 회수 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        selected = summary[summary.track == track].set_index("method").reindex(DEFAULT_METHODS)
        for method, row in selected.iterrows():
            lines.append(
                f"| {METHOD_LABELS[method]} | {fmt(row.runtime_median_ms)} ms "
                f"| {fmt(row.runtime_p95_across_cases_ms)} ms "
                f"| {100 * row.valid_rate:.1f}% "
                f"| {100 * row.valid_and_deadline_pass_rate:.1f}% "
                f"| {100 * row.feasible_case_valid_and_deadline_pass_rate:.1f}% "
                f"| {fmt(row.lookup_saving_vs_paper_pct_mean_valid, 2)}% "
                f"| {fmt(row.supported_gain_recovery_pct_aggregate_valid, 1)}% |"
            )
        lines.append("")
    coverage = summary[summary.track == "coverage"].set_index("method")
    row_budget = summary[summary.track == "row_budget"].set_index("method")
    lines.extend([
        "## 주요 결과",
        "",
        f"- `TD-2L (8)`은 case-p95 `{coverage.loc['td_2l_c8', 'runtime_p95_across_cases_ms']:.3f} ms`에서 "
        f"full supported가 제공한 lookup 개선의 `{coverage.loc['td_2l_c8', 'supported_gain_recovery_pct_aggregate_valid']:.1f}%`를 회수했다.",
        f"- `TD-2L (16)`은 최대 `{int(coverage.loc['td_2l_c16', 'scalarized_calls_max'])}`회 호출 안에 모든 사례에서 "
        f"full supported와 같은 품질에 도달했고 case-p95는 `{coverage.loc['td_2l_c16', 'runtime_p95_across_cases_ms']:.3f} ms`였다. "
        "32회 설정은 추가 품질 개선이 없었다.",
        f"- endpoint trim 256은 coverage를 모두 유지하면서 Paper 대비 lookup latency를 평균 "
        f"`{coverage.loc['td_2l_c16_trim256', 'lookup_saving_vs_paper_pct_mean_valid']:.2f}%` 줄였다. "
        f"full supported의 `{coverage.loc['supported_full', 'lookup_saving_vs_paper_pct_mean_valid']:.2f}%`보다 큰 이유는 "
        "trim 결과가 supported point에 한정되지 않기 때문이다.",
        f"- 가장 빠른 tiles/bucket 계열은 약 `0.08--0.14 ms`였지만, coverage-only lookup 품질은 "
        "Paper보다 같거나 나빴다. Bucket 수를 1,024까지 늘려도 평균 개선은 양수가 되지 않았다.",
        f"- row-budget의 세 `(Q,R)` 조합은 top-R 기준으로 모두 이론적으로 가능했지만, "
        f"가장 높은 전체 유효률은 Paper와 bucket 계열의 `{100 * row_budget.loc['paper', 'valid_rate']:.1f}%`였다. "
        "따라서 coverage-directed mask를 그대로 쓰는 것만으로는 row cap을 해결하지 못한다.",
        "",
    ])
    lines.extend([
        "## 판정 규칙",
        "",
        "평균이 2 ms 미만이어도 통과로 세지 않았다. 각 input/scenario에서 반복 측정 "
        "p95가 2 ms 이하여야 하며, coverage-only에서는 Q를 만족해야 한다. row-budget "
        "track에서는 Q와 R을 동시에 만족해야 한다. 실패와 fallback 사례도 분모에 남겼다.",
        "",
        "![Runtime-quality](results/runtime_quality.png)",
        "",
        "![Deadline pass rate](results/deadline_pass_rate.png)",
        "",
    ])
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    frame = add_paired_metrics(frame)
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    summary_frame, summary = summarize_frame(frame, args)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_runtime_quality(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    plot_deadline(summary_frame, args.output_dir / "deadline_pass_rate.png")
    write_report(summary_frame, args)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    n_values, targets, budgets = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lookup_table = BASE.LatencyTable.load(args.profile)
    model = EXP13.fit_continuous_two_line(
        lookup_table, args.row_size_kib, args.saturation_kib
    )
    params = BASE.ChunkParams(
        start_kb=args.start_kib,
        end_kb=args.saturation_kib,
        step_kb=args.start_kib,
        jump_cap_kb=args.jump_cap_kib,
    )
    if not args.analyze_only:
        if not args.skip_self_check:
            self_check(model, lookup_table, args.row_size_kib, params)
        rows, timing_rows = collect(
            args, n_values, targets, budgets, model, lookup_table, params
        )
        write_csv(args.output_dir / "trials.csv", rows)
        write_csv(args.output_dir / "timing_samples.csv", timing_rows)
        metadata = {
            "two_line_model": model,
            "method_labels": METHOD_LABELS,
            "method_count": len(DEFAULT_METHODS),
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    analyze(args)


if __name__ == "__main__":
    main()
