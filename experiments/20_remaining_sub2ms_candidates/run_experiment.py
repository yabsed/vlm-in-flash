#!/usr/bin/env python3
"""Experiment 20: screen the six remaining sub-2 ms candidate families."""

from __future__ import annotations

import argparse
import csv
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
EXPERIMENT_18 = PROJECT_ROOT / "experiments" / "18_sub2ms_candidates" / "run_experiment.py"


def load_experiment_18():
    spec = importlib.util.spec_from_file_location("experiment_18_for_20", EXPERIMENT_18)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 18 from {EXPERIMENT_18}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP18 = load_experiment_18()
EXP16 = EXP18.EXP16
EXP13 = EXP18.EXP13
EXP2 = EXP18.EXP2
BASE = EXP18.BASE
njit = EXP18.njit

DEFAULT_METHODS = (
    "paper",
    "incumbent_td16_trim256",
    "lambda_bank4",
    "lambda_bank8",
    "lambda_bank16",
    "pwl_j4_c4",
    "pwl_j4_c8",
    "pwl_j8_c4",
    "pwl_j8_c8",
    "paper_local64",
    "paper_local256",
    "td_lookup_c2",
    "td_lookup_c4",
    "td_lookup_c8",
    "predict_correct_c2",
    "predict_correct_c4",
    "predict_correct_c8",
    "capped_label2",
    "capped_label4",
    "capped_label8",
)

METHOD_LABELS = {
    "paper": "Paper",
    "incumbent_td16_trim256": "TD-2L (16) + trim 256",
    "lambda_bank4": "Lambda bank (4)",
    "lambda_bank8": "Lambda bank (8)",
    "lambda_bank16": "Lambda bank (16)",
    "pwl_j4_c4": "PWL DP (J=4, calls=4)",
    "pwl_j4_c8": "PWL DP (J=4, calls=8)",
    "pwl_j8_c4": "PWL DP (J=8, calls=4)",
    "pwl_j8_c8": "PWL DP (J=8, calls=8)",
    "paper_local64": "Paper + local edits (64)",
    "paper_local256": "Paper + local edits (256)",
    "td_lookup_c2": "TD lookup DP (2)",
    "td_lookup_c4": "TD lookup DP (4)",
    "td_lookup_c8": "TD lookup DP (8)",
    "predict_correct_c2": "Lambda predict + correct (2)",
    "predict_correct_c4": "Lambda predict + correct (4)",
    "predict_correct_c8": "Lambda predict + correct (8)",
    "capped_label2": "Capped-label DP (2)",
    "capped_label4": "Capped-label DP (4)",
    "capped_label8": "Capped-label DP (8)",
}

FAMILY_COLORS = {
    "paper": "#D97706",
    "incumbent": "#6D28D9",
    "lambda_bank": "#0891B2",
    "piecewise": "#2563EB",
    "local_edit": "#DC2626",
    "lookup": "#059669",
    "predict": "#7C3AED",
    "capped_label": "#475569",
}


def method_family(method: str) -> str:
    if method == "paper":
        return "paper"
    if method == "incumbent_td16_trim256":
        return "incumbent"
    if method.startswith("lambda_bank"):
        return "lambda_bank"
    if method.startswith("pwl_"):
        return "piecewise"
    if method.startswith("paper_local"):
        return "local_edit"
    if method.startswith("td_lookup"):
        return "lookup"
    if method.startswith("predict_correct"):
        return "predict"
    return "capped_label"


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
    parser.add_argument("--seed", type=int, default=20262001)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[float], list[float]]:
    n_values = sorted(set(args.n_values))
    targets = list(args.target_fractions)
    budgets = list(args.row_budget_fractions)
    if len(DEFAULT_METHODS) != 20 or len(set(DEFAULT_METHODS)) != 20:
        raise RuntimeError("Experiment 20 must contain exactly twenty configurations")
    if not n_values or n_values[0] < 8:
        raise SystemExit("--n-values must contain integers >= 8")
    if args.trials < 1 or args.repetitions < 1:
        raise SystemExit("--trials and --repetitions must be positive")
    if len(targets) != len(budgets):
        raise SystemExit("target and row-budget fractions must have equal lengths")
    if any(not 0.0 < value <= 1.0 for value in targets + budgets):
        raise SystemExit("target and row-budget fractions must lie in (0, 1]")
    if args.cv <= 0 or any(args.cv >= math.sqrt(n - 1) for n in n_values):
        raise SystemExit("--cv must lie in (0, sqrt(N-1)) for every N")
    if args.deadline_ms <= 0 or args.row_size_kib <= 0:
        raise SystemExit("deadline and row size must be positive")
    return n_values, targets, budgets


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def lookup_chunk_ms(length: int, lookup_model: dict) -> float:
    length = int(length)
    cap = int(lookup_model["cap_rows"])
    if length <= cap:
        return float(lookup_model["chunk_costs"][length])
    return float(lookup_model["tail_ms_per_row"] * length)


def lookup_mask_ms(mask: np.ndarray, lookup_model: dict) -> float:
    starts, ends = EXP18.run_bounds(mask)
    return float(sum(
        lookup_chunk_ms(int(end - start), lookup_model)
        for start, end in zip(starts, ends)
    ))


def linear_relaxation_lambda(values: np.ndarray, target: float,
                             marginal_ms: float) -> float:
    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(values[order])
    position = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    threshold = max(float(values[order[position]]), np.finfo(np.float64).tiny)
    return float(marginal_ms / threshold)


def lambda_bank(values: np.ndarray, target: float, model: dict,
                bank_size: int) -> tuple[np.ndarray, dict]:
    """Evaluate a fixed-size deterministic bank around a linear relaxation."""
    if bank_size < 4:
        raise ValueError("lambda bank needs at least four entries")
    center = linear_relaxation_lambda(values, target, float(model["c2_ms_per_row"]))
    high = 2.0 * float(model["c2_ms_per_row"]) * len(values) / max(
        float(values.min()), np.finfo(np.float64).tiny
    )
    if bank_size == 4:
        exponents = np.asarray([-1.0, 1.0])
    elif bank_size == 8:
        exponents = np.arange(-3.0, 3.0, 1.0)
    elif bank_size == 16:
        exponents = np.arange(-4.0, 3.0, 0.5)
    else:
        exponents = np.linspace(-4.0, 3.0, bank_size - 2)
    interior = center * np.exp2(exponents)
    multipliers = np.r_[0.0, interior, high]
    candidates = []
    for multiplier in multipliers:
        result = EXP13._solve_lambda_metrics(
            values, float(multiplier), model["a_ms"], model["c1_ms_per_row"],
            model["c2_ms_per_row"], model["saturation_rows"],
            model["short_max_rows"],
        )
        if float(result[1]) >= target - 1e-14:
            candidates.append((float(result[2]), -float(result[1]), float(multiplier)))
    if not candidates:
        return np.ones(len(values), dtype=bool), {
            "scalarized_calls": bank_size,
            "bank_center_lambda": center,
            "bracket_converged": False,
        }
    _, _, chosen = min(candidates)
    replay = EXP13._solve_lambda_mask(
        values, chosen, model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"], model["short_max_rows"],
    )
    return np.asarray(replay[0], dtype=bool), {
        "scalarized_calls": bank_size,
        "bank_center_lambda": center,
        "lambda": chosen,
        "bracket_converged": True,
    }


def predict_and_correct(values: np.ndarray, target: float, model: dict,
                        max_calls: int) -> tuple[np.ndarray, dict]:
    """Predict lambda from the linear relaxation, then refine its bracket."""
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    predicted = linear_relaxation_lambda(values, target, float(model["c2_ms_per_row"]))
    minimum = max(float(values.min()), np.finfo(np.float64).tiny)
    full_lambda = 2.0 * float(model["c2_ms_per_row"]) * len(values) / minimum
    low = {
        "lambda": 0.0, "importance": 0.0, "cost": 0.0,
        "rows": 0, "chunks": 0,
    }
    high = {
        "lambda": full_lambda, "importance": float(values.sum()),
        "cost": float(model["c2_ms_per_row"]) * len(values),
        "rows": len(values), "chunks": 1,
    }
    calls = 0

    def solve(multiplier: float) -> dict:
        nonlocal calls
        result = EXP13._solve_lambda_metrics(
            values, multiplier, model["a_ms"], model["c1_ms_per_row"],
            model["c2_ms_per_row"], model["saturation_rows"],
            model["short_max_rows"],
        )
        calls += 1
        return {
            "lambda": float(multiplier), "importance": float(result[1]),
            "cost": float(result[2]), "rows": int(result[3]),
            "chunks": int(result[4]),
        }

    first = solve(min(max(predicted, np.finfo(np.float64).tiny), full_lambda))
    if first["importance"] >= target - 1e-14:
        high = first
    else:
        low = first
    converged = False
    while calls < max_calls:
        delta = high["importance"] - low["importance"]
        if delta <= 1e-14:
            break
        multiplier = (high["cost"] - low["cost"]) / delta
        if not np.isfinite(multiplier) or multiplier <= low["lambda"] + 1e-15:
            multiplier = math.sqrt(max(low["lambda"], np.finfo(float).tiny) * high["lambda"])
        if multiplier >= high["lambda"] - 1e-15:
            multiplier = 0.5 * (low["lambda"] + high["lambda"])
        middle = solve(multiplier)
        identity = (
            round(middle["importance"], 12), round(middle["cost"], 12),
            middle["rows"], middle["chunks"],
        )
        endpoint_identities = {
            (round(item["importance"], 12), round(item["cost"], 12),
             item["rows"], item["chunks"])
            for item in (low, high)
        }
        if identity in endpoint_identities:
            converged = True
            break
        if middle["importance"] >= target - 1e-14:
            high = middle
        else:
            low = middle
    replay = EXP13._solve_lambda_mask(
        values, high["lambda"], model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"], model["short_max_rows"],
    )
    return np.asarray(replay[0], dtype=bool), {
        "scalarized_calls": calls,
        "predicted_lambda": predicted,
        "lambda": high["lambda"],
        "bracket_converged": converged,
    }


def target_directed_lookup(values: np.ndarray, target: float, lookup_model: dict,
                           max_calls: int) -> tuple[np.ndarray, dict]:
    """Target-directed intersections using the exact O(Nm) lookup solver."""
    calls = 0

    def solve(multiplier: float) -> dict:
        nonlocal calls
        result = EXP16._solve_lookup_lambda_metrics(
            values, multiplier, lookup_model["chunk_costs"],
            lookup_model["tail_ms_per_row"],
        )
        calls += 1
        return {
            "lambda": float(multiplier), "importance": float(result[1]),
            "cost": float(result[2]), "rows": int(result[3]),
            "chunks": int(result[4]),
        }

    low = solve(0.0)
    high_lambda = 2.0 * lookup_model["tail_ms_per_row"] * len(values) / max(
        float(values.min()), np.finfo(np.float64).tiny
    )
    high = solve(high_lambda)
    converged = False
    while calls < max_calls:
        delta = high["importance"] - low["importance"]
        if delta <= 1e-14:
            break
        multiplier = (high["cost"] - low["cost"]) / delta
        if not np.isfinite(multiplier) or multiplier <= 0:
            break
        middle = solve(multiplier)
        identity = (
            round(middle["importance"], 12), round(middle["cost"], 12),
            middle["rows"], middle["chunks"],
        )
        endpoint_identities = {
            (round(item["importance"], 12), round(item["cost"], 12),
             item["rows"], item["chunks"])
            for item in (low, high)
        }
        if identity in endpoint_identities:
            converged = True
            break
        if not low["importance"] + 1e-12 < middle["importance"] < high["importance"] - 1e-12:
            break
        if middle["importance"] >= target - 1e-14:
            high = middle
        else:
            low = middle
    replay = EXP16._solve_lookup_lambda_mask(
        values, high["lambda"], lookup_model["chunk_costs"],
        lookup_model["tail_ms_per_row"],
    )
    return np.asarray(replay[0], dtype=bool), {
        "scalarized_calls": calls,
        "lambda": high["lambda"],
        "bracket_converged": converged,
    }


def fit_piecewise_lookup(lookup_model: dict, n: int, segment_count: int) -> dict:
    """Fit contiguous affine pieces and retain the exact proportional tail."""
    if segment_count < 2 or segment_count > n:
        raise ValueError("invalid piecewise segment count")
    cap = min(int(lookup_model["cap_rows"]), n)
    if n >= int(lookup_model["cap_rows"]):
        pre_count = segment_count - 1
        pre_end = max(1, cap - 1)
        ends = [int(round(pre_end ** (index / pre_count)))
                for index in range(1, pre_count + 1)]
        ends[-1] = pre_end
        ends = np.maximum.accumulate(np.asarray(ends, dtype=np.int64))
        for index in range(1, len(ends)):
            ends[index] = max(ends[index], ends[index - 1] + 1)
        ends[-1] = pre_end
        lows = []
        highs = []
        slopes = []
        intercepts = []
        low = 1
        for high in ends:
            high = int(high)
            lengths = np.arange(low, high + 1, dtype=np.float64)
            costs = np.asarray([lookup_chunk_ms(int(x), lookup_model) for x in lengths])
            if len(lengths) == 1:
                slope = 0.0
                intercept = float(costs[0])
            else:
                slope, intercept = np.polyfit(lengths, costs, 1)
            lows.append(low); highs.append(high)
            slopes.append(float(slope)); intercepts.append(float(intercept))
            low = high + 1
        lows.append(cap); highs.append(n)
        slopes.append(float(lookup_model["tail_ms_per_row"])); intercepts.append(0.0)
    else:
        ends = [int(round(n ** (index / segment_count)))
                for index in range(1, segment_count + 1)]
        ends = np.maximum.accumulate(np.asarray(ends, dtype=np.int64))
        for index in range(1, len(ends)):
            ends[index] = max(ends[index], ends[index - 1] + 1)
        ends[-1] = n
        lows, highs, slopes, intercepts = [], [], [], []
        low = 1
        for high in ends:
            high = min(int(high), n)
            lengths = np.arange(low, high + 1, dtype=np.float64)
            costs = np.asarray([lookup_chunk_ms(int(x), lookup_model) for x in lengths])
            if len(lengths) == 1:
                slope, intercept = 0.0, float(costs[0])
            else:
                slope, intercept = np.polyfit(lengths, costs, 1)
            lows.append(low); highs.append(high)
            slopes.append(float(slope)); intercepts.append(float(intercept))
            low = high + 1
    return {
        "lows": np.asarray(lows, dtype=np.int32),
        "highs": np.asarray(highs, dtype=np.int32),
        "slopes": np.asarray(slopes, dtype=np.float64),
        "intercepts": np.asarray(intercepts, dtype=np.float64),
        "segments": len(lows),
    }


@njit(cache=False, inline="always")
def _pwl_better(score, importance, cost, incumbent_score,
                incumbent_importance, incumbent_cost):
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
def _solve_pwl_metrics(values, multiplier, lows, highs, slopes, intercepts):
    n = len(values)
    segment_count = len(lows)
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
    score = np.zeros(n + 1, dtype=np.float64)
    importance = np.zeros(n + 1, dtype=np.float64)
    cost = np.zeros(n + 1, dtype=np.float64)
    rows = np.zeros(n + 1, dtype=np.int32)
    chunks = np.zeros(n + 1, dtype=np.int32)
    deque_indices = np.empty((segment_count, n), dtype=np.int32)
    deque_values = np.empty((segment_count, n), dtype=np.float64)
    deque_importance = np.empty((segment_count, n), dtype=np.float64)
    heads = np.zeros(segment_count, dtype=np.int32)
    tails = np.zeros(segment_count, dtype=np.int32)

    for end in range(n):
        best_score = score[end]
        best_importance = importance[end]
        best_cost = cost[end]
        best_rows = rows[end]
        best_chunks = chunks[end]
        for segment in range(segment_count):
            start = end - int(lows[segment]) + 1
            if start >= 0:
                previous = 0 if start == 0 else start - 1
                base = score[previous] - multiplier * prefix[start] + slopes[segment] * start
                base_importance = importance[previous] - prefix[start]
                head = heads[segment]
                tail = tails[segment]
                while tail > head:
                    last = tail - 1
                    if deque_values[segment, last] > base + 1e-13:
                        break
                    if abs(deque_values[segment, last] - base) <= 1e-13 and (
                        deque_importance[segment, last] > base_importance + 1e-14
                    ):
                        break
                    tail -= 1
                deque_indices[segment, tail] = start
                deque_values[segment, tail] = base
                deque_importance[segment, tail] = base_importance
                tail += 1
                tails[segment] = tail
            minimum_start = end - int(highs[segment]) + 1
            head = heads[segment]
            tail = tails[segment]
            while tail > head and deque_indices[segment, head] < minimum_start:
                head += 1
            heads[segment] = head
            if tail <= head:
                continue
            start = int(deque_indices[segment, head])
            previous = 0 if start == 0 else start - 1
            length = end - start + 1
            candidate_importance = importance[previous] + prefix[end + 1] - prefix[start]
            candidate_cost = cost[previous] + intercepts[segment] + slopes[segment] * length
            candidate_score = multiplier * candidate_importance - candidate_cost
            if _pwl_better(
                candidate_score, candidate_importance, candidate_cost,
                best_score, best_importance, best_cost,
            ):
                best_score = candidate_score
                best_importance = candidate_importance
                best_cost = candidate_cost
                best_rows = rows[previous] + length
                best_chunks = chunks[previous] + 1
        score[end + 1] = best_score
        importance[end + 1] = best_importance
        cost[end + 1] = best_cost
        rows[end + 1] = best_rows
        chunks[end + 1] = best_chunks
    return score[n], importance[n], cost[n], rows[n], chunks[n]


@njit(cache=False)
def _solve_pwl_mask(values, multiplier, lows, highs, slopes, intercepts):
    n = len(values)
    segment_count = len(lows)
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
    score = np.zeros(n + 1, dtype=np.float64)
    importance = np.zeros(n + 1, dtype=np.float64)
    cost = np.zeros(n + 1, dtype=np.float64)
    rows = np.zeros(n + 1, dtype=np.int32)
    chunks = np.zeros(n + 1, dtype=np.int32)
    parent_start = np.full(n + 1, -1, dtype=np.int32)
    deque_indices = np.empty((segment_count, n), dtype=np.int32)
    deque_values = np.empty((segment_count, n), dtype=np.float64)
    deque_importance = np.empty((segment_count, n), dtype=np.float64)
    heads = np.zeros(segment_count, dtype=np.int32)
    tails = np.zeros(segment_count, dtype=np.int32)

    for end in range(n):
        best_score = score[end]
        best_importance = importance[end]
        best_cost = cost[end]
        best_rows = rows[end]
        best_chunks = chunks[end]
        best_start = -1
        for segment in range(segment_count):
            start = end - int(lows[segment]) + 1
            if start >= 0:
                previous = 0 if start == 0 else start - 1
                base = score[previous] - multiplier * prefix[start] + slopes[segment] * start
                base_importance = importance[previous] - prefix[start]
                head = heads[segment]
                tail = tails[segment]
                while tail > head:
                    last = tail - 1
                    if deque_values[segment, last] > base + 1e-13:
                        break
                    if abs(deque_values[segment, last] - base) <= 1e-13 and (
                        deque_importance[segment, last] > base_importance + 1e-14
                    ):
                        break
                    tail -= 1
                deque_indices[segment, tail] = start
                deque_values[segment, tail] = base
                deque_importance[segment, tail] = base_importance
                tail += 1
                tails[segment] = tail
            minimum_start = end - int(highs[segment]) + 1
            head = heads[segment]
            tail = tails[segment]
            while tail > head and deque_indices[segment, head] < minimum_start:
                head += 1
            heads[segment] = head
            if tail <= head:
                continue
            start = int(deque_indices[segment, head])
            previous = 0 if start == 0 else start - 1
            length = end - start + 1
            candidate_importance = importance[previous] + prefix[end + 1] - prefix[start]
            candidate_cost = cost[previous] + intercepts[segment] + slopes[segment] * length
            candidate_score = multiplier * candidate_importance - candidate_cost
            if _pwl_better(
                candidate_score, candidate_importance, candidate_cost,
                best_score, best_importance, best_cost,
            ):
                best_score = candidate_score
                best_importance = candidate_importance
                best_cost = candidate_cost
                best_rows = rows[previous] + length
                best_chunks = chunks[previous] + 1
                best_start = start
        score[end + 1] = best_score
        importance[end + 1] = best_importance
        cost[end + 1] = best_cost
        rows[end + 1] = best_rows
        chunks[end + 1] = best_chunks
        parent_start[end + 1] = best_start

    mask = np.zeros(n, dtype=np.bool_)
    position = n
    while position > 0:
        start = int(parent_start[position])
        if start < 0:
            position -= 1
        else:
            for index in range(start, position):
                mask[index] = True
            position = 0 if start == 0 else start - 1
    return mask, score[n], importance[n], cost[n], rows[n], chunks[n]


def target_directed_piecewise(values: np.ndarray, target: float,
                              piecewise: dict, max_calls: int) -> tuple[np.ndarray, dict]:
    calls = 0

    def solve(multiplier: float) -> dict:
        nonlocal calls
        result = _solve_pwl_metrics(
            values, multiplier, piecewise["lows"], piecewise["highs"],
            piecewise["slopes"], piecewise["intercepts"],
        )
        calls += 1
        return {
            "lambda": float(multiplier), "importance": float(result[1]),
            "cost": float(result[2]), "rows": int(result[3]),
            "chunks": int(result[4]),
        }

    low = solve(0.0)
    minimum = max(float(values.min()), np.finfo(np.float64).tiny)
    full_cost = float(piecewise["slopes"][-1] * len(values) + piecewise["intercepts"][-1])
    high_lambda = 2.0 * max(full_cost, np.finfo(float).tiny) / minimum
    high = solve(high_lambda)
    converged = False
    while calls < max_calls:
        delta = high["importance"] - low["importance"]
        if delta <= 1e-14:
            break
        multiplier = (high["cost"] - low["cost"]) / delta
        if not np.isfinite(multiplier) or multiplier <= 0:
            break
        middle = solve(multiplier)
        identity = (
            round(middle["importance"], 12), round(middle["cost"], 12),
            middle["rows"], middle["chunks"],
        )
        endpoint_identities = {
            (round(item["importance"], 12), round(item["cost"], 12),
             item["rows"], item["chunks"])
            for item in (low, high)
        }
        if identity in endpoint_identities:
            converged = True
            break
        if middle["importance"] >= target - 1e-14:
            high = middle
        else:
            low = middle
    replay = _solve_pwl_mask(
        values, high["lambda"], piecewise["lows"], piecewise["highs"],
        piecewise["slopes"], piecewise["intercepts"],
    )
    return np.asarray(replay[0], dtype=bool), {
        "scalarized_calls": calls,
        "piecewise_segments": int(piecewise["segments"]),
        "lambda": high["lambda"],
        "bracket_converged": converged,
    }


def paper_local_edits(mask: np.ndarray, values: np.ndarray, target: float,
                      lookup_model: dict, max_edits: int,
                      row_budget: int | None) -> tuple[np.ndarray, dict]:
    """Lookup-aware profitable merges, endpoint trims, and boundary shifts."""
    output = np.asarray(mask, dtype=bool).copy()
    merges = trims = shifts = 0
    for _ in range(max_edits):
        starts, ends = EXP18.run_bounds(output)
        best_merge = None
        for run in range(len(starts) - 1):
            left_length = int(ends[run] - starts[run])
            right_length = int(ends[run + 1] - starts[run + 1])
            gap = int(starts[run + 1] - ends[run])
            if row_budget is not None and int(output.sum()) + gap > row_budget:
                continue
            saving = (
                lookup_chunk_ms(left_length, lookup_model)
                + lookup_chunk_ms(right_length, lookup_model)
                - lookup_chunk_ms(left_length + gap + right_length, lookup_model)
            )
            if saving > 1e-14:
                candidate = (-saving, gap, int(ends[run]), int(starts[run + 1]))
                if best_merge is None or candidate < best_merge:
                    best_merge = candidate
        if best_merge is not None:
            _, _, left, right = best_merge
            output[left:right] = True
            merges += 1
            continue

        surplus = float(values[output].sum()) - target
        best_trim = None
        for start, end in zip(starts, ends):
            length = int(end - start)
            saving = lookup_chunk_ms(length, lookup_model)
            if length > 1:
                saving -= lookup_chunk_ms(length - 1, lookup_model)
            if saving <= 1e-14:
                continue
            for index in {int(start), int(end - 1)}:
                lost = float(values[index])
                if lost <= surplus + 1e-14:
                    candidate = (lost / saving, lost, index)
                    if best_trim is None or candidate < best_trim:
                        best_trim = candidate
        if best_trim is not None:
            output[best_trim[2]] = False
            trims += 1
            continue

        best_shift = None
        for start, end in zip(starts, ends):
            start = int(start); end = int(end)
            if start > 0 and (start < 2 or not output[start - 2]):
                gain = float(values[start - 1] - values[end - 1])
                if gain > 1e-14:
                    candidate = (-gain, start - 1, end - 1)
                    if best_shift is None or candidate < best_shift:
                        best_shift = candidate
            if end < len(output) and (end + 1 >= len(output) or not output[end + 1]):
                gain = float(values[end] - values[start])
                if gain > 1e-14:
                    candidate = (-gain, end, start)
                    if best_shift is None or candidate < best_shift:
                        best_shift = candidate
        if best_shift is not None:
            _, add_index, remove_index = best_shift
            output[remove_index] = False
            output[add_index] = True
            shifts += 1
            continue
        break
    return output, {
        "local_edits": merges + trims + shifts,
        "local_merges": merges,
        "local_trims": trims,
        "local_shifts": shifts,
    }


@njit(cache=False, inline="always")
def _threshold_candidate_better(importance, cost, threshold,
                                incumbent_importance, incumbent_cost):
    feasible = importance >= threshold - 1e-14
    incumbent_feasible = incumbent_importance >= threshold - 1e-14
    if feasible and not incumbent_feasible:
        return True
    if feasible and incumbent_feasible:
        if cost < incumbent_cost - 1e-14:
            return True
        return abs(cost - incumbent_cost) <= 1e-14 and importance > incumbent_importance + 1e-14
    if not feasible and not incumbent_feasible:
        if importance > incumbent_importance + 1e-14:
            return True
        return abs(importance - incumbent_importance) <= 1e-14 and cost < incumbent_cost - 1e-14
    return False


@njit(cache=False)
def _capped_label_mask(values, target, chunk_costs, tail_per_row,
                       label_cap, row_cap):
    """Keep K target-threshold labels in each capped run-length state."""
    n = len(values)
    m = len(chunk_costs) - 1
    current_i = np.full((m + 1, label_cap), -np.inf, dtype=np.float64)
    current_c = np.full((m + 1, label_cap), np.inf, dtype=np.float64)
    current_r = np.full((m + 1, label_cap), n + 1, dtype=np.int32)
    current_i[0, 0] = 0.0
    current_c[0, 0] = 0.0
    current_r[0, 0] = 0
    parent_state = np.full((n, m + 1, label_cap), -1, dtype=np.int16)
    parent_label = np.full((n, m + 1, label_cap), -1, dtype=np.int8)

    for index in range(n):
        next_i = np.full((m + 1, label_cap), -np.inf, dtype=np.float64)
        next_c = np.full((m + 1, label_cap), np.inf, dtype=np.float64)
        next_r = np.full((m + 1, label_cap), n + 1, dtype=np.int32)
        value = values[index]

        # Skip the current row: any old run-length state becomes state zero.
        for slot in range(label_cap):
            threshold = 0.0 if label_cap == 1 else target * slot / (label_cap - 1)
            best_i = -np.inf; best_c = np.inf; best_r = n + 1
            best_state = -1; best_label = -1
            for state in range(m + 1):
                for label in range(label_cap):
                    candidate_i = current_i[state, label]
                    if np.isneginf(candidate_i):
                        continue
                    candidate_c = current_c[state, label]
                    if _threshold_candidate_better(
                        candidate_i, candidate_c, threshold, best_i, best_c
                    ):
                        best_i = candidate_i; best_c = candidate_c
                        best_r = current_r[state, label]
                        best_state = state; best_label = label
            next_i[0, slot] = best_i; next_c[0, slot] = best_c; next_r[0, slot] = best_r
            parent_state[index, 0, slot] = best_state
            parent_label[index, 0, slot] = best_label

        # Start a run from state zero.
        for slot in range(label_cap):
            threshold = 0.0 if label_cap == 1 else target * slot / (label_cap - 1)
            best_i = -np.inf; best_c = np.inf; best_r = n + 1; best_label = -1
            for label in range(label_cap):
                if current_r[0, label] + 1 > row_cap:
                    continue
                candidate_i = current_i[0, label] + value
                candidate_c = current_c[0, label] + chunk_costs[1]
                if _threshold_candidate_better(
                    candidate_i, candidate_c, threshold, best_i, best_c
                ):
                    best_i = candidate_i; best_c = candidate_c
                    best_r = current_r[0, label] + 1; best_label = label
            next_i[1, slot] = best_i; next_c[1, slot] = best_c; next_r[1, slot] = best_r
            parent_state[index, 1, slot] = 0
            parent_label[index, 1, slot] = best_label

        for state in range(2, m):
            increment = chunk_costs[state] - chunk_costs[state - 1]
            for slot in range(label_cap):
                threshold = 0.0 if label_cap == 1 else target * slot / (label_cap - 1)
                best_i = -np.inf; best_c = np.inf; best_r = n + 1; best_label = -1
                for label in range(label_cap):
                    if current_r[state - 1, label] + 1 > row_cap:
                        continue
                    candidate_i = current_i[state - 1, label] + value
                    candidate_c = current_c[state - 1, label] + increment
                    if _threshold_candidate_better(
                        candidate_i, candidate_c, threshold, best_i, best_c
                    ):
                        best_i = candidate_i; best_c = candidate_c
                        best_r = current_r[state - 1, label] + 1; best_label = label
                next_i[state, slot] = best_i; next_c[state, slot] = best_c
                next_r[state, slot] = best_r
                parent_state[index, state, slot] = state - 1
                parent_label[index, state, slot] = best_label

        # The capped state accepts the boundary transition and the linear tail.
        boundary_increment = chunk_costs[m] - chunk_costs[m - 1]
        for slot in range(label_cap):
            threshold = 0.0 if label_cap == 1 else target * slot / (label_cap - 1)
            best_i = -np.inf; best_c = np.inf; best_r = n + 1
            best_state = -1; best_label = -1
            for source_state in (m - 1, m):
                increment = boundary_increment if source_state == m - 1 else tail_per_row
                for label in range(label_cap):
                    if current_r[source_state, label] + 1 > row_cap:
                        continue
                    candidate_i = current_i[source_state, label] + value
                    candidate_c = current_c[source_state, label] + increment
                    if _threshold_candidate_better(
                        candidate_i, candidate_c, threshold, best_i, best_c
                    ):
                        best_i = candidate_i; best_c = candidate_c
                        best_r = current_r[source_state, label] + 1
                        best_state = source_state; best_label = label
            next_i[m, slot] = best_i; next_c[m, slot] = best_c; next_r[m, slot] = best_r
            parent_state[index, m, slot] = best_state
            parent_label[index, m, slot] = best_label

        current_i = next_i; current_c = next_c; current_r = next_r

    best_i = -np.inf; best_c = np.inf; best_r = n + 1
    best_state = -1; best_label = -1
    for state in range(m + 1):
        for label in range(label_cap):
            candidate_i = current_i[state, label]
            if np.isneginf(candidate_i):
                continue
            candidate_c = current_c[state, label]
            if _threshold_candidate_better(
                candidate_i, candidate_c, target, best_i, best_c
            ):
                best_i = candidate_i; best_c = candidate_c
                best_r = current_r[state, label]
                best_state = state; best_label = label
    mask = np.zeros(n, dtype=np.bool_)
    state = best_state; label = best_label
    for index in range(n - 1, -1, -1):
        if state > 0:
            mask[index] = True
        old_state = int(parent_state[index, state, label])
        old_label = int(parent_label[index, state, label])
        state = old_state; label = old_label
    return mask, best_i, best_c, best_r


def capped_label_dp(values: np.ndarray, target: float, lookup_model: dict,
                    label_cap: int, row_budget: int | None) -> tuple[np.ndarray, dict]:
    cap = len(values) if row_budget is None else int(row_budget)
    result = _capped_label_mask(
        values, target, lookup_model["chunk_costs"],
        lookup_model["tail_ms_per_row"], int(label_cap), cap,
    )
    return np.asarray(result[0], dtype=bool), {
        "label_cap": int(label_cap),
        "label_objective_ms": float(result[2]),
    }


def paper_mask(track: str, values: np.ndarray, target: float, row_budget: int,
               lookup_table, row_size_kib: float, params, windows):
    if track == "coverage":
        return EXP18.exact_paper_coverage(values, target, *windows)
    return EXP18.paper_fixed_rows(values, row_budget, row_size_kib, lookup_table, params)


def core_select(method: str, track: str, values: np.ndarray, target: float,
                row_budget: int, model: dict, lookup_model: dict,
                piecewise_models: dict[int, dict], lookup_table,
                row_size_kib: float, params, windows) -> tuple[np.ndarray, dict]:
    if method == "paper":
        return paper_mask(
            track, values, target, row_budget, lookup_table,
            row_size_kib, params, windows,
        )
    if method == "incumbent_td16_trim256":
        return EXP18.core_select(
            "td_2l_c16_trim256", track, values, target, row_budget, model,
            lookup_table, row_size_kib, params, windows,
        )
    if method.startswith("lambda_bank"):
        return lambda_bank(values, target, model, int(method.removeprefix("lambda_bank")))
    if method.startswith("pwl_j"):
        segments_text, calls_text = method.removeprefix("pwl_j").split("_c")
        segments = int(segments_text); calls = int(calls_text)
        return target_directed_piecewise(values, target, piecewise_models[segments], calls)
    if method.startswith("paper_local"):
        limit = int(method.removeprefix("paper_local"))
        mask, metadata = paper_mask(
            track, values, target, row_budget, lookup_table,
            row_size_kib, params, windows,
        )
        refined, edits = paper_local_edits(
            mask, values, target, lookup_model, limit,
            None if track == "coverage" else row_budget,
        )
        return refined, {**metadata, **edits}
    if method.startswith("td_lookup_c"):
        return target_directed_lookup(
            values, target, lookup_model, int(method.removeprefix("td_lookup_c"))
        )
    if method.startswith("predict_correct_c"):
        return predict_and_correct(
            values, target, model, int(method.removeprefix("predict_correct_c"))
        )
    if method.startswith("capped_label"):
        return capped_label_dp(
            values, target, lookup_model, int(method.removeprefix("capped_label")),
            None if track == "coverage" else row_budget,
        )
    raise KeyError(method)


def select_with_fallback(method: str, track: str, values: np.ndarray,
                         target: float, row_budget: int, model: dict,
                         lookup_model: dict, piecewise_models: dict[int, dict],
                         lookup_table, row_size_kib: float, params,
                         windows) -> tuple[np.ndarray, dict]:
    metadata = {}
    error = ""
    try:
        mask, metadata = core_select(
            method, track, values, target, row_budget, model, lookup_model,
            piecewise_models, lookup_table, row_size_kib, params, windows,
        )
    except Exception as exception:
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
        mask, fallback_added = EXP18.add_top_fallback(mask, values, target, cap)
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


def warm_up(values: np.ndarray, target: float, row_budget: int, model: dict,
            lookup_model: dict, piecewise_models: dict[int, dict], lookup_table,
            row_size_kib: float, params, windows) -> None:
    for method in DEFAULT_METHODS:
        for track in ("coverage", "row_budget"):
            select_with_fallback(
                method, track, values, target, row_budget, model, lookup_model,
                piecewise_models, lookup_table, row_size_kib, params, windows,
            )


def piecewise_cost(mask: np.ndarray, piecewise: dict) -> float:
    starts, ends = EXP18.run_bounds(mask)
    total = 0.0
    for length in ends - starts:
        matches = np.flatnonzero(
            (piecewise["lows"] <= length) & (length <= piecewise["highs"])
        )
        if len(matches) != 1:
            raise RuntimeError("piecewise ranges do not partition run lengths")
        segment = int(matches[0])
        total += piecewise["intercepts"][segment] + piecewise["slopes"][segment] * length
    return float(total)


def self_check(model: dict, lookup_model: dict, lookup_table,
               row_size_kib: float, params) -> None:
    EXP16.self_check_lookup_solver()
    rng = np.random.default_rng(2020)
    values = rng.lognormal(size=9)
    values /= values.sum()
    piecewise = fit_piecewise_lookup(lookup_model, len(values), 4)
    multiplier = 0.7
    actual = _solve_pwl_metrics(
        values, multiplier, piecewise["lows"], piecewise["highs"],
        piecewise["slopes"], piecewise["intercepts"],
    )
    best = (-np.inf, -np.inf, np.inf)
    for encoded in range(1 << len(values)):
        mask = np.asarray([(encoded >> index) & 1 for index in range(len(values))], dtype=bool)
        importance = float(values[mask].sum())
        cost = piecewise_cost(mask, piecewise)
        score = multiplier * importance - cost
        if EXP13._better(score, importance, cost, *best):
            best = (score, importance, cost)
    if not (
        math.isclose(float(actual[0]), best[0], rel_tol=1e-10, abs_tol=1e-11)
        and math.isclose(float(actual[1]), best[1], rel_tol=1e-10, abs_tol=1e-11)
        and math.isclose(float(actual[2]), best[2], rel_tol=1e-10, abs_tol=1e-11)
    ):
        raise RuntimeError("piecewise O(JN) solver failed brute-force self-check")
    replay = _solve_pwl_mask(
        values, multiplier, piecewise["lows"], piecewise["highs"],
        piecewise["slopes"], piecewise["intercepts"],
    )
    if not math.isclose(piecewise_cost(replay[0], piecewise), float(replay[3]), abs_tol=1e-11):
        raise RuntimeError("piecewise mask replay failed")

    target = 0.7
    for labels in (2, 4, 8):
        mask, _ = capped_label_dp(values, target, lookup_model, labels, None)
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise RuntimeError("capped-label mask contract failed")
    base = np.asarray([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=bool)
    target = min(float(values[base].sum()), 0.6)
    before = lookup_mask_ms(base, lookup_model)
    refined, _ = paper_local_edits(base, values, target, lookup_model, 32, None)
    if float(values[refined].sum()) < target - 1e-11:
        raise RuntimeError("local edits violated coverage")
    if lookup_mask_ms(refined, lookup_model) > before + 1e-11:
        raise RuntimeError("local edits increased lookup latency")

    windows = EXP18.paper_windows(len(values), row_size_kib, lookup_table, params)
    pwls = {count: fit_piecewise_lookup(lookup_model, len(values), count) for count in (4, 8)}
    for method in DEFAULT_METHODS:
        mask, metadata = select_with_fallback(
            method, "coverage", values, 0.6, len(values), model, lookup_model,
            pwls, lookup_table, row_size_kib, params, windows,
        )
        if float(values[mask].sum()) < 0.6 - 1e-11 or metadata.get("error"):
            raise RuntimeError(f"method self-check failed for {method}: {metadata}")


def collect(args: argparse.Namespace, n_values: list[int], targets: list[float],
            budgets: list[float], model: dict, lookup_model: dict,
            lookup_table, params):
    trial_rows = []
    timing_rows = []
    rng = np.random.default_rng(args.seed)
    warmed_n = set()
    for n in n_values:
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()
        windows = EXP18.paper_windows(n, args.row_size_kib, lookup_table, params)
        piecewise_models = {
            count: fit_piecewise_lookup(lookup_model, n, count) for count in (4, 8)
        }
        for trial in range(args.trials):
            multiset = EXP2.exact_cv_lognormal(rng, n, args.cv)
            variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
            for spatial_mode, raw_values in variants.items():
                values = np.asarray(raw_values, dtype=np.float64)
                values /= values.sum()
                if n not in warmed_n:
                    warm_up(
                        values, targets[0], max(1, int(round(n * budgets[0]))),
                        model, lookup_model, piecewise_models, lookup_table,
                        args.row_size_kib, params, windows,
                    )
                    warmed_n.add(n)
                for scenario, (target_fraction, budget_fraction) in enumerate(zip(targets, budgets)):
                    target = float(target_fraction * values.sum())
                    row_budget = max(1, min(n, int(round(n * budget_fraction))))
                    top_importance = float(np.partition(values, n - row_budget)[n - row_budget:].sum())
                    intrinsic = top_importance >= target - 1e-11
                    for track in ("coverage", "row_budget"):
                        for method in DEFAULT_METHODS:
                            function = lambda method=method, track=track: select_with_fallback(
                                method, track, values, target, row_budget, model,
                                lookup_model, piecewise_models, lookup_table,
                                args.row_size_kib, params, windows,
                            )
                            mask, metadata, timings = EXP18.benchmark(function, args.repetitions)
                            metrics = EXP18.mask_metrics(
                                mask, values, model, lookup_table, args.row_size_kib
                            )
                            coverage_met = metrics["importance"] >= target - 1e-11
                            row_budget_met = metrics["rows"] <= row_budget
                            valid = coverage_met and (track == "coverage" or row_budget_met)
                            runtime_median = float(np.median(timings))
                            runtime_p95 = float(np.quantile(timings, 0.95))
                            base = {
                                "n": n, "trial": trial, "spatial_mode": spatial_mode,
                                "scenario": scenario, "track": track,
                                "target_fraction": target_fraction,
                                "target_importance": target,
                                "row_budget_fraction": budget_fraction,
                                "row_budget": row_budget,
                                "top_row_budget_importance": top_importance,
                                "row_budget_intrinsically_feasible": intrinsic,
                                "method": method, "method_label": METHOD_LABELS[method],
                                "family": method_family(method),
                            }
                            trial_rows.append({
                                **base, **metrics,
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
                                "local_edits": metadata.get("local_edits", 0),
                                "piecewise_segments": metadata.get("piecewise_segments", 0),
                                "label_cap": metadata.get("label_cap", 0),
                            })
                            for repetition, elapsed in enumerate(timings):
                                timing_rows.append({
                                    **base, "repetition": repetition,
                                    "runtime_ms": elapsed,
                                    "deadline_met": elapsed <= args.deadline_ms,
                                    "valid": valid,
                                    "error": metadata.get("error", ""),
                                })
                print(
                    f"N={n:,} trial={trial + 1}/{args.trials} mode={spatial_mode}",
                    flush=True,
                )
    return trial_rows, timing_rows


def add_paired_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    derived = [
        "paper_lookup_ms", "incumbent_lookup_ms", "lookup_saving_vs_paper_pct",
        "lookup_saving_vs_incumbent_pct",
    ]
    frame = frame.drop(columns=[column for column in derived if column in frame])
    keys = ["n", "trial", "spatial_mode", "scenario", "track"]
    paper = frame[frame.method == "paper"][keys + ["lookup_ms"]].rename(
        columns={"lookup_ms": "paper_lookup_ms"}
    )
    incumbent = frame[frame.method == "incumbent_td16_trim256"][
        keys + ["lookup_ms"]
    ].rename(columns={"lookup_ms": "incumbent_lookup_ms"})
    output = frame.merge(paper, on=keys, how="left").merge(incumbent, on=keys, how="left")
    output["lookup_saving_vs_paper_pct"] = 100.0 * (
        1.0 - output.lookup_ms / output.paper_lookup_ms
    )
    output["lookup_saving_vs_incumbent_pct"] = 100.0 * (
        1.0 - output.lookup_ms / output.incumbent_lookup_ms
    )
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


def summarize_frame(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rows = []
    nested = {}
    for (track, method), group in frame.groupby(["track", "method"], sort=False):
        valid = group[group.valid]
        feasible = group if track == "coverage" else group[group.row_budget_intrinsically_feasible]
        row = {
            "track": track,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "cases": len(group),
            "valid_cases": len(valid),
            "valid_rate": float(group.valid.mean()),
            "coverage_success_rate": float(group.coverage_met.mean()),
            "row_budget_success_rate": float(group.row_budget_met.mean()),
            "deadline_pass_rate": float(group.deadline_met.mean()),
            "valid_and_deadline_pass_rate": float(group.valid_and_deadline_met.mean()),
            "feasible_case_valid_and_deadline_pass_rate": (
                float(feasible.valid_and_deadline_met.mean()) if len(feasible) else math.nan
            ),
            "fallback_rate": float(group.fallback_used.mean()),
            "error_rate": float(group.error.fillna("").astype(bool).mean()),
            "deterministic_rate": float(group.deterministic.mean()),
            "scalarized_calls_median": float(group.scalarized_calls.median()),
            "runtime_median_ms": float(group.runtime_median_ms.median()),
            "runtime_p95_across_cases_ms": float(group.runtime_p95_ms.quantile(0.95)),
            "runtime_worst_case_p95_ms": float(group.runtime_p95_ms.max()),
            "lookup_saving_vs_paper_pct_mean_valid": (
                float(valid.lookup_saving_vs_paper_pct.mean()) if len(valid) else math.nan
            ),
            "lookup_saving_vs_incumbent_pct_mean_valid": (
                float(valid.lookup_saving_vs_incumbent_pct.mean()) if len(valid) else math.nan
            ),
        }
        rows.append(row)
        nested.setdefault(track, {})[method] = {
            **row,
            "runtime_median_ms_distribution": distribution(group.runtime_median_ms),
            "runtime_p95_ms_distribution": distribution(group.runtime_p95_ms),
        }
    summary = {
        "format": "experiment-20-remaining-sub2ms-candidates-v1",
        "method_count": len(DEFAULT_METHODS),
        "methods": list(DEFAULT_METHODS),
        "deadline_ms": args.deadline_ms,
        "timing_scope": (
            "warm host CPU query including selection, mask recovery, local edits, "
            "fallback, validation path, and CPU mask handoff; excluding compilation, "
            "importance production, and actual I/O/compute"
        ),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
            "torch_threads": int(torch.get_num_threads()),
        },
        "configuration": {
            "n_values": sorted(int(value) for value in frame.n.unique()),
            "trials": args.trials, "repetitions": args.repetitions,
            "cv": args.cv, "target_fractions": list(args.target_fractions),
            "row_budget_fractions": list(args.row_budget_fractions),
            "profile": args.profile, "row_size_kib": args.row_size_kib,
            "saturation_kib": args.saturation_kib,
        },
        "tracks": nested,
    }
    return pd.DataFrame(rows), summary


def configure_plot():
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_runtime_quality(summary: pd.DataFrame, path: Path, deadline_ms: float) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), constrained_layout=True)
    markers = (
        "o", "s", "^", "v", "D", "P", "X", "*", "<", ">",
        "h", "p", "d", "H", "8", "+", "x", "1", "2", "3",
    )
    for ax, track in zip(axes, ("coverage", "row_budget")):
        selected = summary[summary.track == track].set_index("method")
        for index, method in enumerate(DEFAULT_METHODS):
            row = selected.loc[method]
            ax.scatter(
                row.runtime_median_ms, row.lookup_saving_vs_paper_pct_mean_valid,
                color=FAMILY_COLORS[method_family(method)], s=62,
                marker=markers[index], label=METHOD_LABELS[method], zorder=3,
            )
        ax.axvline(deadline_ms, color="#111827", linestyle="--", linewidth=1.2)
        ax.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Median end-to-end selector time (ms, log)")
        ax.set_ylabel("Mean lookup-latency saving vs Paper (%)\n(valid cases)")
        ax.set_title("Coverage-only" if track == "coverage" else "Coverage + row budget")
        BASE.polish_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=7.7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_deadline(summary: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.4), sharex=True, constrained_layout=True)
    y = np.arange(len(DEFAULT_METHODS))
    for ax, track in zip(axes, ("coverage", "row_budget")):
        selected = summary[summary.track == track].set_index("method").reindex(DEFAULT_METHODS)
        ax.barh(
            y, 100.0 * selected.valid_and_deadline_pass_rate,
            color=[FAMILY_COLORS[method_family(method)] for method in DEFAULT_METHODS],
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


def fmt_pct(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}%"


def write_report(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Experiment 20 보고서: 남은 sub-2 ms 후보",
        "",
        "실험 18에서 다루지 않은 여섯 후보 계열 18개 설정과 Paper/현 incumbent를 "
        "동일한 측정 계약으로 비교한다. Q는 외부 입력이며 실제 I/O와 compute는 selector "
        "시간에서 제외된다. 따라서 이 결과는 host screening이지 Jetson 2 ms 확인이 아니다.",
        "",
    ]
    for track, title in (("coverage", "Coverage-only"), ("row_budget", "Coverage + row budget")):
        lines.extend([
            f"## {title}", "",
            "| 방법 | 중앙 runtime | case-p95 | 유효률 | 유효+2ms | Paper 대비 lookup | incumbent 대비 lookup | fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        selected = summary[summary.track == track].set_index("method").reindex(DEFAULT_METHODS)
        for method, row in selected.iterrows():
            lines.append(
                f"| {METHOD_LABELS[method]} | {fmt(row.runtime_median_ms)} ms "
                f"| {fmt(row.runtime_p95_across_cases_ms)} ms "
                f"| {100 * row.valid_rate:.1f}% "
                f"| {100 * row.valid_and_deadline_pass_rate:.1f}% "
                f"| {fmt_pct(row.lookup_saving_vs_paper_pct_mean_valid)} "
                f"| {fmt_pct(row.lookup_saving_vs_incumbent_pct_mean_valid)} "
                f"| {100 * row.fallback_rate:.1f}% |"
            )
        lines.append("")
    coverage = summary[summary.track == "coverage"]
    eligible = coverage[
        (coverage.valid_and_deadline_pass_rate == 1.0)
        & np.isfinite(coverage.lookup_saving_vs_paper_pct_mean_valid)
    ].sort_values("lookup_saving_vs_paper_pct_mean_valid", ascending=False)
    if len(eligible):
        winner = eligible.iloc[0]
        conclusion = (
            f"Host 기준 모든 coverage case에서 유효하고 p95<=2 ms인 설정 중 lookup 품질 "
            f"최고는 `{winner.method_label}`이며 Paper 대비 평균 "
            f"`{winner.lookup_saving_vs_paper_pct_mean_valid:.2f}%`다."
        )
    else:
        conclusion = "모든 coverage case에서 유효성과 p95<=2 ms를 동시에 만족한 설정은 없다."
    remaining = coverage[
        ~coverage.method.isin(("paper", "incumbent_td16_trim256"))
    ]
    remaining_eligible = remaining[
        (remaining.valid_and_deadline_pass_rate == 1.0)
        & np.isfinite(remaining.lookup_saving_vs_paper_pct_mean_valid)
    ].sort_values("lookup_saving_vs_paper_pct_mean_valid", ascending=False)
    remaining_quality = remaining[
        np.isfinite(remaining.lookup_saving_vs_paper_pct_mean_valid)
    ].sort_values("lookup_saving_vs_paper_pct_mean_valid", ascending=False)
    if len(remaining_eligible):
        viable = remaining_eligible.iloc[0]
        viable_text = (
            f"남은 여섯 계열만 보면 전 case 2 ms를 통과한 최고 품질 설정은 "
            f"`{viable.method_label}`이다. Paper 대비 `{viable.lookup_saving_vs_paper_pct_mean_valid:.2f}%` "
            f"절감하지만 incumbent 대비로는 `{viable.lookup_saving_vs_incumbent_pct_mean_valid:.2f}%`다."
        )
    else:
        viable_text = "남은 후보 중 모든 coverage case에서 2 ms를 통과한 설정은 없다."
    quality = remaining_quality.iloc[0]
    quality_text = (
        f"시간 제한을 무시한 남은 후보의 최고 lookup 품질은 `{quality.method_label}`의 "
        f"Paper 대비 `{quality.lookup_saving_vs_paper_pct_mean_valid:.2f}%`지만, case-p95가 "
        f"`{quality.runtime_p95_across_cases_ms:.2f} ms`라 2 ms 후보가 아니다."
    )
    lines.extend([
        "## 판정", "", conclusion, "", viable_text, "", quality_text, "",
        "시간 초과, coverage 실패, row-cap 실패, fallback은 제외하지 않고 전체 분모에 남겼다.",
        "",
        "![Runtime-quality](results/runtime_quality.png)", "",
        "![Deadline pass rate](results/deadline_pass_rate.png)", "",
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
    lookup_model = EXP16.lookup_cost_model(lookup_table, args.row_size_kib)
    params = BASE.ChunkParams(
        start_kb=args.start_kib, end_kb=args.saturation_kib,
        step_kb=args.start_kib, jump_cap_kb=args.jump_cap_kib,
    )
    if not args.analyze_only:
        if not args.skip_self_check:
            self_check(model, lookup_model, lookup_table, args.row_size_kib, params)
        rows, timing_rows = collect(
            args, n_values, targets, budgets, model, lookup_model,
            lookup_table, params,
        )
        write_csv(args.output_dir / "trials.csv", rows)
        write_csv(args.output_dir / "timing_samples.csv", timing_rows)
        metadata = {
            "two_line_model": model,
            "lookup_model": {
                "cap_rows": int(lookup_model["cap_rows"]),
                "tail_ms_per_row": float(lookup_model["tail_ms_per_row"]),
            },
            "method_labels": METHOD_LABELS,
            "method_count": len(DEFAULT_METHODS),
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    analyze(args)


if __name__ == "__main__":
    main()
