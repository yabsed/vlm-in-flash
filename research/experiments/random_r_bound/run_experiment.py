#!/usr/bin/env python3
"""Compare VLM-in-a-Flash greedy selection with exact and approximate solvers.

The experiment deliberately separates three optimization problems:

1. Fixed budget: R(M) = R, evaluated under the fitted affine latency
   L_aff(M) = a K(M) + c R.  This is solved exactly with Dinkelbach's
   method and the two-state chain DP from ``research/final_email/idea.md``.
2. Upper budget: 1 <= R(M) <= R, evaluated with the released lookup table.
   The exact solution is the best single contiguous interval of length at
   most R, by the weighted-average theorem in the same document.
3. Importance coverage: I(M) >= alpha I_total.  An exact (r,k) chain DP is
   compared with an O(qN) Lagrangian relaxation that solves q two-state
   chain problems and keeps the feasible supported-frontier candidates, and
   with an O(qN) quantized Pareto DP that preserves q coverage buckets.

The paper heuristic is the repository's actual ``select_chunks`` function,
and the latency-blind baseline is its ``select_topk`` function.  All
comparisons are paired: every method sees the same random importance vector
and row budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
VLM_FLASH = RESEARCH / "vlm-flash"
sys.path.insert(0, str(VLM_FLASH / "src"))

from vlmflash import ChunkParams, LatencyTable, select_chunks, select_topk  # noqa: E402


def affine_fit(table: LatencyTable, row_size_kib: float) -> tuple[float, float, float]:
    """Return intercept, per-row slope, and R^2 for the bundled latency table."""
    pairs = sorted(table.as_dict().items())
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    slope_kib, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope_kib * x
    r2 = 1.0 - float(np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(intercept), float(slope_kib * row_size_kib), r2


def mask_stats(mask: np.ndarray) -> tuple[int, int]:
    """Return (selected rows, maximal contiguous runs)."""
    m = np.asarray(mask, dtype=bool)
    selected = int(m.sum())
    chunks = int(np.sum(m & ~np.r_[False, m[:-1]]))
    return selected, chunks


def affine_latency(mask: np.ndarray, a_ms: float, c_ms_per_row: float) -> float:
    selected, chunks = mask_stats(mask)
    return a_ms * chunks + c_ms_per_row * selected


def _inner_chain_dp(
    values: np.ndarray, selected_rows: int, eta: float, chunk_penalty_ms: float
) -> tuple[np.ndarray, float]:
    """Solve max I(M) - eta*a*K(M) at an exact cardinality.

    Parent pointers are retained so the maximizing mask can be recovered.
    The constant ``-eta*c*R`` is intentionally left outside this inner DP.
    """
    n = len(values)
    rmax = selected_rows
    neg_inf = -np.inf
    end_zero = np.full(rmax + 1, neg_inf, dtype=np.float64)
    end_one = np.full(rmax + 1, neg_inf, dtype=np.float64)
    end_zero[0] = 0.0

    # 0/1 gives the previous ending state; 255 marks an unreachable state.
    parent_zero = np.full((n + 1, rmax + 1), 255, dtype=np.uint8)
    parent_one = np.full((n + 1, rmax + 1), 255, dtype=np.uint8)

    for i, value in enumerate(values, start=1):
        limit = min(i, rmax)
        new_zero = np.full(rmax + 1, neg_inf, dtype=np.float64)
        new_one = np.full(rmax + 1, neg_inf, dtype=np.float64)

        z0 = end_zero[: limit + 1]
        z1 = end_one[: limit + 1]
        take_zero = z0 >= z1
        new_zero[: limit + 1] = np.where(take_zero, z0, z1)
        parent_zero[i, : limit + 1] = np.where(take_zero, 0, 1).astype(np.uint8)

        if limit:
            # Selecting this row either continues state 1 or starts a run
            # from state 0 and pays eta*a.
            continue_run = end_one[:limit]
            start_run = end_zero[:limit] - eta * chunk_penalty_ms
            take_continue = continue_run >= start_run
            new_one[1 : limit + 1] = value + np.where(
                take_continue, continue_run, start_run
            )
            parent_one[i, 1 : limit + 1] = np.where(
                take_continue, 1, 0
            ).astype(np.uint8)

        end_zero, end_one = new_zero, new_one

    state = 0 if end_zero[rmax] >= end_one[rmax] else 1
    score = float(max(end_zero[rmax], end_one[rmax]))
    mask = np.zeros(n, dtype=bool)
    r = rmax
    for i in range(n, 0, -1):
        if state == 0:
            state = int(parent_zero[i, r])
        else:
            mask[i - 1] = True
            state = int(parent_one[i, r])
            r -= 1
    if r != 0 or state != 0:
        raise RuntimeError("chain-DP backtracking reached an invalid initial state")
    return mask, score


def exact_fixed_r_affine(
    values: np.ndarray,
    selected_rows: int,
    a_ms: float,
    c_ms_per_row: float,
    tolerance: float = 1e-12,
    max_iterations: int = 64,
) -> tuple[np.ndarray, int, float]:
    """Exactly maximize I/(aK+cR), up to floating-point tolerance."""
    if not 1 <= selected_rows <= len(values):
        raise ValueError("selected_rows must be in [1, len(values)]")
    eta = 0.0
    for iteration in range(1, max_iterations + 1):
        mask, inner_score = _inner_chain_dp(values, selected_rows, eta, a_ms)
        importance = float(values[mask].sum())
        _, chunks = mask_stats(mask)
        latency = a_ms * chunks + c_ms_per_row * selected_rows
        residual = inner_score - eta * c_ms_per_row * selected_rows
        new_eta = importance / latency
        scale = max(1.0, importance, eta * latency)
        if abs(residual) <= tolerance * scale or abs(new_eta - eta) <= tolerance * max(1.0, new_eta):
            return mask, iteration, residual
        eta = new_eta
    raise RuntimeError(f"Dinkelbach did not converge after {max_iterations} iterations")


class ExactCoverageOracle:
    """Exact offline oracle for min aK+cR subject to an importance bound."""

    def __init__(self, values: np.ndarray, a_ms: float, c_ms_per_row: float):
        self.values = np.asarray(values, dtype=np.float64)
        self.a_ms = a_ms
        self.c_ms_per_row = c_ms_per_row
        self.n = len(values)
        self.kmax = (self.n + 1) // 2
        shape = (self.n + 1, self.kmax + 1)
        neg_inf = -np.inf
        end_zero = np.full(shape, neg_inf, dtype=np.float64)
        end_one = np.full(shape, neg_inf, dtype=np.float64)
        end_zero[0, 0] = 0.0

        parent_shape = (self.n + 1, *shape)
        self.parent_zero = np.full(parent_shape, 255, dtype=np.uint8)
        self.parent_one = np.full(parent_shape, 255, dtype=np.uint8)

        for i, value in enumerate(self.values, start=1):
            take_zero = end_zero >= end_one
            new_zero = np.where(take_zero, end_zero, end_one)
            self.parent_zero[i] = np.where(take_zero, 0, 1).astype(np.uint8)

            continue_run = end_one[:-1, :]
            start_run = np.full_like(continue_run, neg_inf)
            start_run[:, 1:] = end_zero[:-1, :-1]
            take_continue = continue_run >= start_run
            new_one = np.full(shape, neg_inf, dtype=np.float64)
            new_one[1:, :] = value + np.where(take_continue, continue_run, start_run)
            self.parent_one[i, 1:, :] = np.where(
                take_continue, 1, 0
            ).astype(np.uint8)
            end_zero, end_one = new_zero, new_one

        self.end_zero = end_zero
        self.end_one = end_one
        self.importance = np.maximum(end_zero, end_one)
        rows = np.arange(self.n + 1, dtype=np.float64)[:, None]
        chunks = np.arange(self.kmax + 1, dtype=np.float64)[None, :]
        self.cost = self.c_ms_per_row * rows + self.a_ms * chunks

    def solve(self, bound: float) -> dict:
        """Return the exact minimum-cost mask attaining ``importance >= bound``."""
        if not 0 < bound <= float(self.values.sum()) + 1e-12:
            raise ValueError("coverage bound must lie in (0, total importance]")
        feasible = self.importance >= bound - 1e-14
        candidate_cost = np.where(feasible, self.cost, np.inf)
        minimum = float(candidate_cost.min())
        tied = feasible & np.isclose(self.cost, minimum, rtol=1e-12, atol=1e-15)
        tied_importance = np.where(tied, self.importance, -np.inf)
        r, k = np.unravel_index(int(np.argmax(tied_importance)), tied_importance.shape)
        state = 0 if self.end_zero[r, k] >= self.end_one[r, k] else 1
        importance = float(self.importance[r, k])

        mask = np.zeros(self.n, dtype=bool)
        remaining_rows, remaining_chunks = int(r), int(k)
        for i in range(self.n, 0, -1):
            if state == 0:
                state = int(self.parent_zero[i, remaining_rows, remaining_chunks])
            else:
                mask[i - 1] = True
                previous_state = int(
                    self.parent_one[i, remaining_rows, remaining_chunks]
                )
                remaining_rows -= 1
                if previous_state == 0:
                    remaining_chunks -= 1
                state = previous_state
        if remaining_rows != 0 or remaining_chunks != 0 or state != 0:
            raise RuntimeError("coverage-DP backtracking reached an invalid initial state")
        return {
            "mask": mask,
            "importance": importance,
            "rows": int(r),
            "chunks": int(k),
            "aff_ms": minimum,
        }


def solve_lagrangian(
    values: np.ndarray, lam: float, a_ms: float, c_ms_per_row: float
) -> dict:
    """Solve min aK+cR-lambda*I in O(N), returning a minimizing mask."""
    n = len(values)
    end_zero, end_one = 0.0, math.inf
    parent_zero = np.full(n + 1, 255, dtype=np.uint8)
    parent_one = np.full(n + 1, 255, dtype=np.uint8)
    for i, value in enumerate(values, start=1):
        if end_zero <= end_one:
            new_zero, parent_zero[i] = end_zero, 0
        else:
            new_zero, parent_zero[i] = end_one, 1
        continue_run = end_one
        start_run = end_zero + a_ms
        if continue_run <= start_run:
            previous, parent_one[i] = continue_run, 1
        else:
            previous, parent_one[i] = start_run, 0
        new_one = c_ms_per_row - lam * float(value) + previous
        end_zero, end_one = new_zero, new_one

    state = 0 if end_zero <= end_one else 1
    mask = np.zeros(n, dtype=bool)
    for i in range(n, 0, -1):
        if state == 0:
            state = int(parent_zero[i])
        else:
            mask[i - 1] = True
            state = int(parent_one[i])
    rows, chunks = mask_stats(mask)
    importance = float(values[mask].sum())
    return {
        "mask": mask,
        "importance": importance,
        "rows": rows,
        "chunks": chunks,
        "aff_ms": a_ms * chunks + c_ms_per_row * rows,
        "lambda": lam,
    }


def lagrangian_candidates(
    values: np.ndarray,
    a_ms: float,
    c_ms_per_row: float,
    q: int,
    maximum_target: float,
) -> list[dict]:
    """Generate supported frontier points using q logarithmically spaced multipliers."""
    if q < 2:
        raise ValueError("the Lagrangian solver needs q >= 2")
    # Find a data-adaptive upper endpoint that reaches the largest requested
    # coverage. These O(N) probes are bounded by 64 and are reported separately
    # from the q-point grid in the output metadata.
    upper = max(a_ms, c_ms_per_row * len(values))
    probe_count = 0
    while probe_count < 64:
        probe = solve_lagrangian(values, upper, a_ms, c_ms_per_row)
        probe_count += 1
        if probe["importance"] >= maximum_target - 1e-14:
            break
        upper *= 2.0
    else:
        raise RuntimeError("failed to bracket the requested Lagrangian coverage")

    lower = max(upper * 1e-5, np.finfo(np.float64).tiny)
    multipliers = np.r_[0.0, np.geomspace(lower, upper, q - 1)]
    unique: dict[bytes, dict] = {}
    for lam in multipliers:
        solution = solve_lagrangian(values, float(lam), a_ms, c_ms_per_row)
        key = np.packbits(solution["mask"]).tobytes()
        previous = unique.get(key)
        if previous is None or solution["lambda"] < previous["lambda"]:
            unique[key] = solution

    # A feasible fallback is part of the proposed method. It also protects
    # against a coarse grid that stops just below the requested threshold.
    full_mask = np.ones(len(values), dtype=bool)
    full = {
        "mask": full_mask,
        "importance": float(values.sum()),
        "rows": len(values),
        "chunks": 1,
        "aff_ms": a_ms + c_ms_per_row * len(values),
        "lambda": math.inf,
    }
    unique.setdefault(np.packbits(full_mask).tobytes(), full)
    candidates = list(unique.values())
    for candidate in candidates:
        candidate["probe_count"] = probe_count
    return candidates


def select_lagrangian(candidates: list[dict], bound: float) -> dict:
    """Smallest-latency generated candidate meeting an importance bound."""
    feasible = [c for c in candidates if c["importance"] >= bound - 1e-14]
    return min(feasible, key=lambda c: (c["aff_ms"], -c["importance"]))


class QuantizedParetoOracle:
    """Approximate the coverage frontier in O(qN) using importance buckets.

    For each prefix, coverage bucket, and ending bit, the DP retains the
    lowest-cost representative (breaking cost ties toward higher actual
    importance).  Dominated representatives are removed within each ending
    state.  Actual, rather than quantized, importance is used for the final
    feasibility check, so every returned mask satisfies the requested bound.
    """

    def __init__(
        self, values: np.ndarray, a_ms: float, c_ms_per_row: float, q: int
    ) -> None:
        if q < 2:
            raise ValueError("the quantized Pareto solver needs q >= 2")
        self.values = np.asarray(values, dtype=np.float64)
        self.a_ms = a_ms
        self.c_ms_per_row = c_ms_per_row
        self.q = q
        self.total = float(self.values.sum())
        if self.total <= 0:
            raise ValueError("coverage importance must have positive total mass")

        n = len(self.values)
        inf = math.inf
        costs = np.full((2, q + 1), inf, dtype=np.float64)
        importance = np.full((2, q + 1), -np.inf, dtype=np.float64)
        costs[0, 0] = 0.0
        importance[0, 0] = 0.0

        self.parent_state = np.full((n + 1, 2, q + 1), 255, dtype=np.uint8)
        self.parent_bucket = np.full((n + 1, 2, q + 1), -1, dtype=np.int32)
        self.peak_states = 1

        def bucket_of(value: float) -> int:
            scaled = q * value / self.total
            return min(q, max(0, int(math.floor(scaled + 1e-12))))

        def offer(
            new_costs: np.ndarray,
            new_importance: np.ndarray,
            i: int,
            ending: int,
            bucket: int,
            candidate_cost: float,
            candidate_importance: float,
            previous_state: int,
            previous_bucket: int,
        ) -> None:
            old_cost = new_costs[ending, bucket]
            old_importance = new_importance[ending, bucket]
            if candidate_cost < old_cost - 1e-15 or (
                math.isclose(candidate_cost, old_cost, rel_tol=0.0, abs_tol=1e-15)
                and candidate_importance > old_importance
            ):
                new_costs[ending, bucket] = candidate_cost
                new_importance[ending, bucket] = candidate_importance
                self.parent_state[i, ending, bucket] = previous_state
                self.parent_bucket[i, ending, bucket] = previous_bucket

        for i, value in enumerate(self.values, start=1):
            new_costs = np.full((2, q + 1), inf, dtype=np.float64)
            new_importance = np.full((2, q + 1), -np.inf, dtype=np.float64)
            for previous_state in (0, 1):
                for previous_bucket in np.flatnonzero(
                    np.isfinite(costs[previous_state])
                ):
                    old_cost = float(costs[previous_state, previous_bucket])
                    old_importance = float(
                        importance[previous_state, previous_bucket]
                    )
                    offer(
                        new_costs,
                        new_importance,
                        i,
                        0,
                        bucket_of(old_importance),
                        old_cost,
                        old_importance,
                        previous_state,
                        int(previous_bucket),
                    )
                    selected_importance = old_importance + float(value)
                    selected_cost = old_cost + c_ms_per_row
                    if previous_state == 0:
                        selected_cost += a_ms
                    offer(
                        new_costs,
                        new_importance,
                        i,
                        1,
                        bucket_of(selected_importance),
                        selected_cost,
                        selected_importance,
                        previous_state,
                        int(previous_bucket),
                    )

            # With the ending bit fixed, a higher-importance state with no
            # larger cost dominates all future extensions of a lower state.
            for ending in (0, 1):
                best_higher_cost = inf
                for bucket in range(q, -1, -1):
                    candidate_cost = new_costs[ending, bucket]
                    if not math.isfinite(candidate_cost):
                        continue
                    if candidate_cost >= best_higher_cost - 1e-15:
                        new_costs[ending, bucket] = inf
                        new_importance[ending, bucket] = -np.inf
                    else:
                        best_higher_cost = candidate_cost

            costs, importance = new_costs, new_importance
            self.peak_states = max(self.peak_states, int(np.isfinite(costs).sum()))

        self.costs = costs
        self.importance = importance

    def solve(self, bound: float) -> dict:
        """Return the cheapest retained representative meeting an exact bound."""
        if bound <= 0 or bound > self.total + 1e-12:
            raise ValueError("coverage bound must lie in (0, total importance]")
        best_key = (math.inf, math.inf)
        best_state: int | None = None
        best_bucket: int | None = None
        for ending in (0, 1):
            for bucket in np.flatnonzero(np.isfinite(self.costs[ending])):
                actual_importance = float(self.importance[ending, bucket])
                if actual_importance < bound - 1e-14:
                    continue
                key = (float(self.costs[ending, bucket]), -actual_importance)
                if key < best_key:
                    best_key = key
                    best_state = ending
                    best_bucket = int(bucket)

        full_cost = self.a_ms + self.c_ms_per_row * len(self.values)
        if best_state is None or full_cost < best_key[0] - 1e-15:
            mask = np.ones(len(self.values), dtype=bool)
        else:
            mask = np.zeros(len(self.values), dtype=bool)
            state = best_state
            bucket = best_bucket
            for i in range(len(self.values), 0, -1):
                if state == 1:
                    mask[i - 1] = True
                previous_state = int(self.parent_state[i, state, bucket])
                previous_bucket = int(self.parent_bucket[i, state, bucket])
                if previous_state == 255 or previous_bucket < 0:
                    raise RuntimeError(
                        "quantized-Pareto backtracking reached an invalid state"
                    )
                state, bucket = previous_state, previous_bucket

        rows, chunks = mask_stats(mask)
        actual_importance = float(self.values[mask].sum())
        if actual_importance < bound - 1e-12:
            raise RuntimeError("quantized Pareto solver returned an infeasible mask")
        return {
            "mask": mask,
            "importance": actual_importance,
            "rows": rows,
            "chunks": chunks,
            "aff_ms": self.a_ms * chunks + self.c_ms_per_row * rows,
            "peak_states": self.peak_states,
        }


def exact_upper_r_table(
    values: np.ndarray,
    row_bound: int,
    row_size_kib: float,
    table: LatencyTable,
) -> np.ndarray:
    """Best single interval over all lengths 1..row_bound for lookup I/L."""
    prefix = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    best_ratio = -math.inf
    best_start = 0
    best_length = 1
    for length in range(1, row_bound + 1):
        sums = prefix[length:] - prefix[:-length]
        start = int(np.argmax(sums))
        ratio = float(sums[start] / table.read_ms(length * row_size_kib))
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start
            best_length = length
    mask = np.zeros(len(values), dtype=bool)
    mask[best_start : best_start + best_length] = True
    return mask


def random_importance(rng: np.random.Generator, n: int, distribution: str) -> np.ndarray:
    """Generate a positive importance vector and normalize total importance to one."""
    if distribution == "half-normal":
        values = np.abs(rng.standard_normal(n))
    elif distribution == "lognormal":
        values = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    elif distribution == "correlated":
        raw = np.abs(rng.standard_normal(n))
        # A short symmetric moving average supplies local structure without
        # changing the nonnegative-importance assumption.
        kernel = np.asarray([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
        kernel /= kernel.sum()
        values = np.convolve(raw, kernel, mode="same")
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    values = np.maximum(values, np.finfo(np.float64).tiny)
    return values / values.sum()


def ratio(importance: float, latency_ms: float) -> float:
    return importance / latency_ms


def summarize(records: list[dict], budgets: list[int], n: int) -> list[dict]:
    metrics = [
        "greedy_aff_ratio",
        "top_r_aff_ratio",
        "fixed_opt_aff_ratio",
        "fixed_aff_gain_pct",
        "top_r_aff_gain_pct",
        "greedy_table_ratio",
        "top_r_table_ratio",
        "fixed_opt_table_ratio",
        "fixed_lookup_gain_pct",
        "upper_opt_table_ratio",
        "upper_table_gain_pct",
        "greedy_fill_fraction",
        "top_r_fill_fraction",
        "upper_opt_fill_fraction",
        "greedy_importance",
        "top_r_importance",
        "fixed_opt_importance",
        "greedy_aff_ms",
        "top_r_aff_ms",
        "fixed_opt_aff_ms",
        "greedy_table_ms",
        "top_r_table_ms",
        "fixed_opt_table_ms",
        "greedy_chunks",
        "top_r_chunks",
        "fixed_opt_chunks",
        "greedy_coverage_gap_pct",
        "top_r_coverage_gap_pct",
        "fixed_opt_coverage_gap_pct",
        "dinkelbach_iterations",
    ]
    output = []
    for budget in budgets:
        rows = [r for r in records if r["budget_rows"] == budget]
        item = {
            "budget_rows": budget,
            "budget_fraction": budget / n,
            "trials": len(rows),
        }
        for metric in metrics:
            x = np.asarray([r[metric] for r in rows], dtype=np.float64)
            item[metric] = {
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "p05": float(np.quantile(x, 0.05)),
                "p95": float(np.quantile(x, 0.95)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
        output.append(item)
    return output


def summarize_coverage(records: list[dict], targets: list[float]) -> list[dict]:
    metrics = [
        "coverage_achieved",
        "coverage_overshoot",
        "coverage_rows",
        "coverage_row_fraction",
        "coverage_chunks",
        "coverage_aff_ms",
        "coverage_table_ms",
        "lagrangian_achieved",
        "lagrangian_overshoot",
        "lagrangian_rows",
        "lagrangian_row_fraction",
        "lagrangian_chunks",
        "lagrangian_aff_ms",
        "lagrangian_table_ms",
        "lagrangian_target_gap_pct",
        "lagrangian_achieved_gap_pct",
        "lagrangian_candidate_count",
        "lagrangian_probe_count",
        "pareto_achieved",
        "pareto_overshoot",
        "pareto_rows",
        "pareto_row_fraction",
        "pareto_chunks",
        "pareto_aff_ms",
        "pareto_table_ms",
        "pareto_target_gap_pct",
        "pareto_achieved_gap_pct",
        "pareto_peak_states",
    ]
    output = []
    for target in targets:
        rows = [r for r in records if r["coverage_target"] == target]
        item = {"coverage_target": target, "trials": len(rows)}
        for metric in metrics:
            x = np.asarray([r[metric] for r in rows], dtype=np.float64)
            item[metric] = {
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "p05": float(np.quantile(x, 0.05)),
                "p95": float(np.quantile(x, 0.95)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


PLOT_COLORS = {
    "greedy": "#E76F51",
    "fixed": "#277DA1",
    "top_r": "#B565A7",
    "coverage": "#2A9D8F",
    "lagrangian": "#E9A23B",
    "pareto": "#3A506B",
    "lookup": "#6C5B7B",
}


def configure_plot_style(plt) -> None:
    """A compact, consistent style for the experiment's new figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFC",
            "axes.edgecolor": "#4A5568",
            "axes.labelcolor": "#263238",
            "axes.titleweight": "semibold",
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.color": "#37474F",
            "ytick.color": "#37474F",
            "font.size": 10,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "savefig.facecolor": "white",
        }
    )


def polish_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#CBD5E1", linewidth=0.75, alpha=0.55)
    ax.set_axisbelow(True)


def plot_results(records: list[dict], summary: list[dict], path: Path) -> None:
    # Keep Matplotlib's generated cache outside both the repository and the
    # user's (possibly read-only) home directory.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    configure_plot_style(plt)
    x = np.asarray([100 * s["budget_fraction"] for s in summary])

    def band(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.asarray([s[metric][key] for s in summary])
            for key in ("mean", "p05", "p95")
        )

    colors = {
        "greedy": "#D55E00",
        "fixed": "#0072B2",
        "upper": "#009E73",
        "lookup": "#6A3D9A",
        "top_r": "#CC79A7",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    for metric, label, color in (
        ("greedy_aff_ratio", "Paper greedy", colors["greedy"]),
        ("fixed_opt_aff_ratio", "Exact fixed-R DP", colors["fixed"]),
    ):
        mean, lo, hi = band(metric)
        ax.plot(x, mean, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.16, color=color)
    ax.set_title("A. Fixed R: affine objective")
    ax.set_ylabel("Importance / latency (1/ms)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for metric, label, color, style in (
        ("fixed_aff_gain_pct", "On affine objective", colors["fixed"], "-"),
        ("fixed_lookup_gain_pct", "Same masks, lookup re-evaluation", colors["lookup"], "--"),
    ):
        mean, lo, hi = band(metric)
        ax.plot(x, mean, marker="o", linewidth=2, linestyle=style, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.13, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("B. Fixed R: gain over paper greedy")
    ax.set_ylabel("Relative I/L gain (%)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for metric, label, color in (
        ("greedy_table_ratio", "Paper greedy", colors["greedy"]),
        ("upper_opt_table_ratio", "Exact upper-bound interval", colors["upper"]),
    ):
        mean, lo, hi = band(metric)
        ax.plot(x, mean, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.16, color=color)
    ax.set_title("C. 1 <= rows <= R: lookup-table objective")
    ax.set_xlabel("Row budget R / N (%)")
    ax.set_ylabel("Importance / latency (1/ms)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for metric, label, color in (
        ("greedy_fill_fraction", "Paper greedy", colors["greedy"]),
        ("upper_opt_fill_fraction", "Exact upper-bound interval", colors["upper"]),
    ):
        mean, lo, hi = band(metric)
        ax.plot(x, 100 * mean, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.16, color=color)
    ax.set_title("D. Fraction of available row budget used")
    ax.set_xlabel("Row budget R / N (%)")
    ax.set_ylabel("Selected rows / R (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.22)
        ax.set_xlim(x.min() - 2, x.max() + 2)
    fig.suptitle(
        "Random nonnegative importance: paper heuristic vs exact bounded-R solutions\n"
        "Lines are means; bands are 5th-95th percentiles across paired inputs",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_latency_tradeoff(
    summary: list[dict], coverage_summary: list[dict], path: Path
) -> None:
    """Plot importance directly against latency for fixed-R and coverage methods."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    configure_plot_style(plt)
    colors = PLOT_COLORS
    methods = (
        ("greedy", "Paper greedy", colors["greedy"]),
        ("top_r", "Top-R", colors["top_r"]),
        ("fixed_opt", "Exact fixed-R DP", colors["fixed"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for ax, latency_kind, title in (
        (axes[0], "aff", "A. Fitted affine latency"),
        (axes[1], "table", "B. Lookup-table latency"),
    ):
        for prefix, label, color in methods:
            latency = np.asarray([s[f"{prefix}_{latency_kind}_ms"]["mean"] for s in summary])
            importance = np.asarray([s[f"{prefix}_importance"]["mean"] for s in summary])
            ax.plot(latency, importance, marker="o", linewidth=2, label=label, color=color)
            label_offset = {
                "greedy": (3, -11),
                "top_r": (3, 3),
                "fixed_opt": (-34, 3),
            }[prefix]
            for item, x_value, y_value in zip(summary, latency, importance):
                ax.annotate(
                    f"{100 * item['budget_fraction']:.1f}%",
                    (x_value, y_value),
                    xytext=label_offset,
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    alpha=0.85,
                )
        coverage_latency = np.asarray(
            [s[f"coverage_{latency_kind}_ms"]["mean"] for s in coverage_summary]
        )
        coverage_importance = np.asarray(
            [s["coverage_achieved"]["mean"] for s in coverage_summary]
        )
        ax.plot(
            coverage_latency,
            coverage_importance,
            marker="s",
            linewidth=2.3,
            label="Exact coverage DP",
            color=colors["coverage"],
        )
        pareto_latency = np.asarray(
            [s[f"pareto_{latency_kind}_ms"]["mean"] for s in coverage_summary]
        )
        pareto_importance = np.asarray(
            [s["pareto_achieved"]["mean"] for s in coverage_summary]
        )
        ax.plot(
            pareto_latency,
            pareto_importance,
            marker="P",
            markersize=5.5,
            linewidth=2.0,
            linestyle="-.",
            label="Quantized Pareto O(qN)",
            color=colors["pareto"],
        )
        lagrangian_latency = np.asarray(
            [s[f"lagrangian_{latency_kind}_ms"]["mean"] for s in coverage_summary]
        )
        lagrangian_importance = np.asarray(
            [s["lagrangian_achieved"]["mean"] for s in coverage_summary]
        )
        ax.plot(
            lagrangian_latency,
            lagrangian_importance,
            marker="D",
            markersize=5,
            linewidth=1.9,
            linestyle="--",
            label="Lagrangian O(qN)",
            color=colors["lagrangian"],
        )
        for item, x_value, y_value in zip(
            coverage_summary, coverage_latency, coverage_importance
        ):
            target = item["coverage_target"]
            if target in {0.1, 0.5, 0.8, 0.9, 0.95, 0.99}:
                ax.annotate(
                    rf"$\alpha$={target:g}",
                    (x_value, y_value),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=7,
                    color=colors["coverage"],
                )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Latency (ms, log scale)")
        ax.set_ylabel("Retained importance")
        polish_axis(ax)
        ax.legend(frameon=False)
    fig.suptitle(
        "Importance-latency trade-off: fixed-R and coverage policies\n"
        "Fixed-R labels show R / N; coverage labels show the requested lower bound",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_coverage(coverage_summary: list[dict], path: Path) -> None:
    """Standalone diagnostics for exact and approximate coverage solvers."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    configure_plot_style(plt)
    x = np.asarray([100 * s["coverage_target"] for s in coverage_summary])

    def band(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.asarray([s[metric][key] for s in coverage_summary])
            for key in ("mean", "p05", "p95")
        )

    green = PLOT_COLORS["coverage"]
    purple = PLOT_COLORS["lookup"]
    blue = PLOT_COLORS["fixed"]
    orange = PLOT_COLORS["lagrangian"]
    navy = PLOT_COLORS["pareto"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    for metric, label, color, style in (
        ("coverage_aff_ms", "Exact affine optimum", green, "-"),
        ("pareto_aff_ms", "Quantized Pareto O(qN)", navy, "-."),
        ("lagrangian_aff_ms", "Lagrangian O(qN)", orange, "--"),
        ("coverage_table_ms", "Exact masks, lookup re-evaluation", purple, ":"),
    ):
        mean, lo, hi = band(metric)
        ax.plot(x, mean, marker="o", linewidth=2, linestyle=style, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.14, color=color)
    ax.set_title("A. Minimum latency under an importance lower bound")
    ax.set_ylabel("Latency (ms)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    mean, lo, hi = band("coverage_achieved")
    ax.plot(x, 100 * mean, marker="o", color=green, label="Exact achieved")
    ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.16, color=green)
    mean, lo, hi = band("lagrangian_achieved")
    ax.plot(x, 100 * mean, marker="D", linestyle="--", color=orange, label="Lagrangian achieved")
    ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.11, color=orange)
    mean, lo, hi = band("pareto_achieved")
    ax.plot(x, 100 * mean, marker="P", linestyle="-.", color=navy, label="Pareto achieved")
    ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.10, color=navy)
    ax.plot(x, x, color="black", linestyle=":", label="Requested lower bound")
    ax.set_title("B. Requested versus achieved importance")
    ax.set_ylabel("Achieved importance (%)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    mean, lo, hi = band("coverage_row_fraction")
    ax.plot(x, 100 * mean, marker="o", color=blue, label="Exact DP")
    ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.16, color=blue)
    mean, lo, hi = band("lagrangian_row_fraction")
    ax.plot(x, 100 * mean, marker="D", linestyle="--", color=orange, label="Lagrangian")
    ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.11, color=orange)
    mean, lo, hi = band("pareto_row_fraction")
    ax.plot(x, 100 * mean, marker="P", linestyle="-.", color=navy, label="Quantized Pareto")
    ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.10, color=navy)
    ax.set_title("C. Rows selected by each coverage solver")
    ax.set_xlabel("Required importance $\\alpha$ (%)")
    ax.set_ylabel("Selected rows / N (%)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for metric, label, color, style in (
        ("pareto_target_gap_pct", "At requested coverage", navy, "-"),
        ("pareto_achieved_gap_pct", "At achieved coverage", purple, ":"),
    ):
        mean, lo, hi = band(metric)
        ax.plot(x, mean, marker="o", linestyle=style, color=color, label=label)
        ax.fill_between(x, lo, hi, alpha=0.13, color=color)
    ax.axhline(0, color="#455A64", linewidth=0.9)
    ax.set_title("D. Quantized Pareto latency gap to exact DP")
    ax.set_xlabel("Required importance $\\alpha$ (%)")
    ax.set_ylabel("Extra affine latency (%)")
    ax.legend(frameon=False)

    for ax in axes.flat:
        polish_axis(ax)
        ax.set_xlim(x.min() - 2, 101)
    fig.suptitle(
        "Coverage problem: minimize latency subject to retained importance\n"
        "Lines are means; bands are 5th-95th percentiles across paired inputs",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_coverage_gap(summary: list[dict], path: Path) -> None:
    """Compare each fixed-R method with coverage optimum at achieved importance."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    x = np.asarray([100 * s["budget_fraction"] for s in summary])

    def band(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.asarray([s[metric][key] for s in summary])
            for key in ("mean", "p05", "p95")
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    ax = axes[0]
    for metric, label, color in (
        ("greedy_coverage_gap_pct", "Paper greedy", PLOT_COLORS["greedy"]),
        ("fixed_opt_coverage_gap_pct", "Exact fixed-R DP", PLOT_COLORS["fixed"]),
    ):
        mean, lo, hi = band(metric)
        ax.plot(x, mean, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.15, color=color)
    ax.set_title("A. Chunk-aware fixed-R methods")
    ax.set_ylabel("Extra latency over coverage optimum (%)")
    ax.legend(frameon=False)

    ax = axes[1]
    mean, lo, hi = band("top_r_coverage_gap_pct")
    ax.plot(x, mean, marker="o", label="Top-R", color=PLOT_COLORS["top_r"])
    ax.fill_between(x, lo, hi, alpha=0.15, color=PLOT_COLORS["top_r"])
    ax.set_title("B. Latency-blind Top-R")
    ax.set_ylabel("Extra latency over coverage optimum (%)")
    ax.legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("Fixed row budget R / N (%)")
        polish_axis(ax)
        ax.set_xlim(x.min() - 2, x.max() + 2)
        ax.set_ylim(bottom=0)
    fig.suptitle(
        "Latency gap to exact coverage at each method's achieved importance\n"
        "Lines are means; bands are 5th-95th percentiles across paired inputs",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_r_importance(summary: list[dict], path: Path) -> None:
    """Retained importance as a function of the fixed row budget R."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    configure_plot_style(plt)
    x = np.asarray([100 * s["budget_fraction"] for s in summary])
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    for prefix, label, color, marker in (
        ("greedy", "Paper greedy", PLOT_COLORS["greedy"], "o"),
        ("top_r", "Top-R", PLOT_COLORS["top_r"], "s"),
        ("fixed_opt", "Exact fixed-R DP", PLOT_COLORS["fixed"], "D"),
    ):
        metric = f"{prefix}_importance"
        mean = 100 * np.asarray([s[metric]["mean"] for s in summary])
        lo = 100 * np.asarray([s[metric]["p05"] for s in summary])
        hi = 100 * np.asarray([s[metric]["p95"] for s in summary])
        ax.plot(x, mean, marker=marker, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.12, color=color)
    ax.set_title("Retained importance under a fixed row budget")
    ax.set_xlabel("Fixed row budget R / N (%)")
    ax.set_ylabel("Retained importance (%)")
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(0, 102)
    polish_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_r_latency(summary: list[dict], path: Path) -> None:
    """Affine and lookup latency as functions of the fixed row budget R."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    configure_plot_style(plt)
    x = np.asarray([100 * s["budget_fraction"] for s in summary])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    for ax, latency_kind, title in (
        (axes[0], "aff", "A. Fitted affine latency"),
        (axes[1], "table", "B. Lookup-table latency"),
    ):
        for prefix, label, color, marker in (
            ("greedy", "Paper greedy", PLOT_COLORS["greedy"], "o"),
            ("top_r", "Top-R", PLOT_COLORS["top_r"], "s"),
            ("fixed_opt", "Exact fixed-R DP", PLOT_COLORS["fixed"], "D"),
        ):
            metric = f"{prefix}_{latency_kind}_ms"
            mean = np.asarray([s[metric]["mean"] for s in summary])
            lo = np.asarray([s[metric]["p05"] for s in summary])
            hi = np.asarray([s[metric]["p95"] for s in summary])
            ax.plot(x, mean, marker=marker, label=label, color=color)
            ax.fill_between(x, lo, hi, alpha=0.12, color=color)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Fixed row budget R / N (%)")
        ax.set_ylabel("Latency (ms, log scale)")
        ax.set_xlim(x.min() - 2, x.max() + 2)
        polish_axis(ax)
        ax.legend(frameon=False)
    fig.suptitle("Read latency under a fixed row budget", fontsize=13)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def self_check() -> None:
    """Compare all three exact solvers with exhaustive masks on small instances."""
    from itertools import product

    rng = np.random.default_rng(7)
    table = LatencyTable({i: 0.7 + 0.13 * i + 0.02 * (i % 3) for i in range(1, 9)})
    a_ms, c_ms, _ = affine_fit(table, 1.0)
    for n in range(2, 9):
        values = random_importance(rng, n, "half-normal")
        for r in range(1, n + 1):
            fixed_mask, _, _ = exact_fixed_r_affine(values, r, a_ms, c_ms)
            got = ratio(float(values[fixed_mask].sum()), affine_latency(fixed_mask, a_ms, c_ms))
            brute_fixed = -math.inf
            brute_upper = -math.inf
            for bits in product((False, True), repeat=n):
                mask = np.asarray(bits, dtype=bool)
                selected, _ = mask_stats(mask)
                if selected == r:
                    brute_fixed = max(
                        brute_fixed,
                        ratio(float(values[mask].sum()), affine_latency(mask, a_ms, c_ms)),
                    )
                if 1 <= selected <= r:
                    latency = table.mask_elat_ms(torch.from_numpy(mask), 1.0)
                    brute_upper = max(brute_upper, ratio(float(values[mask].sum()), latency))
            assert math.isclose(got, brute_fixed, rel_tol=1e-10, abs_tol=1e-12)

            upper_mask = exact_upper_r_table(values, r, 1.0, table)
            upper_got = ratio(
                float(values[upper_mask].sum()),
                table.mask_elat_ms(torch.from_numpy(upper_mask), 1.0),
            )
            assert math.isclose(upper_got, brute_upper, rel_tol=1e-10, abs_tol=1e-12)

        coverage_oracle = ExactCoverageOracle(values, a_ms, c_ms)
        pareto_oracle = QuantizedParetoOracle(values, a_ms, c_ms, q=256)
        masks = []
        for bits in product((False, True), repeat=n):
            mask = np.asarray(bits, dtype=bool)
            if mask.any():
                masks.append(
                    (
                        float(values[mask].sum()),
                        affine_latency(mask, a_ms, c_ms),
                    )
                )
        for target in (0.2, 0.5, 0.8, 0.95, 1.0):
            got = coverage_oracle.solve(target)
            best_cost = min(cost for importance, cost in masks if importance >= target - 1e-14)
            best_importance = max(
                importance
                for importance, cost in masks
                if math.isclose(cost, best_cost, rel_tol=1e-12, abs_tol=1e-15)
                and importance >= target - 1e-14
            )
            assert math.isclose(got["aff_ms"], best_cost, rel_tol=1e-10, abs_tol=1e-12)
            assert math.isclose(
                got["importance"], best_importance, rel_tol=1e-10, abs_tol=1e-12
            )
            pareto = pareto_oracle.solve(target)
            assert pareto["importance"] >= target - 1e-12
            assert pareto["aff_ms"] >= best_cost - 1e-12
        for lam in (0.0, 0.1, 1.0, 10.0):
            got = solve_lagrangian(values, lam, a_ms, c_ms)
            got_objective = got["aff_ms"] - lam * got["importance"]
            brute_objective = min(cost - lam * importance for importance, cost in masks)
            # Include the empty mask, whose relaxed objective is zero.
            brute_objective = min(0.0, brute_objective)
            assert math.isclose(
                got_objective, brute_objective, rel_tol=1e-10, abs_tol=1e-12
            )

    # A deterministic multi-chunk case guards against accidentally testing
    # only the single-interval regime seen in the default dense random data.
    agx_table = LatencyTable.load("orin-agx")
    agx_a_ms, agx_c_ms, _ = affine_fit(agx_table, 1.0)
    separated = np.zeros(256, dtype=np.float64)
    separated[[0, 127, 255]] = [0.34, 0.33, 0.33]
    exact_separated = ExactCoverageOracle(separated, agx_a_ms, agx_c_ms)
    pareto_separated = QuantizedParetoOracle(
        separated, agx_a_ms, agx_c_ms, q=256
    )
    for target, expected_chunks in ((0.5, 2), (0.9, 3)):
        exact = exact_separated.solve(target)
        pareto = pareto_separated.solve(target)
        assert exact["chunks"] == expected_chunks
        assert pareto["importance"] >= target - 1e-12
        assert math.isclose(
            pareto["aff_ms"], exact["aff_ms"], rel_tol=1e-10, abs_tol=1e-12
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--row-size-kib", type=float, default=1.0)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--distribution", choices=("half-normal", "lognormal", "correlated"), default="half-normal")
    parser.add_argument("--budget-rows", type=int, nargs="+", default=[32, 64, 96, 128, 160, 192, 224])
    parser.add_argument(
        "--coverage-targets",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99],
        help="importance lower bounds for the exact coverage problem",
    )
    parser.add_argument(
        "--lagrangian-q",
        type=int,
        default=256,
        help="number of multiplier evaluations in the O(qN) coverage approximation",
    )
    parser.add_argument(
        "--pareto-q",
        type=int,
        default=256,
        help="number of coverage buckets in the O(qN) quantized Pareto DP",
    )
    parser.add_argument("--start-kib", type=float, default=8.0)
    parser.add_argument("--jump-cap-kib", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.n < 1:
        raise SystemExit("--trials and --n must be positive")
    budgets = sorted(set(args.budget_rows))
    if not budgets or budgets[0] < 1 or budgets[-1] > args.n:
        raise SystemExit("every --budget-rows value must be in [1, n]")
    coverage_targets = sorted(set(args.coverage_targets))
    if not coverage_targets or coverage_targets[0] <= 0 or coverage_targets[-1] > 1:
        raise SystemExit("every --coverage-targets value must be in (0, 1]")
    if args.lagrangian_q < 2:
        raise SystemExit("--lagrangian-q must be at least 2")
    if args.pareto_q < 2:
        raise SystemExit("--pareto-q must be at least 2")

    if not args.skip_self_check:
        self_check()

    table = LatencyTable.load(args.profile)
    a_ms, c_ms_per_row, fit_r2 = affine_fit(table, args.row_size_kib)
    params = ChunkParams(start_kb=args.start_kib, jump_cap_kb=args.jump_cap_kib)
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    coverage_records: list[dict] = []

    for trial in range(args.trials):
        values = random_importance(rng, args.n, args.distribution)
        values_t = torch.from_numpy(values.astype(np.float32))
        coverage_oracle = ExactCoverageOracle(values, a_ms, c_ms_per_row)
        pareto_oracle = QuantizedParetoOracle(
            values, a_ms, c_ms_per_row, args.pareto_q
        )
        lagrangian_pool = lagrangian_candidates(
            values,
            a_ms,
            c_ms_per_row,
            args.lagrangian_q,
            coverage_targets[-1] * float(values.sum()),
        )
        for target in coverage_targets:
            solution = coverage_oracle.solve(target * float(values.sum()))
            coverage_table_ms = table.mask_elat_ms(
                torch.from_numpy(solution["mask"]), args.row_size_kib
            )
            lagrangian = select_lagrangian(
                lagrangian_pool, target * float(values.sum())
            )
            lagrangian_table_ms = table.mask_elat_ms(
                torch.from_numpy(lagrangian["mask"]), args.row_size_kib
            )
            achieved_optimum = coverage_oracle.solve(lagrangian["importance"])[
                "aff_ms"
            ]
            pareto = pareto_oracle.solve(target * float(values.sum()))
            pareto_table_ms = table.mask_elat_ms(
                torch.from_numpy(pareto["mask"]), args.row_size_kib
            )
            pareto_achieved_optimum = coverage_oracle.solve(pareto["importance"])[
                "aff_ms"
            ]
            if pareto["aff_ms"] < solution["aff_ms"] - 1e-12:
                raise RuntimeError("quantized Pareto result beat the exact coverage DP")
            coverage_records.append(
                {
                    "trial": trial,
                    "coverage_target": target,
                    "coverage_achieved": solution["importance"] / float(values.sum()),
                    "coverage_overshoot": solution["importance"] / float(values.sum()) - target,
                    "coverage_rows": solution["rows"],
                    "coverage_row_fraction": solution["rows"] / args.n,
                    "coverage_chunks": solution["chunks"],
                    "coverage_aff_ms": solution["aff_ms"],
                    "coverage_table_ms": coverage_table_ms,
                    "lagrangian_achieved": lagrangian["importance"] / float(values.sum()),
                    "lagrangian_overshoot": lagrangian["importance"] / float(values.sum()) - target,
                    "lagrangian_rows": lagrangian["rows"],
                    "lagrangian_row_fraction": lagrangian["rows"] / args.n,
                    "lagrangian_chunks": lagrangian["chunks"],
                    "lagrangian_aff_ms": lagrangian["aff_ms"],
                    "lagrangian_table_ms": lagrangian_table_ms,
                    "lagrangian_target_gap_pct": 100
                    * (lagrangian["aff_ms"] / solution["aff_ms"] - 1),
                    "lagrangian_achieved_gap_pct": 100
                    * (lagrangian["aff_ms"] / achieved_optimum - 1),
                    "lagrangian_candidate_count": len(lagrangian_pool),
                    "lagrangian_probe_count": lagrangian["probe_count"],
                    "pareto_achieved": pareto["importance"]
                    / float(values.sum()),
                    "pareto_overshoot": pareto["importance"]
                    / float(values.sum())
                    - target,
                    "pareto_rows": pareto["rows"],
                    "pareto_row_fraction": pareto["rows"] / args.n,
                    "pareto_chunks": pareto["chunks"],
                    "pareto_aff_ms": pareto["aff_ms"],
                    "pareto_table_ms": pareto_table_ms,
                    "pareto_target_gap_pct": 100
                    * (pareto["aff_ms"] / solution["aff_ms"] - 1),
                    "pareto_achieved_gap_pct": 100
                    * (pareto["aff_ms"] / pareto_achieved_optimum - 1),
                    "pareto_peak_states": pareto["peak_states"],
                }
            )
        for budget in budgets:
            greedy = select_chunks(
                values_t,
                budget,
                args.row_size_kib,
                table,
                params=params,
                impl="torch",
            )
            top_r = select_topk(values_t, budget, args.row_size_kib, table)
            greedy_mask = greedy.mask.cpu().numpy()
            top_r_mask = top_r.mask.cpu().numpy()
            greedy_rows, greedy_chunks = mask_stats(greedy_mask)
            top_r_rows, top_r_chunks = mask_stats(top_r_mask)
            if greedy_rows == 0:
                raise RuntimeError(
                    f"paper greedy selected no rows at R={budget}; choose compatible chunk parameters"
                )

            # Match the fixed-R oracle to the heuristic's realized cardinality.
            # With the defaults both equal the requested budget exactly.
            fixed_mask, iterations, residual = exact_fixed_r_affine(
                values, greedy_rows, a_ms, c_ms_per_row
            )
            fixed_rows, fixed_chunks = mask_stats(fixed_mask)
            upper_mask = exact_upper_r_table(values, budget, args.row_size_kib, table)
            upper_rows, upper_chunks = mask_stats(upper_mask)

            greedy_importance = float(values[greedy_mask].sum())
            top_r_importance = float(values[top_r_mask].sum())
            fixed_importance = float(values[fixed_mask].sum())
            upper_importance = float(values[upper_mask].sum())
            if top_r_rows != budget:
                raise RuntimeError("Top-R did not select exactly the requested row count")
            if top_r_importance + 1e-10 < max(greedy_importance, fixed_importance):
                raise RuntimeError("Top-R did not maximize importance at fixed cardinality")

            greedy_aff_ms = affine_latency(greedy_mask, a_ms, c_ms_per_row)
            top_r_aff_ms = affine_latency(top_r_mask, a_ms, c_ms_per_row)
            fixed_aff_ms = affine_latency(fixed_mask, a_ms, c_ms_per_row)
            greedy_table_ms = table.mask_elat_ms(torch.from_numpy(greedy_mask), args.row_size_kib)
            top_r_table_ms = table.mask_elat_ms(torch.from_numpy(top_r_mask), args.row_size_kib)
            fixed_table_ms = table.mask_elat_ms(torch.from_numpy(fixed_mask), args.row_size_kib)
            upper_table_ms = table.mask_elat_ms(torch.from_numpy(upper_mask), args.row_size_kib)

            greedy_aff_ratio = ratio(greedy_importance, greedy_aff_ms)
            top_r_aff_ratio = ratio(top_r_importance, top_r_aff_ms)
            fixed_aff_ratio = ratio(fixed_importance, fixed_aff_ms)
            greedy_table_ratio = ratio(greedy_importance, greedy_table_ms)
            top_r_table_ratio = ratio(top_r_importance, top_r_table_ms)
            fixed_table_ratio = ratio(fixed_importance, fixed_table_ms)
            upper_table_ratio = ratio(upper_importance, upper_table_ms)
            if fixed_aff_ratio + 1e-10 < greedy_aff_ratio:
                raise RuntimeError("exact fixed-R result is worse than the feasible greedy mask")
            if fixed_aff_ratio + 1e-10 < top_r_aff_ratio:
                raise RuntimeError("exact fixed-R result is worse than the feasible Top-R mask")
            if upper_table_ratio + 1e-10 < greedy_table_ratio:
                raise RuntimeError("exact upper-bound result is worse than the feasible greedy mask")
            if upper_table_ratio + 1e-10 < top_r_table_ratio:
                raise RuntimeError("exact upper-bound result is worse than the feasible Top-R mask")

            greedy_coverage_ms = coverage_oracle.solve(greedy_importance)["aff_ms"]
            top_r_coverage_ms = coverage_oracle.solve(top_r_importance)["aff_ms"]
            fixed_coverage_ms = coverage_oracle.solve(fixed_importance)["aff_ms"]

            records.append(
                {
                    "trial": trial,
                    "budget_rows": budget,
                    "budget_fraction": budget / args.n,
                    "greedy_rows": greedy_rows,
                    "greedy_chunks": greedy_chunks,
                    "greedy_importance": greedy_importance,
                    "greedy_aff_ms": greedy_aff_ms,
                    "greedy_aff_ratio": greedy_aff_ratio,
                    "greedy_table_ms": greedy_table_ms,
                    "greedy_table_ratio": greedy_table_ratio,
                    "top_r_rows": top_r_rows,
                    "top_r_chunks": top_r_chunks,
                    "top_r_importance": top_r_importance,
                    "top_r_aff_ms": top_r_aff_ms,
                    "top_r_aff_ratio": top_r_aff_ratio,
                    "top_r_table_ms": top_r_table_ms,
                    "top_r_table_ratio": top_r_table_ratio,
                    "top_r_aff_gain_pct": 100 * (top_r_aff_ratio / greedy_aff_ratio - 1),
                    "top_r_coverage_gap_pct": 100 * (top_r_aff_ms / top_r_coverage_ms - 1),
                    "fixed_opt_rows": fixed_rows,
                    "fixed_opt_chunks": fixed_chunks,
                    "fixed_opt_importance": fixed_importance,
                    "fixed_opt_aff_ms": fixed_aff_ms,
                    "fixed_opt_aff_ratio": fixed_aff_ratio,
                    "fixed_opt_table_ms": fixed_table_ms,
                    "fixed_opt_table_ratio": fixed_table_ratio,
                    "fixed_aff_gain_pct": 100 * (fixed_aff_ratio / greedy_aff_ratio - 1),
                    "fixed_lookup_gain_pct": 100 * (fixed_table_ratio / greedy_table_ratio - 1),
                    "fixed_opt_coverage_gap_pct": 100 * (fixed_aff_ms / fixed_coverage_ms - 1),
                    "upper_opt_rows": upper_rows,
                    "upper_opt_chunks": upper_chunks,
                    "upper_opt_importance": upper_importance,
                    "upper_opt_table_ms": upper_table_ms,
                    "upper_opt_table_ratio": upper_table_ratio,
                    "upper_table_gain_pct": 100 * (upper_table_ratio / greedy_table_ratio - 1),
                    "greedy_fill_fraction": greedy_rows / budget,
                    "greedy_coverage_gap_pct": 100 * (greedy_aff_ms / greedy_coverage_ms - 1),
                    "top_r_fill_fraction": top_r_rows / budget,
                    "upper_opt_fill_fraction": upper_rows / budget,
                    "dinkelbach_iterations": iterations,
                    "dinkelbach_residual": residual,
                }
            )
        if (trial + 1) % max(1, args.trials // 10) == 0:
            print(f"completed {trial + 1}/{args.trials} random inputs", flush=True)

    summary = summarize(records, budgets, args.n)
    coverage_summary = summarize_coverage(coverage_records, coverage_targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "trials.csv"
    coverage_raw_path = args.output_dir / "coverage_trials.csv"
    summary_path = args.output_dir / "summary.json"
    figure_path = args.output_dir / "comparison.png"
    latency_importance_path = args.output_dir / "latency_importance.png"
    coverage_figure_path = args.output_dir / "coverage.png"
    coverage_gap_path = args.output_dir / "coverage_gap.png"
    r_importance_path = args.output_dir / "r_importance.png"
    r_latency_path = args.output_dir / "r_latency.png"
    write_csv(raw_path, records)
    write_csv(coverage_raw_path, coverage_records)
    metadata = {
        "format": "random-r-bound-comparison-v2",
        "seed": args.seed,
        "trials": args.trials,
        "n": args.n,
        "distribution": args.distribution,
        "importance_normalization": "sum=1",
        "row_size_kib": args.row_size_kib,
        "profile": args.profile,
        "profile_device": table.meta.get("device"),
        "budgets": budgets,
        "coverage_targets": coverage_targets,
        "lagrangian": {
            "grid_evaluations_q": args.lagrangian_q,
            "grid": "geometric from data-adaptive lower/upper multipliers",
            "fallback": "full-selection mask",
        },
        "quantized_pareto": {
            "coverage_buckets_q": args.pareto_q,
            "representative": "minimum cost per coverage bucket and ending bit",
            "pruning": "exact dominance within each ending bit",
            "feasibility": "checked using unquantized importance",
            "fallback": "full-selection mask",
        },
        "greedy_chunk_params": {
            "start_kib": args.start_kib,
            "jump_cap_kib": args.jump_cap_kib,
            "end_kib": table.max_kb + 1,
            "step_kib": args.start_kib,
        },
        "affine_fit": {
            "a_ms_per_chunk": a_ms,
            "c_ms_per_row": c_ms_per_row,
            "r_squared": fit_r2,
        },
        "summary": summary,
        "coverage_summary": coverage_summary,
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(records, summary, figure_path)
    plot_latency_tradeoff(summary, coverage_summary, latency_importance_path)
    plot_coverage(coverage_summary, coverage_figure_path)
    plot_coverage_gap(summary, coverage_gap_path)
    plot_r_importance(summary, r_importance_path)
    plot_r_latency(summary, r_latency_path)

    print(f"wrote {raw_path}")
    print(f"wrote {coverage_raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path} and {figure_path.with_suffix('.pdf')}")
    print(
        f"wrote {latency_importance_path} and "
        f"{latency_importance_path.with_suffix('.pdf')}"
    )
    print(
        f"wrote {coverage_figure_path} and "
        f"{coverage_figure_path.with_suffix('.pdf')}"
    )
    print(
        f"wrote {coverage_gap_path} and "
        f"{coverage_gap_path.with_suffix('.pdf')}"
    )
    print(f"wrote {r_importance_path} and {r_importance_path.with_suffix('.pdf')}")
    print(f"wrote {r_latency_path} and {r_latency_path.with_suffix('.pdf')}")
    print("\nmean fixed-R affine gain over greedy:")
    for item in summary:
        gain = item["fixed_aff_gain_pct"]
        print(
            f"  R/N={item['budget_fraction']:.3f}: {gain['mean']:.2f}% "
            f"(5th-95th: {gain['p05']:.2f}% to {gain['p95']:.2f}%)"
        )


if __name__ == "__main__":
    main()
