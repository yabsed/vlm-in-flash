#!/usr/bin/env python3
"""Experiment 43: shifted/adaptive Cell-1 with global quality allocation."""

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
    "experiment_39_for_43",
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

METHODS = ("paper", "cell1_x2", "shifted8_x2", "adaptive8_x2")
METHOD_LABELS = {
    "paper": "Paper |x|",
    "cell1_x2": "Cell-1 X²",
    "shifted8_x2": "Shifted-8 X²",
    "adaptive8_x2": "Adaptive-8 X²",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "cell1_x2": "#0F766E",
    "shifted8_x2": "#2563EB",
    "adaptive8_x2": "#7C3AED",
}
METHOD_MARKERS = {
    "paper": "s", "cell1_x2": "o", "shifted8_x2": "^",
    "adaptive8_x2": "D",
}
METHOD_SCORE = {
    "paper": "abs", "cell1_x2": "x2", "shifted8_x2": "x2",
    "adaptive8_x2": "x2",
}
METHOD_INTERNAL = {
    "paper": "paper", "cell1_x2": "c1_l1",
    "shifted8_x2": "shifted8", "adaptive8_x2": "adaptive8",
}
COMPONENTS = EXP39.COMPONENTS
ERROR_CEILINGS = EXP39.ERROR_CEILINGS

# Experiment 39 resolves these names at hook/runtime, so its laptop-safe real
# model measurement path can be reused without copying it.
EXP39.HERE = HERE
EXP39.METHODS = METHODS
EXP39.METHOD_LABELS = METHOD_LABELS
EXP39.METHOD_COLORS = METHOD_COLORS
EXP39.METHOD_MARKERS = METHOD_MARKERS
EXP39.METHOD_SCORE = METHOD_SCORE
EXP39.METHOD_INTERNAL = METHOD_INTERNAL


@njit(cache=False)
def _repair_block(mask, values, row_budget, run_costs):
    """Exact-R repair with one contiguous endpoint block instead of row loops."""
    n = len(mask)
    output = mask.copy()
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    selected = 0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
        selected += int(output[index])
    importance, cost, _ = EXP29._mask_metrics_lookup(output, values, run_costs)
    if selected == row_budget:
        return output, 0, True

    left = np.empty(n, dtype=np.int64)
    right = np.empty(n, dtype=np.int64)
    run_count = EXP29._extract_runs(output, left, right)
    if run_count == 0:
        return output, 0, False

    best_ratio = -np.inf
    best_start = -1
    repair = abs(row_budget - selected)
    if selected < row_budget:
        add = row_budget - selected
        for run in range(run_count):
            run_length = right[run] - left[run] + 1
            # Add immediately to the left. A full gap bridges two runs.
            gap_start = 0 if run == 0 else right[run - 1] + 1
            gap = left[run] - gap_start
            if gap >= add:
                start = left[run] - add
                gain = prefix[left[run]] - prefix[start]
                if run > 0 and gap == add:
                    previous_length = right[run - 1] - left[run - 1] + 1
                    candidate_cost = (
                        cost - run_costs[previous_length] - run_costs[run_length]
                        + run_costs[previous_length + add + run_length]
                    )
                else:
                    candidate_cost = (
                        cost - run_costs[run_length]
                        + run_costs[run_length + add]
                    )
                ratio = (importance + gain) / max(candidate_cost, 1e-300)
                if ratio > best_ratio + 1e-15:
                    best_ratio = ratio
                    best_start = start

            # Add immediately to the right. A full gap bridges two runs.
            gap_end = n if run + 1 == run_count else left[run + 1]
            gap = gap_end - right[run] - 1
            if gap >= add:
                start = right[run] + 1
                gain = prefix[start + add] - prefix[start]
                if run + 1 < run_count and gap == add:
                    next_length = right[run + 1] - left[run + 1] + 1
                    candidate_cost = (
                        cost - run_costs[run_length] - run_costs[next_length]
                        + run_costs[run_length + add + next_length]
                    )
                else:
                    candidate_cost = (
                        cost - run_costs[run_length]
                        + run_costs[run_length + add]
                    )
                ratio = (importance + gain) / max(candidate_cost, 1e-300)
                if ratio > best_ratio + 1e-15:
                    best_ratio = ratio
                    best_start = start
        if best_start < 0:
            return output, repair, False
        for index in range(best_start, best_start + add):
            output[index] = True
        return output, repair, True

    remove = selected - row_budget
    for run in range(run_count):
        run_length = right[run] - left[run] + 1
        if run_length < remove:
            continue
        if run_length == remove:
            candidate_cost = cost - run_costs[run_length]
        else:
            candidate_cost = (
                cost - run_costs[run_length] + run_costs[run_length - remove]
            )
        start = left[run]
        loss = prefix[start + remove] - prefix[start]
        ratio = (importance - loss) / max(candidate_cost, 1e-300)
        if ratio > best_ratio + 1e-15:
            best_ratio = ratio
            best_start = start
        start = right[run] - remove + 1
        loss = prefix[start + remove] - prefix[start]
        ratio = (importance - loss) / max(candidate_cost, 1e-300)
        if ratio > best_ratio + 1e-15:
            best_ratio = ratio
            best_start = start
    if best_start < 0:
        return output, repair, False
    for index in range(best_start, best_start + remove):
        output[index] = False
    return output, repair, True


@njit(cache=False)
def _partition_candidate(values, row_budget, cell_rows, offset, run_costs):
    """Best floor/ceil repaired mask for one shifted s-cell partition."""
    n = len(values)
    if row_budget <= 0 or cell_rows <= 0 or offset >= n:
        return np.zeros(n, dtype=np.bool_), 0.0, 0, 0, 0, False
    cell_count = (n - offset) // cell_rows
    if cell_count <= 0 or row_budget < cell_rows:
        mask = EXP29._best_contiguous(values, row_budget, max(1, cell_rows // 8))
        importance, cost, chunks = EXP29._mask_metrics_lookup(
            mask, values, run_costs
        )
        return mask, importance / max(cost, 1e-300), 0, 0, chunks, True

    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for index in range(n):
        prefix[index + 1] = prefix[index] + values[index]
    weights = np.empty(cell_count, dtype=np.float64)
    for cell in range(cell_count):
        start = offset + cell * cell_rows
        weights[cell] = prefix[start + cell_rows] - prefix[start]
    order = np.argsort(weights)

    floor_count = min(cell_count, row_budget // cell_rows)
    floor_mask = np.zeros(n, dtype=np.bool_)
    for rank in range(cell_count - floor_count, cell_count):
        cell = order[rank]
        start = offset + cell * cell_rows
        for index in range(start, start + cell_rows):
            floor_mask[index] = True
    floor_mask, additions, valid = _repair_block(
        floor_mask, values, row_budget, run_costs
    )
    if not valid:
        return floor_mask, 0.0, 1, additions, 0, False
    importance, cost, chunks = EXP29._mask_metrics_lookup(
        floor_mask, values, run_costs
    )
    best_mask = floor_mask
    best_ratio = importance / max(cost, 1e-300)
    strategy = 1
    repair = additions
    best_chunks = chunks

    ceil_count = min(cell_count, (row_budget + cell_rows - 1) // cell_rows)
    if ceil_count > floor_count:
        ceil_mask = np.zeros(n, dtype=np.bool_)
        for rank in range(cell_count - ceil_count, cell_count):
            cell = order[rank]
            start = offset + cell * cell_rows
            for index in range(start, start + cell_rows):
                ceil_mask[index] = True
        ceil_mask, deletions, ceil_valid = _repair_block(
            ceil_mask, values, row_budget, run_costs
        )
        if ceil_valid:
            ceil_importance, ceil_cost, ceil_chunks = EXP29._mask_metrics_lookup(
                ceil_mask, values, run_costs
            )
            ceil_ratio = ceil_importance / max(ceil_cost, 1e-300)
            if ceil_ratio > best_ratio + 1e-15:
                best_mask = ceil_mask
                best_ratio = ceil_ratio
                strategy = 2
                repair = deletions
                best_chunks = ceil_chunks
    return best_mask, best_ratio, strategy, repair, best_chunks, True


@njit(cache=False)
def _shifted_kernel(values, row_budget, cell_rows, offset_count, run_costs):
    """Evaluate evenly spaced grid origins in one compiled call."""
    n = len(values)
    best_mask = np.zeros(n, dtype=np.bool_)
    best_ratio = -1.0
    best_offset = 0
    best_strategy = 0
    best_repair = 0
    best_chunks = 0
    evaluated = 0
    previous = -1
    for offset_index in range(offset_count):
        offset = int(round(offset_index * cell_rows / offset_count))
        if offset >= cell_rows:
            offset = cell_rows - 1
        if offset == previous:
            continue
        previous = offset
        mask, ratio, strategy, repair, chunks, valid = _partition_candidate(
            values, row_budget, cell_rows, offset, run_costs
        )
        if not valid:
            continue
        evaluated += 1
        if ratio > best_ratio + 1e-15:
            best_mask = mask
            best_ratio = ratio
            best_offset = offset
            best_strategy = strategy
            best_repair = repair
            best_chunks = chunks
    return (
        best_mask, best_ratio, best_offset, best_strategy, best_repair,
        best_chunks, evaluated, best_ratio >= 0.0,
    )


@njit(cache=False)
def _boundary_refine(
    base_mask, values, row_budget, cell_rows, offset, boundary_cells,
    refine_divisor, run_costs,
):
    """Split only low-selected/high-unselected cells near the score boundary."""
    n = len(values)
    cell_count = (n - offset) // cell_rows
    if cell_count <= 1 or boundary_cells <= 0 or refine_divisor <= 1:
        importance, cost, chunks = EXP29._mask_metrics_lookup(
            base_mask, values, run_costs
        )
        return base_mask.copy(), importance / max(cost, 1e-300), chunks, 0, False

    scores = np.zeros(cell_count, dtype=np.float64)
    states = np.zeros(cell_count, dtype=np.int8)
    selected_cells = np.empty(cell_count, dtype=np.int64)
    unselected_cells = np.empty(cell_count, dtype=np.int64)
    selected_count = 0
    unselected_count = 0
    for cell in range(cell_count):
        start = offset + cell * cell_rows
        selected = 0
        score = 0.0
        for index in range(start, start + cell_rows):
            score += values[index]
            selected += int(base_mask[index])
        scores[cell] = score
        if selected == cell_rows:
            states[cell] = 1
            selected_cells[selected_count] = cell
            selected_count += 1
        elif selected == 0:
            states[cell] = -1
            unselected_cells[unselected_count] = cell
            unselected_count += 1

    take_selected = min(boundary_cells, selected_count)
    take_unselected = min(boundary_cells, unselected_count)
    if take_selected == 0 or take_unselected == 0:
        importance, cost, chunks = EXP29._mask_metrics_lookup(
            base_mask, values, run_costs
        )
        return base_mask.copy(), importance / max(cost, 1e-300), chunks, 0, False

    selected_order = np.argsort(scores[selected_cells[:selected_count]])
    unselected_order = np.argsort(scores[unselected_cells[:unselected_count]])
    refine = np.zeros(cell_count, dtype=np.bool_)
    for rank in range(take_selected):
        refine[selected_cells[selected_order[rank]]] = True
    for rank in range(take_unselected):
        source_rank = unselected_count - 1 - rank
        refine[unselected_cells[unselected_order[source_rank]]] = True

    output = base_mask.copy()
    target_in_region = 0
    subblock_count = 0
    for cell in range(cell_count):
        if not refine[cell]:
            continue
        start = offset + cell * cell_rows
        for index in range(start, start + cell_rows):
            target_in_region += int(output[index])
            output[index] = False
        subblock_count += refine_divisor

    starts = np.empty(subblock_count, dtype=np.int64)
    lengths = np.empty(subblock_count, dtype=np.int64)
    densities = np.empty(subblock_count, dtype=np.float64)
    cursor = 0
    for cell in range(cell_count):
        if not refine[cell]:
            continue
        cell_start = offset + cell * cell_rows
        for part in range(refine_divisor):
            start = cell_start + (part * cell_rows) // refine_divisor
            end = cell_start + ((part + 1) * cell_rows) // refine_divisor
            score = 0.0
            for index in range(start, end):
                score += values[index]
            starts[cursor] = start
            lengths[cursor] = end - start
            densities[cursor] = score / max(end - start, 1)
            cursor += 1

    order = np.argsort(densities)
    selected_in_region = 0
    for rank in range(subblock_count - 1, -1, -1):
        if selected_in_region >= target_in_region:
            break
        block = order[rank]
        start = starts[block]
        length = lengths[block]
        for index in range(start, start + length):
            output[index] = True
        selected_in_region += length

    selected_total = 0
    for value in output:
        selected_total += int(value)
    repair = 0
    valid = True
    if selected_total > row_budget:
        output, repair, valid = _repair_block(
            output, values, row_budget, run_costs
        )
    elif selected_total < row_budget:
        output, repair, valid = _repair_block(
            output, values, row_budget, run_costs
        )
    if not valid:
        importance, cost, chunks = EXP29._mask_metrics_lookup(
            base_mask, values, run_costs
        )
        return base_mask.copy(), importance / max(cost, 1e-300), chunks, repair, False
    importance, cost, chunks = EXP29._mask_metrics_lookup(
        output, values, run_costs
    )
    return output, importance / max(cost, 1e-300), chunks, repair, True


class AdaptiveShiftedSelector(EXP37.SuperTileSelector):
    def __init__(self, lookup_table, saturation_kib: float, offset_count: int,
                 boundary_cells: int, refine_divisor: int):
        super().__init__(lookup_table, saturation_kib)
        self.offset_count = int(offset_count)
        self.boundary_cells = int(boundary_cells)
        self.refine_divisor = int(refine_divisor)

    def select(self, method: str, importance: torch.Tensor, row_budget: int,
               d: int) -> tuple[torch.Tensor, dict]:
        if method not in ("shifted8", "adaptive8"):
            return super().select(method, importance, row_budget, d)
        n = int(importance.numel())
        context = self.context(n, d)
        normalized = importance / importance.sum().clamp_min(1e-20)
        values = normalized.detach().to("cpu").numpy().astype(np.float64)
        model = context["model"]
        cell_rows = max(1, int(math.ceil(float(model["saturation_rows"]))))
        row_kib = float(model["saturation_kib"]) / float(model["saturation_rows"])
        run_costs = EXP29.run_costs_for(self.lookup_table, n, row_kib)
        try:
            result = _shifted_kernel(
                values, int(row_budget), cell_rows, self.offset_count, run_costs
            )
            (
                mask_np, ratio, offset, strategy, repair, chunks,
                evaluated, valid,
            ) = result
            refined = False
            refine_repair = 0
            if method == "adaptive8" and valid:
                base_importance, _, _ = EXP29._mask_metrics_lookup(
                    mask_np, values, run_costs
                )
                candidate, candidate_ratio, candidate_chunks, refine_repair, ok = (
                    _boundary_refine(
                        mask_np, values, int(row_budget), cell_rows, offset,
                        self.boundary_cells, self.refine_divisor, run_costs,
                    )
                )
                candidate_importance, _, _ = EXP29._mask_metrics_lookup(
                    candidate, values, run_costs
                )
                # Refinement is the quality-oriented branch. Its possibly
                # higher read cost is judged by the measured frontier after a
                # lower R is chosen, rather than rejected at the same R.
                if ok and candidate_importance > base_importance + 1e-15:
                    mask_np = candidate
                    ratio = candidate_ratio
                    chunks = candidate_chunks
                    refined = True
            if not valid or int(mask_np.sum()) != int(row_budget):
                raise RuntimeError("shifted/adaptive selector failed exact-R invariant")
            metadata = {
                "offset_count": self.offset_count,
                "evaluated_offsets": int(evaluated),
                "winning_offset_rows": int(offset),
                "cell_rows": int(cell_rows),
                "boundary_cells": self.boundary_cells,
                "refine_divisor": self.refine_divisor,
                "boundary_refined": refined,
                "tile_strategy": (
                    "floor_expand" if strategy == 1 else "ceil_trim"
                ),
                "repair_rows": int(repair),
                "refine_repair_rows": int(refine_repair),
                "eligible_candidates": int(chunks),
                "lookup_efficiency": float(ratio),
                "paper_parameter_source": context["paper_parameter_source"],
                "fallback_used": False,
                "error": "",
            }
        except Exception as exception:
            mask_np = EXP29.EXP24.top_r_mask(values, int(row_budget))
            metadata = {
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
    parser.add_argument(
        "--budgets", type=float, nargs="+",
        default=[value / 100.0 for value in range(10, 100, 5)],
    )
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
    parser.add_argument("--offset-count", type=int, default=8)
    parser.add_argument("--boundary-cells", type=int, default=4)
    parser.add_argument("--refine-divisor", type=int, default=2)
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
        raise SystemExit("Experiment 43 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if not args.analyze_only and len(args.models) != 1:
        raise SystemExit("run exactly one model per process, then merge CSVs")
    if args.budgets != sorted(set(args.budgets)) or any(
        not 0.0 < value < 1.0 for value in args.budgets
    ):
        raise SystemExit("--budgets must be unique, increasing, and inside (0, 1)")
    if args.offset_count < 1 or args.boundary_cells < 0 or args.refine_divisor < 2:
        raise SystemExit("invalid adaptive-grid parameters")
    if any(value < 1 for value in (
        args.layer_samples, args.score_repetitions, args.selector_repetitions,
        args.io_repetitions, args.gemm_repetitions, args.cpu_threads,
    )):
        raise SystemExit("sample and thread counts must be positive")


def _interpolate(curve: pd.DataFrame, column: str, error: float) -> float:
    return EXP39._interpolate(curve, column, error)


def _pct_gain(baseline: float, candidate: float) -> float:
    return EXP39._pct_gain(baseline, candidate)


def _lagrangian_allocate(group: pd.DataFrame, loss_column: str,
                         target_loss: float) -> dict | None:
    """Supported multiple-choice solution: one R candidate per projection."""
    cases = []
    for _, case in group.groupby("case_key", sort=True):
        case = case.sort_values("budget_fraction")
        cases.append({
            "actual_time": case.actual_total_ms.to_numpy(dtype=float),
            "decision_time": case.decision_time_ms.to_numpy(dtype=float),
            "loss": case[loss_column].to_numpy(dtype=float),
            "error": case.relative_l2_error.to_numpy(dtype=float),
            "fraction": case.selected_fraction.to_numpy(dtype=float),
        })
    if not cases:
        return None
    time_scale = np.median(np.concatenate([
        case["decision_time"] for case in cases
    ]))
    loss_values = np.concatenate([case["loss"] for case in cases])
    loss_span = max(float(np.nanmax(loss_values) - np.nanmin(loss_values)), 1e-8)
    center = max(time_scale / loss_span, 1e-8)
    lambdas = np.concatenate((
        np.array([0.0]), center * np.logspace(-5, 6, 1200),
        np.array([np.inf]),
    ))
    best = None
    for multiplier in lambdas:
        total_actual = total_decision = 0.0
        total_loss = total_error = total_fraction = 0.0
        for case in cases:
            if np.isinf(multiplier):
                index = int(np.argmin(case["loss"]))
            else:
                index = int(np.argmin(
                    case["decision_time"] + multiplier * case["loss"]
                ))
            total_actual += case["actual_time"][index]
            total_decision += case["decision_time"][index]
            total_loss += case["loss"][index]
            total_error += case["error"][index]
            total_fraction += case["fraction"][index]
        if total_loss <= target_loss + 1e-12 and (
            best is None or total_decision < best["total_decision_time_ms"]
        ):
            count = len(cases)
            best = {
                "cases": count,
                "total_time_ms": total_actual,
                "mean_time_ms": total_actual / count,
                "total_decision_time_ms": total_decision,
                "mean_decision_time_ms": total_decision / count,
                "total_loss": total_loss,
                "mean_loss": total_loss / count,
                "mean_error": total_error / count,
                "mean_selected_fraction": total_fraction / count,
            }
    return best


def _global_allocations(candidates: pd.DataFrame) -> pd.DataFrame:
    holdout = candidates[candidates.split == "holdout"].copy()
    # The optimizer is not allowed to use the held-out timing of the choice it
    # is making. Cost comes from the matching calibration situation/module.
    # Median over the three calibration situations rejects clock-startup and
    # desktop-contention spikes (one observed score sample was >500 ms).
    calibration_costs = candidates[candidates.split == "calibration"].groupby(
        ["model", "module", "method", "budget_index"],
        as_index=False,
    ).actual_total_ms.median().rename(
        columns={"actual_total_ms": "decision_time_ms"}
    )
    holdout = holdout.merge(
        calibration_costs,
        on=["model", "module", "method", "budget_index"],
        how="left", validate="many_to_one",
    )
    if holdout.decision_time_ms.isna().any():
        raise RuntimeError("missing calibration latency for global allocation")
    rows = []
    workload_keys = ["model", "prompt"]
    for workload, workload_frame in holdout.groupby(workload_keys, sort=True):
        paper = workload_frame[workload_frame.method == "paper"]
        for budget_index, baseline in paper.groupby("budget_index", sort=True):
            target_importance_loss = float((1.0 - baseline.abs_importance_retention).sum())
            target_error = float(baseline.relative_l2_error.sum())
            for method in METHODS:
                method_frame = workload_frame[workload_frame.method == method].copy()
                method_frame["abs_loss"] = 1.0 - method_frame.abs_importance_retention
                method_frame["error_loss"] = method_frame.relative_l2_error
                for objective, column, target in (
                    ("importance_bound", "abs_loss", target_importance_loss),
                    ("error_oracle", "error_loss", target_error),
                ):
                    result = _lagrangian_allocate(method_frame, column, target)
                    row = {
                        "model": workload[0], "prompt": workload[1],
                        "paper_budget_index": int(budget_index),
                        "paper_budget_fraction": float(
                            baseline.budget_fraction.iloc[0]
                        ),
                        "method": method, "method_label": METHOD_LABELS[method],
                        "objective": objective, "feasible": result is not None,
                        "target_mean_importance_loss": (
                            target_importance_loss / len(baseline)
                        ),
                        "target_mean_error": target_error / len(baseline),
                    }
                    if result is not None:
                        row.update(result)
                    rows.append(row)
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    holdout = candidates[candidates.split == "holdout"].copy()
    fixed = holdout.groupby(
        ["method", "method_label", "score_kind", "budget_index", "budget_fraction"],
        as_index=False,
    ).agg(
        cases=("case_key", "size"),
        actual_total_ms=("actual_total_ms", "mean"),
        score_median_ms=("score_median_ms", "mean"),
        selector_median_ms=("selector_median_ms", "mean"),
        read_wall_median_ms=("read_wall_median_ms", "mean"),
        gather_median_ms=("gather_median_ms", "mean"),
        gemm_median_ms=("gemm_median_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        importance_retention=("importance_retention", "mean"),
        abs_importance_retention=("abs_importance_retention", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        chunks=("chunks", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    fixed.to_csv(output / "fixed_frontiers.csv", index=False)

    columns = ("actual_total_ms", *COMPONENTS, "selected_fraction", "chunks")
    interpolated_rows = []
    for ceiling in ERROR_CEILINGS:
        for method in METHODS:
            curve = fixed[fixed.method == method]
            row = {
                "error_ceiling": ceiling, "method": method,
                "method_label": METHOD_LABELS[method],
            }
            for column in columns:
                row[column] = _interpolate(curve, column, ceiling)
            interpolated_rows.append(row)
    interpolated = pd.DataFrame(interpolated_rows)
    interpolated.to_csv(output / "same_error_components.csv", index=False)

    comparison_rows = []
    for ceiling, group in interpolated.groupby("error_ceiling"):
        indexed = group.set_index("method")
        row = {"error_ceiling": ceiling}
        for method in METHODS:
            value = indexed.loc[method, "actual_total_ms"]
            row[f"{method}_total_ms"] = value
            row[f"{method}_gain_vs_cell1_pct"] = _pct_gain(
                indexed.loc["cell1_x2", "actual_total_ms"], value
            )
            row[f"{method}_gain_vs_paper_pct"] = _pct_gain(
                indexed.loc["paper", "actual_total_ms"], value
            )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "same_error_comparison.csv", index=False)

    allocations = _global_allocations(candidates)
    allocations.to_csv(output / "global_quality_allocations.csv", index=False)
    allocation_summary = allocations[allocations.feasible].groupby(
        ["objective", "paper_budget_index", "paper_budget_fraction", "method",
         "method_label"], as_index=False,
    ).agg(
        workloads=("prompt", "size"),
        mean_time_ms=("mean_time_ms", "mean"),
        mean_decision_time_ms=("mean_decision_time_ms", "mean"),
        mean_error=("mean_error", "mean"),
        mean_loss=("mean_loss", "mean"),
        mean_selected_fraction=("mean_selected_fraction", "mean"),
        target_mean_error=("target_mean_error", "mean"),
        target_mean_importance_loss=("target_mean_importance_loss", "mean"),
    )
    allocation_summary["fixed_same_error_ms"] = [
        _interpolate(
            fixed[fixed.method == row.method], "actual_total_ms", row.mean_error
        )
        for row in allocation_summary.itertuples(index=False)
    ]
    allocation_summary["gain_vs_fixed_same_error_pct"] = [
        _pct_gain(row.fixed_same_error_ms, row.mean_time_ms)
        for row in allocation_summary.itertuples(index=False)
    ]
    allocation_summary.to_csv(output / "global_quality_summary.csv", index=False)

    per_model_rows = []
    for model, model_frame in holdout.groupby("model", sort=True):
        model_fixed = model_frame.groupby(
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
            for method in METHODS:
                per_model_rows.append({
                    "model": model, "error_ceiling": ceiling,
                    "method": method, "method_label": METHOD_LABELS[method],
                    "actual_total_ms": values[method],
                    "gain_vs_cell1_pct": _pct_gain(
                        values["cell1_x2"], values[method]
                    ),
                })
    per_model = pd.DataFrame(per_model_rows)
    per_model.to_csv(output / "per_model_same_error.csv", index=False)

    pivot = holdout.pivot(
        index=["model", "case_key", "budget_index"], columns="method",
        values=[
            "relative_l2_error", "actual_total_ms", "selector_median_ms",
            "read_wall_median_ms", "chunks",
        ],
    )
    diagnostic_rows = []
    for model in sorted(holdout.model.unique()):
        model_pivot = pivot.xs(model, level="model")
        for method in ("shifted8_x2", "adaptive8_x2"):
            diagnostic_rows.append({
                "model": model, "method": method,
                "same_r_error_improvement_pct": 100.0 * (
                    1.0
                    - model_pivot["relative_l2_error"][method].mean()
                    / model_pivot["relative_l2_error"]["cell1_x2"].mean()
                ),
                "actual_total_delta_ms": (
                    model_pivot["actual_total_ms"][method]
                    - model_pivot["actual_total_ms"]["cell1_x2"]
                ).mean(),
                "selector_delta_ms": (
                    model_pivot["selector_median_ms"][method]
                    - model_pivot["selector_median_ms"]["cell1_x2"]
                ).mean(),
                "read_wall_delta_ms": (
                    model_pivot["read_wall_median_ms"][method]
                    - model_pivot["read_wall_median_ms"]["cell1_x2"]
                ).mean(),
                "chunk_delta": (
                    model_pivot["chunks"][method]
                    - model_pivot["chunks"]["cell1_x2"]
                ).mean(),
            })
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(output / "same_r_diagnostics.csv", index=False)

    _plot(output, fixed, interpolated, allocation_summary)
    _write_report(
        args, candidates, comparison, interpolated, allocation_summary,
        per_model, diagnostics,
    )


def _plot(output: Path, fixed: pd.DataFrame, interpolated: pd.DataFrame,
          allocations: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for method in METHODS:
        group = fixed[fixed.method == method].sort_values("budget_fraction")
        ax.plot(
            group.actual_total_ms, group.relative_l2_error,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linewidth=2.0, markersize=4, label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Measured actual total per projection (ms)")
    ax.set_ylabel("Projection relative L2 error (lower is better)")
    ax.set_title("Adaptive shifted Cell-1: quality-latency frontier")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "adaptive_quality_latency_frontier.png", dpi=200)
    fig.savefig(output / "adaptive_quality_latency_frontier.pdf")
    plt.close(fig)

    common = interpolated.dropna(subset=["actual_total_ms"])
    counts = common.groupby("error_ceiling").method.nunique()
    common = common[common.error_ceiling.isin(counts[counts == len(METHODS)].index)]
    means = common.groupby("method", as_index=False)[list(COMPONENTS)].mean()
    means["order"] = means.method.map({method: i for i, method in enumerate(METHODS)})
    means = means.sort_values("order")
    means.to_csv(output / "same_error_component_means.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    x = np.arange(len(means))
    bottoms = np.zeros(len(means))
    labels = {
        "score_median_ms": "Score", "selector_median_ms": "Selector",
        "read_wall_median_ms": "SSD/upload", "gather_median_ms": "Gather",
        "gemm_median_ms": "Compact GEMM",
    }
    colors = {
        "score_median_ms": "#DB2777", "selector_median_ms": "#7C3AED",
        "read_wall_median_ms": "#2563EB", "gather_median_ms": "#D97706",
        "gemm_median_ms": "#0F766E",
    }
    for column in COMPONENTS:
        values = means[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottoms, color=colors[column], label=labels[column])
        bottoms += values
    for index, total in enumerate(bottoms):
        ax.text(index, total + 0.008, f"{total:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [METHOD_LABELS[method] for method in means.method], rotation=8)
    ax.set_ylabel("Latency at common equal-error points (ms)")
    ax.set_title("Where the adaptive selector wins or loses")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "adaptive_component_breakdown.png", dpi=200)
    fig.savefig(output / "adaptive_component_breakdown.pdf")
    plt.close(fig)

    if len(allocations):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
        objectives = (
            ("importance_bound", "Global importance lower bound"),
            ("error_oracle", "Global measured-error oracle"),
        )
        for ax, (objective, title) in zip(axes, objectives):
            for method in ("cell1_x2", "adaptive8_x2"):
                group = allocations[
                    (allocations.method == method)
                    & (allocations.objective == objective)
                ].sort_values("paper_budget_fraction")
                ax.plot(
                    group.mean_time_ms, group.mean_error,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method], linewidth=2.0,
                    label=METHOD_LABELS[method],
                )
                for point_index, row in enumerate(group.itertuples(index=False)):
                    vertical = 5 if method == "adaptive8_x2" else -10
                    ax.annotate(
                        f"{100 * row.mean_selected_fraction:.0f}%",
                        (row.mean_time_ms, row.mean_error), xytext=(3, vertical),
                        textcoords="offset points", fontsize=5.8,
                    )
            ax.set_xlabel("Measured actual total per projection (ms)")
            ax.set_title(title + "\n(labels are resulting R)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Projection relative L2 error (lower is better)")
        axes[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(output / "global_quality_allocation.png", dpi=200)
        fig.savefig(output / "global_quality_allocation.pdf")
        plt.close(fig)


def _fmt(value: float, digits: int = 3, suffix: str = "") -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}{suffix}"


def _write_report(args, candidates: pd.DataFrame, comparison: pd.DataFrame,
                  interpolated: pd.DataFrame, allocations: pd.DataFrame,
                  per_model: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    common = comparison.dropna()
    adaptive_gain = common.adaptive8_x2_gain_vs_cell1_pct
    shifted_gain = common.shifted8_x2_gain_vs_cell1_pct
    component_common = interpolated.dropna(subset=["actual_total_ms"])
    component_counts = component_common.groupby("error_ceiling").method.nunique()
    component_common = component_common[
        component_common.error_ceiling.isin(
            component_counts[component_counts == len(METHODS)].index
        )
    ]
    component_means = component_common.groupby("method")[
        ["actual_total_ms", *COMPONENTS]
    ].mean()
    lines = [
        "# Experiment 43: adaptive shifted Cell-1", "",
        "고정된 원점의 s-cell만 쓰는 Cell-1을 8개 offset으로 확장하고, 가장 ",
        "불확실한 selected/unselected cell 각각 4개만 s/2 subcell로 다시 ",
        "최적화했다. 모든 방법은 실제 score + selector + O_DIRECT/upload + ",
        "activation gather + compact GEMM 시간을 포함한다.", "",
        "## 동일 projection error", "",
        "양수 gain은 기존 Cell-1 X²보다 빠르다는 뜻이다.", "",
        "| error | Paper | Cell-1 X² | Shifted-8 | gain | Adaptive-8 | gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.error_ceiling:.2f} | {_fmt(row.paper_total_ms, suffix=' ms')} "
            f"| {_fmt(row.cell1_x2_total_ms, suffix=' ms')} "
            f"| {_fmt(row.shifted8_x2_total_ms, suffix=' ms')} "
            f"| {_fmt(row.shifted8_x2_gain_vs_cell1_pct, 2, '%')} "
            f"| {_fmt(row.adaptive8_x2_total_ms, suffix=' ms')} "
            f"| {_fmt(row.adaptive8_x2_gain_vs_cell1_pct, 2, '%')} |"
        )
    lines.extend(["", "## 동일-error 구성요소 평균", "",
                  "| method | score | selector | SSD/upload | gather | GEMM | total |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        if method not in component_means.index:
            lines.append(f"| {METHOD_LABELS[method]} | — | — | — | — | — | — |")
            continue
        row = component_means.loc[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row.score_median_ms:.4f} "
            f"| {row.selector_median_ms:.4f} | {row.read_wall_median_ms:.4f} "
            f"| {row.gather_median_ms:.4f} | {row.gemm_median_ms:.4f} "
            f"| {row.actual_total_ms:.4f} |"
        )

    lines.extend(["", "## R을 없앤 전역 allocation", "",
                  "각 workload의 sampled projection 전체에서 하나의 quality ",
        "budget을 공유한다. 선택 cost는 holdout 실측값이 아니라 동일 ",
        "calibration prompt에서 측정한 module별 median latency를 사용한다. ",
        "`importance_bound`는 공통 mean-|x| retention 하한을 쓰는 measurable ",
        "proxy 상한이며, `error_oracle`은 실측 projection error를 직접 ",
        "사용하는 비배포형 상한이다.", "",
                  "| objective | paper target R | method | resulting R | error | latency |",
                  "|---|---:|---|---:|---:|---:|"])
    shown = allocations[
        allocations.method.isin(("cell1_x2", "adaptive8_x2"))
        & allocations.paper_budget_fraction.isin((0.30, 0.50, 0.70, 0.90))
    ]
    for row in shown.itertuples(index=False):
        lines.append(
            f"| {row.objective} | {100*row.paper_budget_fraction:.0f}% "
            f"| {METHOD_LABELS[row.method]} | {100*row.mean_selected_fraction:.1f}% "
            f"| {row.mean_error:.4f} | {row.mean_time_ms:.4f} ms |"
        )

    lines.extend([
        "", "## 모델별 진단", "",
        "same-R error improvement는 양수가 좋고, 나머지 delta는 음수가 좋다.", "",
        "| model | method | error improvement | selector Δ | SSD/upload Δ | total Δ | chunks Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {METHOD_LABELS[row.method]} "
            f"| {row.same_r_error_improvement_pct:.2f}% "
            f"| {row.selector_delta_ms:+.4f} ms "
            f"| {row.read_wall_delta_ms:+.4f} ms "
            f"| {row.actual_total_delta_ms:+.4f} ms "
            f"| {row.chunk_delta:+.2f} |"
        )
    model_gains = per_model[
        per_model.method.isin(("shifted8_x2", "adaptive8_x2"))
    ].groupby(["model", "method"]).gain_vs_cell1_pct.mean().unstack()
    lines.extend([
        "", "모델별 동일-error 평균 gain:", "",
        "| model | Shifted-8 | Adaptive-8 |", "|---|---:|---:|",
    ])
    for model, row in model_gains.iterrows():
        lines.append(
            f"| {model} | {_fmt(row.get('shifted8_x2', np.nan), 2, '%')} "
            f"| {_fmt(row.get('adaptive8_x2', np.nan), 2, '%')} |"
        )

    fallback_count = int(candidates.fallback_used.sum())
    lines.extend(["", "## 판정", ""])
    if len(adaptive_gain):
        lines.append(
            f"동일-error에서 Shifted-8의 Cell-1 대비 평균 gain은 "
            f"`{shifted_gain.mean():.2f}%`, Adaptive-8은 "
            f"`{adaptive_gain.mean():.2f}%`다."
        )
    else:
        lines.append("모든 방법이 겹치는 error 구간이 없어 gain을 계산하지 못했다.")
    allocation_gains = allocations[
        allocations.method == "cell1_x2"
    ].groupby("objective").gain_vs_fixed_same_error_pct.mean()
    if "importance_bound" in allocation_gains:
        lines.append(
            "반대로 기존 Cell-1의 grid를 그대로 두고 projection별 R만 전역 "
            f"재배분하면, 동일한 실제 error에서 fixed-R Cell-1보다 평균 "
            f"`{allocation_gains['importance_bound']:.2f}%` 빠르다. 실제 error "
            "oracle 상한은 평균 "
            f"`{allocation_gains.get('error_oracle', np.nan):.2f}%`다."
        )
    lines.extend([
        "결론은 selector 자체의 mask 개선이 추가 selector 시간보다 큰지, 그리고 ",
        "고정 R을 없앤 전역 quality allocation이 그보다 더 큰지로 나누어 해석해야 ",
        "한다. error oracle은 달성 가능한 상한이지 배포 가능한 알고리즘이 아니다.",
        "", "## 측정 범위", "",
        f"- 모델 `{candidates.model.nunique()}`개, projection 후보 "
        f"`{len(candidates)}`개; fallback `{fallback_count}`건.",
        f"- offset `{args.offset_count}`개, boundary cell 방향당 "
        f"`{args.boundary_cells}`개, refine divisor `{args.refine_divisor}`.",
        "- R=10%..95%, 5% 간격. 그래프의 allocation 점 위 %는 최적화 결과로 ",
        "나온 평균 selected fraction이다.",
        "", f"![Frontier]({args.output_dir.name}/adaptive_quality_latency_frontier.png)",
        "", f"![Components]({args.output_dir.name}/adaptive_component_breakdown.png)",
        "", f"![Allocation]({args.output_dir.name}/global_quality_allocation.png)", "",
    ])
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines))


def _synthetic_self_check(selector: AdaptiveShiftedSelector) -> None:
    rng = np.random.default_rng(43)
    for n in (127, 256, 509):
        values = torch.from_numpy(rng.random(n).astype(np.float32))
        for rows in (1, n // 10, n // 2, n - 1):
            for method in ("shifted8", "adaptive8"):
                mask, metadata = selector.select(method, values, rows, 1024)
                if int(mask.sum()) != rows or metadata.get("fallback_used"):
                    raise RuntimeError(
                        f"self-check failed: n={n}, rows={rows}, method={method}, "
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
    selector = AdaptiveShiftedSelector(
        lookup, args.saturation_kib, args.offset_count, args.boundary_cells,
        args.refine_divisor,
    )
    _synthetic_self_check(selector)
    vlmflash = EXP32.EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        EXP32.cached_snapshot(MODEL_SPECS[args.models[0]]["repo"]),
        local_files_only=True,
    )
    required = int(config.intermediate_size) * int(config.hidden_size) * 2
    if args.io_blob.stat().st_size < required:
        raise SystemExit(f"I/O blob needs at least {required} bytes")

    model_key = args.models[0]
    print(f"loading and measuring {MODEL_SPECS[model_key]['label']}", flush=True)
    result = EXP39.run_model(
        model_key, selector, native_reader, prompts, args, vlmflash
    )
    (
        model, tokenizer, weight_norms, candidates, selectors, scores,
        io_rows, gemm_rows, model_metadata,
    ) = result
    pd.DataFrame(candidates).to_csv(args.output_dir / "candidates.csv", index=False)
    pd.DataFrame(selectors).to_csv(
        args.output_dir / "selector_samples.csv", index=False
    )
    pd.DataFrame(scores).to_csv(args.output_dir / "score_samples.csv", index=False)
    pd.DataFrame(io_rows).to_csv(args.output_dir / "io_aggregates.csv", index=False)
    pd.DataFrame(gemm_rows).to_csv(args.output_dir / "gemm_aggregates.csv", index=False)

    if not args.skip_end_to_end and any(
        prompt["split"] == "holdout" for prompt in prompts
    ):
        e2e = EXP39.run_end_to_end(
            model, tokenizer, model_key, weight_norms, selector,
            prompts, args, vlmflash,
        )
        pd.DataFrame(e2e).to_csv(args.output_dir / "end_to_end.csv", index=False)
    del model, tokenizer, weight_norms
    gc.collect()
    torch.cuda.empty_cache()

    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-43-adaptive-shifted-cell1-v1",
        "models": [model_metadata],
        "prompts": [
            {key: value for key, value in prompt.items() if key != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "methods": list(METHODS), "budgets": args.budgets,
        "saturation_kib": args.saturation_kib,
        "offset_count": args.offset_count,
        "boundary_cells": args.boundary_cells,
        "refine_divisor": args.refine_divisor,
        "score_repetitions": args.score_repetitions,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions,
        "gemm_repetitions": args.gemm_repetitions,
        "io_threads": args.io_threads,
        "profile": str(args.profile), "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
        "global_allocation": (
            "post-hoc Lagrangian multiple-choice allocation across sampled "
            "projections; decision cost comes from the matching calibration "
            "module median; importance uses common abs-importance loss; "
            "error is oracle"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    analyze(args)


if __name__ == "__main__":
    main()
