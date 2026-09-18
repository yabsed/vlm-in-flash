#!/usr/bin/env python3
"""Experiment 26: search a row-price frontier and adaptively trim to fixed R."""

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
EXPERIMENT_24 = PROJECT_ROOT / "experiments" / "24_fixed_r_ratio" / "run_experiment.py"


def load_experiment_24():
    spec = importlib.util.spec_from_file_location("experiment_24_for_26", EXPERIMENT_24)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 24 from {EXPERIMENT_24}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP24 = load_experiment_24()
EXP22 = EXP24.EXP22
EXP18 = EXP24.EXP18
EXP13 = EXP24.EXP13
EXP2 = EXP24.EXP2
BASE = EXP24.BASE
njit = EXP24.njit
SHAPES = EXP24.SHAPES
DEFAULT_TRACE_INPUT = EXP24.DEFAULT_TRACE_INPUT

METHODS = (
    "paper",
    "top_r",
    "legacy_rho2_mu4",
    "frontier16_trim64",
    "frontier16_trim256",
    "frontier24_trim256_rho3",
)
CANDIDATE_METHODS = METHODS[2:]
FRONTIER_METHODS = METHODS[3:]
METHOD_LABELS = {
    "paper": "Paper",
    "top_r": "Top-R",
    "legacy_rho2_mu4": "Exp24 rho2/mu4",
    "frontier16_trim64": "Frontier16 + adaptive trim64",
    "frontier16_trim256": "Frontier16 + adaptive trim256",
    "frontier24_trim256_rho3": "Frontier24 + adaptive trim256 + rho3",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "top_r": "#475569",
    "legacy_rho2_mu4": "#7C3AED",
    "frontier16_trim64": "#2563EB",
    "frontier16_trim256": "#0F766E",
    "frontier24_trim256_rho3": "#064E3B",
}
FRONTIER_SETTINGS = {
    "frontier16_trim64": {"rho_iterations": 2, "mu_calls": 16, "trim_limit": 64},
    "frontier16_trim256": {"rho_iterations": 2, "mu_calls": 16, "trim_limit": 256},
    "frontier24_trim256_rho3": {
        "rho_iterations": 3, "mu_calls": 24, "trim_limit": 256,
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
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--deadline-ms", type=float, default=2.0)
    parser.add_argument("--paper-impl", choices=("auto", "native", "torch"), default="native")
    parser.add_argument("--oracle-n", type=int, default=18)
    parser.add_argument("--oracle-trials", type=int, default=10)
    parser.add_argument("--oracle-cv", type=float, default=3.30)
    parser.add_argument("--seed", type=int, default=20262601)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--measure-io", action="store_true",
        help="replay every CUDA-track mask through the native O_DIRECT reader",
    )
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
    shapes, tracks = EXP24.validate_args(args)
    if args.io_repetitions < 1 or args.io_warmup < 0 or args.io_threads < 1:
        raise SystemExit("I/O repetition, warmup, and thread arguments are invalid")
    if args.io_max_read_kib < 4:
        raise SystemExit("--io-max-read-kib must be at least 4")
    if args.measure_io:
        if args.io_blob is None or not args.io_blob.is_file():
            raise SystemExit("--measure-io requires an existing --io-blob")
        if "cuda" not in tracks:
            raise SystemExit("--measure-io requires the cuda timing track")
        if (
            not args.analyze_only and args.io_device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise SystemExit("--io-device cuda was requested but CUDA is unavailable")
    return shapes, tracks


@njit(cache=False, inline="always")
def _chunk_ms(length, a_ms, c1_ms, c2_ms, saturation_rows):
    if length <= 0:
        return 0.0
    if length <= saturation_rows:
        return a_ms + c1_ms * length
    return c2_ms * length


@njit(cache=False)
def _adaptive_trim_kernel(mask, values, row_budget, a_ms, c1_ms, c2_ms,
                          saturation_rows):
    """Greedily delete the endpoint giving the best exact ratio after each step."""
    output = mask.copy()
    n = len(output)
    selected = 0
    run_count = 0
    importance = 0.0
    previous = False
    for index in range(n):
        if output[index]:
            selected += 1
            importance += values[index]
            if not previous:
                run_count += 1
            previous = True
        else:
            previous = False

    left = np.empty(run_count, dtype=np.int64)
    right = np.empty(run_count, dtype=np.int64)
    run = -1
    previous = False
    for index in range(n):
        if output[index] and not previous:
            run += 1
            left[run] = index
        if output[index]:
            right[run] = index
        previous = output[index]

    cost = 0.0
    for current_run in range(run_count):
        cost += _chunk_ms(
            right[current_run] - left[current_run] + 1,
            a_ms, c1_ms, c2_ms, saturation_rows,
        )
    start_ratio = importance / cost
    deletions = 0

    while selected > row_budget:
        best_ratio = -np.inf
        best_importance = -np.inf
        best_index = -1
        best_run = -1
        best_saving = 0.0
        for current_run in range(run_count):
            length = right[current_run] - left[current_run] + 1
            if length <= 0:
                continue
            saving = (
                _chunk_ms(length, a_ms, c1_ms, c2_ms, saturation_rows)
                - _chunk_ms(length - 1, a_ms, c1_ms, c2_ms, saturation_rows)
            )
            candidate_cost = cost - saving
            left_index = left[current_run]
            candidate_importance = importance - values[left_index]
            candidate_ratio = candidate_importance / candidate_cost
            if (
                candidate_ratio > best_ratio + 1e-14
                or (
                    abs(candidate_ratio - best_ratio) <= 1e-14
                    and (
                        candidate_importance > best_importance + 1e-14
                        or (
                            abs(candidate_importance - best_importance) <= 1e-14
                            and (best_index < 0 or left_index < best_index)
                        )
                    )
                )
            ):
                best_ratio = candidate_ratio
                best_importance = candidate_importance
                best_index = left_index
                best_run = current_run
                best_saving = saving
            if right[current_run] != left_index:
                right_index = right[current_run]
                candidate_importance = importance - values[right_index]
                candidate_ratio = candidate_importance / candidate_cost
                if (
                    candidate_ratio > best_ratio + 1e-14
                    or (
                        abs(candidate_ratio - best_ratio) <= 1e-14
                        and (
                            candidate_importance > best_importance + 1e-14
                            or (
                                abs(candidate_importance - best_importance) <= 1e-14
                                and right_index < best_index
                            )
                        )
                    )
                ):
                    best_ratio = candidate_ratio
                    best_importance = candidate_importance
                    best_index = right_index
                    best_run = current_run
                    best_saving = saving

        if best_index < 0:
            return output, -1, start_ratio, importance / cost
        output[best_index] = False
        selected -= 1
        deletions += 1
        importance -= values[best_index]
        cost -= best_saving
        if left[best_run] == right[best_run]:
            left[best_run] = 1
            right[best_run] = 0
        elif best_index == left[best_run]:
            left[best_run] += 1
        else:
            right[best_run] -= 1

    return output, deletions, start_ratio, importance / cost


def adaptive_trim(mask: np.ndarray, values: np.ndarray, row_budget: int,
                  model: dict) -> tuple[np.ndarray, dict]:
    selected = int(np.asarray(mask, dtype=bool).sum())
    if selected < row_budget:
        raise ValueError("adaptive trim cannot add rows")
    output, deletions, start_ratio, end_ratio = _adaptive_trim_kernel(
        np.asarray(mask, dtype=np.bool_), np.asarray(values, dtype=np.float64),
        int(row_budget), float(model["a_ms"]), float(model["c1_ms_per_row"]),
        float(model["c2_ms_per_row"]), float(model["saturation_rows"]),
    )
    if deletions < 0 or int(output.sum()) != row_budget:
        raise RuntimeError("adaptive trim failed to reach exact cardinality")
    return np.asarray(output, dtype=bool), {
        "repair_deletions": int(deletions),
        "trim_start_ratio": float(start_ratio),
        "trim_end_ratio": float(end_ratio),
    }


def _solve_price(values: np.ndarray, rho: float, mu: float, model: dict,
                 with_mask: bool):
    adjusted = values - mu
    function = EXP13._solve_lambda_mask if with_mask else EXP13._solve_lambda_metrics
    return function(
        adjusted, 1.0 / rho, model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"], model["short_max_rows"],
    )


def frontier_iteration(values: np.ndarray, row_budget: int, rho: float, model: dict,
                       mu_calls: int, trim_limit: int) -> tuple[np.ndarray, dict]:
    """Evaluate every distinct eligible supported cardinality, then keep best I/L."""
    n = len(values)
    eps = np.finfo(np.float64).eps
    singleton_cost = EXP18.two_line_chunk_ms(1, model)
    low = float(values.min() - rho * singleton_cost - 32.0 * eps)
    high = float(values.max() + 32.0 * eps)
    threshold = float(np.partition(values, n - row_budget)[n - row_budget])
    predicted = float(threshold - rho * model["c2_ms_per_row"])
    predicted = min(max(predicted, np.nextafter(low, high)), np.nextafter(high, low))

    best_mask = EXP24.top_r_mask(values, row_budget)
    best_metrics = EXP24.two_line_metrics(best_mask, values, model)
    best_ratio = float(best_metrics["importance"] / best_metrics["two_line_ms"])
    best_trim = 0
    seen_rows: set[int] = set()
    eligible = 0
    trim_work = 0
    exact_hits = 0
    observed_over = []
    calls = 0

    for call_index in range(mu_calls):
        mu = predicted if call_index == 0 else 0.5 * (low + high)
        solved = _solve_price(values, rho, mu, model, False)
        calls += 1
        rows = int(solved[3])
        if rows >= row_budget:
            low = mu
            observed_over.append(rows - row_budget)
        else:
            high = mu

        overshoot = rows - row_budget
        if 0 <= overshoot <= trim_limit and rows not in seen_rows:
            seen_rows.add(rows)
            replay = _solve_price(values, rho, mu, model, True)
            mask = np.asarray(replay[0], dtype=bool)
            if int(mask.sum()) != rows:
                raise RuntimeError("metrics/mask row count mismatch")
            candidate, trim = adaptive_trim(mask, values, row_budget, model)
            eligible += 1
            trim_work += int(trim["repair_deletions"])
            exact_hits += int(overshoot == 0)
            metrics = EXP24.two_line_metrics(candidate, values, model)
            ratio = float(metrics["importance"] / metrics["two_line_ms"])
            if ratio > best_ratio + 1e-14:
                best_ratio = ratio
                best_mask = candidate
                best_trim = int(trim["repair_deletions"])
        if rows == row_budget:
            break

    return best_mask, {
        "scalarized_calls": calls,
        "frontier_candidates": len(seen_rows),
        "eligible_candidates": eligible,
        "trim_work_deletions": trim_work,
        "repair_deletions": best_trim,
        "mu_exact_hits": exact_hits,
        "minimum_overfill": min(observed_over) if observed_over else n - row_budget,
    }


def frontier_fixed_r_ratio(values: np.ndarray, row_budget: int, model: dict,
                           rho_iterations: int, mu_calls: int,
                           trim_limit: int) -> tuple[np.ndarray, dict]:
    initial = EXP24.top_r_mask(values, row_budget)
    initial_metrics = EXP24.two_line_metrics(initial, values, model)
    best_mask = initial
    best_ratio = float(initial_metrics["importance"] / initial_metrics["two_line_ms"])
    totals = {
        "scalarized_calls": 0, "frontier_candidates": 0,
        "eligible_candidates": 0, "trim_work_deletions": 0,
        "repair_deletions": 0, "mu_exact_hits": 0,
    }
    minimum_overfill = len(values) - row_budget

    for _ in range(rho_iterations):
        candidate, metadata = frontier_iteration(
            values, row_budget, best_ratio, model, mu_calls, trim_limit
        )
        metrics = EXP24.two_line_metrics(candidate, values, model)
        ratio = float(metrics["importance"] / metrics["two_line_ms"])
        for key in totals:
            totals[key] += int(metadata[key])
        minimum_overfill = min(minimum_overfill, int(metadata["minimum_overfill"]))
        if ratio > best_ratio + 1e-14:
            best_ratio = ratio
            best_mask = candidate

    return np.asarray(best_mask, dtype=bool), {
        **totals,
        "minimum_overfill": minimum_overfill,
        "outer_iterations": int(rho_iterations),
        "mu_calls_per_outer": int(mu_calls),
        "trim_limit": int(trim_limit),
        "repair_additions": 0,
        "returned_top_r": bool(np.array_equal(best_mask, initial)),
    }


def safe_select(method: str, values: np.ndarray, row_budget: int,
                model: dict) -> tuple[np.ndarray, dict]:
    try:
        if method == "top_r":
            mask, metadata = EXP24.top_r_mask(values, row_budget), {}
        elif method == "legacy_rho2_mu4":
            mask, metadata = EXP24.fixed_r_ratio(values, row_budget, model, 2, 4)
        else:
            setting = FRONTIER_SETTINGS[method]
            mask, metadata = frontier_fixed_r_ratio(
                values, row_budget, model, setting["rho_iterations"],
                setting["mu_calls"], setting["trim_limit"],
            )
        return mask, {**metadata, "fallback_used": False, "error": ""}
    except Exception as exception:
        return EXP24.top_r_mask(values, row_budget), {
            "fallback_used": True,
            "error": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(limit=3),
        }


def selector_function(method: str, track: str, values_host: np.ndarray,
                      values_cuda: torch.Tensor | None, row_budget: int, model: dict):
    def run():
        values = (
            values_cuda.detach().to("cpu").numpy().astype(np.float64)
            if track == "cuda" else values_host
        )
        mask, metadata = safe_select(method, values, row_budget, model)
        output = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_).copy())
        if track == "cuda":
            output = output.to("cuda")
        return output, metadata
    return run


def self_check(lookup_table, saturation_kib: float) -> None:
    rng = np.random.default_rng(2601)
    shape = next(item for item in SHAPES if item["shape"] == "4864x896")
    model = EXP13.fit_continuous_two_line(
        lookup_table, EXP22.row_size_kib(shape), saturation_kib
    )
    for n in (37, 97):
        values = rng.lognormal(size=n)
        values /= values.sum()
        for row_budget in (n // 4, n // 2, 3 * n // 4):
            top = EXP24.top_r_mask(values, row_budget)
            top_metrics = EXP24.two_line_metrics(top, values, model)
            top_ratio = top_metrics["importance"] / top_metrics["two_line_ms"]
            outputs = {}
            for method in FRONTIER_METHODS:
                mask, metadata = safe_select(method, values, row_budget, model)
                metrics = EXP24.two_line_metrics(mask, values, model)
                ratio = metrics["importance"] / metrics["two_line_ms"]
                if (
                    int(mask.sum()) != row_budget or ratio < top_ratio - 1e-11
                    or metadata["fallback_used"]
                ):
                    raise RuntimeError("frontier adaptive-trim self-check failed")
                outputs[method] = ratio
            if outputs["frontier16_trim256"] < outputs["frontier16_trim64"] - 1e-11:
                raise RuntimeError("wider trim frontier regressed")
            if (
                outputs["frontier24_trim256_rho3"]
                < outputs["frontier16_trim256"] - 1e-11
            ):
                raise RuntimeError("deeper frontier regressed")


def collect_oracle(args: argparse.Namespace, lookup_table) -> list[dict]:
    n = int(args.oracle_n)
    shape = next(item for item in SHAPES if item["shape"] == "4096x14336")
    model = EXP13.fit_continuous_two_line(
        lookup_table, EXP22.row_size_kib(shape), args.saturation_kib
    )
    budgets = np.asarray([
        max(1, min(n, int(round(n * fraction))))
        for fraction in args.row_budget_fractions
    ], dtype=np.int64)
    rng = np.random.default_rng(args.seed + 101)
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    rows = []
    for trial in range(args.oracle_trials):
        multiset = EXP2.exact_cv_lognormal(rng, n, args.oracle_cv)
        variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
        for spatial_mode, raw_values in variants.items():
            values32 = np.asarray(raw_values, dtype=np.float32)
            values32 /= np.float32(values32.astype(np.float64).sum())
            values = values32.astype(np.float64)
            exact = EXP24._exact_fixed_r_ratios(
                values, budgets, model["a_ms"], model["c1_ms_per_row"],
                model["c2_ms_per_row"], model["saturation_rows"],
            )
            for budget_index, row_budget in enumerate(budgets):
                optimum = float(exact[0][budget_index])
                for method in METHODS[1:]:
                    mask, metadata = safe_select(method, values, int(row_budget), model)
                    metrics = EXP24.two_line_metrics(mask, values, model)
                    ratio = metrics["importance"] / metrics["two_line_ms"]
                    rows.append({
                        "n": n, "trial": trial, "spatial_mode": spatial_mode,
                        "budget_index": budget_index,
                        "row_budget_fraction": float(args.row_budget_fractions[budget_index]),
                        "row_budget": int(row_budget), "method": method,
                        "method_label": METHOD_LABELS[method],
                        "two_line_efficiency": ratio,
                        "exact_efficiency": optimum,
                        "optimality_ratio": ratio / optimum,
                        "optimality_gap_pct": 100.0 * (1.0 - ratio / optimum),
                        "row_match": int(mask.sum()) == int(row_budget),
                        "fallback_used": metadata.get("fallback_used", False),
                        "error": metadata.get("error", ""),
                    })
    return rows


def measure_actual_io(native_reader, args: argparse.Namespace, mask: np.ndarray,
                      row_kib: float) -> tuple[dict, list[dict]]:
    """Measure one selected mask from the target flash, including CUDA upload."""
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_).copy())
    if args.io_device == "cuda":
        mask_tensor = mask_tensor.to("cuda")
    row_bytes = int(round(row_kib * 1024.0))
    io_samples, upload_samples, wall_samples, direct_samples = [], [], [], []
    for repetition in range(args.io_warmup + args.io_repetitions):
        if args.io_device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output, io_us, upload_us, direct = native_reader.read_rows(
            str(args.io_blob), mask_tensor, row_bytes, args.io_device,
            args.io_threads, args.io_max_read_kib * 1024, True,
        )
        if args.io_device == "cuda":
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        del output
        if repetition >= args.io_warmup:
            io_samples.append(float(io_us) / 1000.0)
            upload_samples.append(float(upload_us) / 1000.0)
            wall_samples.append(float(wall_ms))
            direct_samples.append(bool(direct))
    if not all(direct_samples):
        raise RuntimeError("O_DIRECT was unavailable during actual mask replay")
    summary = {
        "actual_io_median_ms": float(np.median(io_samples)),
        "actual_io_p95_ms": float(np.quantile(io_samples, 0.95)),
        "actual_upload_median_ms": float(np.median(upload_samples)),
        "actual_upload_p95_ms": float(np.quantile(upload_samples, 0.95)),
        "actual_read_wall_median_ms": float(np.median(wall_samples)),
        "actual_read_wall_p95_ms": float(np.quantile(wall_samples, 0.95)),
        "actual_direct_rate": float(np.mean(direct_samples)),
    }
    samples = [
        {
            "repetition": repetition,
            "actual_io_ms": io_ms,
            "actual_upload_ms": upload_ms,
            "actual_read_wall_ms": wall_ms,
            "actual_direct": direct,
        }
        for repetition, (io_ms, upload_ms, wall_ms, direct) in enumerate(
            zip(io_samples, upload_samples, wall_samples, direct_samples)
        )
    ]
    return summary, samples


def collect(args: argparse.Namespace, traces: list[dict], tracks: list[str],
            lookup_table, native_reader=None):
    rows, timing_rows, model_rows, io_timing_rows = [], [], [], []
    by_name = {item["shape"]: dict(item) for item in SHAPES}
    shape_names = list(dict.fromkeys(trace["shape"] for trace in traces))
    contexts = {}
    for shape_index, shape_name in enumerate(shape_names):
        shape = by_name[shape_name]
        row_kib = EXP22.row_size_kib(shape)
        params = EXP22.make_params(shape, args.saturation_kib)
        model = EXP13.fit_continuous_two_line(
            lookup_table, row_kib, args.saturation_kib
        )
        model_rows.append({
            **shape, "shape_index": shape_index, "row_size_kib": row_kib,
            "activation_trace_count": sum(t["shape"] == shape_name for t in traces),
            **model,
        })
        contexts[shape_name] = shape_index, shape, row_kib, params, model

    for trace_index, trace in enumerate(traces):
        shape_index, shape, row_kib, params, model = contexts[trace["shape"]]
        n = int(shape["n"])
        raw = np.asarray(trace["values"], dtype=np.float32)
        if raw.shape != (n,) or not np.isfinite(raw).all() or np.any(raw < 0.0):
            raise RuntimeError(f"invalid activation trace {trace['trace_id']}")
        total = float(raw.astype(np.float64).sum())
        if total <= 0.0:
            raise RuntimeError(f"zero activation trace {trace['trace_id']}")
        values32 = raw.copy()
        values32 /= np.float32(total)
        values = values32.astype(np.float64)
        values_cuda = torch.from_numpy(values32).to("cuda") if "cuda" in tracks else None
        activation_cv = float(raw.astype(np.float64).std() / raw.astype(np.float64).mean())

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
                    raise RuntimeError("Paper returned an empty mask")
                case = {
                    "trace_id": int(trace["trace_id"]),
                    "input_index": int(trace["input_index"]),
                    "prompt_sha256": trace["prompt_sha256"],
                    "num_tokens": int(trace["num_tokens"]),
                    "module": trace["module"],
                    "projection": str(trace["module"]).rsplit(".", 1)[-1],
                    "module_call_index": int(trace["call_index"]),
                    "activation_cv": activation_cv,
                    "shape": shape["shape"], "shape_index": shape_index,
                    "n": n, "d": int(shape["d"]), "row_size_kib": row_kib,
                    "budget_index": budget_index,
                    "row_budget_fraction": float(budget_fraction),
                    "nominal_row_budget": nominal_budget,
                    "effective_row_budget": effective_r,
                    "paper_underfill_rows": nominal_budget - effective_r,
                    "track": track,
                }

                def append_result(method, mask, metadata, timings):
                    metrics = EXP18.mask_metrics(
                        mask, values, model, lookup_table, row_kib
                    )
                    row_match = metrics["rows"] == effective_r
                    error = metadata.get("error", "")
                    runtime_p95 = float(np.quantile(timings, 0.95))
                    base = {**case, "method": method, "method_label": METHOD_LABELS[method]}
                    actual = {
                        "actual_io_median_ms": math.nan,
                        "actual_io_p95_ms": math.nan,
                        "actual_upload_median_ms": math.nan,
                        "actual_upload_p95_ms": math.nan,
                        "actual_read_wall_median_ms": math.nan,
                        "actual_read_wall_p95_ms": math.nan,
                        "actual_direct_rate": math.nan,
                    }
                    if native_reader is not None and track == "cuda":
                        actual, actual_samples = measure_actual_io(
                            native_reader, args, mask, row_kib
                        )
                        for sample in actual_samples:
                            io_timing_rows.append({**base, **sample})
                    rows.append({
                        **base, **metrics, **actual,
                        "lookup_efficiency": metrics["importance"] / metrics["lookup_ms"],
                        "two_line_efficiency": metrics["importance"] / metrics["two_line_ms"],
                        "row_match": row_match,
                        "valid": row_match and not bool(error),
                        "runtime_median_ms": float(np.median(timings)),
                        "runtime_p95_ms": runtime_p95,
                        "deadline_met": runtime_p95 <= args.deadline_ms,
                        "valid_and_deadline_met": (
                            row_match and not bool(error) and runtime_p95 <= args.deadline_ms
                        ),
                        "deterministic": metadata.get("deterministic", False),
                        "scalarized_calls": metadata.get("scalarized_calls", 0),
                        "outer_iterations": metadata.get("outer_iterations", 0),
                        "mu_exact_hits": metadata.get("mu_exact_hits", 0),
                        "frontier_candidates": metadata.get("frontier_candidates", 0),
                        "eligible_candidates": metadata.get("eligible_candidates", 0),
                        "trim_work_deletions": metadata.get("trim_work_deletions", 0),
                        "minimum_overfill": metadata.get("minimum_overfill", 0),
                        "repair_deletions": metadata.get("repair_deletions", 0),
                        "repair_additions": metadata.get("repair_additions", 0),
                        "returned_top_r": metadata.get("returned_top_r", False),
                        "fallback_used": metadata.get("fallback_used", False),
                        "error": error,
                    })
                    for repetition, elapsed in enumerate(timings):
                        timing_rows.append({
                            **base, "repetition": repetition, "runtime_ms": elapsed,
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
            f"trace={trace_index + 1}/{len(traces)} id={trace['trace_id']} "
            f"shape={shape['shape']} module={trace['module']}", flush=True,
        )
    return rows, timing_rows, model_rows, io_timing_rows


def add_actual_io_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add paired real-read and selector+read metrics relative to Paper."""
    derived = [
        "paper_actual_read_wall_median_ms", "paper_actual_selector_median_ms",
        "paper_actual_total_median_ms",
        "actual_io_efficiency", "paper_actual_io_efficiency",
        "actual_io_efficiency_gain_pct", "actual_read_saving_ms",
        "actual_read_saving_pct", "actual_total_median_ms",
        "actual_total_saving_ms", "actual_total_win",
    ]
    frame = frame.drop(columns=[column for column in derived if column in frame])
    if "actual_read_wall_median_ms" not in frame:
        return frame
    frame["actual_io_efficiency"] = (
        frame.importance / frame.actual_read_wall_median_ms
    )
    keys = ["trace_id", "budget_index", "track"]
    paper = frame[frame.method == "paper"][keys + [
        "actual_read_wall_median_ms", "runtime_median_ms", "actual_io_efficiency",
    ]].rename(columns={
        "actual_read_wall_median_ms": "paper_actual_read_wall_median_ms",
        "runtime_median_ms": "paper_actual_selector_median_ms",
        "actual_io_efficiency": "paper_actual_io_efficiency",
    })
    output = frame.merge(paper, on=keys, how="left")
    output["actual_total_median_ms"] = (
        output.runtime_median_ms + output.actual_read_wall_median_ms
    )
    output["paper_actual_total_median_ms"] = (
        output.paper_actual_selector_median_ms
        + output.paper_actual_read_wall_median_ms
    )
    output["actual_io_efficiency_gain_pct"] = 100.0 * (
        output.actual_io_efficiency / output.paper_actual_io_efficiency - 1.0
    )
    output["actual_read_saving_ms"] = (
        output.paper_actual_read_wall_median_ms - output.actual_read_wall_median_ms
    )
    output["actual_read_saving_pct"] = 100.0 * (
        1.0
        - output.actual_read_wall_median_ms
        / output.paper_actual_read_wall_median_ms
    )
    output["actual_total_saving_ms"] = (
        output.paper_actual_total_median_ms - output.actual_total_median_ms
    )
    output["actual_total_win"] = output.actual_total_saving_ms > 0.0
    return output


def aggregate_group(group: pd.DataFrame) -> dict:
    base = EXP24.aggregate_group(group)
    output = {
        **base,
        "frontier_candidates_mean": float(group.frontier_candidates.mean()),
        "eligible_candidates_mean": float(group.eligible_candidates.mean()),
        "trim_work_deletions_mean": float(group.trim_work_deletions.mean()),
        "minimum_overfill_median": float(group.minimum_overfill.median()),
    }
    if "actual_read_wall_median_ms" in group:
        actual = group[group.actual_read_wall_median_ms.notna()]
        output.update({
            "actual_cases": int(len(actual)),
            "actual_direct_rate": (
                float(actual.actual_direct_rate.mean()) if len(actual) else math.nan
            ),
            "actual_io_median_ms": (
                float(actual.actual_io_median_ms.median()) if len(actual) else math.nan
            ),
            "actual_upload_median_ms": (
                float(actual.actual_upload_median_ms.median()) if len(actual) else math.nan
            ),
            "actual_read_wall_ms_mean": (
                float(actual.actual_read_wall_median_ms.mean()) if len(actual) else math.nan
            ),
            "actual_read_wall_ms_median": (
                float(actual.actual_read_wall_median_ms.median()) if len(actual) else math.nan
            ),
            "actual_read_wall_case_p95_ms": (
                float(actual.actual_read_wall_p95_ms.quantile(0.95))
                if len(actual) else math.nan
            ),
            "actual_io_efficiency_gain_pct_mean": (
                float(actual.actual_io_efficiency_gain_pct.mean())
                if len(actual) else math.nan
            ),
            "actual_read_saving_pct_mean": (
                float(actual.actual_read_saving_pct.mean()) if len(actual) else math.nan
            ),
            "actual_total_ms_mean": (
                float(actual.actual_total_median_ms.mean()) if len(actual) else math.nan
            ),
            "actual_total_ms_median": (
                float(actual.actual_total_median_ms.median()) if len(actual) else math.nan
            ),
            "actual_total_ms_p95": (
                float(actual.actual_total_median_ms.quantile(0.95))
                if len(actual) else math.nan
            ),
            "actual_total_saving_ms_mean": (
                float(actual.actual_total_saving_ms.mean()) if len(actual) else math.nan
            ),
            "actual_total_win_rate": (
                float(actual.actual_total_win.mean()) if len(actual) else math.nan
            ),
        })
    return output


def summarize(frame: pd.DataFrame, oracle: pd.DataFrame | None,
              args: argparse.Namespace):
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    trace_metadata = metadata.get("activation_trace", {})
    summary_rows, nested = [], {}
    for (track, method), group in frame.groupby(["track", "method"], sort=False):
        row = {
            "track": track, "method": method, "method_label": METHOD_LABELS[method],
            **aggregate_group(group),
        }
        summary_rows.append(row)
        nested.setdefault(track, {})[method] = row
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
                "optimality_ratio_min": float(group.optimality_ratio.min()),
                "optimality_gap_pct_p95": float(group.optimality_gap_pct.quantile(0.95)),
                "exact_hit_rate": float((group.optimality_gap_pct <= 1e-9).mean()),
            }
    summary = {
        "format": "experiment-26-frontier-adaptive-trim-v1-real-activations",
        "comparison": "fixed-R I/L frontier search without Paper importance or mask input",
        "deadline_ms": float(args.deadline_ms),
        "timing_scope": {
            "host": "warm host importance to CPU bool mask",
            "cuda": "warm CUDA importance through D2H CPU solve and H2D bool mask",
            "actual_io": (
                "native O_DIRECT mask replay including read-call overhead and GPU upload"
                if args.measure_io else "not measured"
            ),
            "excluded": "real LM trace forward and model compute",
        },
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": metadata.get("torch", torch.__version__),
            "torch_cuda_build": metadata.get("torch_cuda_build", torch.version.cuda),
            "gpu": metadata.get("gpu"),
        },
        "configuration": {
            "shape_count": int(frame["shape"].nunique()),
            "shapes": list(dict.fromkeys(frame["shape"])),
            "activation_trace_count": int(frame.trace_id.nunique()),
            "input_count": int(frame.input_index.nunique()),
            "model": trace_metadata.get("model", args.model),
            "trace_input": metadata.get("trace_file", str(args.trace_input)),
            "max_traces_per_shape": int(args.max_traces_per_shape),
            "repetitions": int(args.repetitions),
            "row_budget_fractions": list(args.row_budget_fractions),
            "tracks": list(dict.fromkeys(frame.track)),
            "profile": args.profile, "saturation_kib": float(args.saturation_kib),
            "frontier_settings": FRONTIER_SETTINGS,
            "actual_io": {
                "enabled": bool(args.measure_io),
                "blob": str(args.io_blob) if args.io_blob else None,
                "repetitions": int(args.io_repetitions),
                "warmup": int(args.io_warmup),
                "threads": int(args.io_threads),
                "max_read_kib": int(args.io_max_read_kib),
                "device": args.io_device,
            },
        },
        "tracks": nested,
        "small_n_oracle": oracle_summary,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(shape_rows), summary


def configure_plot():
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_overall(summary: pd.DataFrame, path: Path, deadline_ms: float) -> None:
    plt = configure_plot()
    tracks = list(dict.fromkeys(summary.track))
    fig, axes = plt.subplots(1, len(tracks), figsize=(7.2 * len(tracks), 5.5),
                             squeeze=False, constrained_layout=True)
    for ax, track in zip(axes[0], tracks):
        selected = summary[summary.track == track].set_index("method").reindex(METHODS)
        for method, row in selected.iterrows():
            ax.scatter(
                row.runtime_median_ms, row.lookup_efficiency_gain_pct_mean_valid,
                s=85, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
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


def plot_by_shape(shape_summary: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    primary = "cuda" if "cuda" in set(shape_summary.track) else "host"
    selected = shape_summary[shape_summary.track == primary]
    order = (
        selected[selected.method == "paper"].sort_values(["n", "d"])["shape"].tolist()
    )
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 9.0), constrained_layout=True)
    for method in CANDIDATE_METHODS:
        data = selected[selected.method == method].set_index("shape").reindex(order)
        axes[0].plot(
            x, data.lookup_efficiency_gain_pct_mean_valid, marker="o",
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
        axes[1].plot(
            x, data.runtime_case_p95_ms, marker="o",
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    axes[0].axhline(0.0, color="#64748B", linestyle=":")
    axes[1].axhline(2.0, color="#111827", linestyle="--")
    axes[0].set_ylabel("Mean lookup I/L gain vs Paper (%)")
    axes[1].set_ylabel("95th percentile of case p95 (ms)")
    axes[0].set_title(f"{primary}: fixed-R quality by shape")
    axes[1].set_title(f"{primary}: selector latency by shape")
    for ax in axes:
        ax.set_xticks(x, order, rotation=45, ha="right")
        ax.legend(frameon=False, ncol=2)
        BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_oracle(oracle: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    methods = METHODS[1:]
    ax.boxplot(
        [100.0 * oracle.loc[oracle.method == method, "optimality_ratio"] for method in methods],
        tick_labels=[METHOD_LABELS[method] for method in methods], showfliers=True,
    )
    ax.axhline(100.0, color="#111827", linestyle="--")
    ax.set_ylabel("Exact fixed-R optimum recovered (%)")
    ax.tick_params(axis="x", rotation=15)
    BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_actual_total(summary: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    primary = "cuda" if "cuda" in set(summary.track) else "host"
    selected = summary[
        (summary.track == primary) & summary.actual_total_ms_mean.notna()
    ].set_index("method").reindex(METHODS)
    labels = [METHOD_LABELS[method] for method in selected.index]
    y = np.arange(len(selected))
    selector = selected.actual_total_ms_mean - selected.actual_read_wall_ms_mean
    fig, ax = plt.subplots(figsize=(11.5, 6.3), constrained_layout=True)
    ax.barh(y, selector, color="#7C3AED", label="Selector")
    ax.barh(
        y, selected.actual_read_wall_ms_mean, left=selector,
        color="#0F766E", label="O_DIRECT read + GPU upload",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean measured time per case (ms)")
    ax.set_title("Laptop-native selector + flash-read total")
    ax.legend(frameon=False)
    BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, shape_summary: pd.DataFrame,
                 oracle: pd.DataFrame | None, args: argparse.Namespace,
                 trials: pd.DataFrame | None = None) -> None:
    report_path = args.report_output or (HERE / "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    image_prefix = Path(os.path.relpath(args.output_dir, report_path.parent)).as_posix()
    primary_track = "cuda" if "cuda" in set(summary.track) else "host"
    indexed = summary[summary.track == primary_track].set_index("method").reindex(METHODS)
    winner_method = indexed.loc[list(FRONTIER_METHODS)].sort_values(
        "lookup_efficiency_gain_pct_mean_valid", ascending=False
    ).index[0]
    winner = indexed.loc[winner_method]
    legacy = indexed.loc["legacy_rho2_mu4"]
    lines = [
        "# Experiment 26 보고서: Frontier + Adaptive Fixed-R Trim", "",
        f"`{args.model}`의 실제 `vlm-flash` activation trace에서 fixed-R `I/L`을 더 "
        "충실하게 탐색했다. selector는 importance vector, R, latency model만 사용하며 "
        "Paper mask나 `I(M_paper)`를 입력으로 사용하지 않는다. Paper는 paired 평가에서 "
        "R을 맞추고 품질을 비교하는 기준선일 뿐이다.", "",
        "각 μ에서 나온 모든 서로 다른 eligible cardinality를 exact-R로 adaptive trim한 뒤 "
        "실제 I/L이 가장 높은 해를 보존한다. trim은 매 삭제마다 현재 비율을 다시 계산한다.",
        "", "## 전체 결과", "",
    ]
    for track, label in (("host", "Host input -> CPU mask"),
                         ("cuda", "CUDA input -> CUDA mask")):
        data = summary[summary.track == track]
        if data.empty:
            continue
        lines.extend([
            f"### {label}", "",
            "| 방법 | 중앙 runtime | case-p95 | 유효+2ms | lookup I/L | two-line I/L | importance | 가중 lookup 절감 | Top-R 반환 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method, row in data.set_index("method").reindex(METHODS).iterrows():
            lines.append(
                f"| {METHOD_LABELS[method]} | {row.runtime_median_ms:.3f} ms "
                f"| {row.runtime_case_p95_ms:.3f} ms "
                f"| {100 * row.valid_and_deadline_pass_rate:.1f}% "
                f"| {row.lookup_efficiency_gain_pct_mean_valid:.2f}% "
                f"| {row.two_line_efficiency_gain_pct_mean_valid:.2f}% "
                f"| {row.importance_gain_pct_mean_valid:.2f}% "
                f"| {row.lookup_saving_pct_weighted_valid:.2f}% "
                f"| {100 * row.returned_top_r_rate:.1f}% |"
            )
        lines.append("")

    if "actual_total_ms_mean" in indexed and indexed.actual_total_ms_mean.notna().any():
        actual = indexed[indexed.actual_total_ms_mean.notna()]
        fastest_method = actual.actual_total_ms_mean.idxmin()
        lines.extend([
            "## 노트북 실제 O_DIRECT 총시간", "",
            "각 CUDA-track mask를 target SSD에서 직접 읽고 GPU로 upload했다. 아래 총시간은 "
            "case별 `selector median + native read wall median`의 평균이며 LM compute는 제외한다.",
            "", "| 방법 | 실제 read wall | selector+read | Paper 대비 절감 | Paper보다 빠른 case |",
            "|---|---:|---:|---:|---:|",
        ])
        for method, row in actual.reindex(METHODS).iterrows():
            win = "-" if method == "paper" else f"{100 * row.actual_total_win_rate:.1f}%"
            lines.append(
                f"| {METHOD_LABELS[method]} | {row.actual_read_wall_ms_mean:.3f} ms "
                f"| {row.actual_total_ms_mean:.3f} ms "
                f"| {row.actual_total_saving_ms_mean:+.3f} ms "
                f"| {win} |"
            )
        lines.extend([
            "",
            f"직접 측정한 총시간이 가장 낮은 방법은 `{METHOD_LABELS[fastest_method]}`로 "
            f"평균 `{actual.loc[fastest_method, 'actual_total_ms_mean']:.3f} ms`다.",
            "",
        ])
        if trials is not None:
            measured = trials[trials.actual_io_median_ms.notna()]
            io_ratio = measured.actual_io_median_ms / measured.lookup_ms
            wall_ratio = measured.actual_read_wall_median_ms / measured.lookup_ms
            lines.extend([
                f"Profile 예측과 native I/O-only 실측의 Pearson 상관은 "
                f"`r={measured.actual_io_median_ms.corr(measured.lookup_ms):.4f}`지만, "
                f"실측/예측 비율 중앙값은 `{io_ratio.median():.3f}x`다. GPU upload와 "
                f"reader 호출 전체를 포함한 wall-clock은 `r="
                f"{measured.actual_read_wall_median_ms.corr(measured.lookup_ms):.4f}`, "
                f"중앙값 `{wall_ratio.median():.3f}x`다. 따라서 table은 mask 순위에는 "
                "유용하지만 노트북의 절대 총시간을 대신하지 못한다.", "",
                "### Shape별 실제 총시간", "",
                "| Shape | Paper | Exp24 rho2/mu4 | 가장 빠른 frontier | 해당 방법 |",
                "|---|---:|---:|---:|---|",
            ])
            actual_shapes = shape_summary[
                (shape_summary.track == primary_track)
                & shape_summary.actual_total_ms_mean.notna()
            ]
            for shape in actual_shapes[actual_shapes.method == "paper"]["shape"]:
                indexed_shape = actual_shapes[
                    actual_shapes["shape"] == shape
                ].set_index("method")
                frontier_method = indexed_shape.loc[list(FRONTIER_METHODS)][
                    "actual_total_ms_mean"
                ].idxmin()
                lines.append(
                    f"| {shape} | {indexed_shape.loc['paper', 'actual_total_ms_mean']:.3f} ms "
                    f"| {indexed_shape.loc['legacy_rho2_mu4', 'actual_total_ms_mean']:.3f} ms "
                    f"| {indexed_shape.loc[frontier_method, 'actual_total_ms_mean']:.3f} ms "
                    f"| {METHOD_LABELS[frontier_method]} |"
                )
            lines.append("")

    lines.extend(["## Small-N exact oracle", ""])
    if oracle is None or not len(oracle):
        lines.extend(["Oracle 실행을 생략했다.", ""])
    else:
        lines.extend([
            f"별도 synthetic `N={args.oracle_n}` exhaustive optimum과 비교했다.", "",
            "| 방법 | 평균 optimum 회수 | 최악 회수 | p95 gap | exact hit |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in METHODS[1:]:
            group = oracle[oracle.method == method]
            lines.append(
                f"| {METHOD_LABELS[method]} | {100 * group.optimality_ratio.mean():.3f}% "
                f"| {100 * group.optimality_ratio.min():.3f}% "
                f"| {group.optimality_gap_pct.quantile(0.95):.3f}% "
                f"| {100 * (group.optimality_gap_pct <= 1e-9).mean():.1f}% |"
            )
        lines.append("")

    winner_shapes = shape_summary[
        (shape_summary.track == primary_track) & (shape_summary.method == winner_method)
    ].sort_values(["n", "d"])
    lines.extend([
        "## 판정", "",
        f"새 adaptive-trim 후보 중 실제 trace의 평균 lookup I/L이 가장 높은 설정은 "
        f"`{METHOD_LABELS[winner_method]}`다. "
        f"Paper 대비 lookup I/L `{winner.lookup_efficiency_gain_pct_mean_valid:.2f}%`, "
        f"two-line I/L `{winner.two_line_efficiency_gain_pct_mean_valid:.2f}%`, importance "
        f"`{winner.importance_gain_pct_mean_valid:.2f}%`다.", "",
        f"Experiment 24 기준선 `{METHOD_LABELS['legacy_rho2_mu4']}` 대비 lookup I/L 평균 변화는 "
        f"`{winner.lookup_efficiency_gain_pct_mean_valid - legacy.lookup_efficiency_gain_pct_mean_valid:+.2f}` "
        f"percentage points이고, `{primary_track}` 2ms 통과율은 "
        f"`{100 * winner.valid_and_deadline_pass_rate:.1f}%`다.", "",
        f"따라서 새 frontier/trim 후보는 기존 24의 평균 I/L "
        f"`{legacy.lookup_efficiency_gain_pct_mean_valid:.2f}%`를 넘지 못했다. 대신 새 후보의 "
        f"importance 변화는 `{winner.importance_gain_pct_mean_valid:.2f}%`로 기존 기준선의 "
        f"`{legacy.importance_gain_pct_mean_valid:.2f}%`보다 높아 다른 trade-off를 만들었다.", "",
        f"선택된 해가 Top-R로 되돌아간 비율은 `{100 * winner.returned_top_r_rate:.1f}%`, "
        f"평균 eligible frontier 후보 수는 `{winner.eligible_candidates_mean:.2f}`개, "
        f"rho 반복에서 선택된 trim 삭제 수의 합은 평균 "
        f"`{winner.repair_deletions_mean:.1f}`개다.", "",
        f"## Shape별 {METHOD_LABELS[winner_method]} ({primary_track})", "",
        "| Shape | lookup I/L | two-line I/L | importance | 가중 lookup 절감 | case-p95 | 유효+2ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in winner_shapes.iterrows():
        lines.append(
            f"| {row['shape']} | {row.lookup_efficiency_gain_pct_mean_valid:.2f}% "
            f"| {row.two_line_efficiency_gain_pct_mean_valid:.2f}% "
            f"| {row.importance_gain_pct_mean_valid:.2f}% "
            f"| {row.lookup_saving_pct_weighted_valid:.2f}% "
            f"| {row.runtime_case_p95_ms:.3f} ms "
            f"| {100 * row.valid_and_deadline_pass_rate:.1f}% |"
        )
    lines.extend([
        "", "## 측정 한계", "",
        f"- 실제 activation은 `{args.model}`의 세 짧은 text prompt와 한 모델에서 얻었다.",
        "- small-N oracle만 synthetic이며 실제 trace 결과와 분리했다.",
        f"- mask 선택의 lookup 목적은 `{args.profile}` table을 사용했다.",
        (
            "- 별도 실제 read 결과는 native O_DIRECT와 GPU upload를 직접 측정했다."
            if args.measure_io else
            "- 실제 NVMe I/O는 실행하지 않았고 lookup latency는 table 예측값이다."
        ),
        "- adaptive endpoint trim은 interior deletion이나 cardinality-preserving swap을 탐색하지 않는다.",
        "- μ scalarization이 unsupported exact-R 해를 건너뛸 수 있어 전역 최적 보장은 없다.",
        "", f"![Runtime-quality]({image_prefix}/runtime_quality.png)", "",
    ])
    if args.measure_io:
        lines.extend([f"![Measured total]({image_prefix}/actual_total.png)", ""])
    lines.extend([f"![Shape comparison]({image_prefix}/shape_comparison.png)", ""])
    if oracle is not None and len(oracle):
        lines.extend([f"![Small-N oracle]({image_prefix}/oracle_optimality.png)", ""])
    report_path.write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    if "trace_id" not in frame:
        raise SystemExit("Experiment 26 requires real-model activation results")
    frame = EXP24.add_paired_metrics(frame)
    frame = add_actual_io_metrics(frame)
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    oracle_path = args.output_dir / "oracle_trials.csv"
    oracle = pd.read_csv(oracle_path) if oracle_path.exists() else None
    summary_frame, shape_frame, summary = summarize(frame, oracle, args)
    measured = frame[frame.actual_io_median_ms.notna()]
    if len(measured):
        summary["actual_io_replay"] = {
            "cases": int(len(measured)),
            "all_direct": bool((measured.actual_direct_rate == 1.0).all()),
            "profile_vs_io_pearson_r": float(
                measured.lookup_ms.corr(measured.actual_io_median_ms)
            ),
            "io_over_profile_ratio_median": float(
                (measured.actual_io_median_ms / measured.lookup_ms).median()
            ),
            "profile_vs_read_wall_pearson_r": float(
                measured.lookup_ms.corr(measured.actual_read_wall_median_ms)
            ),
            "read_wall_over_profile_ratio_median": float(
                (measured.actual_read_wall_median_ms / measured.lookup_ms).median()
            ),
        }
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    plot_by_shape(shape_frame, args.output_dir / "shape_comparison.png")
    if "actual_total_ms_mean" in summary_frame and summary_frame.actual_total_ms_mean.notna().any():
        plot_actual_total(summary_frame, args.output_dir / "actual_total.png")
    if oracle is not None and len(oracle):
        plot_oracle(oracle, args.output_dir / "oracle_optimality.png")
    write_report(summary_frame, shape_frame, oracle, args, frame)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    _, tracks = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return

    traces, trace_metadata, trace_path = EXP24.prepare_activation_traces(args)
    shape_names = list(dict.fromkeys(trace["shape"] for trace in traces))
    shape_specs = [
        dict(next(item for item in SHAPES if item["shape"] == name)) for name in shape_names
    ]
    try:
        recorded_trace_path = str(trace_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        recorded_trace_path = str(trace_path.resolve())
    lookup_table = BASE.LatencyTable.load(args.profile)
    profile_path = Path(args.profile)
    profile_record = {
        "requested": str(args.profile),
        "max_kib": int(lookup_table.max_kb),
        "metadata": lookup_table.meta,
        "sha256": (
            EXP22.sha256_file(profile_path) if profile_path.is_file() else None
        ),
    }
    metadata = {
        "activation_trace": trace_metadata,
        "trace_file": recorded_trace_path,
        "trace_sha256": EXP22.sha256_file(trace_path),
        "captured_trace_count": int(trace_metadata["trace_count"]),
        "selected_trace_count": len(traces),
        "selected_trace_ids": [int(trace["trace_id"]) for trace in traces],
        "max_traces_per_shape": int(args.max_traces_per_shape),
        "shape_specs": shape_specs,
        "method_labels": METHOD_LABELS,
        "frontier_settings": FRONTIER_SETTINGS,
        "latency_profile": profile_record,
        "actual_io": {
            "enabled": bool(args.measure_io),
            "blob": str(args.io_blob) if args.io_blob else None,
            "blob_sha256": (
                EXP22.sha256_file(args.io_blob) if args.measure_io else None
            ),
            "repetitions": int(args.io_repetitions),
            "warmup": int(args.io_warmup),
            "threads": int(args.io_threads),
            "max_read_kib": int(args.io_max_read_kib),
            "device": args.io_device,
        },
        "requested_tracks": args.tracks, "executed_tracks": tracks,
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
        if args.io_blob.stat().st_size < required_bytes:
            raise SystemExit(
                f"--io-blob has {args.io_blob.stat().st_size} bytes; "
                f"at least {required_bytes} are required"
            )
        EXP22.load_vlmflash()
        from vlmflash._native import native, unavailable_reason
        native_reader = native()
        if native_reader is None:
            raise SystemExit(
                f"actual I/O replay needs the native reader: {unavailable_reason()}"
            )
    if not args.skip_self_check:
        self_check(lookup_table, args.saturation_kib)
    if not args.skip_oracle:
        EXP24.write_csv(args.output_dir / "oracle_trials.csv", collect_oracle(args, lookup_table))
    rows, timings, models, io_timings = collect(
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
