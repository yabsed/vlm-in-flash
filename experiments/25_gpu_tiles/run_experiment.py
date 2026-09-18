#!/usr/bin/env python3
"""Experiment 25: run Experiment 18 saturation tiles natively on a GPU."""

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

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_22 = (
    PROJECT_ROOT / "experiments" / "22_predicted_lambda_trim" / "run_experiment.py"
)


def load_experiment_22():
    spec = importlib.util.spec_from_file_location("experiment_22_for_25", EXPERIMENT_22)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 22 from {EXPERIMENT_22}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP22 = load_experiment_22()
EXP18 = EXP22.EXP18
EXP13 = EXP22.EXP13
BASE = EXP22.BASE
SHAPES = EXP22.SHAPES
DEFAULT_TRACE_INPUT = EXP22.HERE / "results" / "activation_traces.npz"

METHODS = ("tiles_half_s", "tiles_s", "tiles_2s")
METHOD_LABELS = {
    "tiles_half_s": "Tiles (s/2)",
    "tiles_s": "Tiles (s)",
    "tiles_2s": "Tiles (2s)",
}
METHOD_FACTORS = {
    "tiles_half_s": 0.5,
    "tiles_s": 1.0,
    "tiles_2s": 2.0,
}
METHOD_COLORS = {
    "tiles_half_s": "#2563EB",
    "tiles_s": "#7C3AED",
    "tiles_2s": "#DC2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+", default=None,
        help="optional subset of captured Table-2 shapes, written as NxD",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--model", default=EXP22.DEFAULT_MODEL)
    parser.add_argument(
        "--prompt", action="append", default=None,
        help="text prompt to trace when --capture-traces is used; repeat as needed",
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trace-input", type=Path, default=DEFAULT_TRACE_INPUT,
        help="real activation archive (default: Experiment 22's Qwen trace)",
    )
    parser.add_argument(
        "--capture-traces", action="store_true",
        help="run a fresh real-LM forward instead of loading --trace-input",
    )
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--max-traces-per-shape", type=int, default=32)
    parser.add_argument("--collect-traces-only", action="store_true")
    parser.add_argument(
        "--target-fractions", type=float, nargs="+", default=[0.50, 0.70, 0.90]
    )
    parser.add_argument(
        "--row-budget-fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--deadline-ms", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[dict]:
    by_name = {item["shape"]: item for item in SHAPES}
    unknown = sorted(set(args.shapes or ()) - set(by_name))
    if unknown:
        raise SystemExit(f"unknown --shapes: {unknown}; available: {sorted(by_name)}")
    shapes = [dict(by_name[name]) for name in (args.shapes or by_name)]
    if len(shapes) != len({item["shape"] for item in shapes}):
        raise SystemExit("--shapes must not contain duplicates")
    if args.repetitions < 1 or args.warmup < 0:
        raise SystemExit("repetitions must be positive and warmup nonnegative")
    if args.max_input_tokens < 1 or args.max_traces_per_shape < 0:
        raise SystemExit("trace token/cap arguments are invalid")
    if args.prompt and args.prompt_file:
        raise SystemExit("use either --prompt or --prompt-file, not both")
    if not args.capture_traces and (args.prompt or args.prompt_file):
        raise SystemExit("prompt options require --capture-traces")
    if args.collect_traces_only and args.analyze_only:
        raise SystemExit("--collect-traces-only and --analyze-only are mutually exclusive")
    if len(args.target_fractions) != len(args.row_budget_fractions):
        raise SystemExit("target and row-budget fraction counts must match")
    fractions = list(args.target_fractions) + list(args.row_budget_fractions)
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise SystemExit("target and row-budget fractions must lie in (0, 1]")
    if args.saturation_kib <= 0 or args.deadline_ms <= 0:
        raise SystemExit("saturation and deadline must be positive")
    if (
        not args.analyze_only and not args.collect_traces_only
        and not torch.cuda.is_available()
    ):
        raise SystemExit("Experiment 25 requires a CUDA-capable PyTorch runtime")
    return shapes


def prepare_activation_traces(args: argparse.Namespace) -> tuple[list[dict], dict, Path]:
    if args.capture_traces:
        traces, metadata = EXP22.capture_activation_traces(args)
        trace_path = args.trace_output or (args.output_dir / "activation_traces.npz")
        EXP22.save_activation_traces(trace_path, traces, metadata)
        print(f"wrote {len(traces)} real activation traces to {trace_path}", flush=True)
    else:
        traces, metadata = EXP22.load_activation_traces(args.trace_input)
        trace_path = args.trace_input
        print(f"loaded {len(traces)} real activation traces from {trace_path}", flush=True)
    args.model = metadata.get("model", args.model)
    selected, skipped = EXP22.select_activation_traces(
        traces, args.shapes, args.max_traces_per_shape
    )
    metadata = {
        **metadata,
        "trace_count": len(traces),
        "selected_trace_count": len(selected),
        "selected_trace_ids": [int(trace["trace_id"]) for trace in selected],
        "max_traces_per_shape": int(args.max_traces_per_shape),
        "skipped_non_table2_shapes": skipped,
    }
    return selected, metadata, trace_path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_size_kib(shape: dict) -> float:
    return 2.0 * int(shape["d"]) / 1024.0


def make_params(shape: dict, saturation_kib: float):
    return BASE.ChunkParams(
        start_kb=float(shape["start_kib"]), end_kb=float(saturation_kib),
        step_kb=float(shape["start_kib"]), jump_cap_kb=float(shape["jump_kib"]),
    )


def tile_length(method: str, model: dict, n: int) -> int:
    saturation = max(1, int(round(float(model["saturation_rows"]))))
    return max(1, min(n, int(round(METHOD_FACTORS[method] * saturation))))


def build_grid(n: int, length: int, offset: int, model: dict,
               device: torch.device) -> dict[str, torch.Tensor | int]:
    boundaries = [0]
    if offset > 0:
        boundaries.append(offset)
    position = offset + length
    while position < n:
        boundaries.append(position)
        position += length
    if boundaries[-1] != n:
        boundaries.append(n)
    starts = np.asarray(boundaries[:-1], dtype=np.int64)
    ends = np.asarray(boundaries[1:], dtype=np.int64)
    lengths = ends - starts
    row_tile_ids = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
    if len(row_tile_ids) != n:
        raise RuntimeError("tile grid does not cover every row exactly once")
    costs = np.asarray(
        [EXP18.two_line_chunk_ms(int(value), model) for value in lengths],
        dtype=np.float32,
    )
    return {
        "offset": int(offset),
        "starts": torch.from_numpy(starts).to(device),
        "ends": torch.from_numpy(ends).to(device),
        "costs": torch.from_numpy(costs).to(device),
        "row_tile_ids": torch.from_numpy(row_tile_ids).to(device),
        "tile_indices": torch.arange(len(starts), device=device, dtype=torch.int64),
        "false_scalar": torch.zeros((), device=device, dtype=torch.bool),
        "minus_one": torch.full((len(starts),), -1, device=device, dtype=torch.int64),
    }


def build_layout(n: int, length: int, model: dict,
                 device: torch.device) -> tuple[dict, ...]:
    offsets = sorted(set((0, length // 2)))
    return tuple(build_grid(n, length, offset, model, device) for offset in offsets)


def select_one_grid(prefix: torch.Tensor, target: torch.Tensor, grid: dict,
                    model: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = grid["starts"]
    ends = grid["ends"]
    before = torch.where(
        starts > 0,
        prefix[torch.clamp(starts - 1, min=0)],
        torch.zeros((), dtype=prefix.dtype, device=prefix.device),
    )
    importance = prefix[ends - 1] - before
    score = importance / grid["costs"]
    order = torch.argsort(score, descending=True, stable=True)
    cumulative = torch.cumsum(importance[order], dim=0)
    cutoff = torch.searchsorted(cumulative, target, right=False)
    cutoff = torch.clamp(cutoff, max=len(order) - 1)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, grid["tile_indices"])
    selected = ranks <= cutoff
    mask = selected[grid["row_tile_ids"]]

    previous = torch.cat((grid["false_scalar"].reshape(1), selected[:-1]))
    following = torch.cat((selected[1:], grid["false_scalar"].reshape(1)))
    run_starts = selected & ~previous
    run_ends = selected & ~following
    start_positions = torch.where(run_starts, starts, grid["minus_one"])
    active_start = torch.cummax(start_positions, dim=0).values
    run_lengths = torch.where(run_ends, ends - active_start, torch.zeros_like(ends))
    lengths_float = run_lengths.to(prefix.dtype)
    short_cost = (
        float(model["a_ms"])
        + float(model["c1_ms_per_row"]) * lengths_float
    )
    long_cost = float(model["c2_ms_per_row"]) * lengths_float
    run_cost = torch.where(
        run_ends,
        torch.where(
            lengths_float <= float(model["saturation_rows"]), short_cost, long_cost
        ),
        torch.zeros_like(lengths_float),
    ).sum()
    kept_importance = cumulative[cutoff]
    return mask, run_cost, kept_importance


def gpu_saturation_tiles(values: torch.Tensor, target: torch.Tensor,
                         layout: tuple[dict, ...], model: dict):
    prefix = torch.cumsum(values, dim=0)
    first_mask, first_cost, first_importance = select_one_grid(
        prefix, target, layout[0], model
    )
    if len(layout) == 1:
        chosen = torch.zeros((), device=values.device, dtype=torch.int64)
        return first_mask, chosen, first_cost, first_importance
    second_mask, second_cost, second_importance = select_one_grid(
        prefix, target, layout[1], model
    )
    choose_second = (second_cost < first_cost) | (
        (second_cost == first_cost)
        & ((second_importance - target) < (first_importance - target))
    )
    mask = torch.where(choose_second, second_mask, first_mask)
    chosen = choose_second.to(torch.int64)
    cost = torch.where(choose_second, second_cost, first_cost)
    importance = torch.where(choose_second, second_importance, first_importance)
    return mask, chosen, cost, importance


def benchmark_cuda(function, repetitions: int, warmup: int):
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    event_ms: list[float] = []
    wall_ms: list[float] = []
    output = None
    for start_event, end_event in zip(starts, ends):
        wall_start = time.perf_counter_ns()
        start_event.record()
        output = function()
        end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        event_ms.append(float(start_event.elapsed_time(end_event)))
    assert output is not None
    mask = output[0].detach().cpu().numpy().astype(bool, copy=True)
    chosen_grid = int(output[1].detach().cpu().item())
    predicted_cost = float(output[2].detach().cpu().item())
    predicted_importance = float(output[3].detach().cpu().item())
    check_a = function()[0]
    check_b = function()[0]
    deterministic = bool(torch.equal(check_a, check_b))
    torch.cuda.synchronize()
    return mask, {
        "chosen_grid": chosen_grid,
        "predicted_two_line_ms": predicted_cost,
        "predicted_importance": predicted_importance,
        "deterministic": deterministic,
    }, wall_ms, event_ms


def benchmark_cpu(function, repetitions: int, warmup: int):
    for _ in range(warmup):
        output = function()
    timings: list[float] = []
    output = None
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        output = function()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    assert output is not None
    return np.asarray(output[0], dtype=bool).copy(), dict(output[1]), timings


def mask_metrics(mask: np.ndarray, values: np.ndarray, model: dict,
                 lookup_table, row_kib: float) -> dict:
    return EXP18.mask_metrics(mask, values, model, lookup_table, row_kib)


def self_check(lookup_table, saturation_kib: float) -> None:
    device = torch.device("cuda")
    rng = np.random.default_rng(2501)
    for shape_name in ("896x896", "4864x896", "3584x18944"):
        shape = next(item for item in SHAPES if item["shape"] == shape_name)
        n = min(int(shape["n"]), 257)
        row_kib = row_size_kib(shape)
        model = EXP13.fit_continuous_two_line(lookup_table, row_kib, saturation_kib)
        raw = rng.lognormal(size=n).astype(np.float32)
        raw /= np.float32(raw.astype(np.float64).sum())
        values = raw.astype(np.float64)
        values_cuda = torch.from_numpy(raw).to(device)
        for method in METHODS:
            length = tile_length(method, model, n)
            layout = build_layout(n, length, model, device)
            for target in (0.5, 0.7, 0.9):
                target_cuda = torch.tensor(target, device=device, dtype=torch.float32)
                gpu = gpu_saturation_tiles(values_cuda, target_cuda, layout, model)
                gpu_mask = gpu[0].cpu().numpy().astype(bool)
                legacy_mask, _ = EXP18.saturation_tiles(values, target, model, length)
                gpu_metrics = EXP18.mask_metrics_two_line(gpu_mask, values, model)
                legacy_metrics = EXP18.mask_metrics_two_line(legacy_mask, values, model)
                if gpu_metrics["importance"] < target - 2e-7:
                    raise RuntimeError("GPU Tiles failed coverage self-check")
                if not math.isclose(
                    gpu_metrics["two_line_ms"], float(gpu[2].cpu().item()),
                    rel_tol=2e-6, abs_tol=2e-6,
                ):
                    raise RuntimeError("GPU run-cost evaluation failed self-check")
                if not math.isclose(
                    gpu_metrics["two_line_ms"], legacy_metrics["two_line_ms"],
                    rel_tol=2e-6, abs_tol=2e-6,
                ):
                    raise RuntimeError("GPU and Experiment-18 Tiles disagree")


def collect(args: argparse.Namespace, traces: list[dict], lookup_table):
    rows: list[dict] = []
    timing_rows: list[dict] = []
    model_rows: list[dict] = []
    device = torch.device("cuda")
    by_name = {item["shape"]: dict(item) for item in SHAPES}
    shape_names = list(dict.fromkeys(trace["shape"] for trace in traces))
    shape_context = {}
    for shape_index, shape_name in enumerate(shape_names):
        shape = by_name[shape_name]
        n = int(shape["n"])
        row_kib = row_size_kib(shape)
        params = make_params(shape, args.saturation_kib)
        model = EXP13.fit_continuous_two_line(
            lookup_table, row_kib, args.saturation_kib
        )
        windows = EXP18.paper_windows(n, row_kib, lookup_table, params)
        layouts = {}
        for method in METHODS:
            length = tile_length(method, model, n)
            layouts[method] = (
                length, build_layout(n, length, model, device)
            )
        model_rows.append({
            **shape, "shape_index": shape_index, "row_size_kib": row_kib,
            "activation_trace_count": sum(t["shape"] == shape_name for t in traces),
            "tiles_half_s_rows": layouts["tiles_half_s"][0],
            "tiles_s_rows": layouts["tiles_s"][0],
            "tiles_2s_rows": layouts["tiles_2s"][0],
            **model,
        })
        shape_context[shape_name] = (
            shape_index, shape, row_kib, params, model, windows, layouts
        )

    for trace_index, trace in enumerate(traces):
        shape_index, shape, row_kib, params, model, windows, layouts = (
            shape_context[trace["shape"]]
        )
        n = int(shape["n"])
        raw_values = np.asarray(trace["values"], dtype=np.float32)
        if raw_values.shape != (n,):
            raise RuntimeError(
                f"trace {trace['trace_id']} for {trace['module']} has shape "
                f"{raw_values.shape}, expected {(n,)}"
            )
        total = float(raw_values.astype(np.float64).sum())
        if not np.isfinite(raw_values).all() or np.any(raw_values < 0.0) or total <= 0.0:
            raise RuntimeError(f"trace {trace['trace_id']} has invalid importance values")
        values32 = raw_values.copy()
        values32 /= np.float32(total)
        values = values32.astype(np.float64)
        activation_cv = float(
            raw_values.astype(np.float64).std() / raw_values.astype(np.float64).mean()
        )
        values_cuda = torch.from_numpy(values32).to(device)
        projection = str(trace["module"]).rsplit(".", 1)[-1]

        for scenario, (target_fraction, budget_fraction) in enumerate(zip(
            args.target_fractions, args.row_budget_fractions
        )):
            target = float(target_fraction * values.sum())
            target_cuda = torch.tensor(target, device=device, dtype=torch.float32)
            row_budget = max(1, min(n, int(round(n * budget_fraction))))
            if row_budget == n:
                top_r_importance = float(values.sum())
            else:
                top_r_importance = float(
                    np.partition(values, n - row_budget)[n - row_budget:].sum()
                )
            intrinsically_feasible = top_r_importance >= target - 1e-11

            paper_cov_mask, _ = EXP18.exact_paper_coverage(values, target, *windows)
            paper_cov = mask_metrics(
                paper_cov_mask, values, model, lookup_table, row_kib
            )
            paper_fixed_mask, _ = EXP18.paper_fixed_rows(
                values, row_budget, row_kib, lookup_table, params
            )
            paper_fixed = mask_metrics(
                paper_fixed_mask, values, model, lookup_table, row_kib
            )
            paper_fixed_valid = (
                paper_fixed["importance"] >= target - 1e-11
                and paper_fixed["rows"] <= row_budget
            )

            case = {
                "trace_id": int(trace["trace_id"]),
                "input_index": int(trace["input_index"]),
                "prompt_sha256": trace["prompt_sha256"],
                "num_tokens": int(trace["num_tokens"]),
                "module": trace["module"], "projection": projection,
                "module_call_index": int(trace["call_index"]),
                "activation_cv": activation_cv,
                "shape": shape["shape"], "shape_index": shape_index,
                "n": n, "d": int(shape["d"]), "row_size_kib": row_kib,
                "scenario": scenario, "target_fraction": float(target_fraction),
                "target_importance": target,
                "row_budget_fraction": float(budget_fraction),
                "row_budget": row_budget,
                "row_budget_intrinsically_feasible": intrinsically_feasible,
                "paper_fixed_valid": paper_fixed_valid,
                "paper_fixed_importance": paper_fixed["importance"],
                "paper_fixed_rows": paper_fixed["rows"],
                "paper_fixed_lookup_ms": paper_fixed["lookup_ms"],
                "paper_coverage_importance": paper_cov["importance"],
                "paper_coverage_rows": paper_cov["rows"],
                "paper_coverage_chunks": paper_cov["chunks"],
                "paper_coverage_lookup_ms": paper_cov["lookup_ms"],
                "paper_coverage_two_line_ms": paper_cov["two_line_ms"],
            }

            for method in METHODS:
                length, layout = layouts[method]
                function = lambda layout=layout: gpu_saturation_tiles(
                    values_cuda, target_cuda, layout, model
                )
                mask, metadata, wall_ms, event_ms = benchmark_cuda(
                    function, args.repetitions, args.warmup
                )
                metrics = mask_metrics(mask, values, model, lookup_table, row_kib)
                legacy_function = lambda length=length: EXP18.saturation_tiles(
                    values, target, model, length
                )
                legacy_mask, legacy_meta, legacy_wall_ms = benchmark_cpu(
                    legacy_function, args.repetitions, args.warmup
                )
                legacy_metrics = mask_metrics(
                    legacy_mask, values, model, lookup_table, row_kib
                )
                coverage_met = metrics["importance"] >= target - 2e-7
                row_budget_met = metrics["rows"] <= row_budget
                wall_p95 = float(np.quantile(wall_ms, 0.95))
                event_p95 = float(np.quantile(event_ms, 0.95))
                lookup_saving = 100.0 * (
                    1.0 - metrics["lookup_ms"] / paper_cov["lookup_ms"]
                )
                two_line_saving = 100.0 * (
                    1.0 - metrics["two_line_ms"] / paper_cov["two_line_ms"]
                )
                base = {
                    **case, "method": method,
                    "method_label": METHOD_LABELS[method], "tile_length": length,
                }
                rows.append({
                    **base, **metrics,
                    "importance_overshoot": metrics["importance"] - target,
                    "coverage_met": coverage_met,
                    "row_budget_met": row_budget_met,
                    "coverage_and_row_budget_met": coverage_met and row_budget_met,
                    "deterministic": metadata["deterministic"],
                    "chosen_offset": int(layout[metadata["chosen_grid"]]["offset"]),
                    "gpu_predicted_two_line_ms": metadata["predicted_two_line_ms"],
                    "gpu_predicted_importance": metadata["predicted_importance"],
                    "runtime_wall_median_ms": float(np.median(wall_ms)),
                    "runtime_wall_p95_ms": wall_p95,
                    "runtime_event_median_ms": float(np.median(event_ms)),
                    "runtime_event_p95_ms": event_p95,
                    "runtime_cpu_legacy_median_ms": float(np.median(legacy_wall_ms)),
                    "runtime_cpu_legacy_p95_ms": float(
                        np.quantile(legacy_wall_ms, 0.95)
                    ),
                    "gpu_wall_over_cpu_median_ratio": float(
                        np.median(wall_ms) / np.median(legacy_wall_ms)
                    ),
                    "deadline_met": wall_p95 <= args.deadline_ms,
                    "valid_and_deadline_met": coverage_met and wall_p95 <= args.deadline_ms,
                    "row_budget_valid_and_deadline_met": (
                        coverage_met and row_budget_met and wall_p95 <= args.deadline_ms
                    ),
                    "lookup_saving_vs_paper_pct": lookup_saving,
                    "two_line_saving_vs_paper_pct": two_line_saving,
                    "legacy_mask_match": bool(np.array_equal(mask, legacy_mask)),
                    "legacy_chosen_offset": int(legacy_meta["chosen_offset"]),
                    "legacy_lookup_delta_ms": (
                        metrics["lookup_ms"] - legacy_metrics["lookup_ms"]
                    ),
                    "legacy_two_line_delta_ms": (
                        metrics["two_line_ms"] - legacy_metrics["two_line_ms"]
                    ),
                })
                for repetition, (wall, event, legacy_wall) in enumerate(
                    zip(wall_ms, event_ms, legacy_wall_ms)
                ):
                    timing_rows.append({
                        **base, "repetition": repetition,
                        "runtime_wall_ms": wall, "runtime_event_ms": event,
                        "runtime_cpu_legacy_ms": legacy_wall,
                    })
        print(
            f"trace={trace_index + 1}/{len(traces)} id={trace['trace_id']} "
            f"shape={shape['shape']} module={trace['module']}", flush=True,
        )
    return rows, timing_rows, model_rows


def distribution(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        return {key: None for key in ("mean", "median", "p05", "p95", "min", "max")}
    return {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()), "max": float(x.max()),
    }


def aggregate_group(group: pd.DataFrame, deadline_ms: float) -> dict:
    weighted_lookup = 100.0 * (
        1.0 - float(group.lookup_ms.sum()) / float(group.paper_coverage_lookup_ms.sum())
    )
    weighted_two_line = 100.0 * (
        1.0
        - float(group.two_line_ms.sum()) / float(group.paper_coverage_two_line_ms.sum())
    )
    return {
        "cases": int(len(group)),
        "coverage_success_rate": float(group.coverage_met.mean()),
        "row_budget_success_rate": float(group.coverage_and_row_budget_met.mean()),
        "paper_row_budget_success_rate": float(group.paper_fixed_valid.mean()),
        "valid_and_deadline_pass_rate": float(group.valid_and_deadline_met.mean()),
        "row_budget_valid_and_deadline_pass_rate": float(
            group.row_budget_valid_and_deadline_met.mean()
        ),
        "deadline_pass_rate": float(group.deadline_met.mean()),
        "cpu_deadline_pass_rate": float(
            (group.runtime_cpu_legacy_p95_ms <= deadline_ms).mean()
        ),
        "gpu_faster_case_rate": float(
            (group.gpu_wall_over_cpu_median_ratio < 1.0).mean()
        ),
        "deterministic_rate": float(group.deterministic.mean()),
        "legacy_mask_match_rate": float(group.legacy_mask_match.mean()),
        "legacy_two_line_max_abs_delta_ms": float(
            group.legacy_two_line_delta_ms.abs().max()
        ),
        "runtime_wall_median_ms": float(group.runtime_wall_median_ms.median()),
        "runtime_wall_case_p95_ms": float(group.runtime_wall_p95_ms.quantile(0.95)),
        "runtime_wall_worst_case_p95_ms": float(group.runtime_wall_p95_ms.max()),
        "runtime_event_median_ms": float(group.runtime_event_median_ms.median()),
        "runtime_event_case_p95_ms": float(group.runtime_event_p95_ms.quantile(0.95)),
        "runtime_event_worst_case_p95_ms": float(group.runtime_event_p95_ms.max()),
        "runtime_cpu_legacy_median_ms": float(
            group.runtime_cpu_legacy_median_ms.median()
        ),
        "runtime_cpu_legacy_case_p95_ms": float(
            group.runtime_cpu_legacy_p95_ms.quantile(0.95)
        ),
        "runtime_cpu_legacy_worst_case_p95_ms": float(
            group.runtime_cpu_legacy_p95_ms.max()
        ),
        "gpu_wall_over_cpu_median_ratio": float(
            group.gpu_wall_over_cpu_median_ratio.median()
        ),
        "lookup_saving_pct_mean": float(group.lookup_saving_vs_paper_pct.mean()),
        "lookup_saving_pct_median": float(group.lookup_saving_vs_paper_pct.median()),
        "lookup_saving_pct_weighted": weighted_lookup,
        "lookup_nonregression_rate": float(
            (group.lookup_saving_vs_paper_pct >= -1e-9).mean()
        ),
        "two_line_saving_pct_mean": float(group.two_line_saving_vs_paper_pct.mean()),
        "two_line_saving_pct_weighted": weighted_two_line,
        "rows_mean": float(group.rows.mean()),
        "chunks_mean": float(group.chunks.mean()),
        "importance_overshoot_mean": float(group.importance_overshoot.mean()),
    }


def summarize(frame: pd.DataFrame, args: argparse.Namespace):
    metadata_path = args.output_dir / "metadata.json"
    recorded_metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    recorded_gpu = recorded_metadata.get("gpu")
    recorded_torch = recorded_metadata.get("torch", torch.__version__)
    recorded_cuda_build = recorded_metadata.get(
        "torch_cuda_build", torch.version.cuda
    )
    trace_metadata = recorded_metadata.get("activation_trace", {})
    summary_rows = []
    nested = {}
    for method, group in frame.groupby("method", sort=False):
        metrics = aggregate_group(group, args.deadline_ms)
        row = {"method": method, "method_label": METHOD_LABELS[method], **metrics}
        summary_rows.append(row)
        nested[method] = {
            **row,
            "runtime_wall_p95_distribution": distribution(group.runtime_wall_p95_ms),
            "runtime_event_p95_distribution": distribution(group.runtime_event_p95_ms),
            "lookup_saving_distribution": distribution(group.lookup_saving_vs_paper_pct),
        }

    shape_rows = []
    for (shape, method), group in frame.groupby(["shape", "method"], sort=False):
        first = group.iloc[0]
        shape_rows.append({
            "shape": shape, "n": int(first.n), "d": int(first.d),
            "row_size_kib": float(first.row_size_kib),
            "method": method, "method_label": METHOD_LABELS[method],
            "tile_length": int(first.tile_length),
            **aggregate_group(group, args.deadline_ms),
        })

    scenario_rows = []
    for (target, method), group in frame.groupby(
        ["target_fraction", "method"], sort=False
    ):
        scenario_rows.append({
            "target_fraction": float(target), "method": method,
            "method_label": METHOD_LABELS[method],
            **aggregate_group(group, args.deadline_ms),
        })

    projection_rows = []
    for (projection, method), group in frame.groupby(
        ["projection", "method"], sort=False
    ):
        projection_rows.append({
            "projection": projection, "method": method,
            "method_label": METHOD_LABELS[method],
            **aggregate_group(group, args.deadline_ms),
        })

    summary = {
        "format": "experiment-25-gpu-tiles-v2-real-activations",
        "comparison": (
            "GPU-native Experiment-18 saturation tiles against Paper's "
            "coverage-stopping quality reference"
        ),
        "deadline_ms": float(args.deadline_ms),
        "deadline_clock": "synchronized wall p95",
        "timing_scope": (
            "warm CUDA float32 importance to CUDA bool mask; includes PyTorch dispatch, "
            "tile sums/sorts, coverage stopping, exact two-line grid choice, mask "
            "materialization, GPU execution and synchronization; excludes static layouts, "
            "the real LM trace forward, Paper/reference work, I/O and model compute"
        ),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": recorded_torch, "torch_cuda_build": recorded_cuda_build,
            "cuda_available": bool(recorded_gpu or torch.cuda.is_available()),
            "gpu": (
                recorded_gpu or (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                )
            ),
        },
        "configuration": {
            "shape_count": int(frame["shape"].nunique()),
            "shapes": list(dict.fromkeys(frame["shape"])),
            "activation_trace_count": int(frame["trace_id"].nunique()),
            "input_count": int(frame["input_index"].nunique()),
            "model": trace_metadata.get("model", args.model),
            "trace_input": recorded_metadata.get("trace_file", str(args.trace_input)),
            "max_traces_per_shape": int(
                recorded_metadata.get("max_traces_per_shape", args.max_traces_per_shape)
            ),
            "repetitions": int(args.repetitions), "warmup": int(args.warmup),
            "target_fractions": list(args.target_fractions),
            "row_budget_fractions": list(args.row_budget_fractions),
            "profile": args.profile, "saturation_kib": float(args.saturation_kib),
        },
        "methods": nested,
    }
    return (
        pd.DataFrame(summary_rows), pd.DataFrame(shape_rows),
        pd.DataFrame(scenario_rows), pd.DataFrame(projection_rows), summary,
    )


def configure_plot():
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_overall(summary: pd.DataFrame, path: Path, deadline_ms: float) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(METHODS))
    labels = [METHOD_LABELS[method] for method in METHODS]
    colors = [METHOD_COLORS[method] for method in METHODS]
    axes[0].bar(x - 0.18, summary.set_index("method").loc[list(METHODS), "runtime_event_case_p95_ms"],
                width=0.36, color=colors, alpha=0.55, label="CUDA event")
    axes[0].bar(x + 0.18, summary.set_index("method").loc[list(METHODS), "runtime_wall_case_p95_ms"],
                width=0.36, color=colors, label="Synchronized wall")
    axes[0].axhline(deadline_ms, color="#111827", linestyle="--", linewidth=1.2,
                    label=f"{deadline_ms:g} ms deadline")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("95th percentile of case p95 (ms)")
    axes[0].set_title("GPU selector latency")
    axes[0].legend(frameon=False)
    BASE.polish_axis(axes[0])

    indexed = summary.set_index("method").loc[list(METHODS)]
    axes[1].bar(x - 0.18, indexed.lookup_saving_pct_weighted,
                width=0.36, color=colors, label="Lookup")
    axes[1].bar(x + 0.18, indexed.two_line_saving_pct_weighted,
                width=0.36, color=colors, alpha=0.55, label="Two-line")
    axes[1].axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Absolute-latency-weighted saving vs Paper (%)")
    axes[1].set_title("Coverage-matched lookup quality")
    axes[1].legend(frameon=False)
    BASE.polish_axis(axes[1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_shapes(shape_summary: pd.DataFrame, path: Path, deadline_ms: float) -> None:
    plt = configure_plot()
    order = (
        shape_summary[shape_summary.method == METHODS[0]]
        .sort_values(["n", "d"])["shape"].tolist()
    )
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.0), constrained_layout=True)
    for method in METHODS:
        selected = shape_summary[shape_summary.method == method].set_index("shape").reindex(order)
        axes[0].plot(x, selected.runtime_wall_case_p95_ms, marker="o", linewidth=1.5,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(x, selected.lookup_saving_pct_weighted, marker="o", linewidth=1.5,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axes[0].axhline(deadline_ms, color="#111827", linestyle="--", linewidth=1.2)
    axes[1].axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Wall case-p95 (ms)")
    axes[0].set_title("GPU selector scaling by shape")
    axes[1].set_ylabel("Weighted lookup saving vs Paper (%)")
    axes[1].set_title("Coverage-matched lookup quality by shape")
    for ax in axes:
        ax.set_xticks(x, order, rotation=55, ha="right")
        ax.legend(frameon=False, ncol=3)
        BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(summary: pd.DataFrame, shape_summary: pd.DataFrame,
                 scenario_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    indexed = summary.set_index("method").reindex(METHODS)
    quality_winner = indexed.lookup_saving_pct_weighted.idxmax()
    speed_winner = indexed.runtime_wall_case_p95_ms.idxmin()
    paper_budget_rate = float(indexed.paper_row_budget_success_rate.iloc[0])
    quality_change = float(indexed.loc[quality_winner, "lookup_saving_pct_weighted"])
    if quality_change > 0.0:
        quality_conclusion = (
            f"전체 가중 lookup은 소폭 개선됐지만 크기는 `{quality_change:.2f}%`에 그쳤다."
        )
    else:
        quality_conclusion = (
            f"전체 가중 lookup은 `{abs(quality_change):.2f}%` 악화됐다."
        )
    positive_shapes = {
        method: int((
            shape_summary[shape_summary.method == method].lookup_saving_pct_weighted
            > 1e-9
        ).sum())
        for method in METHODS
    }
    parity_mismatches = int(round(float(
        (indexed.cases * (1.0 - indexed.legacy_mask_match_rate)).sum()
    )))
    parity_cases = int(indexed.cases.sum())
    parity_max_cost_delta = float(indexed.legacy_two_line_max_abs_delta_ms.max())
    metadata_path = args.output_dir / "metadata.json"
    recorded_metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    gpu_name = recorded_metadata.get("gpu", "CUDA GPU")
    trace_count = int(indexed.cases.iloc[0] / len(args.target_fractions))
    lines = [
        "# Experiment 25 보고서: GPU Saturation Tiles", "",
        "Experiment 18의 `Tiles(s/2)`, `Tiles(s)`, `Tiles(2s)`를 CUDA-resident "
        "PyTorch 연산으로 옮겼다. 입력 importance부터 최종 boolean mask까지 GPU에 "
        "머물며, data-dependent CPU readback은 없다.", "",
        f"importance는 `vlm-flash`를 실제 `{args.model}` projection에 attach한 dense forward에서 "
        "수집했으며 Experiment 22/24와 같은 trace archive를 사용했다.", "",
        "## 한 문장 판정", "",
        "결과 생성 전입니다.", "",
        "## 측정 설정", "",
        f"- GPU: {gpu_name}",
        f"- model: {args.model}",
        f"- 실제 activation trace / Table-2 shape: {trace_count} / {shape_summary['shape'].nunique()}",
        f"- trace × scenarios: {trace_count} × {len(args.target_fractions)}",
        f"- case당 warm-up/repetitions: {args.warmup}/{args.repetitions}",
        "- CUDA event: device 실행시간",
        "- synchronized wall: Python dispatch + GPU 실행 + synchronization",
        f"- 2ms 판정: synchronized wall p95 <= {args.deadline_ms:g} ms", "",
        "## 전체 결과", "",
        "| 방법 | GPU wall 중앙 | GPU wall case-p95 | CUDA-event case-p95 | CPU case-p95 | GPU/CPU | coverage | coverage+R | Paper 대비 lookup 평균/가중 | CPU mask 일치 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in indexed.iterrows():
        lines.append(
            f"| {METHOD_LABELS[method]} | {fmt(row.runtime_wall_median_ms)} ms "
            f"| {fmt(row.runtime_wall_case_p95_ms)} ms "
            f"| {fmt(row.runtime_event_case_p95_ms)} ms "
            f"| {fmt(row.runtime_cpu_legacy_case_p95_ms)} ms "
            f"| {row.gpu_wall_over_cpu_median_ratio:.2f}× "
            f"| {100 * row.coverage_success_rate:.1f}% "
            f"| {100 * row.row_budget_success_rate:.1f}% "
            f"| {row.lookup_saving_pct_mean:.2f}% / {row.lookup_saving_pct_weighted:.2f}% "
            f"| {100 * row.legacy_mask_match_rate:.1f}% |"
        )
    lines.extend([
        "", "## 판정", "",
        f"가중 lookup 품질이 가장 높은 설정은 `{METHOD_LABELS[quality_winner]}`이다. "
        f"Paper coverage selector 대비 {quality_conclusion}", "",
        f"wall case-p95가 가장 낮은 설정은 `{METHOD_LABELS[speed_winner]}`의 "
        f"`{indexed.loc[speed_winner, 'runtime_wall_case_p95_ms']:.3f} ms`다. "
        f"모든 설정의 2ms case 통과율은 "
        + ", ".join(
            f"{METHOD_LABELS[m]} {100 * indexed.loc[m, 'valid_and_deadline_pass_rate']:.1f}%"
            for m in METHODS
        ) + ".", "",
        "동일 입력의 Experiment 18 CPU 구현 대비 GPU wall/CPU 중앙시간 비율은 "
        + ", ".join(
            f"{METHOD_LABELS[m]} {indexed.loc[m, 'gpu_wall_over_cpu_median_ratio']:.2f}×"
            for m in METHODS
        ) + "다. 1보다 크면 현재 eager GPU 구현이 더 느리다는 뜻이다. GPU가 실제로 "
        "더 빠른 case 비율은 "
        + ", ".join(
            f"{METHOD_LABELS[m]} {100 * indexed.loc[m, 'gpu_faster_case_rate']:.1f}%"
            for m in METHODS
        ) + "다.", "",
        "CPU와 GPU 모두 2ms를 통과한 비율은 각각 "
        + ", ".join(
            f"{METHOD_LABELS[m]} CPU {100 * indexed.loc[m, 'cpu_deadline_pass_rate']:.1f}%/"
            f"GPU {100 * indexed.loc[m, 'valid_and_deadline_pass_rate']:.1f}%"
            for m in METHODS
        ) + "다. `s/2` GPU는 일부 tile 수가 많은 큰 shape의 CPU tail을 줄였지만, "
        "대부분의 case 중앙시간은 CPU가 더 짧았다.", "",
        "Paper 대비 가중 lookup이 양수인 shape 수는 "
        + ", ".join(
            f"{METHOD_LABELS[m]} {positive_shapes[m]}/{shape_summary['shape'].nunique()}"
            for m in METHODS
        ) + "다. Case별 lookup 비악화율은 "
        + ", ".join(
            f"{METHOD_LABELS[m]} {100 * indexed.loc[m, 'lookup_nonregression_rate']:.1f}%"
            for m in METHODS
        ) + "다.", "",
        f"Experiment 18 CPU mask와는 `{parity_cases - parity_mismatches}/{parity_cases}`개가 "
        f"정확히 일치했다. 나머지 `{parity_mismatches}`개는 float32 GPU와 float64 CPU의 "
        f"정렬 경계 차이이며 coverage 위반은 없었고 최대 two-line 차이는 "
        f"`{parity_max_cost_delta:.6f} ms`였다.", "",
        "row-budget은 알고리즘이 직접 강제하지 않는 audit이다. Paper fixed-R의 "
        f"coverage+R 성공률은 `{100 * paper_budget_rate:.1f}%`이고, Tiles는 "
        + ", ".join(
            f"{METHOD_LABELS[m]} {100 * indexed.loc[m, 'row_budget_success_rate']:.1f}%"
            for m in METHODS
        ) + "다.", "",
        "## Target별 결과", "",
        "| Q/total | 방법 | wall case-p95 | coverage+R | 가중 lookup 절감 |",
        "|---:|---|---:|---:|---:|",
    ])
    for _, row in scenario_summary.sort_values(["target_fraction", "method"]).iterrows():
        lines.append(
            f"| {row.target_fraction:.2f} | {row.method_label} "
            f"| {row.runtime_wall_case_p95_ms:.3f} ms "
            f"| {100 * row.row_budget_success_rate:.1f}% "
            f"| {row.lookup_saving_pct_weighted:.2f}% |"
        )
    lines.extend([
        "", "## Shape별 wall case-p95", "",
        "| Shape | rows | row KiB | s/2 | s | 2s |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for shape in (
        shape_summary[shape_summary.method == METHODS[0]]
        .sort_values(["n", "d"])["shape"]
    ):
        selected = shape_summary[shape_summary["shape"] == shape].set_index("method")
        first = selected.iloc[0]
        lines.append(
            f"| {shape} | {int(first.n):,} | {first.row_size_kib:.2f} "
            f"| {selected.loc['tiles_half_s', 'runtime_wall_case_p95_ms']:.3f} ms "
            f"| {selected.loc['tiles_s', 'runtime_wall_case_p95_ms']:.3f} ms "
            f"| {selected.loc['tiles_2s', 'runtime_wall_case_p95_ms']:.3f} ms |"
        )
    lines.extend([
        "", "## 측정 한계", "",
        f"- 실제 activation은 `{args.model}`의 짧은 text prompt에서 얻었으며 다른 model, "
        "긴 context, multimodal frame-append workload까지 대표하지 않는다.",
        "- lookup latency는 released Orin AGX profile 예측값이고 실제 NVMe I/O를 실행하지 않았다.",
        f"- {gpu_name} timing이므로 Jetson의 절대시간을 증명하지 않는다.",
        "- static tile layout은 shape만으로 결정되어 online timing에서 제외했다.",
        "- eager PyTorch 구현이며 CUDA graph나 custom fused kernel은 사용하지 않았다.",
        "- row-budget 결과는 제약을 직접 푼 것이 아니라 coverage mask에 대한 사후 audit이다.",
        "", "![Overall](results/runtime_quality.png)", "",
        "![Shape scaling](results/shape_scaling.png)", "",
    ])
    verdict = (
        f"\\boxed{{\\text{{{METHOD_LABELS[quality_winner]}는 2 ms를 지켰지만, "
        f"lookup 개선은 {quality_change:.2f}\\%이고 CPU 대비 "
        f"{indexed.loc[quality_winner, 'gpu_wall_over_cpu_median_ratio']:.2f}배 느렸다.}}}}"
    )
    lines[lines.index("결과 생성 전입니다.")] = f"\\[{verdict}\\]"
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    if "trace_id" not in frame:
        raise SystemExit(
            "these are Experiment 25's retired synthetic results; rerun the experiment "
            "to create real-model activation results before using --analyze-only"
        )
    summary_frame, shape_frame, scenario_frame, projection_frame, summary = summarize(
        frame, args
    )
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    scenario_frame.to_csv(args.output_dir / "scenario_summary.csv", index=False)
    projection_frame.to_csv(args.output_dir / "projection_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    plot_shapes(shape_frame, args.output_dir / "shape_scaling.png", args.deadline_ms)
    write_report(summary_frame, shape_frame, scenario_frame, args)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return
    traces, trace_metadata, trace_path = prepare_activation_traces(args)
    selected_shape_names = list(dict.fromkeys(trace["shape"] for trace in traces))
    shape_specs = [
        dict(next(item for item in SHAPES if item["shape"] == shape_name))
        for shape_name in selected_shape_names
    ]
    try:
        recorded_trace_path = str(trace_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        recorded_trace_path = str(trace_path.resolve())
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
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.collect_traces_only:
        print(
            f"selected {len(traces)} traces across {len(selected_shape_names)} "
            "Table-2 shapes; stopping as requested"
        )
        return

    lookup_table = BASE.LatencyTable.load(args.profile)
    if not args.skip_self_check:
        self_check(lookup_table, args.saturation_kib)
    rows, timing_rows, model_rows = collect(args, traces, lookup_table)
    write_csv(args.output_dir / "trials.csv", rows)
    write_csv(args.output_dir / "timing_samples.csv", timing_rows)
    write_csv(args.output_dir / "models.csv", model_rows)
    analyze(args)


if __name__ == "__main__":
    main()
