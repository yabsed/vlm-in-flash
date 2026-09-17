#!/usr/bin/env python3
"""Experiment 22: Paper versus predicted-lambda TD-2L(8) plus endpoint trim."""

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
import warnings

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_20 = (
    PROJECT_ROOT / "experiments" / "20_remaining_sub2ms_candidates"
    / "run_experiment.py"
)


def load_experiment_20():
    spec = importlib.util.spec_from_file_location("experiment_20_for_22", EXPERIMENT_20)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 20 from {EXPERIMENT_20}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP20 = load_experiment_20()
EXP18 = EXP20.EXP18
EXP13 = EXP20.EXP13
EXP2 = EXP20.EXP2
BASE = EXP20.BASE


# Table 2, AGX settings.  Shape is input rows x output columns, so an FP16
# weight row occupies 2 * output_columns bytes.
SHAPES = (
    {"shape": "3584x3584", "n": 3584, "d": 3584, "start_kib": 20.0, "jump_kib": 20.0},
    {"shape": "8960x1536", "n": 8960, "d": 1536, "start_kib": 16.0, "jump_kib": 16.0},
    {"shape": "896x4864", "n": 896, "d": 4864, "start_kib": 8.0, "jump_kib": 8.0},
    {"shape": "4096x1024", "n": 4096, "d": 1024, "start_kib": 12.0, "jump_kib": 12.0},
    {"shape": "3584x18944", "n": 3584, "d": 18944, "start_kib": 8.0, "jump_kib": 8.0},
    {"shape": "4096x4096", "n": 4096, "d": 4096, "start_kib": 20.0, "jump_kib": 20.0},
    {"shape": "18944x3584", "n": 18944, "d": 3584, "start_kib": 32.0, "jump_kib": 32.0},
    {"shape": "1536x1536", "n": 1536, "d": 1536, "start_kib": 16.0, "jump_kib": 12.0},
    {"shape": "1536x256", "n": 1536, "d": 256, "start_kib": 8.0, "jump_kib": 8.0},
    {"shape": "896x128", "n": 896, "d": 128, "start_kib": 8.0, "jump_kib": 8.0},
    {"shape": "14336x4096", "n": 14336, "d": 4096, "start_kib": 32.0, "jump_kib": 32.0},
    {"shape": "4864x896", "n": 4864, "d": 896, "start_kib": 12.0, "jump_kib": 16.0},
    {"shape": "3584x512", "n": 3584, "d": 512, "start_kib": 8.0, "jump_kib": 12.0},
    {"shape": "896x896", "n": 896, "d": 896, "start_kib": 8.0, "jump_kib": 8.0},
    {"shape": "4096x14336", "n": 4096, "d": 14336, "start_kib": 8.0, "jump_kib": 8.0},
    {"shape": "1536x8960", "n": 1536, "d": 8960, "start_kib": 8.0, "jump_kib": 8.0},
)

METHODS = (
    "paper",
    "predict_correct_c8",
    "predict_correct_c8_trim64",
    "predict_correct_c8_trim256",
)
METHOD_LABELS = {
    "paper": "Paper",
    "predict_correct_c8": "Predicted-lambda TD-2L(8)",
    "predict_correct_c8_trim64": "Predicted-lambda TD-2L(8) + trim 64",
    "predict_correct_c8_trim256": "Predicted-lambda TD-2L(8) + trim 256",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "predict_correct_c8": "#7C3AED",
    "predict_correct_c8_trim64": "#0F766E",
    "predict_correct_c8_trim256": "#064E3B",
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
    parser.add_argument("--seed", type=int, default=20262201)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
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
    if args.trials < 1 or args.repetitions < 1:
        raise SystemExit("--trials and --repetitions must be positive")
    if not args.row_budget_fractions or any(
        not 0.0 < value <= 1.0 for value in args.row_budget_fractions
    ):
        raise SystemExit("--row-budget-fractions must lie in (0, 1]")
    if args.cv <= 0 or any(args.cv >= math.sqrt(item["n"] - 1) for item in shapes):
        raise SystemExit("--cv must lie in (0, sqrt(N-1)) for every shape")
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


def row_size_kib(shape: dict) -> float:
    return 2.0 * int(shape["d"]) / 1024.0


def make_params(shape: dict, saturation_kib: float):
    return BASE.ChunkParams(
        start_kb=float(shape["start_kib"]), end_kb=float(saturation_kib),
        step_kb=float(shape["start_kib"]), jump_cap_kb=float(shape["jump_kib"]),
    )


def endpoint_trim(mask: np.ndarray, values: np.ndarray, target: float,
                  model: dict, max_deletions: int) -> tuple[np.ndarray, dict]:
    """Greedily delete exposed endpoints with positive two-line savings."""
    trimmed = np.asarray(mask, dtype=bool).copy()
    surplus = float(values[trimmed].sum()) - float(target)
    if surplus < -1e-11:
        raise RuntimeError("endpoint trim received an infeasible mask")
    starts, ends = EXP18.run_bounds(trimmed)
    left = starts.astype(np.int64).copy()
    right = (ends - 1).astype(np.int64).copy()
    versions = np.zeros(len(left), dtype=np.int64)
    heap: list[tuple[float, float, int, int, int, float]] = []

    def offer(run: int) -> None:
        length = int(right[run] - left[run] + 1)
        if length <= 0:
            return
        before = EXP18.two_line_chunk_ms(length, model)
        after = 0.0 if length == 1 else EXP18.two_line_chunk_ms(length - 1, model)
        saving = float(before - after)
        # The algorithm's invariant requires strictly lower latency.
        if saving <= 0.0:
            return
        for index in {int(left[run]), int(right[run])}:
            lost = float(values[index])
            heapq.heappush(
                heap, (lost / saving, lost, run, index, int(versions[run]), saving)
            )

    for run in range(len(left)):
        offer(run)

    deletions = 0
    predicted_saving = 0.0
    while heap and deletions < max_deletions:
        _, lost, run, index, version, saving = heapq.heappop(heap)
        if version != versions[run] or not trimmed[index]:
            continue
        if index not in (left[run], right[run]):
            continue
        if lost > surplus + 1e-14:
            # Surplus only decreases, so this endpoint can never become feasible later.
            continue
        trimmed[index] = False
        surplus -= lost
        predicted_saving += saving
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

    final_importance = float(values[trimmed].sum())
    if final_importance < target - 1e-11:
        raise RuntimeError("endpoint trim violated target coverage")
    return trimmed, {
        "trim_deletions": deletions,
        "trim_limit": int(max_deletions),
        "trim_two_line_saving_ms": predicted_saving,
        "trim_remaining_surplus": final_importance - float(target),
    }


def paper_select(values_tensor: torch.Tensor, row_budget: int, row_kib: float,
                 lookup_table, params, implementation: str) -> tuple[torch.Tensor, dict]:
    result = BASE.select_chunks(
        values_tensor, row_budget, row_kib, lookup_table,
        params=params, impl=implementation,
    )
    return result.mask, {
        "paper_estimated_ms": float(result.est_cost_ms),
        "paper_selected_rows": int(result.num_selected),
    }


def proposed_select(values: np.ndarray, target: float, model: dict,
                    trim_limit: int) -> tuple[np.ndarray, dict]:
    error = ""
    try:
        mask, metadata = EXP20.predict_and_correct(values, target, model, 8)
    except Exception as exception:  # preserve failed cases in the output
        error = f"{type(exception).__name__}: {exception}"
        metadata = {"traceback": traceback.format_exc(limit=3)}
        mask = np.zeros(len(values), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    fallback_added = 0
    if float(values[mask].sum()) < target - 1e-12:
        mask, fallback_added = EXP18.add_top_fallback(mask, values, target, None)
    if trim_limit:
        mask, trim_metadata = endpoint_trim(mask, values, target, model, trim_limit)
        metadata = {**metadata, **trim_metadata}
    metadata = dict(metadata)
    metadata.update({
        "fallback_used": bool(error or fallback_added),
        "fallback_added_rows": int(fallback_added),
        "error": error,
    })
    return mask, metadata


def materialize_proposed_output(mask: np.ndarray, track: str) -> torch.Tensor:
    handoff = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_).copy())
    if track == "cuda":
        handoff = handoff.to("cuda")
    return handoff


def benchmark(function, repetitions: int, uses_cuda: bool) -> tuple[np.ndarray, dict, list[float]]:
    masks: list[np.ndarray] = []
    metadata: list[dict] = []
    timings: list[float] = []
    for _ in range(repetitions):
        if uses_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output, details = function()
        if uses_cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if isinstance(output, torch.Tensor):
            mask = output.detach().to("cpu").numpy().astype(bool, copy=True)
        else:
            mask = np.asarray(output, dtype=bool).copy()
        masks.append(mask)
        metadata.append(dict(details))
        timings.append(elapsed)
    reference = masks[0]
    details = metadata[-1]
    details["deterministic"] = all(
        np.array_equal(reference, candidate) for candidate in masks[1:]
    )
    return reference, details, timings


def paper_function(track: str, values_host: np.ndarray, values_cuda: torch.Tensor | None,
                   row_budget: int, row_kib: float, lookup_table, params,
                   implementation: str):
    def run():
        source = (
            values_cuda if track == "cuda"
            else torch.from_numpy(np.asarray(values_host, dtype=np.float32))
        )
        return paper_select(
            source, row_budget, row_kib, lookup_table, params, implementation
        )
    return run


def proposed_function(method: str, track: str, values_host: np.ndarray,
                      values_cuda: torch.Tensor | None, target: float, model: dict):
    trim_limit = {
        "predict_correct_c8": 0,
        "predict_correct_c8_trim64": 64,
        "predict_correct_c8_trim256": 256,
    }[method]

    def run():
        if track == "cuda":
            # Online CPU selector contract: GPU importance -> CPU solve -> GPU mask.
            values = values_cuda.detach().to("cpu").numpy().astype(np.float64)
        else:
            values = values_host
        mask, metadata = proposed_select(values, target, model, trim_limit)
        return materialize_proposed_output(mask, track), metadata
    return run


def mask_metrics(mask: np.ndarray, values: np.ndarray, model: dict,
                 lookup_table, row_kib: float) -> dict:
    return EXP18.mask_metrics(mask, values, model, lookup_table, row_kib)


def self_check(lookup_table, saturation_kib: float) -> None:
    rng = np.random.default_rng(2201)
    shape = next(item for item in SHAPES if item["shape"] == "4864x896")
    row_kib = row_size_kib(shape)
    model = EXP13.fit_continuous_two_line(lookup_table, row_kib, saturation_kib)

    # Fixed-lambda DP agrees with exhaustive enumeration on a small problem.
    values = rng.lognormal(size=10)
    values /= values.sum()
    multiplier = 0.4
    solved = EXP13._solve_lambda_metrics(
        values, multiplier, model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"], model["short_max_rows"],
    )
    best = -np.inf
    for encoded in range(1 << len(values)):
        candidate = np.asarray(
            [(encoded >> index) & 1 for index in range(len(values))], dtype=bool
        )
        candidate_metrics = EXP18.mask_metrics_two_line(candidate, values, model)
        score = multiplier * candidate_metrics["importance"] - candidate_metrics["two_line_ms"]
        best = max(best, score)
    if not math.isclose(float(solved[0]), best, rel_tol=1e-10, abs_tol=1e-11):
        raise RuntimeError("fixed-lambda O(N) DP failed exhaustive self-check")

    values = rng.lognormal(size=97)
    values /= values.sum()
    target = 0.7
    base, _ = EXP20.predict_and_correct(values, target, model, 8)
    base_metrics = EXP18.mask_metrics_two_line(base, values, model)
    if base_metrics["importance"] < target - 1e-11:
        raise RuntimeError("predicted-lambda search returned an infeasible mask")
    for limit in (64, 256):
        trimmed, metadata = endpoint_trim(base, values, target, model, limit)
        after = EXP18.mask_metrics_two_line(trimmed, values, model)
        if (
            after["importance"] < target - 1e-11
            or after["two_line_ms"] > base_metrics["two_line_ms"] + 1e-12
            or np.any(trimmed & ~np.asarray(base, dtype=bool))
            or metadata["trim_deletions"] > limit
        ):
            raise RuntimeError("endpoint trim invariant self-check failed")


def collect(args: argparse.Namespace, shapes: list[dict], tracks: list[str],
            lookup_table) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    timing_rows: list[dict] = []
    model_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for shape_index, shape in enumerate(shapes):
        n = int(shape["n"])
        row_kib = row_size_kib(shape)
        params = make_params(shape, args.saturation_kib)
        model = EXP13.fit_continuous_two_line(
            lookup_table, row_kib, args.saturation_kib
        )
        model_rows.append({
            **shape, "row_size_kib": row_kib, **model,
        })
        hotness = np.linspace(1.0, -1.0, n)
        hotness = (hotness - hotness.mean()) / hotness.std()

        for trial in range(args.trials):
            multiset = EXP2.exact_cv_lognormal(rng, n, args.cv)
            variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
            for spatial_mode, raw_values in variants.items():
                # Both methods see precisely the same float32-representable scores.
                values32 = np.asarray(raw_values, dtype=np.float32)
                values32 /= np.float32(values32.astype(np.float64).sum())
                values = values32.astype(np.float64)
                values_cuda = (
                    torch.from_numpy(values32).to("cuda") if "cuda" in tracks else None
                )
                for budget_index, budget_fraction in enumerate(args.row_budget_fractions):
                    row_budget = max(1, min(n, int(round(n * budget_fraction))))
                    for track in tracks:
                        uses_cuda_paper = bool(torch.cuda.is_available())
                        pfun = paper_function(
                            track, values, values_cuda, row_budget, row_kib,
                            lookup_table, params, args.paper_impl,
                        )
                        # Warm every case; compilation and first-touch effects are excluded.
                        pfun()
                        paper_mask, paper_meta, paper_timings = benchmark(
                            pfun, args.repetitions, uses_cuda_paper
                        )
                        paper_metrics = mask_metrics(
                            paper_mask, values, model, lookup_table, row_kib
                        )
                        target = float(paper_metrics["importance"])

                        case_base = {
                            "shape": shape["shape"], "shape_index": shape_index,
                            "n": n, "d": int(shape["d"]), "row_size_kib": row_kib,
                            "paper_start_kib": float(shape["start_kib"]),
                            "paper_jump_kib": float(shape["jump_kib"]),
                            "trial": trial, "spatial_mode": spatial_mode,
                            "budget_index": budget_index,
                            "row_budget_fraction": float(budget_fraction),
                            "row_budget": row_budget, "target_importance": target,
                            "track": track,
                        }

                        def append_result(method: str, mask: np.ndarray, metadata: dict,
                                          timings: list[float]) -> None:
                            metrics = mask_metrics(mask, values, model, lookup_table, row_kib)
                            coverage_met = metrics["importance"] >= target - 1e-11
                            runtime_p95 = float(np.quantile(timings, 0.95))
                            base = {
                                **case_base, "method": method,
                                "method_label": METHOD_LABELS[method],
                            }
                            rows.append({
                                **base, **metrics,
                                "importance_delta_vs_target": metrics["importance"] - target,
                                "rows_delta_vs_paper": metrics["rows"] - paper_metrics["rows"],
                                "coverage_met": coverage_met,
                                "valid": coverage_met and not bool(metadata.get("error", "")),
                                "runtime_median_ms": float(np.median(timings)),
                                "runtime_p95_ms": runtime_p95,
                                "deadline_met": runtime_p95 <= args.deadline_ms,
                                "valid_and_deadline_met": (
                                    coverage_met and not bool(metadata.get("error", ""))
                                    and runtime_p95 <= args.deadline_ms
                                ),
                                "deterministic": metadata.get("deterministic", False),
                                "scalarized_calls": metadata.get("scalarized_calls", 0),
                                "bracket_converged": metadata.get("bracket_converged", False),
                                "trim_deletions": metadata.get("trim_deletions", 0),
                                "trim_two_line_saving_ms": metadata.get(
                                    "trim_two_line_saving_ms", 0.0
                                ),
                                "fallback_used": metadata.get("fallback_used", False),
                                "fallback_added_rows": metadata.get("fallback_added_rows", 0),
                                "error": metadata.get("error", ""),
                            })
                            for repetition, elapsed in enumerate(timings):
                                timing_rows.append({
                                    **base, "repetition": repetition,
                                    "runtime_ms": elapsed,
                                })

                        append_result("paper", paper_mask, paper_meta, paper_timings)
                        for method in METHODS[1:]:
                            function = proposed_function(
                                method, track, values, values_cuda, target, model
                            )
                            function()
                            mask, metadata, timings = benchmark(
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
        "paper_lookup_ms", "paper_two_line_ms", "untrimmed_lookup_ms",
        "untrimmed_two_line_ms", "lookup_saving_vs_paper_pct",
        "two_line_saving_vs_paper_pct", "lookup_trim_gain_pct",
        "two_line_trim_gain_pct",
    ]
    frame = frame.drop(columns=[column for column in derived if column in frame])
    keys = [
        "shape", "trial", "spatial_mode", "budget_index", "track",
    ]
    paper = frame[frame.method == "paper"][keys + ["lookup_ms", "two_line_ms"]].rename(
        columns={"lookup_ms": "paper_lookup_ms", "two_line_ms": "paper_two_line_ms"}
    )
    untrimmed = frame[frame.method == "predict_correct_c8"][
        keys + ["lookup_ms", "two_line_ms"]
    ].rename(columns={
        "lookup_ms": "untrimmed_lookup_ms",
        "two_line_ms": "untrimmed_two_line_ms",
    })
    output = frame.merge(paper, on=keys, how="left").merge(untrimmed, on=keys, how="left")
    output["lookup_saving_vs_paper_pct"] = 100.0 * (
        1.0 - output.lookup_ms / output.paper_lookup_ms
    )
    output["two_line_saving_vs_paper_pct"] = 100.0 * (
        1.0 - output.two_line_ms / output.paper_two_line_ms
    )
    output["lookup_trim_gain_pct"] = 100.0 * (
        1.0 - output.lookup_ms / output.untrimmed_lookup_ms
    )
    output["two_line_trim_gain_pct"] = 100.0 * (
        1.0 - output.two_line_ms / output.untrimmed_two_line_ms
    )
    is_trim = output.method.isin((
        "predict_correct_c8_trim64", "predict_correct_c8_trim256",
    ))
    output.loc[~is_trim, ["lookup_trim_gain_pct", "two_line_trim_gain_pct"]] = np.nan
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
    trim_values = valid.lookup_trim_gain_pct.dropna()
    weighted_lookup_saving = (
        100.0 * (1.0 - float(valid.lookup_ms.sum()) / float(valid.paper_lookup_ms.sum()))
        if len(valid) and float(valid.paper_lookup_ms.sum()) > 0.0 else math.nan
    )
    return {
        "cases": int(len(group)), "valid_cases": int(len(valid)),
        "valid_rate": float(group.valid.mean()),
        "coverage_success_rate": float(group.coverage_met.mean()),
        "deadline_pass_rate": float(group.deadline_met.mean()),
        "valid_and_deadline_pass_rate": float(group.valid_and_deadline_met.mean()),
        "deterministic_rate": float(group.deterministic.mean()),
        "fallback_rate": float(group.fallback_used.mean()),
        "error_rate": float(group.error.fillna("").astype(bool).mean()),
        "runtime_median_ms": float(group.runtime_median_ms.median()),
        "runtime_case_p95_ms": float(group.runtime_p95_ms.quantile(0.95)),
        "runtime_worst_case_p95_ms": float(group.runtime_p95_ms.max()),
        "lookup_saving_vs_paper_pct_mean_valid": (
            float(valid.lookup_saving_vs_paper_pct.mean()) if len(valid) else math.nan
        ),
        "lookup_saving_vs_paper_pct_median_valid": (
            float(valid.lookup_saving_vs_paper_pct.median()) if len(valid) else math.nan
        ),
        "lookup_saving_vs_paper_pct_weighted_valid": weighted_lookup_saving,
        "lookup_win_rate_valid": (
            float((valid.lookup_saving_vs_paper_pct > 1e-9).mean())
            if len(valid) else math.nan
        ),
        "lookup_nonregression_rate_valid": (
            float((valid.lookup_saving_vs_paper_pct >= -1e-9).mean())
            if len(valid) else math.nan
        ),
        "lookup_regression_cases_valid": int(
            (valid.lookup_saving_vs_paper_pct < -1e-9).sum()
        ),
        "lookup_saving_vs_paper_pct_min_valid": (
            float(valid.lookup_saving_vs_paper_pct.min()) if len(valid) else math.nan
        ),
        "two_line_saving_vs_paper_pct_mean_valid": (
            float(valid.two_line_saving_vs_paper_pct.mean()) if len(valid) else math.nan
        ),
        "lookup_trim_gain_pct_mean_valid": (
            float(valid.lookup_trim_gain_pct.mean()) if len(valid) else math.nan
        ),
        "two_line_trim_gain_pct_mean_valid": (
            float(valid.two_line_trim_gain_pct.mean()) if len(valid) else math.nan
        ),
        "trim_lookup_regression_rate_valid": (
            float((trim_values < -1e-9).mean()) if len(trim_values) else math.nan
        ),
        "rows_delta_vs_paper_mean_valid": (
            float(valid.rows_delta_vs_paper.mean()) if len(valid) else math.nan
        ),
        "trim_deletions_mean": float(group.trim_deletions.mean()),
        "scalarized_calls_median": float(group.scalarized_calls.median()),
    }


def summarize(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary_rows = []
    nested = {}
    for (track, method), group in frame.groupby(["track", "method"], sort=False):
        metrics = aggregate_group(group)
        row = {
            "track": track, "method": method, "method_label": METHOD_LABELS[method],
            **metrics,
        }
        summary_rows.append(row)
        nested.setdefault(track, {})[method] = {
            **row,
            "runtime_median_ms_distribution": distribution(group.runtime_median_ms),
            "runtime_p95_ms_distribution": distribution(group.runtime_p95_ms),
            "lookup_saving_vs_paper_pct_valid_distribution": distribution(
                group.loc[group.valid, "lookup_saving_vs_paper_pct"]
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

    summary = {
        "format": "experiment-22-predicted-lambda-trim-v1",
        "comparison": (
            "Paper fixed-R mask versus proposed mask at paired target "
            "Q = importance(Paper mask)"
        ),
        "deadline_ms": float(args.deadline_ms),
        "timing_scope": {
            "host": (
                "warm host importance to CPU bool mask; includes selection, mask recovery, "
                "trim, fallback/validation, and output materialization"
            ),
            "cuda": (
                "warm CUDA-resident float32 importance to CUDA bool mask; CPU proposal "
                "includes D2H, float64 conversion, selection, trim, H2D, and synchronization"
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
        },
        "tracks": nested,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(shape_rows), summary


def configure_plot():
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def plot_overall(summary: pd.DataFrame, path: Path, deadline_ms: float) -> None:
    plt = configure_plot()
    tracks = list(dict.fromkeys(summary.track))
    fig, axes = plt.subplots(1, len(tracks), figsize=(7.0 * len(tracks), 5.5),
                             squeeze=False, constrained_layout=True)
    for ax, track in zip(axes[0], tracks):
        selected = summary[summary.track == track].set_index("method").reindex(METHODS)
        for method, row in selected.iterrows():
            ax.scatter(
                row.runtime_median_ms, row.lookup_saving_vs_paper_pct_mean_valid,
                s=85, color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3,
            )
        ax.axvline(deadline_ms, color="#111827", linestyle="--", linewidth=1.2)
        ax.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Median selector time (ms)")
        ax.set_ylabel("Mean lookup saving vs Paper (%)")
        ax.set_title("Host input / CPU mask" if track == "host" else "CUDA input / CUDA mask")
        BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_by_shape(shape_summary: pd.DataFrame, path: Path) -> None:
    plt = configure_plot()
    tracks = list(dict.fromkeys(shape_summary.track))
    fig, axes = plt.subplots(
        len(tracks), 2, figsize=(15.0, 5.5 * len(tracks)),
        squeeze=False, constrained_layout=True,
    )
    for row_index, track in enumerate(tracks):
        selected = shape_summary[shape_summary.track == track]
        order = (
            selected[selected.method == "paper"]
            .sort_values(["n", "d"])["shape"].tolist()
        )
        x = np.arange(len(order))
        ax_quality, ax_runtime = axes[row_index]
        for method, offset in (
            ("predict_correct_c8_trim64", -0.18),
            ("predict_correct_c8_trim256", 0.18),
        ):
            method_frame = selected[selected.method == method].set_index("shape").reindex(order)
            ax_quality.bar(
                x + offset, method_frame.lookup_saving_vs_paper_pct_mean_valid,
                width=0.34, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            )
        for method in METHODS:
            method_frame = selected[selected.method == method].set_index("shape").reindex(order)
            ax_runtime.plot(
                x, method_frame.runtime_case_p95_ms, marker="o", linewidth=1.4,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            )
        ax_runtime.axhline(2.0, color="#111827", linestyle="--", linewidth=1.1)
        for ax in (ax_quality, ax_runtime):
            ax.set_xticks(x, order, rotation=55, ha="right")
            BASE.polish_axis(ax)
        label = "Host" if track == "host" else "CUDA round trip"
        ax_quality.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
        ax_quality.set_ylabel("Mean lookup saving vs Paper (%)")
        ax_quality.set_title(f"{label}: paired lookup quality")
        ax_runtime.set_ylabel("95th percentile of case p95 (ms)")
        ax_runtime.set_title(f"{label}: selector scaling")
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    handles2, labels2 = axes[-1, 1].get_legend_handles_labels()
    merged = dict(zip(labels + labels2, handles + handles2))
    fig.legend(merged.values(), merged.keys(), loc="outside lower center", ncol=2, frameon=False)
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
                 args: argparse.Namespace) -> None:
    lines = [
        "# Experiment 22 보고서: Paper vs Predicted-lambda TD-2L(8) + Endpoint Trim",
        "",
        "Paper는 논문의 고정 row-budget 알고리즘을 그대로 실행했다. 각 paired input에서 "
        "Paper가 달성한 importance를 `Q = I(M_paper)`로 두고 proposed method가 최소한 같은 "
        "importance를 유지하도록 했다. 따라서 아래 lookup 비교는 importance-matched 비교다.",
        "",
        "논문 Table 2에는 총 16개 matrix shape가 있으며 모두 포함했다. importance는 실제 "
        f"activation trace가 아니라 CV `{args.cv:g}`인 synthetic lognormal이고, lookup latency는 "
        f"노트북 NVMe 실측값이 아니라 `{args.profile}` 공개 profile의 예측값이다.",
        "",
        "## 전체 결과",
        "",
    ]
    for track, track_label in (("host", "Host input -> CPU mask"),
                               ("cuda", "CUDA input -> CUDA mask")):
        selected = summary[summary.track == track]
        if selected.empty:
            continue
        lines.extend([
            f"### {track_label}", "",
            "| 방법 | 중앙 runtime | case-p95 | worst case-p95 | 유효+2ms | Paper 대비 평균/가중 lookup | Paper보다 비악화 | trim 순증분 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        indexed = selected.set_index("method").reindex(METHODS)
        for method, row in indexed.iterrows():
            lines.append(
                f"| {METHOD_LABELS[method]} | {fmt(row.runtime_median_ms)} ms "
                f"| {fmt(row.runtime_case_p95_ms)} ms "
                f"| {fmt(row.runtime_worst_case_p95_ms)} ms "
                f"| {100 * row.valid_and_deadline_pass_rate:.1f}% "
                f"| {fmt_pct(row.lookup_saving_vs_paper_pct_mean_valid)} / "
                f"{fmt_pct(row.lookup_saving_vs_paper_pct_weighted_valid)} "
                f"| {100 * row.lookup_nonregression_rate_valid:.1f}% "
                f"| {fmt_pct(row.lookup_trim_gain_pct_mean_valid)} |"
            )
        lines.append("")

    primary_track = "cuda" if "cuda" in set(summary.track) else "host"
    primary = summary[summary.track == primary_track].set_index("method")
    candidates = primary.loc[list(METHODS[2:])]
    passing = candidates[candidates.valid_and_deadline_pass_rate == 1.0]
    if len(passing):
        winner = passing.sort_values(
            "lookup_saving_vs_paper_pct_mean_valid", ascending=False
        ).iloc[0]
        verdict = (
            f"`{primary_track}` track에서 모든 case가 유효하고 2 ms를 통과한 trim 설정 중 "
            f"lookup 품질이 가장 높은 것은 `{winner.method_label}`이다. Paper 대비 평균 "
            f"lookup 절감은 `{winner.lookup_saving_vs_paper_pct_mean_valid:.2f}%`, case-p95는 "
            f"`{winner.runtime_case_p95_ms:.3f} ms`다."
        )
    else:
        winner = candidates.sort_values(
            "lookup_saving_vs_paper_pct_mean_valid", ascending=False
        ).iloc[0]
        verdict = (
            f"`{primary_track}` track에서는 모든 case에서 2 ms를 통과한 trim 설정이 없다. "
            f"품질 최고 설정은 `{winner.method_label}`이고 Paper 대비 평균 lookup 절감은 "
            f"`{winner.lookup_saving_vs_paper_pct_mean_valid:.2f}%`다."
        )
    regression_text = (
        f"`{winner.method_label}`은 {int(winner.valid_cases)}개 paired case 중 "
        f"`{int(winner.lookup_regression_cases_valid)}`개에서 Paper보다 느린 lookup을 "
        f"보였고, 비악화율은 `{100 * winner.lookup_nonregression_rate_valid:.1f}%`다. "
        f"절대 latency로 가중한 전체 절감은 "
        f"`{winner.lookup_saving_vs_paper_pct_weighted_valid:.2f}%`다. Endpoint trim 자체는 "
        f"평균 `{winner.lookup_trim_gain_pct_mean_valid:.2f}%`를 추가로 줄였지만, noisy lookup "
        f"table 기준으로는 `{100 * winner.trim_lookup_regression_rate_valid:.1f}%` case에서 "
        "소폭 악화됐다. Two-line 목적의 악화는 0건이다."
    )
    primary_shapes = shape_summary[
        (shape_summary.track == primary_track)
        & (shape_summary.method == winner.name)
    ]
    fully_passing_shapes = int((primary_shapes.valid_and_deadline_pass_rate == 1.0).sum())
    slower_shapes = primary_shapes[
        primary_shapes.lookup_saving_vs_paper_pct_mean_valid < 0.0
    ]["shape"].tolist()
    shape_text = (
        f"Shape 단위로는 `{fully_passing_shapes}/{len(primary_shapes)}`개가 모든 case에서 "
        "2 ms를 통과했다. "
        + (
            f"평균 lookup이 Paper보다 나쁜 shape는 `{', '.join(slower_shapes)}`다."
            if slower_shapes else "모든 shape에서 평균 lookup이 Paper보다 낮았다."
        )
    )
    lines.extend(["## 판정", "", verdict, "", regression_text, "", shape_text, ""])

    for track, track_label in (("host", "Host"), ("cuda", "CUDA round trip")):
        selected = shape_summary[
            (shape_summary.track == track)
            & (shape_summary.method == "predict_correct_c8_trim256")
        ].sort_values(["n", "d"])
        if selected.empty:
            continue
        lines.extend([
            f"## Shape별 trim 256 ({track_label})", "",
            "| Shape | rows | row KiB | runtime case-p95 | worst p95 | Paper 대비 lookup | 유효+2ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for _, row in selected.iterrows():
            lines.append(
                f"| {row['shape']} | {int(row.n):,} | {row.row_size_kib:.2f} "
                f"| {fmt(row.runtime_case_p95_ms)} ms "
                f"| {fmt(row.runtime_worst_case_p95_ms)} ms "
                f"| {fmt_pct(row.lookup_saving_vs_paper_pct_mean_valid)} "
                f"| {100 * row.valid_and_deadline_pass_rate:.1f}% |"
            )
        lines.append("")

    lines.extend([
        "## 측정 범위와 한계", "",
        "- `host`: host float32 importance에서 CPU bool mask까지 측정했다. CUDA가 있으면 Paper native 구현의 GPU sort도 포함된다.",
        "- `cuda`: 미리 만들어 둔 CUDA float32 importance에서 시작해 CPU proposal의 D2H, float64 변환, 8회 이하 DP, mask 복원, trim, H2D 및 synchronization을 모두 포함했다.",
        "- 최초 JIT/native compilation, importance 생성, 실제 NVMe I/O, model compute는 제외했다.",
        "- lookup latency는 Orin AGX profile 기반이므로 RTX 3050 노트북 selector 시간과 서로 다른 축이다. Jetson의 2 ms 충족 여부는 Jetson에서 다시 측정해야 한다.",
        "- endpoint trim은 two-line 목적을 단조 감소시키지만 측정 잡음이 있는 lookup table의 각 개별 점까지 단조 감소한다고 보장하지는 않는다.",
        "",
        "![Overall runtime-quality](results/runtime_quality.png)", "",
        "![Shape comparison](results/shape_comparison.png)", "",
    ])
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    frame = add_paired_metrics(frame)
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    summary_frame, shape_frame, summary = summarize(frame, args)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    plot_by_shape(shape_frame, args.output_dir / "shape_comparison.png")
    write_report(summary_frame, shape_frame, args)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    shapes, tracks = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lookup_table = BASE.LatencyTable.load(args.profile)
    if not args.analyze_only:
        if not args.skip_self_check:
            self_check(lookup_table, args.saturation_kib)
        rows, timing_rows, model_rows = collect(args, shapes, tracks, lookup_table)
        write_csv(args.output_dir / "trials.csv", rows)
        write_csv(args.output_dir / "timing_samples.csv", timing_rows)
        write_csv(args.output_dir / "models.csv", model_rows)
        metadata = {
            "shape_specs": shapes,
            "method_labels": METHOD_LABELS,
            "requested_tracks": args.tracks,
            "executed_tracks": tracks,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    analyze(args)


if __name__ == "__main__":
    main()
