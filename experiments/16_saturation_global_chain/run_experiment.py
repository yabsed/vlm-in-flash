#!/usr/bin/env python3
"""Experiment 16: exact supported frontier, Quant, and Paper greedy.

The optimization surrogate is the continuous two-line latency model.  The
primary evaluator is exactly the Experiment 15 released Orin AGX lookup rule:
table lookup through 255 KiB and endpoint-proportional scaling afterwards.

``Global supported`` adaptively enumerates the complete strongly-supported
frontier of exact scalarized binary-chain solutions. ``Quant`` is Experiment
13's fixed q=131,072 lambda grid. Neither is an exact oracle for unsupported
points of the constrained coverage problem.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_13 = (
    PROJECT_ROOT / "experiments" / "13_two_line_lagrangian" / "run_experiment.py"
)
DEFAULT_SOURCE = PROJECT_ROOT / "experiments" / "13_two_line_lagrangian" / "results"
DEFAULT_INPUT_SUMMARY = (
    PROJECT_ROOT / "experiments" / "10_exact_lookup_recheck" / "results" / "summary.json"
)
GLOBAL_COLOR = "#2A9D8F"
QUANT_COLOR = "#E6A700"


def load_experiment_13():
    spec = importlib.util.spec_from_file_location("experiment_13_for_16", EXPERIMENT_13)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 13 from {EXPERIMENT_13}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP13 = load_experiment_13()
BASE = EXP13.BASE
SPATIAL_MODES = EXP13.SPATIAL_MODES
SPATIAL_LABELS = EXP13.SPATIAL_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--input-limit", type=int, default=None,
        help="development-only number of spatial inputs",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def summarize(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def configure_plot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    return plt


def run_lengths(encoded: str) -> np.ndarray:
    return EXP13.EXP11.run_lengths(encoded)


def shortfall_cost(lengths: np.ndarray, model: dict) -> float:
    c2 = float(model["c2_ms_per_row"])
    delta = float(model["d_ms_per_excess_row"])
    saturation = float(model["saturation_rows"])
    return float(
        c2 * lengths.sum()
        + delta * np.maximum(saturation - lengths, 0).sum()
    )


def capped_run_penalized(values: np.ndarray, multiplier: float, c1: float,
                         c2: float, saturation: int) -> tuple[float, float, float]:
    """Reference O(Ns) binary-chain DP for an integer saturation length."""
    delta = c2 - c1
    opening = delta * saturation + c1
    scores = np.full(saturation + 1, -np.inf, dtype=np.float64)
    importance = np.full(saturation + 1, -np.inf, dtype=np.float64)
    costs = np.full(saturation + 1, np.inf, dtype=np.float64)
    scores[0] = importance[0] = costs[0] = 0.0

    def better(score, kept, cost, old_score, old_kept, old_cost) -> bool:
        if np.isneginf(old_score):
            return not np.isneginf(score)
        if np.isneginf(score):
            return False
        scale = max(1.0, abs(score), abs(old_score))
        tolerance = 1e-13 * scale
        return bool(
            score > old_score + tolerance
            or (
                abs(score - old_score) <= tolerance
                and (
                    kept > old_kept + 1e-14
                    or (
                        abs(kept - old_kept) <= 1e-14
                        and cost < old_cost - 1e-14
                    )
                )
            )
        )

    for value in np.asarray(values, dtype=np.float64):
        next_scores = np.full_like(scores, -np.inf)
        next_importance = np.full_like(importance, -np.inf)
        next_costs = np.full_like(costs, np.inf)

        for state in range(saturation + 1):
            if better(
                scores[state], importance[state], costs[state],
                next_scores[0], next_importance[0], next_costs[0],
            ):
                next_scores[0] = scores[state]
                next_importance[0] = importance[state]
                next_costs[0] = costs[state]

        score = scores[0] + multiplier * value - opening
        kept = importance[0] + value
        cost = costs[0] + opening
        if better(score, kept, cost, next_scores[1], next_importance[1], next_costs[1]):
            next_scores[1], next_importance[1], next_costs[1] = score, kept, cost

        for state in range(1, saturation):
            target = state + 1
            score = scores[state] + multiplier * value - c1
            kept = importance[state] + value
            cost = costs[state] + c1
            if better(
                score, kept, cost,
                next_scores[target], next_importance[target], next_costs[target],
            ):
                next_scores[target] = score
                next_importance[target] = kept
                next_costs[target] = cost

        score = scores[saturation] + multiplier * value - c2
        kept = importance[saturation] + value
        cost = costs[saturation] + c2
        if better(
            score, kept, cost,
            next_scores[saturation], next_importance[saturation], next_costs[saturation],
        ):
            next_scores[saturation] = score
            next_importance[saturation] = kept
            next_costs[saturation] = cost

        scores, importance, costs = next_scores, next_importance, next_costs

    best = 0
    for state in range(1, saturation + 1):
        if better(
            scores[state], importance[state], costs[state],
            scores[best], importance[best], costs[best],
        ):
            best = state
    return float(scores[best]), float(importance[best]), float(costs[best])


def self_check_capped_dp() -> None:
    """Check the explicit O(Ns) recurrence against the exact O(N) solver."""
    rng = np.random.default_rng(1616)
    saturation = 4
    c1 = 0.07
    c2 = 0.13
    a = (c2 - c1) * saturation
    for n in (7, 12, 17):
        for _ in range(3):
            values = rng.lognormal(size=n)
            values /= values.sum()
            for multiplier in (0.0, 0.1, 1.0, 10.0):
                actual = capped_run_penalized(
                    values, multiplier, c1, c2, saturation
                )
                expected = EXP13._solve_lambda_metrics(
                    values, multiplier, a, c1, c2, float(saturation), saturation
                )[:3]
                if not all(
                    math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-11)
                    for left, right in zip(actual, expected)
                ):
                    raise RuntimeError(
                        f"capped-run DP self-check failed: {actual} != {expected}"
                    )


def node_key(node: dict) -> tuple:
    return (
        round(float(node["importance"]), 13),
        round(float(node["two_line_ms"]), 13),
        int(node["rows"]),
        int(node["chunks"]),
    )


class ExactSupportedOracle:
    """Enumerate all strongly supported scalarized two-line optima.

    Adjacent candidate points define the only multiplier at which another
    supported point can lie above their chord. Recursively solving at those
    intersections enumerates the lower convex envelope without a lambda grid.
    Random continuous importance makes collinear weakly-supported points a
    probability-zero event; those points are outside this oracle's claim.
    """

    def __init__(self, values: np.ndarray, model: dict) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.model = model
        self.solve_calls = 0
        self._cache: dict[float, dict] = {}

        left = self._solve_metrics(0.0)
        minimum_value = float(self.values.min())
        high = 2.0 * float(model["c2_ms_per_row"]) * len(values) / minimum_value
        right = self._solve_metrics(high)
        for _ in range(20):
            if right["importance"] >= float(self.values.sum()) - 1e-12:
                break
            high *= 2.0
            right = self._solve_metrics(high)
        else:
            raise RuntimeError("could not reach the full-selection endpoint")

        nodes = {node_key(left): left, node_key(right): right}
        stack = [(left, right)]
        visited_pairs: set[tuple] = set()
        while stack:
            low, high_node = stack.pop()
            pair_key = (node_key(low), node_key(high_node))
            if pair_key in visited_pairs:
                continue
            visited_pairs.add(pair_key)
            delta_i = high_node["importance"] - low["importance"]
            if delta_i <= 1e-13:
                continue
            multiplier = (
                high_node["two_line_ms"] - low["two_line_ms"]
            ) / delta_i
            if multiplier <= 0 or not np.isfinite(multiplier):
                continue
            middle = self._solve_metrics(multiplier)
            middle_key = node_key(middle)
            if middle_key in (node_key(low), node_key(high_node)):
                continue
            tolerance = 1e-11
            if not (
                low["importance"] + tolerance
                < middle["importance"]
                < high_node["importance"] - tolerance
            ):
                raise RuntimeError("supported-frontier recursion lost importance order")
            nodes[middle_key] = middle
            stack.append((low, middle))
            stack.append((middle, high_node))
            if len(nodes) > 10000:
                raise RuntimeError("supported-frontier enumeration exceeded safety cap")

        self.nodes = sorted(nodes.values(), key=lambda item: item["importance"])
        costs = np.asarray([node["two_line_ms"] for node in self.nodes])
        if np.any(np.diff(costs) < -1e-11):
            raise RuntimeError("supported frontier is not cost-monotone")

    def _solve_metrics(self, multiplier: float) -> dict:
        key = float(multiplier)
        if key not in self._cache:
            result = EXP13._solve_lambda_metrics(
                self.values, key,
                self.model["a_ms"], self.model["c1_ms_per_row"],
                self.model["c2_ms_per_row"], self.model["saturation_rows"],
                self.model["short_max_rows"],
            )
            self.solve_calls += 1
            self._cache[key] = {
                "lambda": key,
                "score": float(result[0]),
                "importance": float(result[1]),
                "two_line_ms": float(result[2]),
                "rows": int(result[3]),
                "chunks": int(result[4]),
                "excess_rows": float(result[5]),
            }
        return self._cache[key]

    def candidate(self, bound: float) -> dict:
        feasible = [
            node for node in self.nodes
            if node["importance"] >= bound - 1e-14
        ]
        if not feasible:
            raise RuntimeError("supported frontier missed a feasible full endpoint")
        return min(
            feasible,
            key=lambda node: (node["two_line_ms"], -node["importance"]),
        )

    def solve(self, bound: float) -> dict:
        node = self.candidate(bound)
        replay = EXP13._solve_lambda_mask(
            self.values, node["lambda"],
            self.model["a_ms"], self.model["c1_ms_per_row"],
            self.model["c2_ms_per_row"], self.model["saturation_rows"],
            self.model["short_max_rows"],
        )
        if not (
            math.isclose(float(replay[2]), node["importance"], rel_tol=1e-10, abs_tol=1e-11)
            and math.isclose(float(replay[3]), node["two_line_ms"], rel_tol=1e-10, abs_tol=1e-11)
        ):
            raise RuntimeError("supported mask replay differs from frontier metrics")
        return {**node, "mask": replay[0]}


def self_check_supported_oracle() -> None:
    """Compare adaptive enumeration with a brute-force lower convex hull."""
    rng = np.random.default_rng(1617)
    saturation = 3
    c1 = 0.07
    c2 = 0.13
    model = {
        "a_ms": (c2 - c1) * saturation,
        "c1_ms_per_row": c1,
        "c2_ms_per_row": c2,
        "d_ms_per_excess_row": c2 - c1,
        "saturation_rows": float(saturation),
        "short_max_rows": saturation,
    }
    for _ in range(3):
        n = 9
        values = rng.lognormal(size=n)
        values /= values.sum()
        points = []
        for bits in range(1 << n):
            mask = np.asarray([(bits >> index) & 1 for index in range(n)], dtype=bool)
            encoded = EXP13.EXP10.encode_runs(mask)
            lengths = run_lengths(encoded)
            points.append((
                float(values[mask].sum()),
                shortfall_cost(lengths, model),
            ))

        # Remove dominated points, then take the lower convex envelope. Slopes
        # of consecutive envelope segments must be strictly increasing.
        points.sort(key=lambda item: (item[0], item[1]))
        nondominated_reversed = []
        minimum_cost = math.inf
        for point in reversed(points):
            if point[1] < minimum_cost - 1e-12:
                nondominated_reversed.append(point)
                minimum_cost = point[1]
        nondominated = list(reversed(nondominated_reversed))
        hull: list[tuple[float, float]] = []
        for point in nondominated:
            while len(hull) >= 2:
                first, second = hull[-2], hull[-1]
                left_slope = (second[1] - first[1]) / (second[0] - first[0])
                right_slope = (point[1] - second[1]) / (point[0] - second[0])
                if left_slope < right_slope - 1e-11:
                    break
                hull.pop()
            hull.append(point)

        oracle = ExactSupportedOracle(values, model)
        expected = {(round(i, 11), round(cost, 11)) for i, cost in hull}
        actual = {
            (round(node["importance"], 11), round(node["two_line_ms"], 11))
            for node in oracle.nodes
        }
        if actual != expected:
            raise RuntimeError(
                f"supported-frontier self-check failed: {actual ^ expected}"
            )


def metric_fields(prefix: str, metrics: dict) -> dict:
    fields = (
        "importance", "rows", "chunks", "runs", "run_length_median",
        "rows_beyond_profile_fraction", "aff_ms", "two_line_ms",
        "released_ms", "tail_linear_ms", "block_split_ms",
    )
    return {f"{prefix}_{field}": metrics[field] for field in fields}


def validate_and_record_chunks(base: dict, method: str, metrics: dict,
                               model: dict, output: list[dict]) -> None:
    lengths = run_lengths(metrics["runs"])
    if len(lengths) != int(metrics["chunks"]):
        raise RuntimeError(f"run-count mismatch for {method}")
    if int(lengths.sum()) != int(metrics["rows"]):
        raise RuntimeError(f"row-count mismatch for {method}")
    reconstructed = shortfall_cost(lengths, model)
    if not math.isclose(
        reconstructed, float(metrics["two_line_ms"]), rel_tol=1e-10, abs_tol=1e-11
    ):
        raise RuntimeError(f"shortfall identity mismatch for {method}")
    for length in lengths:
        output.append({
            **base,
            "method": method,
            "chunk_rows": int(length),
            "is_saturated": bool(length >= float(model["saturation_rows"])),
        })


def build_trials(source: pd.DataFrame, values_by_input: dict, model: dict,
                 affine: tuple[float, float], row_size_kib: float, policies,
                 input_limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired_rows: list[dict] = []
    length_rows: list[dict] = []
    input_rows: list[dict] = []
    grouped = source.groupby(["trial", "target_cv", "spatial_mode"], sort=True)

    for input_index, ((trial, target_cv, spatial_mode), group) in enumerate(grouped):
        if input_limit is not None and input_index >= input_limit:
            break
        input_id = (int(trial), round(float(target_cv), 10), spatial_mode)
        values = values_by_input[input_id]
        started = time.perf_counter_ns()
        oracle = ExactSupportedOracle(values, model)
        build_ms = (time.perf_counter_ns() - started) / 1e6
        input_rows.append({
            "trial": int(trial),
            "target_cv": float(target_cv),
            "spatial_mode": spatial_mode,
            "supported_points": len(oracle.nodes),
            "scalarized_solve_calls": oracle.solve_calls,
            "global_build_runtime_ms": build_ms,
        })

        global_cache: dict[tuple, dict] = {}
        for row in group.sort_values("budget_fraction").to_dict("records"):
            base = {
                "trial": int(trial),
                "target_cv": float(target_cv),
                "spatial_mode": spatial_mode,
                "budget_fraction": float(row["budget_fraction"]),
                "budget_rows": int(row["budget_rows"]),
                "target_importance": float(row["target_importance"]),
            }
            paper_mask = EXP13.EXP10.decode_runs(row["paper_runs"], len(values))
            quant_mask = EXP13.EXP10.decode_runs(row["lag_runs"], len(values))
            paper = EXP13.mask_metrics(
                paper_mask, values, model, affine, row_size_kib, policies
            )
            quant = EXP13.mask_metrics(
                quant_mask, values, model, affine, row_size_kib, policies
            )

            node = oracle.candidate(base["target_importance"])
            key = node_key(node)
            if key not in global_cache:
                solution = oracle.solve(base["target_importance"])
                global_cache[key] = EXP13.mask_metrics(
                    solution["mask"], values, model, affine, row_size_kib, policies
                )
            global_supported = global_cache[key]

            for method, metrics in (
                ("paper", paper), ("quant", quant),
                ("global_supported", global_supported),
            ):
                if metrics["importance"] < base["target_importance"] - 1e-12:
                    raise RuntimeError(f"{method} missed the Paper importance target")
                validate_and_record_chunks(base, method, metrics, model, length_rows)

            if global_supported["two_line_ms"] > quant["two_line_ms"] + 1e-11:
                raise RuntimeError("complete supported frontier is worse than Quant")

            output = {
                **base,
                **metric_fields("paper", paper),
                **metric_fields("quant", quant),
                **metric_fields("global_supported", global_supported),
                "global_quant_same_mask": float(
                    global_supported["runs"] == quant["runs"]
                ),
            }
            for method in ("quant", "global_supported"):
                output[f"{method}_importance_overshoot"] = (
                    output[f"{method}_importance"] - output["target_importance"]
                )
                for policy in ("two_line", "released", "tail_linear", "block_split"):
                    method_ms = output[f"{method}_{policy}_ms"]
                    paper_ms = output[f"paper_{policy}_ms"]
                    output[f"{method}_saving_vs_paper_{policy}_pct"] = 100.0 * (
                        1.0 - method_ms / paper_ms
                    )
                    output[f"{method}_win_{policy}"] = float(
                        method_ms < paper_ms - 1e-12
                    )
            for policy in ("two_line", "released", "tail_linear", "block_split"):
                output[f"global_saving_vs_quant_{policy}_pct"] = 100.0 * (
                    1.0
                    - output[f"global_supported_{policy}_ms"]
                    / output[f"quant_{policy}_ms"]
                )
            paired_rows.append(output)

        print(
            f"completed {input_index + 1}/{len(grouped)} inputs: "
            f"{len(oracle.nodes)} supported points, {build_ms:.1f} ms",
            flush=True,
        )

    return (
        pd.DataFrame(paired_rows),
        pd.DataFrame(length_rows),
        pd.DataFrame(input_rows),
    )


def build_summary(source_summary: dict, input_summary: dict,
                  paired: pd.DataFrame, chunks: pd.DataFrame,
                  inputs: pd.DataFrame, args: argparse.Namespace, policies) -> dict:
    latency_policies = ("two_line", "released", "tail_linear", "block_split")
    method_structure = {}
    for method in ("global_supported", "quant", "paper"):
        method_chunks = chunks[chunks.method == method]
        shortfall = []
        for encoded in paired[f"{method}_runs"]:
            lengths = run_lengths(encoded)
            shortfall.append(
                float(np.maximum(
                    source_summary["two_line_model"]["saturation_rows"] - lengths, 0
                ).sum())
            )
        method_structure[method] = {
            "rows": summarize(paired[f"{method}_rows"]),
            "chunks_per_solution": summarize(paired[f"{method}_chunks"]),
            "shortfall_rows_per_solution": summarize(shortfall),
            "chunk_rows": summarize(method_chunks.chunk_rows),
            "saturated_chunk_fraction": float(method_chunks.is_saturated.mean()),
        }

    comparisons = {}
    for method in ("global_supported", "quant"):
        comparisons[method] = {
            policy: {
                **summarize(paired[f"{method}_saving_vs_paper_{policy}_pct"]),
                "strict_win_rate": float(paired[f"{method}_win_{policy}"].mean()),
            }
            for policy in latency_policies
        }

    return {
        "format": "experiment-16-released-global-quant-v2",
        "source_experiment": "13_two_line_lagrangian",
        "source_results": str(args.source_results.resolve()),
        "source_sha256": {
            name: file_sha256(args.source_results / name)
            for name in ("summary.json", "paper_matched_trials.csv")
        },
        "n": int(source_summary["n"]),
        "q": int(source_summary["q"]),
        "primary_latency_evaluator": {
            "name": "released Orin AGX lookup with endpoint-proportional tail",
            "measured_max_kib": policies.max_kib,
            "measured_max_rows": policies.cutoff_rows,
            "tail_ms_per_row": policies.released_tail_ms_per_row,
            "same_rule_as_experiment_15": True,
        },
        "optimization_surrogate": source_summary["two_line_model"],
        "method_scope": {
            "global_supported": (
                "adaptive enumeration of all strongly supported exact scalarized "
                "two-line optima; not an exact constrained-coverage oracle"
            ),
            "quant": (
                f"fixed q={source_summary['q']} lambda grid of exact scalarized "
                "two-line optima; not an exact constrained-coverage oracle"
            ),
            "paper": "Neuron Chunking Paper greedy at its fixed-R operating points",
        },
        "cases": {
            "spatial_inputs": int(len(inputs)),
            "paired_targets": int(len(paired)),
            "global_supported_chunks": int((chunks.method == "global_supported").sum()),
            "quant_chunks": int((chunks.method == "quant").sum()),
            "paper_chunks": int((chunks.method == "paper").sum()),
        },
        "solver_validation": (
            "The explicit O(Ns) capped-run recurrence was checked against the exact "
            "O(N) interval solver; the row-plus-shortfall identity was checked for "
            "every saved mask."
        ),
        "saving_vs_paper_pct": comparisons,
        "global_saving_vs_quant_pct": {
            policy: summarize(paired[f"global_saving_vs_quant_{policy}_pct"])
            for policy in latency_policies
        },
        "global_quant_same_mask_rate": float(paired.global_quant_same_mask.mean()),
        "importance_overshoot": {
            method: summarize(paired[f"{method}_importance_overshoot"])
            for method in ("global_supported", "quant")
        },
        "solution_structure": method_structure,
        "global_runtime_ms": summarize(inputs.global_build_runtime_ms),
        "global_supported_points": summarize(inputs.supported_points),
        "global_scalarized_solve_calls": summarize(inputs.scalarized_solve_calls),
        "by_spatial_mode_released_saving_pct": {
            method: {
                mode: summarize(group[f"{method}_saving_vs_paper_released_pct"])
                for mode, group in paired.groupby("spatial_mode", sort=True)
            }
            for method in ("global_supported", "quant")
        },
        "by_cv_released_saving_pct": {
            method: {
                str(cv): summarize(group[f"{method}_saving_vs_paper_released_pct"])
                for cv, group in paired.groupby("target_cv", sort=True)
            }
            for method in ("global_supported", "quant")
        },
        "by_budget_released_saving_pct": {
            method: {
                str(budget): summarize(group[f"{method}_saving_vs_paper_released_pct"])
                for budget, group in paired.groupby("budget_fraction", sort=True)
            }
            for method in ("global_supported", "quant")
        },
        "input_provenance": {
            "summary": str(args.input_summary.resolve()),
            "sha256": file_sha256(args.input_summary),
            "seed": input_summary["seed"],
        },
    }


def plot_method_grid(frame: pd.DataFrame, path: Path, x_fields: dict,
                     y_fields: dict, xlabel: str, ylabel: str,
                     x_log: bool = False) -> None:
    plt = configure_plot()
    display_cvs = (1.25, 3.30, 4.55)
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 14.5), constrained_layout=True)
    specs = (
        ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "--", 1.8),
        ("quant", "Quant (q=131,072)", QUANT_COLOR, "P", ":", 2.0),
        ("global_supported", "Global supported", GLOBAL_COLOR, "X", "-", 2.2),
    )
    for row_index, cv in enumerate(display_cvs):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            selected = frame[
                np.isclose(frame.target_cv, cv) & (frame.spatial_mode == mode)
            ]
            grouped = selected.groupby("budget_fraction", sort=True)
            for prefix, label, color, marker, linestyle, linewidth in specs:
                curve = grouped[[x_fields[prefix], y_fields[prefix]]].mean()
                ax.plot(
                    curve[x_fields[prefix]], curve[y_fields[prefix]],
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=linewidth, label=label,
                )
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


def plot_chunk_histogram(chunks: pd.DataFrame, saturation_rows: float,
                         path: Path) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=True,
                             constrained_layout=True)
    specs = (
        ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "--", 1.8),
        ("quant", "Quant (q=131,072)", QUANT_COLOR, ":", 2.0),
        ("global_supported", "Global supported", GLOBAL_COLOR, "-", 2.2),
    )
    maximum = int(chunks.chunk_rows.max())
    edges = np.unique(np.r_[
        0.5,
        np.rint(np.geomspace(1, maximum + 1, 45)) + 0.5,
    ])
    for ax, mode in zip(axes, SPATIAL_MODES):
        selected = chunks[chunks.spatial_mode == mode]
        for method, label, color, linestyle, linewidth in specs:
            values = selected[selected.method == method].chunk_rows.to_numpy()
            ax.hist(
                values, bins=edges, weights=np.ones(len(values)) / len(values),
                histtype="step", linewidth=linewidth, linestyle=linestyle,
                color=color, label=label,
            )
        ax.axvline(
            saturation_rows, color="#334155", linewidth=1.2, linestyle="-.",
            label=f"two-line s = {saturation_rows:.1f}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Chunk length (rows, log scale)")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Fraction of chunks")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_plots(paired: pd.DataFrame, chunks: pd.DataFrame,
                 model: dict, output_dir: Path) -> None:
    methods = ("global_supported", "quant", "paper")
    plot_method_grid(
        paired, output_dir / "importance_latency.png",
        {method: f"{method}_released_ms" for method in methods},
        {method: f"{method}_importance" for method in methods},
        "Released lookup latency L (ms, log scale)", "Retained importance I", True,
    )
    plot_method_grid(
        paired, output_dir / "importance_rows.png",
        {method: f"{method}_rows" for method in methods},
        {method: f"{method}_importance" for method in methods},
        "Selected rows R", "Retained importance I",
    )
    plot_method_grid(
        paired, output_dir / "latency_rows.png",
        {method: f"{method}_rows" for method in methods},
        {method: f"{method}_released_ms" for method in methods},
        "Selected rows R", "Released lookup latency L (ms)",
    )
    plot_chunk_histogram(
        chunks, float(model["saturation_rows"]),
        output_dir / "chunk_length_histogram.png",
    )


def main() -> None:
    args = parse_args()
    summary_path = args.source_results / "summary.json"
    trials_path = args.source_results / "paper_matched_trials.csv"
    for path in (summary_path, trials_path, args.input_summary):
        if not path.is_file():
            raise SystemExit(f"missing source result: {path}")

    source_summary = json.loads(summary_path.read_text())
    input_summary = json.loads(args.input_summary.read_text())
    if source_summary.get("format") != "experiment-13-two-line-lagrangian-v1":
        raise SystemExit("--source-results is not an Experiment 13 result directory")
    if input_summary.get("format") != "experiment-10-exact-lookup-recheck-v1":
        raise SystemExit("--input-summary is not an Experiment 10 summary")

    self_check_capped_dp()
    self_check_supported_oracle()
    model = source_summary["two_line_model"]
    row_size_kib = float(input_summary["row_size_kib"])
    table = BASE.LatencyTable.load(input_summary["profile"])
    affine_a, affine_c, _ = BASE.affine_fit(table, row_size_kib)
    affine = (affine_a, affine_c)
    policies = EXP13.EXP11.LatencyPolicies(table, row_size_kib, 128)
    values_by_input = EXP13.generated_inputs(input_summary)
    source = pd.read_csv(trials_path)

    warm_values = np.full(16, 1 / 16, dtype=np.float64)
    EXP13._solve_lambda_metrics(
        warm_values, 1.0, model["a_ms"], model["c1_ms_per_row"],
        model["c2_ms_per_row"], model["saturation_rows"],
        model["short_max_rows"],
    )

    paired, chunks, inputs = build_trials(
        source, values_by_input, model, affine, row_size_kib, policies,
        args.input_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_trials.csv", index=False)
    chunks.to_csv(args.output_dir / "chunk_lengths.csv", index=False)
    inputs.to_csv(args.output_dir / "input_trials.csv", index=False)
    summary = build_summary(
        source_summary, input_summary, paired, chunks, inputs, args, policies
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_plots(paired, chunks, model, args.output_dir)
    print(json.dumps({
        method: summary["saving_vs_paper_pct"][method]["released"]
        for method in ("global_supported", "quant")
    }, indent=2))
    print(f"wrote Experiment 16 results to {args.output_dir}")


if __name__ == "__main__":
    main()
