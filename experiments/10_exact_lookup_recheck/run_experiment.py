#!/usr/bin/env python3
"""Experiment 10: re-evaluate Experiment 07 with the released lookup table.

Experiment 07's expensive O(N^3) affine DP tables are not rebuilt.  Its saved
exact terminal states (R, K, importance) are replayed with a DP truncated to
the largest required K, allowing the corresponding masks to be reconstructed
in O(N R K_max) time.  Every mask is persisted as selected-run RLE so future
latency tables can be evaluated without any selection or DP recomputation.
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
import torch

try:
    from numba import njit
except ImportError as error:  # pragma: no cover
    raise SystemExit("Experiment 10 requires numba") from error


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_07 = PROJECT_ROOT / "experiments" / "07_exact_n4864" / "run_experiment.py"
DEFAULT_SOURCE = PROJECT_ROOT / "experiments" / "07_exact_n4864" / "results"


def load_experiment_07():
    spec = importlib.util.spec_from_file_location("experiment_07_for_10", EXPERIMENT_07)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 07 from {EXPERIMENT_07}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP7 = load_experiment_07()
BASE = EXP7.BASE
EXP2 = EXP7.EXP2
EXP6 = EXP7.EXP6
CONFIG = EXP7.CONFIG
SPATIAL_MODES = EXP7.SPATIAL_MODES
SPATIAL_LABELS = EXP7.SPATIAL_LABELS

INTEGER_COLUMNS = {
    "trial", "budget_rows", "paper_rows", "paper_chunks", "dpr_rows",
    "dpr_chunks", "top_rows", "top_chunks", "paper_cover_rows",
    "paper_cover_chunks", "dpr_cover_rows", "dpr_cover_chunks",
    "top_cover_rows", "top_cover_chunks", "dpr_iterations", "cover_rows",
    "cover_chunks",
}
TEXT_COLUMNS = {"spatial_mode"}
EXACT_PREFIXES = ("paper_cover", "dpr_cover", "top_cover")


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in TEXT_COLUMNS:
                    row[key] = value
                elif key in INTEGER_COLUMNS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


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


def assert_close(actual: float, expected: float, label: str,
                 tolerance: float = 2e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"Experiment 07 replay mismatch for {label}: {actual} != {expected}")


def input_key(row: dict) -> tuple:
    return int(row["trial"]), round(float(row["target_cv"]), 10), row["spatial_mode"]


def fixed_key(row: dict) -> tuple:
    return (*input_key(row), round(float(row["budget_fraction"]), 10))


def coverage_key(row: dict) -> tuple:
    return (*input_key(row), round(float(row["coverage_target"]), 10))


def run_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    selected = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, selected, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end - start)) for start, end in zip(starts, ends)]


def encode_runs(mask: np.ndarray) -> str:
    """Persist a full mask as semicolon-separated zero-based start:length runs."""
    return ";".join(f"{start}:{length}" for start, length in run_intervals(mask))


def decode_runs(encoded: str, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if encoded:
        for item in encoded.split(";"):
            start_text, length_text = item.split(":")
            start, length = int(start_text), int(length_text)
            mask[start:start + length] = True
    return mask


def lookup_latency(mask: np.ndarray, row_size_kib: float, table) -> float:
    return float(sum(
        table.read_ms(length * row_size_kib)
        for _, length in run_intervals(mask)
    ))


def mask_metrics(mask: np.ndarray, values: np.ndarray, a_ms: float,
                 c_ms: float, row_size_kib: float, table) -> dict:
    rows, chunks = BASE.mask_stats(mask)
    return {
        "importance": float(values[mask].sum()),
        "rows": rows,
        "chunks": chunks,
        "aff_ms": float(a_ms * chunks + c_ms * rows),
        "lookup_ms": lookup_latency(mask, row_size_kib, table),
        "runs": encode_runs(mask),
    }


@njit(cache=False)
def _exact_state_parents_kernel(values, rmax, kmax):
    """Exact chain DP truncated to states needed by Experiment 07 outputs.

    One byte stores both predecessor ending bits: bit 0 for an ending-zero
    state and bit 1 for an ending-one state.  This retains backtracking while
    avoiding the full O(N^3) parent tensor of the original generic oracle.
    """
    n = len(values)
    neg_inf = -np.inf
    end_zero = np.full((rmax + 1, kmax + 1), neg_inf, dtype=np.float64)
    end_one = np.full((rmax + 1, kmax + 1), neg_inf, dtype=np.float64)
    new_zero = np.full((rmax + 1, kmax + 1), neg_inf, dtype=np.float64)
    new_one = np.full((rmax + 1, kmax + 1), neg_inf, dtype=np.float64)
    parents = np.zeros((n + 1, rmax + 1, kmax + 1), dtype=np.uint8)
    end_zero[0, 0] = 0.0

    for i in range(1, n + 1):
        new_zero[:, :] = neg_inf
        new_one[:, :] = neg_inf
        value = values[i - 1]
        row_limit = min(i, rmax)
        for rows in range(row_limit + 1):
            zero_kmax = min(kmax, rows, i - rows)
            for chunks in range(zero_kmax + 1):
                skip_zero = end_zero[rows, chunks]
                skip_one = end_one[rows, chunks]
                if skip_zero >= skip_one:
                    new_zero[rows, chunks] = skip_zero
                else:
                    new_zero[rows, chunks] = skip_one
                    parents[i, rows, chunks] = np.uint8(1)

            if rows > 0:
                one_kmax = min(kmax, rows, i - rows + 1)
                for chunks in range(1, one_kmax + 1):
                    continue_run = end_one[rows - 1, chunks]
                    start_run = end_zero[rows - 1, chunks - 1]
                    if continue_run >= start_run:
                        new_one[rows, chunks] = value + continue_run
                        parents[i, rows, chunks] |= np.uint8(2)
                    else:
                        new_one[rows, chunks] = value + start_run

        swap = end_zero
        end_zero = new_zero
        new_zero = swap
        swap = end_one
        end_one = new_one
        new_one = swap
    return end_zero, end_one, parents


class ExactStateMaskOracle:
    """Reconstruct only the exact (R,K) states saved by Experiment 07."""

    def __init__(self, values: np.ndarray, rmax: int, kmax: int):
        self.values = np.asarray(values, dtype=np.float64)
        self.rmax = int(rmax)
        self.kmax = int(kmax)
        self.end_zero, self.end_one, self.parents = _exact_state_parents_kernel(
            self.values, self.rmax, self.kmax
        )
        self.buffer_bytes = (
            self.end_zero.nbytes + self.end_one.nbytes + self.parents.nbytes
        )

    def solve_state(self, rows: int, chunks: int) -> dict:
        rows, chunks = int(rows), int(chunks)
        if not (0 <= rows <= self.rmax and 0 <= chunks <= self.kmax):
            raise ValueError("requested state lies outside reconstructed DP")
        z = float(self.end_zero[rows, chunks])
        o = float(self.end_one[rows, chunks])
        state = 0 if z >= o else 1
        importance = z if state == 0 else o
        if not np.isfinite(importance):
            raise RuntimeError(f"unreachable exact state R={rows}, K={chunks}")

        mask = np.zeros(len(self.values), dtype=bool)
        remaining_rows, remaining_chunks = rows, chunks
        for i in range(len(self.values), 0, -1):
            code = int(self.parents[i, remaining_rows, remaining_chunks])
            if state == 0:
                state = code & 1
            else:
                mask[i - 1] = True
                previous_state = (code >> 1) & 1
                remaining_rows -= 1
                if previous_state == 0:
                    remaining_chunks -= 1
                state = previous_state
        if remaining_rows != 0 or remaining_chunks != 0 or state != 0:
            raise RuntimeError("exact-state backtracking reached an invalid initial state")
        return {"mask": mask, "importance": importance}


def distribution(values) -> dict:
    return EXP2.distribution_stats([float(value) for value in values])


def reconstruct_exact_masks(values: np.ndarray, coverage_group: list[dict],
                            fixed_group: list[dict]) -> tuple[dict, float, int]:
    expected_by_state: dict[tuple[int, int], float] = {}
    for row in coverage_group:
        state = int(row["cover_rows"]), int(row["cover_chunks"])
        expected = float(row["cover_importance"])
        if state in expected_by_state:
            assert_close(expected_by_state[state], expected, f"state {state} importance")
        expected_by_state[state] = expected
    for row in fixed_group:
        for prefix in EXACT_PREFIXES:
            state = int(row[f"{prefix}_rows"]), int(row[f"{prefix}_chunks"])
            expected = float(row[f"{prefix}_importance"])
            if state in expected_by_state:
                assert_close(expected_by_state[state], expected, f"state {state} importance")
            expected_by_state[state] = expected

    rmax = max(state[0] for state in expected_by_state)
    kmax = max(state[1] for state in expected_by_state)
    oracle, build_ms = timed_call(ExactStateMaskOracle, values, rmax, kmax)
    masks = {}
    for state, expected in expected_by_state.items():
        solution = oracle.solve_state(*state)
        assert_close(solution["importance"], expected, f"exact state {state}")
        rows, chunks = BASE.mask_stats(solution["mask"])
        if (rows, chunks) != state:
            raise RuntimeError(f"reconstructed mask has {(rows, chunks)}, expected {state}")
        masks[state] = solution["mask"]
    return masks, build_ms, oracle.buffer_bytes


def add_solution(row: dict, prefix: str, metrics: dict) -> None:
    for key in ("importance", "rows", "chunks", "aff_ms"):
        expected_key = f"{prefix}_{key}"
        if expected_key in row:
            expected = row[expected_key]
            if key in ("rows", "chunks"):
                if int(metrics[key]) != int(expected):
                    raise RuntimeError(f"{prefix} replay mismatch for {key}")
            else:
                assert_close(float(metrics[key]), float(expected), f"{prefix} {key}")
    row[f"{prefix}_lookup_ms"] = metrics["lookup_ms"]
    row[f"{prefix}_runs"] = metrics["runs"]


def aggregate_curve(rows: list[dict], target_cv: float, spatial_mode: str,
                    x_key: str, metrics: tuple[str, ...]) -> list[dict]:
    selected = [
        row for row in rows
        if float(row["target_cv"]) == target_cv
        and row["spatial_mode"] == spatial_mode
    ]
    return EXP2.summarize(selected, (x_key,), metrics)


def plot_lookup_frontiers(coverage_rows: list[dict], fixed_rows: list[dict],
                          path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    cvs = sorted({float(row["target_cv"]) for row in coverage_rows})
    display_cvs = [min(cvs, key=lambda value: abs(value - wanted))
                   for wanted in (1.25, 3.30, 4.55)]
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.0), constrained_layout=True)
    for row_index, cv in enumerate(display_cvs):
        for column_index, mode in enumerate(SPATIAL_MODES):
            ax = axes[row_index, column_index]
            coverage = aggregate_curve(
                coverage_rows, cv, mode, "coverage_target",
                ("cover_lookup_ms", "cover_importance"),
            )
            fixed = aggregate_curve(
                fixed_rows, cv, mode, "budget_fraction",
                ("paper_lookup_ms", "paper_importance", "dpr_lookup_ms",
                 "dpr_importance", "top_lookup_ms", "top_importance"),
            )
            specs = (
                (coverage, "cover", "Affine DP-Cover", BASE.PLOT_COLORS["coverage"], "X", "--"),
                (fixed, "paper", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o", "-"),
                (fixed, "dpr", "DP-R exact", BASE.PLOT_COLORS["fixed"], "s", "-"),
                (fixed, "top", "Top-R", BASE.PLOT_COLORS["top_r"], "^", ":"),
            )
            for points, prefix, label, color, marker, linestyle in specs:
                ax.plot(
                    [point[f"{prefix}_lookup_ms"]["mean"] for point in points],
                    [point[f"{prefix}_importance"]["mean"] for point in points],
                    color=color, marker=marker, linestyle=linestyle, label=label,
                )
            ax.set_xscale("log")
            ax.set_title(f"{SPATIAL_LABELS[mode]}, CV={cv:g}")
            ax.set_xlabel("Released lookup-table latency (ms, log scale)")
            ax.set_ylabel("Retained importance")
            BASE.polish_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_matched_ratios(fixed_rows: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    rows = [row for row in fixed_rows if float(row["target_cv"]) < 9.0]
    summaries = EXP2.summarize(
        rows, ("spatial_mode", "budget_fraction"),
        ("paper_lookup_cover_ratio", "dpr_lookup_cover_ratio",
         "top_lookup_cover_ratio"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True,
                             constrained_layout=True)
    for ax, mode in zip(axes, SPATIAL_MODES):
        points = [item for item in summaries if item["spatial_mode"] == mode]
        points.sort(key=lambda item: item["budget_fraction"])
        for key, label, color, marker in (
            ("paper_lookup_cover_ratio", "Paper greedy", BASE.PLOT_COLORS["greedy"], "o"),
            ("dpr_lookup_cover_ratio", "DP-R exact", BASE.PLOT_COLORS["fixed"], "s"),
            ("top_lookup_cover_ratio", "Top-R", BASE.PLOT_COLORS["top_r"], "^"),
        ):
            ax.plot(
                [item["budget_fraction"] for item in points],
                [item[key]["median"] for item in points],
                color=color, marker=marker, label=label,
            )
        ax.axhline(1.0, color=BASE.PLOT_COLORS["coverage"], linewidth=1.2,
                   linestyle="--", label="Same lookup latency")
        ax.set_yscale("log")
        ax.set_xlabel("Fixed-R budget / N")
        ax.set_title(SPATIAL_LABELS[mode])
        BASE.polish_axis(ax)
    axes[0].set_ylabel("Method lookup / matched affine-DP-Cover lookup (median, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_affine_lookup_comparison(fixed_rows: list[dict], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    rows = [row for row in fixed_rows if float(row["target_cv"]) < 9.0]
    methods = (
        ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
        ("dpr", "DP-R exact", BASE.PLOT_COLORS["fixed"]),
        ("top", "Top-R", BASE.PLOT_COLORS["top_r"]),
    )
    x = np.arange(len(methods), dtype=float)
    width = 0.34
    affine = [np.median([row[f"{key}_cover_ratio"] for row in rows])
              for key, _, _ in methods]
    lookup = [np.median([row[f"{key}_lookup_cover_ratio"] for row in rows])
              for key, _, _ in methods]
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, affine, width, color="#94A3B8", label="Affine model")
    bars = ax.bar(x + width / 2, lookup, width,
                  color="#2A9D8F", label="Lookup table")
    ax.axhline(1.0, color="#334155", linewidth=1.1, linestyle="--")
    ax.set_xticks(x, [label for _, label, _ in methods])
    ax.set_yscale("log")
    ax.set_ylabel("Median latency / matched affine-DP-Cover latency (log)")
    ax.set_title("The affine-optimal mask is not the lookup-table optimum")
    ax.legend(frameon=False)
    for bar, value in zip(bars, lookup):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}x",
                ha="center", va="bottom", fontsize=9)
    BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def self_check() -> None:
    rng = np.random.default_rng(1010)
    for n in (8, 17, 31):
        values = rng.lognormal(size=n)
        reference = BASE.ExactCoverageOracle(values, 0.011, 0.0007)
        oracle = ExactStateMaskOracle(values, n, (n + 1) // 2)
        for rows in range(n + 1):
            for chunks in range((n + 1) // 2 + 1):
                expected = max(reference.end_zero[rows, chunks],
                               reference.end_one[rows, chunks])
                actual = max(oracle.end_zero[rows, chunks],
                             oracle.end_one[rows, chunks])
                if np.isfinite(expected):
                    assert_close(actual, expected, "self-check state", 1e-11)
                    solution = oracle.solve_state(rows, chunks)
                    assert_close(float(values[solution["mask"]].sum()), expected,
                                 "self-check mask", 1e-11)
                elif np.isfinite(actual):
                    raise RuntimeError("truncated exact-state DP created a spurious state")

    table = BASE.LatencyTable.load("orin-agx")
    mask = np.asarray([0, 1, 1, 0, 1, 0], dtype=bool)
    encoded = encode_runs(mask)
    if not np.array_equal(decode_runs(encoded, len(mask)), mask):
        raise RuntimeError("run-length encoding self-check failed")
    expected = table.read_ms(3.5) + table.read_ms(1.75)
    assert_close(lookup_latency(mask, 1.75, table), expected, "lookup latency", 1e-12)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--dtype-bytes", type=float, default=2.0)
    parser.add_argument("--local-rho", type=float, default=0.95)
    parser.add_argument("--hot-rho", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = ("summary.json", "fixed_r_trials.csv", "coverage_trials.csv",
                "input_trials.csv")
    for name in required:
        if not (args.source_results / name).is_file():
            raise SystemExit(f"missing Experiment 07 result: {args.source_results / name}")
    if not args.skip_self_check:
        self_check()

    source_summary = json.loads((args.source_results / "summary.json").read_text())
    if source_summary.get("format") != "experiment-07-exact-n4864-v1":
        raise SystemExit("--source-results is not an Experiment 07 result directory")
    if int(source_summary["n"]) != 4864:
        raise SystemExit("Experiment 10 requires the N=4,864 Experiment 07 results")

    source_fixed = read_csv(args.source_results / "fixed_r_trials.csv")
    source_coverage = read_csv(args.source_results / "coverage_trials.csv")
    source_inputs = read_csv(args.source_results / "input_trials.csv")
    fixed_groups: dict[tuple, list[dict]] = {}
    coverage_groups: dict[tuple, list[dict]] = {}
    for row in source_fixed:
        fixed_groups.setdefault(input_key(row), []).append(row)
    for row in source_coverage:
        coverage_groups.setdefault(input_key(row), []).append(row)
    input_by_key = {input_key(row): row for row in source_inputs}
    if len(input_by_key) != len(source_inputs):
        raise RuntimeError("duplicate Experiment 07 input keys")

    n = int(source_summary["n"])
    row_size_kib = float(CONFIG["cols"]) * args.dtype_bytes / 1024.0
    table = BASE.LatencyTable.load(args.profile)
    a_ms = float(source_summary["affine_fit"]["a_ms_per_chunk"])
    c_ms = float(source_summary["affine_fit"]["c_ms_per_row"])
    params = BASE.ChunkParams(start_kb=CONFIG["start_kib"],
                              jump_cap_kb=CONFIG["jump_cap_kib"])
    trials = int(source_summary["trials_per_cv"])
    cv_targets = [float(value) for value in source_summary["cv_targets"]]
    coverage_targets = [float(value) for value in source_summary["coverage_targets"]]
    budget_fractions = [float(value) for value in source_summary["budget_fractions"]]
    rng = np.random.default_rng(int(source_summary["seed"]))
    hotness = np.linspace(1.0, -1.0, n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    fixed_output: list[dict] = []
    coverage_output: list[dict] = []
    reconstruction_rows: list[dict] = []
    total_inputs = trials * len(cv_targets) * len(SPATIAL_MODES)
    completed = 0

    # Compile variable-shape kernels before measured work.
    ExactStateMaskOracle(np.ones(8), 8, 4)

    for trial in range(trials):
        for target_cv in cv_targets:
            base_values = EXP2.exact_cv_lognormal(rng, n, target_cv)
            variants = EXP2.spatial_variants(
                base_values, rng, hotness, args.local_rho, args.hot_rho
            )
            for mode, values in variants.items():
                key = (trial, round(target_cv, 10), mode)
                source_input = input_by_key[key]
                actual_cv = EXP2.coefficient_of_variation(values)
                lag1, first_half = EXP2.spatial_stats(values)
                assert_close(actual_cv, source_input["actual_cv"], "actual CV")
                assert_close(lag1, source_input["lag1_corr"], "lag-1 correlation")
                assert_close(first_half, source_input["first_half_mass"], "first-half mass")

                fixed_group = sorted(fixed_groups[key], key=lambda row: row["budget_fraction"])
                coverage_group = sorted(
                    coverage_groups[key], key=lambda row: row["coverage_target"]
                )
                exact_masks, reconstruction_ms, buffer_bytes = reconstruct_exact_masks(
                    values, coverage_group, fixed_group
                )
                reconstruction_rows.append({
                    "trial": trial, "target_cv": target_cv, "spatial_mode": mode,
                    "distinct_exact_states": len(exact_masks),
                    "max_exact_rows": max(state[0] for state in exact_masks),
                    "max_exact_chunks": max(state[1] for state in exact_masks),
                    "reconstruction_ms": reconstruction_ms,
                    "reconstruction_buffer_gib": buffer_bytes / 2**30,
                })

                for source in coverage_group:
                    row = dict(source)
                    state = int(row["cover_rows"]), int(row["cover_chunks"])
                    metrics = mask_metrics(
                        exact_masks[state], values, a_ms, c_ms, row_size_kib, table
                    )
                    add_solution(row, "cover", metrics)
                    coverage_output.append(row)

                values_t = torch.from_numpy(values.astype(np.float32))
                top_order = np.argsort(-values, kind="stable")
                for source in fixed_group:
                    row = dict(source)
                    budget = int(row["budget_rows"])
                    paper = BASE.select_chunks(
                        values_t, budget, row_size_kib, table,
                        params=params, impl="torch",
                    )
                    paper_mask = paper.mask.cpu().numpy()
                    add_solution(row, "paper", mask_metrics(
                        paper_mask, values, a_ms, c_ms, row_size_kib, table
                    ))

                    dpr_mask, iterations, residual = BASE.exact_fixed_r_affine(
                        values, budget, a_ms, c_ms
                    )
                    add_solution(row, "dpr", mask_metrics(
                        dpr_mask, values, a_ms, c_ms, row_size_kib, table
                    ))
                    if iterations != int(row["dpr_iterations"]):
                        raise RuntimeError("DP-R replay used a different iteration count")
                    assert_close(residual, row["dpr_residual"], "DP-R residual", 1e-9)

                    top_mask = np.zeros(n, dtype=bool)
                    top_mask[top_order[:budget]] = True
                    add_solution(row, "top", mask_metrics(
                        top_mask, values, a_ms, c_ms, row_size_kib, table
                    ))

                    for prefix in EXACT_PREFIXES:
                        state = int(row[f"{prefix}_rows"]), int(row[f"{prefix}_chunks"])
                        add_solution(row, prefix, mask_metrics(
                            exact_masks[state], values, a_ms, c_ms, row_size_kib, table
                        ))

                    for method in ("paper", "dpr", "top"):
                        matched = f"{method}_cover"
                        row[f"{method}_lookup_cover_ratio"] = (
                            row[f"{method}_lookup_ms"] / row[f"{matched}_lookup_ms"]
                        )
                        row[f"{method}_lookup_delta_pct"] = 100.0 * (
                            row[f"{method}_lookup_cover_ratio"] - 1.0
                        )
                    fixed_output.append(row)

                completed += 1
                print(
                    f"completed {completed}/{total_inputs} inputs "
                    f"({reconstruction_ms / 1000:.2f}s reconstruction, "
                    f"{buffer_bytes / 2**30:.2f} GiB)",
                    flush=True,
                )
                del exact_masks

    if len(fixed_output) != len(source_fixed):
        raise RuntimeError("did not reproduce every Experiment 07 fixed-R row")
    if len(coverage_output) != len(source_coverage):
        raise RuntimeError("did not reproduce every Experiment 07 coverage row")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fixed_r_lookup_trials.csv", fixed_output)
    write_csv(args.output_dir / "coverage_lookup_trials.csv", coverage_output)
    write_csv(args.output_dir / "reconstruction_trials.csv", reconstruction_rows)

    vlm_fixed = [row for row in fixed_output if float(row["target_cv"]) < 9.0]
    method_summary = {}
    for method in ("paper", "dpr", "top"):
        ratios = [row[f"{method}_lookup_cover_ratio"] for row in vlm_fixed]
        method_summary[method] = {
            "matched_lookup_ratio": distribution(ratios),
            "lookup_delta_pct": distribution(
                [row[f"{method}_lookup_delta_pct"] for row in vlm_fixed]
            ),
            "lower_lookup_than_affine_dp_cover_rate": float(np.mean(
                np.asarray(ratios) < 1.0 - 1e-12
            )),
            "same_lookup_as_affine_dp_cover_rate": float(np.mean(
                np.isclose(ratios, 1.0, rtol=1e-12, atol=1e-12)
            )),
        }
    summary = {
        "format": "experiment-10-exact-lookup-recheck-v1",
        "source_experiment": "07_exact_n4864",
        "source_results": str(args.source_results.resolve()),
        "source_sha256": {
            name: file_sha256(args.source_results / name) for name in required
        },
        "n": n, "model": source_summary["model"], "shape": source_summary["shape"],
        "profile": args.profile, "row_size_kib": row_size_kib,
        "trials_per_cv": trials, "seed": source_summary["seed"],
        "cv_targets": cv_targets, "coverage_targets": coverage_targets,
        "budget_fractions": budget_fractions,
        "methods": ["Affine DP-Cover", "Paper greedy", "DP-R exact", "Top-R"],
        "interpretation_warning": (
            "DP-Cover is exact for fitted affine latency, then evaluated under the "
            "lookup table; it is not an exact lookup-latency oracle."
        ),
        "reconstruction": {
            "algorithm": "saved (R,K,I) states + exact O(N R K_max) backtracking DP",
            "full_o_n3_frontier_recomputed": False,
            "runtime_ms": distribution(
                [row["reconstruction_ms"] for row in reconstruction_rows]
            ),
            "buffer_gib": distribution(
                [row["reconstruction_buffer_gib"] for row in reconstruction_rows]
            ),
            "max_required_chunks": int(max(
                row["max_exact_chunks"] for row in reconstruction_rows
            )),
            "persisted_representation": "zero-based selected runs as start:length",
        },
        "matched_lookup_comparison_vlm_cvs": method_summary,
        "cases": {
            "inputs": len(reconstruction_rows),
            "coverage_points": len(coverage_output),
            "fixed_r_points": len(fixed_output),
            "primary_vlm_fixed_r_points": len(vlm_fixed),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_lookup_frontiers(
        coverage_output, fixed_output,
        args.output_dir / "importance_lookup_frontiers.png",
    )
    plot_matched_ratios(
        fixed_output, args.output_dir / "matched_lookup_ratio.png"
    )
    plot_affine_lookup_comparison(
        fixed_output, args.output_dir / "affine_lookup_comparison.png"
    )
    print(f"wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
