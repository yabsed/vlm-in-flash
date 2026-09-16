#!/usr/bin/env python3
"""Experiment 11: explain why the affine and lookup comparisons disagree.

The experiment reuses Experiment 10's persisted run-length encodings.  It
does not rerun Paper, DP-R, Top-R, or any coverage DP.  It measures run-length
support, decomposes each matched ratio exactly, and tests sensitivity to the
released table's beyond-profile extrapolation rule.
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

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_10 = PROJECT_ROOT / "experiments" / "10_exact_lookup_recheck" / "run_experiment.py"
DEFAULT_SOURCE = PROJECT_ROOT / "experiments" / "10_exact_lookup_recheck" / "results"


def load_experiment_10():
    spec = importlib.util.spec_from_file_location("experiment_10_for_11", EXPERIMENT_10)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 10 from {EXPERIMENT_10}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP10 = load_experiment_10()
BASE = EXP10.BASE
SPATIAL_MODES = EXP10.SPATIAL_MODES
SPATIAL_LABELS = EXP10.SPATIAL_LABELS
METHODS = ("paper", "dpr", "top")
METHOD_LABELS = {
    "paper": "Paper greedy",
    "dpr": "DP-R exact",
    "top": "Top-R",
}
METHOD_COLORS = {
    "paper": BASE.PLOT_COLORS["greedy"],
    "dpr": BASE.PLOT_COLORS["fixed"],
    "top": BASE.PLOT_COLORS["top_r"],
}
POLICY_LABELS = {
    "affine_fit": "Affine fit",
    "released": "Released extrapolation",
    "tail_linear": "Anchored tail-linear",
    "block_split": "255-KiB block split",
}
POLICY_COLORS = {
    "affine_fit": "#94A3B8",
    "released": "#2A9D8F",
    "tail_linear": "#F4A261",
    "block_split": "#577590",
}


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


def run_lengths(encoded: str) -> np.ndarray:
    if not encoded:
        return np.empty(0, dtype=np.int64)
    return np.asarray(
        [int(item.split(":")[1]) for item in encoded.split(";")],
        dtype=np.int64,
    )


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


class LatencyPolicies:
    def __init__(self, table, row_size_kib: float, tail_fit_start_kib: int):
        self.table = table
        self.row_size_kib = float(row_size_kib)
        pairs = sorted(table.as_dict().items())
        self.keys = np.asarray([key for key, _ in pairs], dtype=np.float64)
        self.values = np.asarray([value for _, value in pairs], dtype=np.float64)
        self.max_kib = float(self.keys[-1])
        self.max_ms = float(self.values[-1])
        selected = self.keys >= tail_fit_start_kib
        if selected.sum() < 8:
            raise ValueError("tail fit needs at least eight table points")
        self.tail_slope_ms_per_kib, self.tail_intercept_ms = np.polyfit(
            self.keys[selected], self.values[selected], 1
        )
        predicted = (
            self.tail_intercept_ms
            + self.tail_slope_ms_per_kib * self.keys[selected]
        )
        residual = np.sum((self.values[selected] - predicted) ** 2)
        total = np.sum((self.values[selected] - self.values[selected].mean()) ** 2)
        self.tail_fit_r2 = float(1.0 - residual / total)

    @property
    def cutoff_rows(self) -> float:
        return self.max_kib / self.row_size_kib

    @property
    def released_tail_ms_per_row(self) -> float:
        return self.max_ms / self.max_kib * self.row_size_kib

    @property
    def fitted_tail_ms_per_row(self) -> float:
        return self.tail_slope_ms_per_kib * self.row_size_kib

    def chunk_ms(self, rows: int, policy: str) -> float:
        kib = float(rows) * self.row_size_kib
        if policy == "released":
            return float(self.table.read_ms(kib))
        if policy == "tail_linear":
            if kib <= self.max_kib:
                return float(self.table.read_ms(kib))
            return float(
                self.max_ms + self.tail_slope_ms_per_kib * (kib - self.max_kib)
            )
        if policy == "block_split":
            blocks = int(kib // self.max_kib)
            remainder = kib - blocks * self.max_kib
            latency = blocks * self.max_ms
            if remainder > 1e-12:
                latency += float(self.table.read_ms(remainder))
            return float(latency)
        raise ValueError(f"unknown latency policy {policy!r}")

    def mask_ms(self, encoded: str, policy: str) -> float:
        return float(sum(self.chunk_ms(int(length), policy)
                         for length in run_lengths(encoded)))


def run_summary_rows(data: pd.DataFrame, policies: LatencyPolicies) -> list[dict]:
    rows = []
    prefixes = (
        ("paper", "Paper greedy"),
        ("paper_cover", "Matched affine DP-Cover (Paper importance)"),
        ("dpr", "DP-R exact"),
        ("dpr_cover", "Matched affine DP-Cover (DP-R importance)"),
        ("top", "Top-R"),
        ("top_cover", "Matched affine DP-Cover (Top-R importance)"),
    )
    for mode in ("all", *SPATIAL_MODES):
        selected = data if mode == "all" else data[data.spatial_mode == mode]
        for prefix, label in prefixes:
            all_lengths: list[int] = []
            case_mean_lengths = []
            case_row_fractions = []
            case_chunk_fractions = []
            for encoded in selected[f"{prefix}_runs"]:
                lengths = run_lengths(encoded)
                outside = lengths > policies.cutoff_rows
                all_lengths.extend(int(value) for value in lengths)
                case_mean_lengths.append(float(lengths.mean()))
                case_chunk_fractions.append(float(outside.mean()))
                case_row_fractions.append(float(lengths[outside].sum() / lengths.sum()))
            lengths = np.asarray(all_lengths, dtype=np.float64)
            outside = lengths > policies.cutoff_rows
            rows.append({
                "spatial_mode": mode,
                "method": prefix,
                "label": label,
                "cases": len(selected),
                "chunks": len(lengths),
                "mean_chunks_per_case": float(selected[f"{prefix}_chunks"].mean()),
                "global_rows_per_chunk": float(lengths.sum() / len(lengths)),
                "run_length_mean": float(lengths.mean()),
                "run_length_median": float(np.median(lengths)),
                "run_length_p05": float(np.quantile(lengths, 0.05)),
                "run_length_p95": float(np.quantile(lengths, 0.95)),
                "case_mean_run_length_median": float(np.median(case_mean_lengths)),
                "mean_case_chunk_fraction_beyond_profile": float(
                    np.mean(case_chunk_fractions)
                ),
                "mean_case_row_fraction_beyond_profile": float(
                    np.mean(case_row_fractions)
                ),
                "global_chunk_fraction_beyond_profile": float(outside.mean()),
                "global_row_fraction_beyond_profile": float(
                    lengths[outside].sum() / lengths.sum()
                ),
                "lookup_over_affine_mean": float(
                    (selected[f"{prefix}_lookup_ms"]
                     / selected[f"{prefix}_aff_ms"]).mean()
                ),
                "lookup_over_affine_median": float(
                    (selected[f"{prefix}_lookup_ms"]
                     / selected[f"{prefix}_aff_ms"]).median()
                ),
            })
    return rows


def decomposition_rows(data: pd.DataFrame) -> list[dict]:
    rows = []
    for source in data.to_dict("records"):
        for method in METHODS:
            matched = f"{method}_cover"
            affine_ratio = float(source[f"{method}_cover_ratio"])
            method_rescale = float(
                source[f"{method}_lookup_ms"] / source[f"{method}_aff_ms"]
            )
            cover_rescale = float(
                source[f"{matched}_lookup_ms"] / source[f"{matched}_aff_ms"]
            )
            correction = method_rescale / cover_rescale
            lookup_ratio = float(source[f"{method}_lookup_cover_ratio"])
            rows.append({
                "trial": int(source["trial"]),
                "target_cv": float(source["target_cv"]),
                "spatial_mode": source["spatial_mode"],
                "budget_fraction": float(source["budget_fraction"]),
                "method": method,
                "affine_matched_ratio": affine_ratio,
                "method_lookup_over_affine": method_rescale,
                "cover_lookup_over_affine": cover_rescale,
                "relative_rescale_correction": correction,
                "lookup_matched_ratio": lookup_ratio,
                "factorization_error": lookup_ratio - affine_ratio * correction,
            })
    return rows


def sensitivity_rows(data: pd.DataFrame, policies: LatencyPolicies) -> list[dict]:
    rows = []
    for source in data.to_dict("records"):
        for method in METHODS:
            matched = f"{method}_cover"
            for policy in ("affine_fit", "released", "tail_linear", "block_split"):
                if policy == "affine_fit":
                    method_ms = float(source[f"{method}_aff_ms"])
                    cover_ms = float(source[f"{matched}_aff_ms"])
                else:
                    method_ms = policies.mask_ms(source[f"{method}_runs"], policy)
                    cover_ms = policies.mask_ms(source[f"{matched}_runs"], policy)
                rows.append({
                    "trial": int(source["trial"]),
                    "target_cv": float(source["target_cv"]),
                    "spatial_mode": source["spatial_mode"],
                    "budget_fraction": float(source["budget_fraction"]),
                    "method": method,
                    "policy": policy,
                    "method_ms": method_ms,
                    "matched_cover_ms": cover_ms,
                    "matched_ratio": method_ms / cover_ms,
                })
    return rows


def plot_latency_and_runs(data: pd.DataFrame, policies: LatencyPolicies,
                          a_ms: float, c_ms: float, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)
    lengths = np.unique(np.rint(np.geomspace(1, 4864, 500)).astype(int))
    affine = a_ms + c_ms * lengths
    released = np.asarray([policies.chunk_ms(int(x), "released") for x in lengths])
    tail = np.asarray([policies.chunk_ms(int(x), "tail_linear") for x in lengths])

    ax = axes[0, 0]
    ax.plot(lengths, affine, color=POLICY_COLORS["affine_fit"], label="Affine a+cℓ")
    ax.plot(lengths, released, color=POLICY_COLORS["released"], label="Released table evaluator")
    ax.plot(lengths, tail, color=POLICY_COLORS["tail_linear"], linestyle="--",
            label="Anchored tail-linear sensitivity")
    ax.axvline(policies.cutoff_rows, color="#334155", linestyle=":",
               label=f"Profile limit ({policies.cutoff_rows:.1f} rows)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Chunk length ℓ (rows)")
    ax.set_ylabel("Latency of one chunk (ms)")
    ax.set_title("A. Per-chunk cost")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(lengths, released / affine, color=POLICY_COLORS["released"],
            label="Released / affine")
    ax.plot(lengths, tail / affine, color=POLICY_COLORS["tail_linear"],
            linestyle="--", label="Tail-linear / affine")
    ax.axhline(1.0, color="#334155", linestyle=":")
    ax.axvline(policies.cutoff_rows, color="#334155", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Chunk length ℓ (rows)")
    ax.set_ylabel("Per-chunk latency ratio")
    ax.set_title("B. Model mismatch by chunk length")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for prefix, label, color in (
        ("paper", "Paper greedy", BASE.PLOT_COLORS["greedy"]),
        ("paper_cover", "Matched affine DP-Cover", BASE.PLOT_COLORS["coverage"]),
    ):
        values = np.sort(np.concatenate([run_lengths(x) for x in data[f"{prefix}_runs"]]))
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=color, label=label)
    ax.axvline(policies.cutoff_rows, color="#334155", linestyle=":",
               label="Profile limit")
    ax.set_xscale("log")
    ax.set_xlabel("Chunk length (rows, log scale)")
    ax.set_ylabel("Fraction of chunks ≤ length")
    ax.set_title("C. Chunk-length ECDF")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    prefixes = ("paper", "paper_cover", "dpr", "dpr_cover", "top", "top_cover")
    labels = ("Paper", "Cover@Paper", "DP-R", "Cover@DP-R", "Top-R", "Cover@Top-R")
    fractions = []
    for prefix in prefixes:
        values = []
        for encoded in data[f"{prefix}_runs"]:
            chunk_lengths = run_lengths(encoded)
            values.append(float(
                chunk_lengths[chunk_lengths > policies.cutoff_rows].sum()
                / chunk_lengths.sum()
            ))
        fractions.append(100 * np.mean(values))
    colors = [BASE.PLOT_COLORS["greedy"], BASE.PLOT_COLORS["coverage"],
              BASE.PLOT_COLORS["fixed"], BASE.PLOT_COLORS["coverage"],
              BASE.PLOT_COLORS["top_r"], BASE.PLOT_COLORS["coverage"]]
    bars = ax.bar(np.arange(len(prefixes)), fractions, color=colors)
    ax.set_xticks(np.arange(len(prefixes)), labels, rotation=25, ha="right")
    ax.set_ylabel("Rows in chunks beyond profile (%)")
    ax.set_title("D. Dependence on extrapolation")
    for bar, value in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=8)

    for ax in axes.flat:
        BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_paper_decomposition(decomposition: pd.DataFrame,
                             run_summary: pd.DataFrame, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    paper = decomposition[decomposition.method == "paper"]
    modes = list(SPATIAL_MODES)
    x = np.arange(len(modes), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)

    affine = [paper[paper.spatial_mode == mode].affine_matched_ratio.mean()
              for mode in modes]
    lookup = [paper[paper.spatial_mode == mode].lookup_matched_ratio.mean()
              for mode in modes]
    ax = axes[0]
    ax.bar(x - width / 2, affine, width, color=POLICY_COLORS["affine_fit"],
           label="Affine")
    ax.bar(x + width / 2, lookup, width, color=POLICY_COLORS["released"],
           label="Released lookup")
    ax.axhline(1.0, color="#334155", linestyle="--", linewidth=1.1)
    ax.set_ylabel("Paper / matched DP-Cover latency")
    ax.set_title("A. Apparent Paper gap")
    ax.legend(frameon=False)

    method_scale = [
        paper[paper.spatial_mode == mode].method_lookup_over_affine.mean()
        for mode in modes
    ]
    cover_scale = [
        paper[paper.spatial_mode == mode].cover_lookup_over_affine.mean()
        for mode in modes
    ]
    ax = axes[1]
    ax.bar(x - width / 2, method_scale, width, color=BASE.PLOT_COLORS["greedy"],
           label="Paper")
    ax.bar(x + width / 2, cover_scale, width, color=BASE.PLOT_COLORS["coverage"],
           label="Matched DP-Cover")
    ax.axhline(1.0, color="#334155", linestyle="--", linewidth=1.1)
    ax.set_ylabel("Lookup latency / affine latency")
    ax.set_title("B. Unequal model rescaling")
    ax.legend(frameon=False)

    paper_rows = run_summary[
        (run_summary.method == "paper") & run_summary.spatial_mode.isin(modes)
    ].set_index("spatial_mode")
    cover_rows = run_summary[
        (run_summary.method == "paper_cover") & run_summary.spatial_mode.isin(modes)
    ].set_index("spatial_mode")
    paper_fraction = [100 * paper_rows.loc[mode].mean_case_row_fraction_beyond_profile
                      for mode in modes]
    cover_fraction = [100 * cover_rows.loc[mode].mean_case_row_fraction_beyond_profile
                      for mode in modes]
    ax = axes[2]
    ax.bar(x - width / 2, paper_fraction, width, color=BASE.PLOT_COLORS["greedy"],
           label="Paper")
    ax.bar(x + width / 2, cover_fraction, width, color=BASE.PLOT_COLORS["coverage"],
           label="Matched DP-Cover")
    ax.set_ylabel("Rows beyond measured table (%)")
    ax.set_title("C. Extrapolation exposure")
    ax.legend(frameon=False)

    for ax in axes:
        ax.set_xticks(x, [SPATIAL_LABELS[mode] for mode in modes], rotation=18,
                      ha="right")
        BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_extrapolation_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    BASE.configure_plot_style(plt)
    policies = ("affine_fit", "released", "tail_linear", "block_split")
    x = np.arange(len(METHODS), dtype=float)
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for index, policy in enumerate(policies):
        selected = sensitivity[sensitivity.policy == policy]
        means = [selected[selected.method == method].matched_ratio.mean()
                 for method in METHODS]
        below = [100 * np.mean(
            selected[selected.method == method].matched_ratio < 1.0 - 1e-12
        ) for method in METHODS]
        offset = (index - (len(policies) - 1) / 2) * width
        axes[0].bar(x + offset, means, width, color=POLICY_COLORS[policy],
                    label=POLICY_LABELS[policy])
        axes[1].bar(x + offset, below, width, color=POLICY_COLORS[policy],
                    label=POLICY_LABELS[policy])
    axes[0].axhline(1.0, color="#334155", linestyle="--", linewidth=1.1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean method / matched DP-Cover latency (log)")
    axes[0].set_title("A. Matched ratio is extrapolation-sensitive")
    axes[1].set_ylabel("Method cheaper than affine DP-Cover (%)")
    axes[1].set_title("B. Apparent reversals")
    for ax in axes:
        ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
        BASE.polish_axis(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def self_check(policies: LatencyPolicies) -> None:
    cutoff = policies.cutoff_rows
    below_rows = max(1, int(math.floor(cutoff)))
    for policy in ("released", "tail_linear", "block_split"):
        value = policies.chunk_ms(below_rows, policy)
        expected = policies.table.read_ms(below_rows * policies.row_size_kib)
        if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"{policy} differs inside the measured table")
    if not math.isclose(
        policies.chunk_ms(1000, "released"),
        policies.table.read_ms(1000 * policies.row_size_kib),
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise RuntimeError("released extrapolation implementation mismatch")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--tail-fit-start-kib", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = ("summary.json", "fixed_r_lookup_trials.csv")
    for name in required:
        if not (args.source_results / name).is_file():
            raise SystemExit(f"missing Experiment 10 result: {args.source_results / name}")
    source_summary = json.loads((args.source_results / "summary.json").read_text())
    if source_summary.get("format") != "experiment-10-exact-lookup-recheck-v1":
        raise SystemExit("--source-results is not an Experiment 10 result directory")
    if args.profile != source_summary["profile"]:
        raise SystemExit("--profile must match the Experiment 10 source profile")

    data = pd.read_csv(args.source_results / "fixed_r_lookup_trials.csv")
    primary = data[data.target_cv < 9.0].copy()
    row_size_kib = float(source_summary["row_size_kib"])
    table = BASE.LatencyTable.load(args.profile)
    policies = LatencyPolicies(table, row_size_kib, args.tail_fit_start_kib)
    self_check(policies)
    a_ms, c_ms, affine_r2 = BASE.affine_fit(table, row_size_kib)

    runs = run_summary_rows(primary, policies)
    decomposition = decomposition_rows(primary)
    sensitivity = sensitivity_rows(primary, policies)
    decomposition_frame = pd.DataFrame(decomposition)
    max_error = float(np.max(np.abs(decomposition_frame.factorization_error)))
    if max_error > 1e-12:
        raise RuntimeError(f"ratio factorization failed: max error {max_error}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_length_summary.csv", runs)
    write_csv(args.output_dir / "ratio_decomposition.csv", decomposition)
    write_csv(args.output_dir / "extrapolation_sensitivity.csv", sensitivity)

    run_frame = pd.DataFrame(runs)
    sensitivity_frame = pd.DataFrame(sensitivity)
    overall_runs = run_frame[run_frame.spatial_mode == "all"].set_index("method")
    overall_decomposition = decomposition_frame.groupby("method", sort=False)
    sensitivity_summary = {}
    for policy in ("affine_fit", "released", "tail_linear", "block_split"):
        selected = sensitivity_frame[sensitivity_frame.policy == policy]
        sensitivity_summary[policy] = {
            method: {
                "matched_ratio": summarize(
                    selected[selected.method == method].matched_ratio
                ),
                "method_cheaper_rate": float(np.mean(
                    selected[selected.method == method].matched_ratio < 1.0 - 1e-12
                )),
            }
            for method in METHODS
        }

    paper_group = overall_decomposition.get_group("paper")
    summary = {
        "format": "experiment-11-affine-lookup-gap-v1",
        "source_experiment": "10_exact_lookup_recheck",
        "source_results": str(args.source_results.resolve()),
        "source_sha256": {
            name: file_sha256(args.source_results / name) for name in required
        },
        "n": int(source_summary["n"]),
        "profile": args.profile,
        "row_size_kib": row_size_kib,
        "primary_vlm_cases": len(primary),
        "measured_table": {
            "max_kib": policies.max_kib,
            "max_rows": policies.cutoff_rows,
            "released_extrapolation_ms_per_row": policies.released_tail_ms_per_row,
            "tail_fit_start_kib": args.tail_fit_start_kib,
            "tail_fit_slope_ms_per_kib": policies.tail_slope_ms_per_kib,
            "tail_fit_ms_per_row": policies.fitted_tail_ms_per_row,
            "tail_fit_r_squared": policies.tail_fit_r2,
        },
        "affine_fit": {
            "a_ms_per_chunk": a_ms,
            "c_ms_per_row": c_ms,
            "r_squared": affine_r2,
            "released_affine_break_even_rows": (
                a_ms / (policies.released_tail_ms_per_row - c_ms)
            ),
        },
        "run_length_evidence": {
            prefix: {
                "mean_chunks": float(overall_runs.loc[prefix].mean_chunks_per_case),
                "run_length_median": float(overall_runs.loc[prefix].run_length_median),
                "case_mean_run_length_median": float(
                    overall_runs.loc[prefix].case_mean_run_length_median
                ),
                "mean_case_rows_beyond_profile_pct": 100.0 * float(
                    overall_runs.loc[prefix].mean_case_row_fraction_beyond_profile
                ),
                "lookup_over_affine_mean": float(
                    overall_runs.loc[prefix].lookup_over_affine_mean
                ),
            }
            for prefix in ("paper", "paper_cover", "dpr", "dpr_cover", "top", "top_cover")
        },
        "paper_gap_factorization": {
            "affine_ratio": summarize(paper_group.affine_matched_ratio),
            "paper_lookup_over_affine": summarize(paper_group.method_lookup_over_affine),
            "cover_lookup_over_affine": summarize(paper_group.cover_lookup_over_affine),
            "relative_rescale_correction": summarize(
                paper_group.relative_rescale_correction
            ),
            "released_lookup_ratio": summarize(paper_group.lookup_matched_ratio),
            "max_identity_error": max_error,
        },
        "extrapolation_sensitivity": sensitivity_summary,
        "interpretation": (
            "The released evaluator is measured only through 255 KiB. "
            "Beyond that it scales the endpoint proportionally. Affine DP-Cover "
            "is much more exposed to this extrapolated region than Paper."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_latency_and_runs(
        primary, policies, a_ms, c_ms,
        args.output_dir / "latency_model_and_runs.png",
    )
    plot_paper_decomposition(
        decomposition_frame, run_frame,
        args.output_dir / "paper_gap_decomposition.png",
    )
    plot_extrapolation_sensitivity(
        sensitivity_frame, args.output_dir / "extrapolation_sensitivity.png",
    )
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
