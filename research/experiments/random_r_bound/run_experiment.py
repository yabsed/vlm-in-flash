#!/usr/bin/env python3
"""Compare VLM-in-a-Flash greedy selection with exact bounded-R solvers.

The experiment deliberately separates two optimization problems:

1. Fixed budget: R(M) = R, evaluated under the fitted affine latency
   L_aff(M) = a K(M) + c R.  This is solved exactly with Dinkelbach's
   method and the two-state chain DP from ``research/final_email/idea.md``.
2. Upper budget: 1 <= R(M) <= R, evaluated with the released lookup table.
   The exact solution is the best single contiguous interval of length at
   most R, by the weighted-average theorem in the same document.

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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(records: list[dict], summary: list[dict], path: Path) -> None:
    # Keep Matplotlib's generated cache outside both the repository and the
    # user's (possibly read-only) home directory.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

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


def plot_latency_tradeoff(summary: list[dict], path: Path) -> None:
    """Plot importance directly against latency for the three fixed-R methods."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
    import matplotlib.pyplot as plt

    colors = {"greedy": "#D55E00", "fixed": "#0072B2", "top_r": "#CC79A7"}
    methods = (
        ("greedy", "Paper greedy", colors["greedy"]),
        ("top_r", "Top-R", colors["top_r"]),
        ("fixed_opt", "Exact fixed-R DP", colors["fixed"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
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
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Latency (ms, log scale)")
        ax.set_ylabel("Retained importance")
        ax.grid(alpha=0.22, which="both")
        ax.legend(frameon=False)
    fig.suptitle(
        "Fixed-R importance-latency trade-off\n"
        "Each label is R / N; points are means across paired random inputs",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def self_check() -> None:
    """Compare both exact solvers with exhaustive masks on small instances."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--row-size-kib", type=float, default=1.0)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--distribution", choices=("half-normal", "lognormal", "correlated"), default="half-normal")
    parser.add_argument("--budget-rows", type=int, nargs="+", default=[32, 64, 96, 128, 160, 192, 224])
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

    if not args.skip_self_check:
        self_check()

    table = LatencyTable.load(args.profile)
    a_ms, c_ms_per_row, fit_r2 = affine_fit(table, args.row_size_kib)
    params = ChunkParams(start_kb=args.start_kib, jump_cap_kb=args.jump_cap_kib)
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    for trial in range(args.trials):
        values = random_importance(rng, args.n, args.distribution)
        values_t = torch.from_numpy(values.astype(np.float32))
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
                    "fixed_opt_rows": fixed_rows,
                    "fixed_opt_chunks": fixed_chunks,
                    "fixed_opt_importance": fixed_importance,
                    "fixed_opt_aff_ms": fixed_aff_ms,
                    "fixed_opt_aff_ratio": fixed_aff_ratio,
                    "fixed_opt_table_ms": fixed_table_ms,
                    "fixed_opt_table_ratio": fixed_table_ratio,
                    "fixed_aff_gain_pct": 100 * (fixed_aff_ratio / greedy_aff_ratio - 1),
                    "fixed_lookup_gain_pct": 100 * (fixed_table_ratio / greedy_table_ratio - 1),
                    "upper_opt_rows": upper_rows,
                    "upper_opt_chunks": upper_chunks,
                    "upper_opt_importance": upper_importance,
                    "upper_opt_table_ms": upper_table_ms,
                    "upper_opt_table_ratio": upper_table_ratio,
                    "upper_table_gain_pct": 100 * (upper_table_ratio / greedy_table_ratio - 1),
                    "greedy_fill_fraction": greedy_rows / budget,
                    "top_r_fill_fraction": top_r_rows / budget,
                    "upper_opt_fill_fraction": upper_rows / budget,
                    "dinkelbach_iterations": iterations,
                    "dinkelbach_residual": residual,
                }
            )
        if (trial + 1) % max(1, args.trials // 10) == 0:
            print(f"completed {trial + 1}/{args.trials} random inputs", flush=True)

    summary = summarize(records, budgets, args.n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "trials.csv"
    summary_path = args.output_dir / "summary.json"
    figure_path = args.output_dir / "comparison.png"
    latency_importance_path = args.output_dir / "latency_importance.png"
    write_csv(raw_path, records)
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
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(records, summary, figure_path)
    plot_latency_tradeoff(summary, latency_importance_path)

    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path} and {figure_path.with_suffix('.pdf')}")
    print(
        f"wrote {latency_importance_path} and "
        f"{latency_importance_path.with_suffix('.pdf')}"
    )
    print("\nmean fixed-R affine gain over greedy:")
    for item in summary:
        gain = item["fixed_aff_gain_pct"]
        print(
            f"  R/N={item['budget_fraction']:.3f}: {gain['mean']:.2f}% "
            f"(5th-95th: {gain['p05']:.2f}% to {gain['p95']:.2f}%)"
        )


if __name__ == "__main__":
    main()
