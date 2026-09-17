#!/usr/bin/env python3
"""Experiment 19: plot latency on x and retained importance on y."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_18 = (
    PROJECT_ROOT / "experiments" / "18_sub2ms_candidates" / "run_experiment.py"
)
DEFAULT_EXP18_RESULTS = (
    PROJECT_ROOT / "experiments" / "18_sub2ms_candidates" / "results"
)
DEFAULT_TARGETS = tuple(float(value) for value in np.linspace(0.05, 0.95, 19)) + (0.99,)
DEFAULT_CVS = (1.25, 3.30, 4.55)


def load_experiment_18():
    spec = importlib.util.spec_from_file_location("experiment_18_for_19", EXPERIMENT_18)
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
METHODS = EXP18.DEFAULT_METHODS
LABELS = EXP18.METHOD_LABELS

PAPER_COLOR = "#D97706"
SUPPORTED_COLOR = "#6D28D9"
METHOD_STYLE = {
    "paper": (PAPER_COLOR, "o", "--"),
    "supported_full": (SUPPORTED_COLOR, "s", "-"),
    "td_2l_c4": ("#5AB4AC", "^", ":"),
    "td_2l_c8": ("#2A9D8F", "v", "-"),
    "td_2l_c16": ("#16877A", "D", "-"),
    "td_2l_c32": ("#08766B", "P", "--"),
    "td_2l_c8_trim64": ("#0F766E", "X", "-"),
    "td_2l_c16_trim256": ("#115E59", "*", "-"),
    "tiles_half_s": ("#60A5FA", "<", "-"),
    "tiles_s": ("#2563EB", ">", "-"),
    "tiles_2s": ("#1D4ED8", "h", "--"),
    "paper_bucket64": ("#F87171", "p", "-"),
    "paper_bucket256": ("#DC2626", "8", "-"),
    "paper_bucket1024": ("#B91C1C", "d", "--"),
}
PANELS = (
    ("Target-directed DP", (
        "paper", "supported_full", "td_2l_c4", "td_2l_c8",
        "td_2l_c16", "td_2l_c32",
    )),
    ("Endpoint trim", (
        "paper", "supported_full", "td_2l_c8", "td_2l_c8_trim64",
        "td_2l_c16", "td_2l_c16_trim256",
    )),
    ("Saturation tiles", (
        "paper", "supported_full", "tiles_half_s", "tiles_s", "tiles_2s",
    )),
    ("Paper bucket", (
        "paper", "supported_full", "paper_bucket64", "paper_bucket256",
        "paper_bucket1024",
    )),
)
FRONTIER_METHODS = (
    "paper",
    "supported_full",
    "td_2l_c16",
    "td_2l_c16_trim256",
    "tiles_s",
    "paper_bucket256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4864)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cvs", type=float, nargs="+", default=DEFAULT_CVS)
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--profile", default="orin-agx")
    parser.add_argument("--row-size-kib", type=float, default=1.75)
    parser.add_argument("--saturation-kib", type=float, default=236.0)
    parser.add_argument("--start-kib", type=float, default=12.0)
    parser.add_argument("--jump-cap-kib", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=20261801)
    parser.add_argument("--deadline-ms", type=float, default=2.0)
    parser.add_argument("--exp18-results", type=Path, default=DEFAULT_EXP18_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    targets = sorted(set(float(value) for value in args.targets))
    cvs = sorted(set(float(value) for value in args.cvs))
    if args.n < 2 or args.trials < 1:
        raise SystemExit("--n must be >= 2 and --trials must be positive")
    if not cvs or any(cv <= 0 or cv >= math.sqrt(args.n - 1) for cv in cvs):
        raise SystemExit("every --cvs value must lie in (0, sqrt(N-1))")
    if not targets or targets[0] <= 0 or targets[-1] > 1:
        raise SystemExit("targets must lie in (0, 1]")
    return targets, cvs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect(args: argparse.Namespace, targets: list[float], cvs: list[float], model: dict,
            lookup_table, params) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    hotness = np.linspace(1.0, -1.0, args.n)
    hotness = (hotness - hotness.mean()) / hotness.std()
    windows = EXP18.paper_windows(args.n, args.row_size_kib, lookup_table, params)
    rows = []
    warmed = False
    for target_cv in cvs:
        for trial in range(args.trials):
            multiset = EXP2.exact_cv_lognormal(rng, args.n, target_cv)
            variants = EXP2.spatial_variants(multiset, rng, hotness, 0.95, 0.75)
            observed_cv = EXP2.coefficient_of_variation(multiset)
            for spatial_mode, raw_values in variants.items():
                values = np.asarray(raw_values, dtype=np.float64)
                values /= values.sum()
                if not warmed:
                    EXP18.warm_up(
                        values, targets[0], args.n, model, lookup_table,
                        args.row_size_kib, params, windows,
                    )
                    warmed = True
                supported = EXP16.ExactSupportedOracle(values, model)
                for target_fraction in targets:
                    target = float(target_fraction * values.sum())
                    for method in METHODS:
                        if method == "supported_full":
                            node = supported.solve(target)
                            mask = np.asarray(node["mask"], dtype=bool)
                            metadata = {"fallback_used": False, "error": ""}
                        else:
                            mask, metadata = EXP18.select_with_fallback(
                                method, "coverage", values, target, args.n,
                                model, lookup_table, args.row_size_kib, params, windows,
                            )
                        metrics = EXP18.mask_metrics(
                            mask, values, model, lookup_table, args.row_size_kib
                        )
                        rows.append({
                            "n": args.n,
                            "target_cv": target_cv,
                            "observed_cv": observed_cv,
                            "trial": trial,
                            "spatial_mode": spatial_mode,
                            "target_fraction": target_fraction,
                            "method": method,
                            "method_label": LABELS[method],
                            "importance_fraction": metrics["importance"] / values.sum(),
                            "importance_overshoot": metrics["importance"] - target,
                            "rows": metrics["rows"],
                            "chunks": metrics["chunks"],
                            "two_line_ms": metrics["two_line_ms"],
                            "lookup_ms": metrics["lookup_ms"],
                            "coverage_met": metrics["importance"] >= target - 1e-11,
                            "fallback_used": metadata.get("fallback_used", False),
                            "error": metadata.get("error", ""),
                        })
                print(
                    f"CV={target_cv:g} trial={trial + 1}/{args.trials} "
                    f"mode={spatial_mode}", flush=True
                )
    return rows


def configure_plot():
    import matplotlib.pyplot as plt
    BASE.configure_plot_style(plt)
    return plt


def curve_points(data: pd.DataFrame, method: str, x_field: str,
                 aggregate: bool, include_origin: bool) -> tuple[np.ndarray, np.ndarray]:
    selected = data[data.method == method]
    if aggregate:
        grouped = selected.groupby("target_fraction", sort=True)
        x = grouped[x_field].mean().to_numpy(dtype=np.float64)
        y = grouped.importance_fraction.mean().to_numpy(dtype=np.float64)
    else:
        selected = selected.sort_values("target_fraction")
        x = selected[x_field].to_numpy(dtype=np.float64)
        y = selected.importance_fraction.to_numpy(dtype=np.float64)
    if include_origin:
        return np.r_[0.0, x], np.r_[0.0, y]
    return x, y


def plot_family_grid(data: pd.DataFrame, x_field: str, xlabel: str,
                     title: str, path: Path, deadline_ms: float | None = None,
                     aggregate: bool = True) -> None:
    plt = configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2), constrained_layout=True)
    for ax, (panel_title, methods) in zip(axes.flat, PANELS):
        for method in methods:
            x, y = curve_points(
                data, method, x_field, aggregate,
                include_origin=deadline_ms is None,
            )
            color, marker, linestyle = METHOD_STYLE[method]
            ax.plot(
                x, y, color=color, marker=marker, linestyle=linestyle,
                linewidth=2.0, markersize=4.5, label=LABELS[method],
            )
        if deadline_ms is not None:
            ax.axvline(
                deadline_ms, color="#111827", linestyle="--", linewidth=1.2,
                label=f"{deadline_ms:g} ms deadline",
            )
            ax.set_xscale("log")
        ax.set_xlim(left=max(0.0, ax.get_xlim()[0]) if deadline_ms is None else None)
        ax.set_ylim(0.0, 1.01)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Retained importance")
        ax.set_title(panel_title)
        BASE.polish_axis(ax)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_condition_frontiers(
    data: pd.DataFrame,
    methods: tuple[str, ...],
    x_field: str,
    xlabel: str,
    path: Path,
) -> None:
    """Match the earlier CV-by-spatial-mode frontier layout."""
    plt = configure_plot()
    cvs = sorted(float(value) for value in data.target_cv.unique())
    spatial_modes = tuple(EXP2.SPATIAL_MODES)
    fig, axes = plt.subplots(
        len(cvs), len(spatial_modes),
        figsize=(14.2, max(4.0, 3.55 * len(cvs))),
        squeeze=False,
        constrained_layout=True,
    )
    handles = {}
    for row_index, target_cv in enumerate(cvs):
        for col_index, spatial_mode in enumerate(spatial_modes):
            ax = axes[row_index, col_index]
            condition = data[
                np.isclose(data.target_cv, target_cv)
                & (data.spatial_mode == spatial_mode)
            ]
            if condition.empty:
                raise RuntimeError(
                    f"missing curve for CV={target_cv:g}, mode={spatial_mode}"
                )
            for method in methods:
                x, y = curve_points(
                    condition, method, x_field, aggregate=True,
                    include_origin=False,
                )
                positive = x > 0
                color, marker, linestyle = METHOD_STYLE[method]
                line, = ax.plot(
                    x[positive], y[positive], color=color, marker=marker,
                    linestyle=linestyle, linewidth=1.9, markersize=3.8,
                    label=LABELS[method],
                )
                handles.setdefault(method, line)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.01)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Retained importance")
            ax.set_title(
                f"{EXP2.SPATIAL_LABELS[spatial_mode]}, CV={target_cv:g}"
            )
            BASE.polish_axis(ax)
    fig.legend(
        [handles[method] for method in methods],
        [LABELS[method] for method in methods],
        loc="outside upper center", ncol=min(len(methods), 6),
        frameon=False, fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_comparisons(frame: pd.DataFrame, latency_field: str) -> pd.DataFrame:
    keys = ["target_cv", "trial", "spatial_mode", "target_fraction"]
    paper = frame[frame.method == "paper"][keys + [latency_field]].rename(
        columns={latency_field: "paper_ms"}
    )
    supported = frame[frame.method == "supported_full"][keys + [latency_field]].rename(
        columns={latency_field: "supported_ms"}
    )
    output = frame.merge(paper, on=keys, how="left").merge(
        supported, on=keys, how="left"
    )
    output["saving_vs_paper_pct"] = 100.0 * (
        1.0 - output[latency_field] / output.paper_ms
    )
    output["saving_vs_supported_pct"] = 100.0 * (
        1.0 - output[latency_field] / output.supported_ms
    )
    return output


def build_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    nested = {}
    for latency_model, field in (("lookup", "lookup_ms"), ("two_line", "two_line_ms")):
        compared = add_comparisons(frame, field)
        nested[latency_model] = {}
        for method in METHODS:
            selected = compared[compared.method == method]
            row = {
                "latency_model": latency_model,
                "method": method,
                "method_label": LABELS[method],
                "points": len(selected),
                "coverage_success_rate": float(selected.coverage_met.mean()),
                "fallback_rate": float(selected.fallback_used.mean()),
                "mean_saving_vs_paper_pct": float(selected.saving_vs_paper_pct.mean()),
                "median_saving_vs_paper_pct": float(selected.saving_vs_paper_pct.median()),
                "strict_win_rate_vs_paper": float(
                    (selected[field] < selected.paper_ms - 1e-12).mean()
                ),
                "mean_saving_vs_supported_pct": float(
                    selected.saving_vs_supported_pct.mean()
                ),
                "strict_win_rate_vs_supported": float(
                    (selected[field] < selected.supported_ms - 1e-12).mean()
                ),
            }
            rows.append(row)
            nested[latency_model][method] = row
    return pd.DataFrame(rows), {
        "format": "experiment-19-candidate-importance-latency-v1",
        "methods": list(METHODS),
        "axis_contract": {
            "x": "latency",
            "y": "retained importance",
            "selector_latency": "Experiment 18 warm host runtime",
            "io_latency": "model-predicted, not measured I/O",
        },
        "latency_models": nested,
    }


def build_condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for latency_model, field in (("lookup", "lookup_ms"), ("two_line", "two_line_ms")):
        compared = add_comparisons(frame, field)
        grouped = compared.groupby(
            ["target_cv", "spatial_mode", "method"], sort=True
        )
        for (target_cv, spatial_mode, method), selected in grouped:
            rows.append({
                "latency_model": latency_model,
                "target_cv": target_cv,
                "spatial_mode": spatial_mode,
                "method": method,
                "method_label": LABELS[method],
                "points": len(selected),
                "coverage_success_rate": float(selected.coverage_met.mean()),
                "fallback_rate": float(selected.fallback_used.mean()),
                "mean_saving_vs_paper_pct": float(
                    selected.saving_vs_paper_pct.mean()
                ),
                "mean_saving_vs_supported_pct": float(
                    selected.saving_vs_supported_pct.mean()
                ),
                "strict_win_rate_vs_paper": float(
                    (selected[field] < selected.paper_ms - 1e-12).mean()
                ),
                "strict_win_rate_vs_supported": float(
                    (selected[field] < selected.supported_ms - 1e-12).mean()
                ),
            })
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Experiment 19 보고서: latency–importance curves",
        "",
        "모든 그림은 가로축이 latency, 세로축이 retained importance다. Selector "
        "latency와 predicted I/O latency는 서로 다른 그림으로 분리했다.",
        "",
        "## Lookup-model I/O curve summary",
        "",
        "| 방법 | Paper 대비 평균 절감 | Paper 승률 | supported 대비 평균 절감 |",
        "|---|---:|---:|---:|",
    ]
    lookup = summary[summary.latency_model == "lookup"].set_index("method")
    for method in METHODS:
        row = lookup.loc[method]
        lines.append(
            f"| {LABELS[method]} | {row.mean_saving_vs_paper_pct:.2f}% "
            f"| {100 * row.strict_win_rate_vs_paper:.1f}% "
            f"| {row.mean_saving_vs_supported_pct:.2f}% |"
        )
    td16 = lookup.loc["td_2l_c16"]
    trim = lookup.loc["td_2l_c16_trim256"]
    lines.extend([
        "",
        "## 해석",
        "",
        f"Dense target sweep에서도 TD-2L (16)은 full supported 대비 평균 "
        f"`{td16.mean_saving_vs_supported_pct:.3f}%` 차이다. TD-2L (16)+trim256은 "
        f"supported보다 lookup latency를 평균 `{trim.mean_saving_vs_supported_pct:.2f}%` "
        "더 줄인다. 이는 trim mask가 strongly-supported point에 제한되지 않기 때문이다.",
        "",
        "Selector latency plot의 2 ms 선은 host pre-screen 기준이며 Jetson 측정 결과가 "
        "아니다. Lookup/two-line 그림의 x축은 predicted I/O latency이므로 2 ms selector "
        "deadline과 직접 비교하면 안 된다.",
        "",
        "![Selector latency versus importance](results/selector_latency_importance.png)",
        "",
        "![Lookup I/O latency versus importance](results/lookup_latency_importance.png)",
        "",
        "![Two-line I/O latency versus importance](results/two_line_latency_importance.png)",
        "",
        "## CV × spatial-mode frontiers",
        "",
        "각 행은 CV 1.25/3.30/4.55, 각 열은 Random order/Locally clustered/"
        "Persistent hot-cold다. 아래 주 그림은 각 후보 계열의 대표 설정을 비교하며, "
        "모든 설정은 `results/conditioned/`의 계열별 그림에 포함했다.",
        "",
        "![Lookup CV-spatial frontiers](results/lookup_importance_latency_frontiers.png)",
        "",
        "![Two-line CV-spatial frontiers](results/two_line_importance_latency_frontiers.png)",
        "",
    ])
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "curve_trials.csv")
    if not frame.coverage_met.all():
        raise RuntimeError("coverage-only curve contains a failed target")
    summary_frame, summary = build_summary(frame)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    condition_summary = build_condition_summary(frame)
    condition_summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_family_grid(
        frame, "lookup_ms", "Predicted lookup I/O latency (ms)",
        "Released-lookup latency vs retained importance",
        args.output_dir / "lookup_latency_importance.png",
    )
    plot_family_grid(
        frame, "two_line_ms", "Predicted two-line I/O latency (ms)",
        "Two-line latency vs retained importance",
        args.output_dir / "two_line_latency_importance.png",
    )
    for latency_name, field, xlabel in (
        ("lookup", "lookup_ms", "Predicted lookup I/O latency (ms, log scale)"),
        ("two_line", "two_line_ms", "Predicted two-line I/O latency (ms, log scale)"),
    ):
        plot_condition_frontiers(
            frame, FRONTIER_METHODS, field, xlabel,
            args.output_dir / f"{latency_name}_importance_latency_frontiers.png",
        )
        for panel_title, methods in PANELS:
            slug = panel_title.lower().replace(" ", "_").replace("-", "_")
            plot_condition_frontiers(
                frame, methods, field, xlabel,
                args.output_dir / "conditioned" /
                f"{latency_name}_{slug}_frontiers.png",
            )

    exp18 = pd.read_csv(args.exp18_results / "trials.csv")
    selector = exp18[exp18.track == "coverage"].copy()
    expected_methods = set(METHODS)
    if set(selector.method.unique()) != expected_methods:
        raise RuntimeError("Experiment 18 method set does not match Experiment 19")
    plot_family_grid(
        selector, "runtime_median_ms", "Measured host selector latency (ms, log)",
        "Selector latency vs retained importance (Experiment 18)",
        args.output_dir / "selector_latency_importance.png",
        deadline_ms=args.deadline_ms,
    )
    write_report(summary_frame, args)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    targets, cvs = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lookup_table = BASE.LatencyTable.load(args.profile)
    model = EXP13.fit_continuous_two_line(
        lookup_table, args.row_size_kib, args.saturation_kib
    )
    params = BASE.ChunkParams(
        start_kb=args.start_kib,
        end_kb=args.saturation_kib,
        step_kb=args.start_kib,
        jump_cap_kb=args.jump_cap_kib,
    )
    if not args.analyze_only:
        if not args.skip_self_check:
            EXP18.self_check(model, lookup_table, args.row_size_kib, params)
        rows = collect(args, targets, cvs, model, lookup_table, params)
        write_csv(args.output_dir / "curve_trials.csv", rows)
        (args.output_dir / "metadata.json").write_text(json.dumps({
            "n": args.n,
            "trials": args.trials,
            "cvs": cvs,
            "targets": targets,
            "two_line_model": model,
            "profile": args.profile,
            "row_size_kib": args.row_size_kib,
        }, indent=2) + "\n")
    analyze(args)


if __name__ == "__main__":
    main()
