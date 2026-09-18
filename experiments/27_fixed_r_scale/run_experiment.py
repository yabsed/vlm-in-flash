#!/usr/bin/env python3
"""Experiment 27: scale only Experiment 24's rho and mu call budgets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPERIMENT_24 = PROJECT_ROOT / "experiments" / "24_fixed_r_ratio" / "run_experiment.py"


def load_experiment_24():
    spec = importlib.util.spec_from_file_location("experiment_24_for_27", EXPERIMENT_24)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Experiment 24 from {EXPERIMENT_24}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP24 = load_experiment_24()
EXP22 = EXP24.EXP22
BASE = EXP24.BASE
SHAPES = EXP24.SHAPES
DEFAULT_TRACE_INPUT = EXP24.DEFAULT_TRACE_INPUT

# This is deliberately only a compute-budget sweep. Every setting calls the
# unchanged Experiment 24 fixed_r_ratio implementation.
METHODS = (
    "paper",
    "top_r",
    "rho2_mu4",
    "rho2_mu8",
    "rho2_mu16",
    "rho3_mu8",
    "rho3_mu16",
    "rho4_mu16",
    "rho4_mu32",
)
SCALED_METHODS = METHODS[2:]
METHOD_SETTINGS = {
    "rho2_mu4": (2, 4),
    "rho2_mu8": (2, 8),
    "rho2_mu16": (2, 16),
    "rho3_mu8": (3, 8),
    "rho3_mu16": (3, 16),
    "rho4_mu16": (4, 16),
    "rho4_mu32": (4, 32),
}
METHOD_LABELS = {
    "paper": "Paper",
    "top_r": "Top-R",
    **{
        method: f"Exp24 rho{setting[0]}/mu{setting[1]}"
        for method, setting in METHOD_SETTINGS.items()
    },
}
METHOD_COLORS = {
    "paper": "#D97706",
    "top_r": "#475569",
    "rho2_mu4": "#7C3AED",
    "rho2_mu8": "#8B5CF6",
    "rho2_mu16": "#A78BFA",
    "rho3_mu8": "#2563EB",
    "rho3_mu16": "#0EA5E9",
    "rho4_mu16": "#0F766E",
    "rho4_mu32": "#064E3B",
}

# Experiment 24's helpers resolve these module globals at call time. Rebinding
# them lets us benchmark the exact same selector over a larger settings grid.
EXP24.METHODS = METHODS
EXP24.FIXED_R_METHODS = SCALED_METHODS
EXP24.METHOD_SETTINGS = METHOD_SETTINGS
EXP24.METHOD_LABELS = METHOD_LABELS
EXP24.METHOD_COLORS = METHOD_COLORS


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
    # Keep Experiment 24's seed so the separate small-N oracle is paired too.
    parser.add_argument("--seed", type=int, default=20262401)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def plot_scale(summary: pd.DataFrame, path: Path, deadline_ms: float) -> None:
    plt = EXP24.configure_plot()
    tracks = list(dict.fromkeys(summary.track))
    fig, axes = plt.subplots(
        len(tracks), 2, figsize=(13.8, 5.0 * len(tracks)), squeeze=False,
        constrained_layout=True,
    )
    rho_colors = {2: "#7C3AED", 3: "#2563EB", 4: "#0F766E"}
    for row_index, track in enumerate(tracks):
        selected = summary[summary.track == track].set_index("method")
        for rho in (2, 3, 4):
            methods = [
                method for method, setting in METHOD_SETTINGS.items()
                if setting[0] == rho
            ]
            methods.sort(key=lambda method: METHOD_SETTINGS[method][1])
            calls = [rho * METHOD_SETTINGS[method][1] for method in methods]
            quality = [
                selected.loc[method, "lookup_efficiency_gain_pct_mean_valid"]
                for method in methods
            ]
            latency = [selected.loc[method, "runtime_case_p95_ms"] for method in methods]
            label = f"rho={rho}"
            axes[row_index, 0].plot(
                calls, quality, marker="o", color=rho_colors[rho], label=label,
            )
            axes[row_index, 1].plot(
                calls, latency, marker="o", color=rho_colors[rho], label=label,
            )
            for x, y, method in zip(calls, quality, methods):
                axes[row_index, 0].annotate(
                    f"mu={METHOD_SETTINGS[method][1]}", (x, y),
                    xytext=(3, 5), textcoords="offset points", fontsize=8,
                )
        axes[row_index, 0].axhline(0.0, color="#64748B", linestyle=":")
        axes[row_index, 1].axhline(deadline_ms, color="#111827", linestyle="--")
        label = "Host" if track == "host" else "CUDA round trip"
        axes[row_index, 0].set_title(f"{label}: quality vs maximum DP calls")
        axes[row_index, 1].set_title(f"{label}: latency vs maximum DP calls")
        axes[row_index, 0].set_ylabel("Mean lookup I/L gain vs Paper (%)")
        axes[row_index, 1].set_ylabel("95th percentile of case p95 (ms)")
        for ax in axes[row_index]:
            ax.set_xlabel("rho_iterations × mu_calls")
            ax.legend(frameon=False)
            BASE.polish_axis(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, shape_summary: pd.DataFrame,
                 oracle: pd.DataFrame | None, args: argparse.Namespace) -> None:
    primary = "cuda" if "cuda" in set(summary.track) else "host"
    indexed = summary[summary.track == primary].set_index("method").reindex(METHODS)
    candidates = indexed.loc[list(SCALED_METHODS)]
    best_quality_method = candidates.lookup_efficiency_gain_pct_mean_valid.idxmax()
    best_quality = candidates.loc[best_quality_method]
    baseline = indexed.loc["rho2_mu4"]
    near_best = candidates[
        candidates.lookup_efficiency_gain_pct_mean_valid
        >= best_quality.lookup_efficiency_gain_pct_mean_valid - 0.05
    ]
    knee_method = near_best.runtime_case_p95_ms.idxmin()
    knee = candidates.loc[knee_method]
    deadline_candidates = candidates[candidates.valid_and_deadline_pass_rate >= 0.95]
    if deadline_candidates.empty:
        best_deadline_method = candidates.valid_and_deadline_pass_rate.idxmax()
    else:
        best_deadline_method = deadline_candidates.lookup_efficiency_gain_pct_mean_valid.idxmax()
    best_deadline = candidates.loc[best_deadline_method]

    lines = [
        "# Experiment 27 보고서: Exp24 계산량 스케일업", "",
        "Experiment 24의 `fixed_r_ratio()`를 한 줄도 바꾸지 않고, "
        "`rho_iterations`와 `mu_calls`만 늘렸다. 모든 설정은 Paper mask나 "
        "`I(M_paper)`를 입력으로 받지 않는다. Paper가 실제로 반환한 행 수만 공통 "
        "fixed-R 예산으로 사용한다.", "", "## 전체 결과", "",
    ]
    for track, label in (("host", "Host input -> CPU mask"),
                         ("cuda", "CUDA input -> CUDA mask")):
        data = summary[summary.track == track]
        if data.empty:
            continue
        table = data.set_index("method").reindex(METHODS)
        lines.extend([
            f"### {label}", "",
            "| 방법 | 최대 DP calls | 중앙 runtime | case-p95 | 유효+2ms | lookup I/L | two-line I/L | importance | 가중 lookup 절감 | Top-R 반환 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method, row in table.iterrows():
            setting = METHOD_SETTINGS.get(method)
            calls = "-" if setting is None else str(setting[0] * setting[1])
            lines.append(
                f"| {METHOD_LABELS[method]} | {calls} | {row.runtime_median_ms:.3f} ms "
                f"| {row.runtime_case_p95_ms:.3f} ms "
                f"| {100 * row.valid_and_deadline_pass_rate:.1f}% "
                f"| {row.lookup_efficiency_gain_pct_mean_valid:.3f}% "
                f"| {row.two_line_efficiency_gain_pct_mean_valid:.3f}% "
                f"| {row.importance_gain_pct_mean_valid:.3f}% "
                f"| {row.lookup_saving_pct_weighted_valid:.3f}% "
                f"| {100 * row.returned_top_r_rate:.1f}% |"
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
        (shape_summary.track == primary)
        & (shape_summary.method == best_quality_method)
    ].sort_values(["n", "d"])
    quality_delta = (
        best_quality.lookup_efficiency_gain_pct_mean_valid
        - baseline.lookup_efficiency_gain_pct_mean_valid
    )
    lines.extend([
        "## 판정", "",
        f"평균 lookup I/L이 가장 높은 scaled 설정은 `{METHOD_LABELS[best_quality_method]}`다. "
        f"Paper 대비 `{best_quality.lookup_efficiency_gain_pct_mean_valid:.3f}%`이며, "
        f"Exp24 기준 `rho2/mu4` 대비 `{quality_delta:+.3f}` percentage points다.", "",
        f"이 설정의 `{primary}` case-p95는 `{best_quality.runtime_case_p95_ms:.3f} ms`, "
        f"2ms 유효 통과율은 `{100 * best_quality.valid_and_deadline_pass_rate:.1f}%`다.", "",
        f"계산량 대비 포화점은 `{METHOD_LABELS[knee_method]}`다. 최고 설정보다 lookup I/L이 "
        f"`{best_quality.lookup_efficiency_gain_pct_mean_valid - knee.lookup_efficiency_gain_pct_mean_valid:.3f}` "
        f"percentage points만 낮지만, CUDA case-p95는 "
        f"`{knee.runtime_case_p95_ms:.3f} ms`로 최고 설정의 "
        f"`{best_quality.runtime_case_p95_ms:.3f} ms`보다 "
        f"`{100 * (1.0 - knee.runtime_case_p95_ms / best_quality.runtime_case_p95_ms):.1f}%` 낮다. "
        f"따라서 2ms를 완화하고 품질을 우선한다면 이 설정이 실용적인 scale-up이다.", "",
        f"95% 이상의 2ms 통과율을 만족하는 후보가 있으면 그 안에서, 없으면 통과율이 "
        f"가장 높은 후보로 고른 latency 쪽 설정은 `{METHOD_LABELS[best_deadline_method]}`다 "
        f"(lookup I/L `{best_deadline.lookup_efficiency_gain_pct_mean_valid:.3f}%`, "
        f"통과율 `{100 * best_deadline.valid_and_deadline_pass_rate:.1f}%`).", "",
        f"## Shape별 {METHOD_LABELS[best_quality_method]} ({primary})", "",
        "| Shape | lookup I/L | two-line I/L | importance | case-p95 | 유효+2ms |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in winner_shapes.iterrows():
        lines.append(
            f"| {row['shape']} | {row.lookup_efficiency_gain_pct_mean_valid:.3f}% "
            f"| {row.two_line_efficiency_gain_pct_mean_valid:.3f}% "
            f"| {row.importance_gain_pct_mean_valid:.3f}% "
            f"| {row.runtime_case_p95_ms:.3f} ms "
            f"| {100 * row.valid_and_deadline_pass_rate:.1f}% |"
        )
    lines.extend([
        "", "## 해석 범위", "",
        f"- 실제 activation은 `{args.model}`의 저장된 Qwen forward trace다.",
        "- 후보 간 차이는 오직 rho 반복 수와 각 반복의 mu bisection 호출 수다.",
        "- small-N oracle만 synthetic이며 실제 trace 결론과 분리했다.",
        "- lookup latency는 Orin AGX profile 예측값이며 실제 NVMe I/O 측정값이 아니다.",
        "- selector timing은 warm importance-to-mask 구간이며 LM forward는 제외한다.",
        "", "![Scale sweep](results/scale_sweep.png)", "",
        "![Runtime-quality](results/runtime_quality.png)", "",
        "![Best setting by shape](results/shape_comparison.png)", "",
    ])
    if oracle is not None and len(oracle):
        lines.extend(["![Small-N oracle](results/oracle_optimality.png)", ""])
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    if "trace_id" not in frame:
        raise SystemExit("Experiment 27 requires real-model activation results")
    frame = EXP24.add_paired_metrics(frame)
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    oracle_path = args.output_dir / "oracle_trials.csv"
    oracle = pd.read_csv(oracle_path) if oracle_path.exists() else None
    summary_frame, shape_frame, summary = EXP24.summarize(frame, oracle, args)
    summary["format"] = "experiment-27-fixed-r-scale-v1-real-activations"
    summary["comparison"] = (
        "unchanged Experiment 24 fixed-R selector with only rho_iterations "
        "and mu_calls scaled"
    )
    summary["configuration"]["method_settings"] = {
        method: {"rho_iterations": setting[0], "mu_calls": setting[1]}
        for method, setting in METHOD_SETTINGS.items()
    }
    primary = "cuda" if "cuda" in set(summary_frame.track) else "host"
    indexed = summary_frame[summary_frame.track == primary].set_index("method")
    candidates = indexed.loc[list(SCALED_METHODS)]
    best_method = candidates.lookup_efficiency_gain_pct_mean_valid.idxmax()
    near_best = candidates[
        candidates.lookup_efficiency_gain_pct_mean_valid
        >= candidates.loc[best_method, "lookup_efficiency_gain_pct_mean_valid"] - 0.05
    ]
    knee_method = near_best.runtime_case_p95_ms.idxmin()
    summary["scale_assessment"] = {
        "primary_track": primary,
        "maximum_quality_method": best_method,
        "maximum_quality_lookup_gain_vs_paper_pct": float(
            candidates.loc[best_method, "lookup_efficiency_gain_pct_mean_valid"]
        ),
        "maximum_quality_case_p95_ms": float(
            candidates.loc[best_method, "runtime_case_p95_ms"]
        ),
        "quality_knee_method_within_0_05pp": knee_method,
        "quality_knee_lookup_gain_vs_paper_pct": float(
            candidates.loc[knee_method, "lookup_efficiency_gain_pct_mean_valid"]
        ),
        "quality_knee_case_p95_ms": float(
            candidates.loc[knee_method, "runtime_case_p95_ms"]
        ),
    }

    comparison_rows = []
    for track in list(dict.fromkeys(frame.track)):
        quality = frame[frame.track == track].pivot(
            index=["trace_id", "budget_index"], columns="method",
            values="lookup_efficiency",
        )
        baseline = quality["rho2_mu4"]
        for method in SCALED_METHODS:
            delta = 100.0 * (quality[method] / baseline - 1.0)
            setting = METHOD_SETTINGS[method]
            comparison_rows.append({
                "track": track,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "rho_iterations": setting[0],
                "mu_calls": setting[1],
                "maximum_scalarized_calls": setting[0] * setting[1],
                "lookup_efficiency_delta_vs_rho2_mu4_pct_mean": float(delta.mean()),
                "lookup_efficiency_win_rate_vs_rho2_mu4": float((delta > 1e-9).mean()),
                "lookup_efficiency_equal_rate_vs_rho2_mu4": float(
                    (delta.abs() <= 1e-9).mean()
                ),
                "lookup_efficiency_loss_rate_vs_rho2_mu4": float((delta < -1e-9).mean()),
            })
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(args.output_dir / "scale_comparison.csv", index=False)
    summary["scale_comparison_vs_rho2_mu4"] = comparison_rows
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    shape_frame.to_csv(args.output_dir / "shape_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    EXP24.plot_overall(summary_frame, args.output_dir / "runtime_quality.png", args.deadline_ms)
    chosen = (
        summary_frame[
            (summary_frame.track == primary)
            & (summary_frame.method.isin(SCALED_METHODS))
        ].sort_values("lookup_efficiency_gain_pct_mean_valid", ascending=False).iloc[0].method
    )
    EXP24.plot_by_shape(shape_frame, chosen, args.output_dir / "shape_comparison.png")
    plot_scale(summary_frame, args.output_dir / "scale_sweep.png", args.deadline_ms)
    if oracle is not None and len(oracle):
        EXP24.plot_oracle(oracle, args.output_dir / "oracle_optimality.png")
    write_report(summary_frame, shape_frame, oracle, args)
    print(summary_frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    _, tracks = EXP24.validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return

    traces, trace_metadata, trace_path = EXP24.prepare_activation_traces(args)
    shape_names = list(dict.fromkeys(trace["shape"] for trace in traces))
    shape_specs = [
        dict(next(item for item in SHAPES if item["shape"] == name))
        for name in shape_names
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
        "method_settings": {
            method: {"rho_iterations": setting[0], "mu_calls": setting[1]}
            for method, setting in METHOD_SETTINGS.items()
        },
        "requested_tracks": args.tracks,
        "executed_tracks": tracks,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.collect_traces_only:
        print(f"selected {len(traces)} traces; stopping as requested")
        return

    lookup_table = BASE.LatencyTable.load(args.profile)
    if not args.skip_self_check:
        EXP24.self_check(lookup_table, args.saturation_kib)
    if not args.skip_oracle:
        EXP24.write_csv(
            args.output_dir / "oracle_trials.csv",
            EXP24.collect_oracle(args, lookup_table),
        )
    rows, timings, models = EXP24.collect(args, traces, tracks, lookup_table)
    EXP24.write_csv(args.output_dir / "trials.csv", rows)
    EXP24.write_csv(args.output_dir / "timing_samples.csv", timings)
    EXP24.write_csv(args.output_dir / "models.csv", models)
    analyze(args)


if __name__ == "__main__":
    main()
