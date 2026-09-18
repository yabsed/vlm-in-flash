#!/usr/bin/env python3
"""Experiment 42: directly measure the laptop SSD latency after s."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PROFILE_SCRIPT = (
    PROJECT_ROOT / "preliminary_research" / "vlm-flash" / "scripts"
    / "profile_flash.py"
)
OLD_PROFILE = (
    PROJECT_ROOT / "experiments" / "26_frontier_adaptive_trim"
    / "results_laptop" / "laptop_sn850x_profile.json"
)
sys.path.insert(0, str(PROFILE_SCRIPT.parent))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE = _load_module("profile_flash_for_42", PROFILE_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-profile", type=Path, default=OLD_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_laptop")
    parser.add_argument("--blob", type=Path, default=HERE / "results_laptop" / "profile_blob.dat")
    parser.add_argument("--min-kib", type=int, default=192)
    parser.add_argument("--max-kib", type=int, default=768)
    parser.add_argument("--step-kib", type=int, default=4)
    parser.add_argument("--saturation-kib", type=int, default=240)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--blob-mb", type=int, default=128)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.old_profile.is_file():
        raise SystemExit(f"missing old profile: {args.old_profile}")
    if not (1 <= args.min_kib < args.saturation_kib < args.max_kib <= 768):
        raise SystemExit("require 1 <= min < s < max <= 768 KiB")
    if min(args.step_kib, args.iters, args.threads, args.blob_mb) < 1:
        raise SystemExit("step, iterations, threads, and blob size must be positive")


def measurement_sizes(args: argparse.Namespace) -> list[int]:
    values = set(range(args.min_kib, args.max_kib + 1, args.step_kib))
    s = args.saturation_kib
    values.update(s + delta for delta in (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32))
    values.add(args.max_kib)
    return sorted(value for value in values if args.min_kib <= value <= args.max_kib)


def latency_table(raw: dict) -> dict[int, float]:
    table = {}
    for kib, runs in raw.items():
        throughput_mib_s = float(PROFILE.saturation_throughput(runs, kib))
        table[int(kib)] = (float(kib) / 1024.0) / throughput_mib_s * 1000.0
    return table


def metrics(observed: np.ndarray, predicted: np.ndarray, parameters: int) -> dict:
    residual = observed - predicted
    rss = float(np.square(residual).sum())
    total = float(np.square(observed - observed.mean()).sum())
    n = len(observed)
    return {
        "r_squared": float(1.0 - rss / total) if total else float("nan"),
        "rmse_ms": float(np.sqrt(np.mean(np.square(residual)))),
        "mape_pct": float(np.mean(np.abs(residual / observed)) * 100.0),
        "max_abs_pct": float(np.max(np.abs(residual / observed)) * 100.0),
        "bias_pct": float(np.mean(residual / observed) * 100.0),
        "bic": float(n * np.log(max(rss / n, 1e-30)) + parameters * np.log(n)),
    }


def fit_models(frame: pd.DataFrame, saturation_kib: int, old_fit: dict) -> dict:
    post = frame[frame.r_kib > saturation_kib]
    r = post.r_kib.to_numpy(dtype=float)
    latency = post.latency_ms.to_numpy(dtype=float)

    c2 = float(np.dot(r, latency) / np.dot(r, r))
    origin_prediction = c2 * r
    affine_slope, affine_intercept = np.polyfit(r, latency, 1)
    affine_prediction = affine_intercept + affine_slope * r
    quadratic = np.polyfit(r, latency, 2)
    quadratic_prediction = np.polyval(quadratic, r)
    old_prediction = old_fit["c2_ms_per_row"] * r

    ratio = latency / r
    ratio_slope, ratio_intercept = np.polyfit(r, ratio, 1)
    fitted_start = ratio_intercept + ratio_slope * float(r.min())
    fitted_end = ratio_intercept + ratio_slope * float(r.max())

    direction_c2 = {}
    for column in ("latency_asc_ms", "latency_desc_ms"):
        values = post[column].to_numpy(dtype=float)
        direction_c2[column] = float(np.dot(r, values) / np.dot(r, r))

    disagreement = np.abs(
        post.latency_asc_ms.to_numpy(dtype=float)
        - post.latency_desc_ms.to_numpy(dtype=float)
    ) / latency * 100.0
    stable = disagreement <= 10.0
    stable_r = r[stable]
    stable_latency = latency[stable]
    stable_c2 = float(np.dot(stable_r, stable_latency) / np.dot(stable_r, stable_r))

    bins = [
        (saturation_kib, 384), (384, 512), (512, 640), (640, 768),
    ]
    local = []
    for lo, hi in bins:
        group = frame[(frame.r_kib > lo) & (frame.r_kib <= hi)]
        if not len(group):
            continue
        x = group.r_kib.to_numpy(dtype=float)
        y = group.latency_ms.to_numpy(dtype=float)
        local_c2 = float(np.dot(x, y) / np.dot(x, x))
        local.append({
            "lo_exclusive_kib": lo, "hi_inclusive_kib": hi,
            "points": len(group), "c2_ms_per_kib": local_c2,
            "mean_l_over_r_ms_per_kib": float((y / x).mean()),
            "cv_l_over_r_pct": float(100.0 * (y / x).std(ddof=1) / (y / x).mean()),
        })

    return {
        "post_points": len(post),
        "post_range_kib": [int(r.min()), int(r.max())],
        "through_origin": {"c2_ms_per_kib": c2, **metrics(latency, origin_prediction, 1)},
        "affine": {
            "intercept_ms": float(affine_intercept),
            "slope_ms_per_kib": float(affine_slope),
            **metrics(latency, affine_prediction, 2),
        },
        "quadratic": {
            "quadratic_ms_per_kib2": float(quadratic[0]),
            "linear_ms_per_kib": float(quadratic[1]),
            "intercept_ms": float(quadratic[2]),
            **metrics(latency, quadratic_prediction, 3),
        },
        "old_twoline_extrapolation": {
            "c2_ms_per_kib": float(old_fit["c2_ms_per_row"]),
            **metrics(latency, old_prediction, 0),
        },
        "l_over_r": {
            "mean_ms_per_kib": float(ratio.mean()),
            "std_ms_per_kib": float(ratio.std(ddof=1)),
            "cv_pct": float(100.0 * ratio.std(ddof=1) / ratio.mean()),
            "trend_ms_per_kib2": float(ratio_slope),
            "fitted_start_ms_per_kib": float(fitted_start),
            "fitted_end_ms_per_kib": float(fitted_end),
            "fitted_drift_pct": float(100.0 * (fitted_end - fitted_start) / fitted_start),
        },
        "direction_c2_ms_per_kib": direction_c2,
        "direction_c2_spread_pct": float(
            100.0 * abs(direction_c2["latency_asc_ms"] - direction_c2["latency_desc_ms"])
            / np.mean(list(direction_c2.values()))
        ),
        "robust_sensitivity": {
            "direction_disagreement_threshold_pct": 10.0,
            "excluded_r_kib": [int(value) for value in r[~stable]],
            "points": int(stable.sum()),
            "c2_ms_per_kib": stable_c2,
            **metrics(stable_latency, stable_c2 * stable_r, 1),
        },
        "local_ranges": local,
    }


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    frame = pd.read_csv(output / "post_s_measurements.csv")
    old_document = json.loads(args.old_profile.read_text())
    old_table = {int(key): float(value) for key, value in old_document["table"].items()}

    # Same continuous fit used by Experiment 41, expressed per KiB.
    old_x = np.asarray(sorted(old_table), dtype=float)
    old_y = np.asarray([old_table[int(value)] for value in old_x], dtype=float)
    s = float(args.saturation_kib)
    design = np.empty((len(old_x), 2), dtype=float)
    short = old_x <= s
    design[short, 0] = old_x[short] - s
    design[short, 1] = s
    design[~short, 0] = 0.0
    design[~short, 1] = old_x[~short]
    c1, c2 = np.linalg.lstsq(design, old_y, rcond=None)[0]
    old_fit = {
        "a_ms": float((c2 - c1) * s),
        "c1_ms_per_row": float(c1),
        "c2_ms_per_row": float(c2),
        "saturation_kib": s,
    }
    fit = fit_models(frame, args.saturation_kib, old_fit)

    post = frame[frame.r_kib > args.saturation_kib]
    combined_x = np.concatenate((old_x, post.r_kib.to_numpy(dtype=float)))
    combined_y = np.concatenate((old_y, post.latency_ms.to_numpy(dtype=float)))
    combined_design = np.empty((len(combined_x), 2), dtype=float)
    combined_short = combined_x <= s
    combined_design[combined_short, 0] = combined_x[combined_short] - s
    combined_design[combined_short, 1] = s
    combined_design[~combined_short, 0] = 0.0
    combined_design[~combined_short, 1] = combined_x[~combined_short]
    updated_c1, updated_c2 = np.linalg.lstsq(
        combined_design, combined_y, rcond=None
    )[0]
    updated_prediction = combined_design @ np.asarray([updated_c1, updated_c2])
    fit["updated_continuous_two_line"] = {
        "a_ms": float((updated_c2 - updated_c1) * s),
        "c1_ms_per_kib": float(updated_c1),
        "c2_ms_per_kib": float(updated_c2),
        "saturation_kib": s,
        **metrics(combined_y, updated_prediction, 2),
        "pre_mape_pct": float(np.mean(np.abs(
            (combined_y[combined_short] - updated_prediction[combined_short])
            / combined_y[combined_short]
        )) * 100.0),
        "post_mape_pct": float(np.mean(np.abs(
            (combined_y[~combined_short] - updated_prediction[~combined_short])
            / combined_y[~combined_short]
        )) * 100.0),
    }
    lookup_c2 = float(old_table[args.saturation_kib] / args.saturation_kib)
    lookup_prediction = lookup_c2 * post.r_kib.to_numpy(dtype=float)
    fit["old_lookup_extrapolation"] = {
        "c2_ms_per_kib": lookup_c2,
        **metrics(post.latency_ms.to_numpy(dtype=float), lookup_prediction, 0),
    }

    overlap = frame[frame.r_kib <= args.saturation_kib].copy()
    overlap["old_latency_ms"] = [old_table[int(r)] for r in overlap.r_kib]
    overlap["relative_delta"] = (
        overlap.latency_ms - overlap.old_latency_ms
    ) / overlap.old_latency_ms
    repeatability = {
        "asc_desc_mape_pct": float(np.mean(np.abs(
            frame.latency_asc_ms - frame.latency_desc_ms
        ) / frame.latency_ms) * 100.0),
        "asc_desc_bias_pct": float(np.mean(
            (frame.latency_asc_ms - frame.latency_desc_ms) / frame.latency_ms
        ) * 100.0),
        "overlap_old_profile_mape_pct": float(
            np.mean(np.abs(overlap.relative_delta)) * 100.0
        ),
        "overlap_old_profile_bias_pct": float(overlap.relative_delta.mean() * 100.0),
        "overlap_points": len(overlap),
        "max_direction_disagreement_pct": float(np.max(np.abs(
            frame.latency_asc_ms - frame.latency_desc_ms
        ) / frame.latency_ms) * 100.0),
        "max_direction_disagreement_r_kib": int(frame.iloc[np.argmax(np.abs(
            frame.latency_asc_ms - frame.latency_desc_ms
        ) / frame.latency_ms)].r_kib),
    }
    fit["repeatability"] = repeatability
    fit["old_experiment_41_fit"] = old_fit
    (output / "fit_summary.json").write_text(json.dumps(fit, indent=2) + "\n")
    pd.DataFrame(fit["local_ranges"]).to_csv(output / "local_slopes.csv", index=False)

    make_plots(output, frame, old_x, old_y, old_fit, fit, args.saturation_kib)
    write_report(output, frame, fit, args)


def make_plots(output: Path, frame: pd.DataFrame, old_x: np.ndarray,
               old_y: np.ndarray, old_fit: dict, fit: dict,
               saturation_kib: int) -> None:
    r = frame.r_kib.to_numpy(dtype=float)
    latency = frame.latency_ms.to_numpy(dtype=float)
    x_model = np.linspace(1.0, float(r.max()), 1000)
    old_prediction = np.where(
        x_model <= saturation_kib,
        old_fit["a_ms"] + old_fit["c1_ms_per_row"] * x_model,
        old_fit["c2_ms_per_row"] * x_model,
    )
    new_c2 = fit["through_origin"]["c2_ms_per_kib"]
    lookup_c2 = fit["old_lookup_extrapolation"]["c2_ms_per_kib"]

    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    ax.plot(old_x, old_y, color="#94A3B8", linewidth=1.0,
            label="Original profile (1–240 KiB)")
    ax.scatter(r, frame.latency_asc_ms, color="#93C5FD", s=10, alpha=0.5,
               label="New ascending sweep")
    ax.scatter(r, frame.latency_desc_ms, color="#A7F3D0", s=10, alpha=0.5,
               label="New descending sweep")
    ax.plot(r, latency, color="#0F766E", linewidth=1.3,
            label="New two-direction mean")
    ax.plot(x_model, old_prediction, color="#DC2626", linewidth=1.8,
            linestyle="--", label="Experiment 41 2-line extrapolation")
    post_x = x_model[x_model >= saturation_kib]
    ax.plot(post_x, lookup_c2 * post_x, color="#D97706", linewidth=1.5,
            linestyle=":", label="Original lookup extrapolation")
    ax.plot(post_x, new_c2 * post_x, color="#7C3AED", linewidth=1.8,
            label="Post-s through-origin fit")
    ax.axvline(saturation_kib, color="#475569", linestyle=":", linewidth=1.2)
    ax.set_xlabel("r: contiguous read size (KiB)")
    ax.set_ylabel("L(r): profiled read latency (ms)")
    ax.set_title("Experiment 42: measured latency beyond s")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "twoline_fit_extended.png", dpi=220)
    fig.savefig(output / "twoline_fit_extended.pdf")
    plt.close(fig)

    post = frame[frame.r_kib > saturation_kib]
    full_r = np.concatenate((old_x, post.r_kib.to_numpy(dtype=float)))
    full_ratio = np.concatenate((old_y / old_x, post.l_over_r_ms_per_kib))
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 8.0), sharex=True)
    axes[0].plot(full_r, full_ratio, color="#0F172A", linewidth=1.25,
                 label="Measured L/r")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("L(r) / r (ms/KiB, log scale)")
    axes[0].set_title("Full-range L/r: startup overhead amortizes before saturation")

    axes[1].scatter(
        frame.r_kib, frame.latency_asc_ms / frame.r_kib,
        color="#60A5FA", s=12, alpha=0.5, label="Ascending L/r",
    )
    axes[1].scatter(
        frame.r_kib, frame.latency_desc_ms / frame.r_kib,
        color="#34D399", s=12, alpha=0.5, label="Descending L/r",
    )
    axes[1].plot(full_r, full_ratio, color="#0F172A", linewidth=1.25,
                 label="Measured L/r")
    axes[1].set_ylabel("L(r) / r (ms/KiB)")
    axes[1].set_title("Linear-scale view near the post-s plateau")
    zoom_values = full_ratio[full_r >= 128.0]
    axes[1].set_ylim(
        max(0.0, float(np.quantile(zoom_values, 0.01)) * 0.90),
        float(np.quantile(zoom_values, 0.99)) * 1.10,
    )

    for ax in axes:
        ax.axhline(old_fit["c2_ms_per_row"], color="#DC2626", linestyle="--",
                   linewidth=1.5, label="Experiment 41 c₂")
        ax.axhline(lookup_c2, color="#D97706", linestyle=":", linewidth=1.5,
                   label="Original lookup slope")
        ax.axhline(new_c2, color="#7C3AED", linestyle="-.", linewidth=1.5,
                   label="Measured post-s c₂ fit")
        ax.axvline(saturation_kib, color="#475569", linestyle=":", linewidth=1.2,
                   label="s = 240 KiB")
        ax.set_xlim(0.0, float(frame.r_kib.max()))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, ncol=3)
    axes[1].legend(fontsize=7.5, ncol=3)
    axes[1].set_xlabel("r (KiB)")
    fig.suptitle("L(r)/r from r = 1 KiB through 768 KiB")
    fig.tight_layout()
    fig.savefig(output / "r_vs_l_over_r.png", dpi=220)
    fig.savefig(output / "r_vs_l_over_r.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    old_post_prediction = old_fit["c2_ms_per_row"] * post.r_kib
    new_post_prediction = new_c2 * post.r_kib
    axes[0].plot(
        post.r_kib,
        100.0 * (post.latency_ms - old_post_prediction) / post.latency_ms,
        color="#DC2626", label="Residual vs Experiment 41 extrapolation",
    )
    axes[0].plot(
        post.r_kib,
        100.0 * (post.latency_ms - new_post_prediction) / post.latency_ms,
        color="#7C3AED", label="Residual vs refitted c₂r",
    )
    axes[0].axhline(0.0, color="#64748B", linewidth=1)
    axes[0].set_ylabel("Signed residual (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    throughput = 1000.0 * frame.r_kib / frame.latency_ms / 1024.0
    axes[1].plot(frame.r_kib, throughput, color="#0F766E", linewidth=1.2)
    axes[1].axvline(saturation_kib, color="#475569", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("r (KiB)")
    axes[1].set_ylabel("Logical throughput (MiB/s)")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Post-s residual and throughput diagnostics")
    fig.tight_layout()
    fig.savefig(output / "post_s_diagnostics.png", dpi=220)
    fig.savefig(output / "post_s_diagnostics.pdf")
    plt.close(fig)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_report(output: Path, frame: pd.DataFrame, fit: dict,
                 args: argparse.Namespace) -> None:
    origin = fit["through_origin"]
    affine = fit["affine"]
    quadratic = fit["quadratic"]
    old = fit["old_twoline_extrapolation"]
    old_lookup = fit["old_lookup_extrapolation"]
    ratio = fit["l_over_r"]
    repeat = fit["repeatability"]
    robust = fit["robust_sensitivity"]
    updated = fit["updated_continuous_two_line"]
    linear_enough = (
        origin["r_squared"] >= 0.98
        and origin["mape_pct"] <= 5.0
        and ratio["cv_pct"] <= 5.0
    )
    verdict = (
        "**측정 범위에서는 post-s latency가 `c₂r`에 충분히 가깝다.**"
        if linear_enough else
        "**측정 범위에서 post-s latency를 순수한 `c₂r`로 보기 어렵다.**"
    )
    lines = [
        "# Experiment 42: s 이후 SSD latency 선형성", "",
        "노트북 SN850X에서 기존 saturation `s=240 KiB` 전후를 실제 O_DIRECT로 "
        "다시 측정했다. 크기 순서에 따른 열·시간 drift를 보기 위해 같은 점을 "
        "오름차순과 내림차순으로 각각 측정하고 평균했다.", "",
        "## 판정", "", verdict, "",
        f"240 KiB 이후 `{fit['post_points']}`개 점에서 origin-constrained fit은 "
        f"`L(r)={origin['c2_ms_per_kib']:.9f}r`이며, "
        f"`R²={origin['r_squared']:.5f}`, `MAPE={origin['mape_pct']:.2f}%`, "
        f"최대 절대오차 `{origin['max_abs_pct']:.2f}%`다.", "",
        f"직접적인 선형성 지표인 `L/r`의 평균은 "
        f"`{ratio['mean_ms_per_kib']:.9f} ms/KiB`, CV는 "
        f"`{ratio['cv_pct']:.2f}%`다. 선형 trend가 예측하는 측정 구간 처음→끝 "
        f"drift는 `{ratio['fitted_drift_pct']:+.2f}%`다.", "",
        f"오름/내림 방향 차이가 10%를 넘은 점 `{robust['excluded_r_kib']}`을 "
        f"제외해도 `c₂={robust['c2_ms_per_kib']:.9f}`, "
        f"`R²={robust['r_squared']:.5f}`, `MAPE={robust['mape_pct']:.2f}%`로 "
        "결론은 변하지 않는다.", "",
        "## 기존 Experiment 41 외삽 검증", "",
        f"기존 `c₂={old['c2_ms_per_kib']:.9f} ms/KiB`의 post-s 실측 MAPE는 "
        f"`{old['mape_pct']:.2f}%`, bias는 `{old['bias_pct']:+.2f}%`다. "
        f"새 post-s fit과의 c₂ 차이는 "
        f"`{100.0 * (origin['c2_ms_per_kib'] / old['c2_ms_per_kib'] - 1):+.2f}%`다.", "",
        f"원래 lookup의 범위 밖 규칙은 마지막 점을 비례 확장하므로 slope가 "
        f"`{old_lookup['c2_ms_per_kib']:.9f} ms/KiB`다. 이 extrapolation의 "
        f"post-s MAPE는 `{old_lookup['mape_pct']:.2f}%`, bias는 "
        f"`{old_lookup['bias_pct']:+.2f}%`로, 기존 2-line보다 오차가 크다.", "",
        "기존 1–240 KiB와 새 241–768 KiB를 함께 연속 2-line으로 다시 "
        f"적합하면 `a={updated['a_ms']:.6f} ms`, "
        f"`c₁={updated['c1_ms_per_kib']:.9f} ms/KiB`, "
        f"`c₂={updated['c2_ms_per_kib']:.9f} ms/KiB`다. 전체 "
        f"`R²={updated['r_squared']:.5f}`, MAPE `{updated['mape_pct']:.2f}%` "
        f"(pre-s `{updated['pre_mape_pct']:.2f}%`, post-s "
        f"`{updated['post_mape_pct']:.2f}%`)다.", "",
        "## 모델 비교", "",
        "| model | parameters | R² | MAPE | RMSE | BIC |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| `c₂r` | 1 | {origin['r_squared']:.5f} | {origin['mape_pct']:.2f}% "
        f"| {origin['rmse_ms']:.6f} ms | {origin['bic']:.1f} |",
        f"| `b₀+b₁r` | 2 | {affine['r_squared']:.5f} | {affine['mape_pct']:.2f}% "
        f"| {affine['rmse_ms']:.6f} ms | {affine['bic']:.1f} |",
        f"| quadratic | 3 | {quadratic['r_squared']:.5f} | {quadratic['mape_pct']:.2f}% "
        f"| {quadratic['rmse_ms']:.6f} ms | {quadratic['bic']:.1f} |", "",
        f"Quadratic의 BIC가 `c₂r`보다 "
        f"`{origin['bic'] - quadratic['bic']:.1f}` 낮으므로 작은 곡률은 "
        "검출된다. 따라서 `c₂r`은 좋은 공학적 근사이지 정확한 물리 법칙은 아니다.", "",
        "## 구간별 L/r", "",
        "| range (KiB) | points | fitted c₂ (ms/KiB) | mean L/r | CV |",
        "|:---|---:|---:|---:|---:|",
    ]
    for row in fit["local_ranges"]:
        lines.append(
            f"| ({row['lo_exclusive_kib']}, {row['hi_inclusive_kib']}] "
            f"| {row['points']} | {row['c2_ms_per_kib']:.9f} "
            f"| {row['mean_l_over_r_ms_per_kib']:.9f} "
            f"| {row['cv_l_over_r_pct']:.2f}% |"
        )
    selected = frame[frame.r_kib.isin(
        [240, 241, 242, 256, 320, 384, 512, 640, 768]
    )]
    lines.extend([
        "", "## 대표 실측값", "",
        "| r (KiB) | L(r) (ms) | L/r (ms/KiB) | throughput (MiB/s) |",
        "|---:|---:|---:|---:|",
    ])
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {int(row.r_kib)} | {row.latency_ms:.6f} "
            f"| {row.l_over_r_ms_per_kib:.9f} "
            f"| {row.logical_throughput_mib_s:.1f} |"
        )
    lines.extend([
        "", "## 반복성", "",
        f"- 오름차순/내림차순 MAPE: `{repeat['asc_desc_mape_pct']:.2f}%`; "
        f"방향 bias: `{repeat['asc_desc_bias_pct']:+.2f}%`.",
        f"- 방향별 c₂ 차이: `{fit['direction_c2_spread_pct']:.2f}%`.",
        f"- 최대 방향 불일치: `{repeat['max_direction_disagreement_pct']:.2f}%` "
        f"at `r={repeat['max_direction_disagreement_r_kib']} KiB`.",
        f"- 192–240 KiB overlap의 기존 profile 대비 MAPE: "
        f"`{repeat['overlap_old_profile_mape_pct']:.2f}%`; bias: "
        f"`{repeat['overlap_old_profile_bias_pct']:+.2f}%`.", "",
        "이 실험의 L(r)은 논문 profiler와 동일하게 여러 chunk-count에서 얻은 "
        "포화 throughput을 단일 chunk latency로 환산한 값이다. 단일 read의 "
        "wall-clock latency 자체는 아니다.", "",
        f"- 측정점: `{len(frame)}`개 × 두 방향.",
        f"- 범위: `{args.min_kib}–{args.max_kib} KiB`; 기본 간격 "
        f"`{args.step_kib} KiB`와 s 주변 추가점.",
        f"- 각 chunk-count마다 warmup `{args.warmup}`, sample `{args.iters}`, "
        f"O_DIRECT thread `{args.threads}`.", "",
        "![Extended fit](results_laptop/twoline_fit_extended.png)", "",
        "![r versus L over r](results_laptop/r_vs_l_over_r.png)", "",
        "![Diagnostics](results_laptop/post_s_diagnostics.png)", "",
    ])
    (HERE / "report_laptop.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = PROFILE.native()
    if extension is None:
        raise SystemExit(f"native reader unavailable: {PROFILE.unavailable_reason()}")
    num_rows = args.blob_mb * 1024 // PROFILE.ROW_KB
    PROFILE.prepare_blob(str(args.blob), num_rows)
    sizes = measurement_sizes(args)
    print(f"measuring {len(sizes)} sizes ascending", flush=True)
    asc_raw = PROFILE.measure(
        extension, str(args.blob), num_rows, sizes,
        args.iters, args.warmup, args.threads,
    )
    print(f"measuring {len(sizes)} sizes descending", flush=True)
    desc_raw = PROFILE.measure(
        extension, str(args.blob), num_rows, list(reversed(sizes)),
        args.iters, args.warmup, args.threads,
    )
    asc = latency_table(asc_raw)
    desc = latency_table(desc_raw)
    rows = []
    for r_kib in sizes:
        mean = statistics.mean((asc[r_kib], desc[r_kib]))
        rows.append({
            "r_kib": r_kib,
            "latency_asc_ms": asc[r_kib],
            "latency_desc_ms": desc[r_kib],
            "latency_ms": mean,
            "direction_half_span_ms": abs(asc[r_kib] - desc[r_kib]) / 2.0,
            "l_over_r_ms_per_kib": mean / r_kib,
            "logical_throughput_mib_s": 1000.0 * r_kib / mean / 1024.0,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "post_s_measurements.csv", index=False)
    metadata = {
        "format": "experiment-42-post-s-ssd-linearity-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "flash": "WD_BLACK SN850X 1000GB",
        "old_profile": str(args.old_profile),
        "blob": str(args.blob.resolve()),
        "blob_mb": args.blob_mb,
        "sizes_kib": sizes,
        "saturation_kib": args.saturation_kib,
        "iters_per_direction": args.iters,
        "warmup_per_direction": args.warmup,
        "threads": args.threads,
        "directions": ["ascending", "descending"],
        "chunk_schedule": PROFILE.NUM_CHUNKS_SCHED,
        "max_native_read_kib": 768,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.analyze_only:
        analyze(args)
        return
    with PROFILE.exclusive(" ".join(sys.argv)):
        run(args)
    analyze(args)


if __name__ == "__main__":
    main()
