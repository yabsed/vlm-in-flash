#!/usr/bin/env python3
"""Experiment 39: weight-aware Cell-1 importance on real laptop workloads."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import re
import time
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/vlmflash_torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm-flash-mpl-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXP37_PATH = (
    PROJECT_ROOT / "experiments" / "37_super_saturation_tiles"
    / "run_experiment.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP37 = _load_module("experiment_37_for_39", EXP37_PATH)
EXP36 = EXP37.EXP36
EXP35 = EXP37.EXP35
EXP34 = EXP35.EXP34
EXP32 = EXP37.EXP32
MODEL_SPECS = EXP37.MODEL_SPECS
PROMPTS = EXP37.PROMPTS
LOCAL_PROFILE = EXP37.LOCAL_PROFILE
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")

METHODS = ("paper", "cell1_abs", "cell1_x2", "wcell1")
METHOD_LABELS = {
    "paper": "Paper |x|",
    "cell1_abs": "Cell-1 |x|",
    "cell1_x2": "Cell-1 X²",
    "wcell1": "WCell-1 X²·W²",
}
METHOD_COLORS = {
    "paper": "#D97706",
    "cell1_abs": "#0F766E",
    "cell1_x2": "#2563EB",
    "wcell1": "#7C3AED",
}
METHOD_MARKERS = {
    "paper": "s", "cell1_abs": "o", "cell1_x2": "^", "wcell1": "D",
}
METHOD_SCORE = {
    "paper": "abs", "cell1_abs": "abs", "cell1_x2": "x2", "wcell1": "weighted",
}
METHOD_INTERNAL = {
    "paper": "paper", "cell1_abs": "c1_l1", "cell1_x2": "c1_l1",
    "wcell1": "c1_l1",
}
ERROR_CEILINGS = (0.50, 0.45, 0.42, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15)
KL_CEILINGS = (12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0)
COMPONENTS = (
    "score_median_ms", "selector_median_ms", "read_wall_median_ms",
    "gather_median_ms", "gemm_median_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS),
        default=["qwen05"],
        help=(
            "exactly one model per process; run separate processes and merge "
            "their outputs with merge_results.py"
        ),
    )
    parser.add_argument(
        "--budgets", type=float, nargs="+",
        default=[value / 100.0 for value in range(10, 100, 5)],
    )
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--layer-samples", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--score-repetitions", type=int, default=3)
    parser.add_argument("--score-warmup", type=int, default=1)
    parser.add_argument("--selector-repetitions", type=int, default=2)
    parser.add_argument("--io-repetitions", type=int, default=2)
    parser.add_argument("--io-warmup", type=int, default=1)
    parser.add_argument("--gemm-repetitions", type=int, default=3)
    parser.add_argument("--gemm-warmup", type=int, default=1)
    parser.add_argument("--io-threads", type=int, default=2)
    parser.add_argument("--io-max-read-kib", type=int, default=768)
    parser.add_argument("--io-blob", type=Path)
    parser.add_argument("--profile", type=Path, default=LOCAL_PROFILE)
    parser.add_argument("--saturation-kib", type=float, default=240.0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.55)
    parser.add_argument("--norm-chunk-rows", type=int, default=128)
    parser.add_argument("--projection-throttle-ms", type=float, default=25.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--process-nice", type=int, default=10)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.analyze_only and not torch.cuda.is_available():
        raise SystemExit("Experiment 39 requires the laptop CUDA GPU")
    if not args.profile.is_file():
        raise SystemExit(f"latency profile does not exist: {args.profile}")
    if not args.analyze_only and (args.io_blob is None or not args.io_blob.is_file()):
        raise SystemExit("a real --io-blob is required")
    if not args.analyze_only and len(args.models) != 1:
        raise SystemExit(
            "laptop-safe mode requires exactly one --models value per process; "
            "run each model separately, then use merge_results.py"
        )
    if args.budgets != sorted(set(args.budgets)) or any(
        not 0.0 < value < 1.0 for value in args.budgets
    ):
        raise SystemExit("--budgets must be unique, increasing, and inside (0, 1)")
    counts = (
        args.score_repetitions, args.selector_repetitions, args.io_repetitions,
        args.gemm_repetitions,
    )
    if any(value < 1 for value in counts) or args.score_warmup < 0:
        raise SystemExit("sample counts must be positive and warmup non-negative")
    if not 0.4 <= args.cuda_memory_fraction <= 0.85:
        raise SystemExit("--cuda-memory-fraction must be in [0.4, 0.85]")
    if args.cpu_threads < 1 or not 0 <= args.process_nice <= 19:
        raise SystemExit("invalid CPU thread or nice setting")
    if args.norm_chunk_rows < 1 or args.projection_throttle_ms < 0:
        raise SystemExit("invalid norm chunk or throttle setting")


def make_score(x: torch.Tensor, weight_norm_sq: torch.Tensor,
               kind: str) -> torch.Tensor:
    n = int(x.shape[-1])
    if kind == "abs":
        # Match the released VLMFlash importance calculation exactly.
        return x.detach().abs().reshape(-1, n).mean(dim=0).float()
    energy = x.detach().float().reshape(-1, n).square().mean(dim=0)
    if kind == "x2":
        return energy
    if kind == "weighted":
        return energy * weight_norm_sq
    raise ValueError(f"unknown score kind: {kind}")


def benchmark_score(x: torch.Tensor, weight_norm_sq: torch.Tensor, kind: str,
                    warmup: int, repetitions: int):
    values, score = [], None
    for repetition in range(warmup + repetitions):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        score = make_score(x, weight_norm_sq, kind)
        end.record()
        torch.cuda.synchronize()
        if repetition >= warmup:
            values.append(float(start.elapsed_time(end)))
    assert score is not None
    return score, values


def precompute_weight_norms(model, vlmflash, chunk_rows: int):
    norms, samples = {}, []
    metadata_bytes = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not (
                re.search(vlmflash.DEFAULT_INCLUDE, name)
                and type(module) is torch.nn.Linear
            ):
                continue
            started = time.perf_counter_ns()
            # Convert only a small output-row slab to FP32 at a time. This
            # avoids both a full-size GPU temporary and the system-RAM/swap
            # pressure caused by retaining a CPU model for norm extraction.
            norm = torch.zeros(
                module.in_features, device=module.weight.device,
                dtype=torch.float32,
            )
            for start in range(0, module.out_features, chunk_rows):
                block = module.weight.detach()[start:start + chunk_rows].float()
                norm.add_(block.square().sum(dim=0))
                del block
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - started) / 1e6)
            norms[name] = norm
            metadata_bytes += norm.numel() * norm.element_size()
    return norms, samples, metadata_bytes


def make_projection_hook(
    state: dict, model_key: str, model_label: str, module_name: str, module,
    weight_norm_sq: torch.Tensor, selector, native_reader, args,
    candidate_rows: list[dict], selector_rows: list[dict],
    score_rows: list[dict], io_rows: list[dict], gemm_rows: list[dict],
):
    layer_match = LAYER_RE.search(module_name)
    layer_index = int(layer_match.group(1)) if layer_match else -1
    projection = EXP32.projection_name(module_name)
    n, d = int(module.in_features), int(module.out_features)
    context = selector.context(n, d)

    def hook(_module, inputs, output):
        x = inputs[0].detach()
        reference = output.detach()
        case_key = f"{model_key}|{state['prompt']}|{module_name}"
        scores, score_samples = {}, {}
        for kind in ("abs", "x2", "weighted"):
            score, samples = benchmark_score(
                x, weight_norm_sq, kind, args.score_warmup, args.score_repetitions
            )
            scores[kind] = score
            score_samples[kind] = samples
            for repetition, elapsed in enumerate(samples):
                score_rows.append({
                    "case_key": case_key, "score_kind": kind,
                    "repetition": repetition, "runtime_ms": elapsed,
                })

        measured_masks: dict[str, dict] = {}

        def get_measurement(mask_np: np.ndarray):
            digest = EXP34.mask_digest(mask_np)
            if digest not in measured_masks:
                measured = EXP34.measure_mask(
                    x, reference, module, mask_np, native_reader, args
                )
                measured_masks[digest] = measured
                io_rows.append({
                    "case_key": case_key, "mask_sha256": digest,
                    "read_wall_median_ms": measured["read_wall_median_ms"],
                    "read_wall_p95_ms": measured["read_wall_p95_ms"],
                    "io_median_ms": measured["io_median_ms"],
                    "upload_median_ms": measured["upload_median_ms"],
                    "direct_rate": measured["direct_rate"],
                })
                gemm_rows.append({
                    "case_key": case_key, "mask_sha256": digest,
                    "gather_median_ms": measured["gather_median_ms"],
                    "gather_p95_ms": measured["gather_p95_ms"],
                    "gemm_median_ms": measured["gemm_median_ms"],
                    "gemm_p95_ms": measured["gemm_p95_ms"],
                })
            return digest, measured_masks[digest]

        warm_budget = max(1, min(n - 1, int(round(n * args.budgets[0]))))
        for method in METHODS:
            selector.select(
                METHOD_INTERNAL[method], scores[METHOD_SCORE[method]], warm_budget, d
            )

        for budget_index, fraction in enumerate(args.budgets):
            row_budget = max(1, min(n - 1, int(round(n * fraction))))
            for method in METHODS:
                kind = METHOD_SCORE[method]
                score = scores[kind]
                internal = METHOD_INTERNAL[method]
                mask_np, metadata, selection_samples, deterministic = (
                    EXP36.benchmark_selector(
                        selector, internal, score, row_budget, d,
                        args.selector_repetitions,
                    )
                )
                digest, measured = get_measurement(mask_np)
                selected = int(mask_np.sum())
                score_median = float(np.median(score_samples[kind]))
                selector_median = float(np.median(selection_samples))
                nonselector = float(measured["nonselector_total_ms"])
                candidate_rows.append({
                    "case_key": case_key, "model": model_key,
                    "model_label": model_label, "prompt": state["prompt"],
                    "split": state["split"], "situation": state["situation"],
                    "num_tokens": state["num_tokens"], "module": module_name,
                    "layer": layer_index, "projection": projection,
                    "shape": f"{n}x{d}", "n": n, "d": d,
                    "row_size_kib": 2.0 * d / 1024.0,
                    "paper_parameter_source": context["paper_parameter_source"],
                    "method": method, "method_label": METHOD_LABELS[method],
                    "score_kind": kind, "budget_index": budget_index,
                    "budget_fraction": fraction, "row_budget": row_budget,
                    "selected_rows": selected, "selected_fraction": selected / n,
                    "row_match": selected == row_budget, "mask_sha256": digest,
                    "importance_retention": EXP34.retention(score, mask_np),
                    "abs_importance_retention": EXP34.retention(scores["abs"], mask_np),
                    "weighted_importance_retention": EXP34.retention(
                        scores["weighted"], mask_np
                    ),
                    "score_median_ms": score_median,
                    "score_p95_ms": float(np.quantile(score_samples[kind], 0.95)),
                    "selector_median_ms": selector_median,
                    "selector_p95_ms": float(np.quantile(selection_samples, 0.95)),
                    "selector_total_ms": score_median + selector_median,
                    "legacy_total_ms": selector_median + nonselector,
                    "actual_total_ms": score_median + selector_median + nonselector,
                    "deterministic": deterministic,
                    "fallback_used": bool(metadata.get("fallback_used", False)),
                    "error": metadata.get("error", ""), **measured,
                })
                for repetition, elapsed in enumerate(selection_samples):
                    selector_rows.append({
                        "case_key": case_key, "method": method,
                        "budget_index": budget_index, "repetition": repetition,
                        "runtime_ms": elapsed,
                    })
        print(
            f"  measured {model_key} {state['prompt']} {module_name}: "
            f"{len(METHODS)*len(args.budgets)} candidates, "
            f"{len(measured_masks)} masks",
            flush=True,
        )
        # Return temporary compact-weight and score allocations to the driver
        # after every projection so the desktop compositor keeps VRAM headroom.
        torch.cuda.empty_cache()
        if args.projection_throttle_ms:
            time.sleep(args.projection_throttle_ms / 1000.0)

    return hook


def run_model(model_key, selector, native_reader, prompts, args, vlmflash):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    spec = MODEL_SPECS[model_key]
    snapshot = EXP32.cached_snapshot(spec["repo"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float16, local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    weight_norms, norm_samples, norm_bytes = precompute_weight_norms(
        model, vlmflash, args.norm_chunk_rows
    )
    torch.cuda.empty_cache()
    sampled_layers = EXP32.layer_samples(
        int(model.config.num_hidden_layers), args.layer_samples
    )
    targets = []
    for name, module in model.named_modules():
        match = LAYER_RE.search(name)
        if (
            match and int(match.group(1)) in sampled_layers
            and re.search(vlmflash.DEFAULT_INCLUDE, name)
            and type(module) is torch.nn.Linear
        ):
            targets.append((name, module))
    if len(targets) != len(sampled_layers) * 7:
        raise RuntimeError(f"{model_key}: sampled projection count mismatch")

    candidate_rows, selector_rows, score_rows, io_rows, gemm_rows = (
        [], [], [], [], []
    )
    state = {"prompt": "", "split": "", "situation": "", "num_tokens": 0}
    hooks = [
        module.register_forward_hook(make_projection_hook(
            state, model_key, spec["label"], name, module, weight_norms[name],
            selector, native_reader, args, candidate_rows, selector_rows,
            score_rows, io_rows, gemm_rows,
        ))
        for name, module in targets
    ]
    try:
        with torch.inference_mode():
            for prompt in prompts:
                inputs = EXP32.tokenize(tokenizer, prompt["text"], args.max_input_tokens)
                state.update({
                    "prompt": prompt["name"], "split": prompt["split"],
                    "situation": prompt["situation"],
                    "num_tokens": int(inputs["input_ids"].numel()),
                })
                output = model(**inputs, use_cache=False)
                torch.cuda.synchronize()
                del output, inputs
                print(f"captured {model_key} {prompt['name']}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    metadata = {
        "model": model_key, "label": spec["label"], "repo": spec["repo"],
        "snapshot": snapshot.name, "model_type": model.config.model_type,
        "layers": int(model.config.num_hidden_layers),
        "sampled_layers": sampled_layers, "sampled_projection_count": len(targets),
        "shapes": sorted({f"{m.in_features}x{m.out_features}" for _, m in targets}),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "dtype": str(next(model.parameters()).dtype),
        "weight_norm_vectors": len(weight_norms),
        "weight_norm_metadata_bytes": int(norm_bytes),
        "weight_norm_precompute_ms": float(sum(norm_samples)),
    }
    return (
        model, tokenizer, weight_norms, candidate_rows, selector_rows, score_rows,
        io_rows, gemm_rows, metadata,
    )


class EndToEndPolicy:
    def __init__(self, selector, selection_type, weight_norm_sq, d, budgets, collector):
        self.selector = selector
        self.selection_type = selection_type
        self.weight_norm_sq = weight_norm_sq
        self.d = int(d)
        self.budgets = budgets
        self.collector = collector
        self.method = "paper"
        self.budget_index = 0
        self.prepared_score = None

    def configure(self, method: str, budget_index: int) -> None:
        self.method = method
        self.budget_index = int(budget_index)
        self.prepared_score = None

    def prepare(self, module, inputs) -> None:
        kind = METHOD_SCORE[self.method]
        if kind == "abs":
            self.prepared_score = None
            return
        self.prepared_score = make_score(inputs[0], self.weight_norm_sq, kind)

    def __call__(self, importance, _num_load_rows, _row_size_kib):
        n = int(importance.numel())
        score = importance if self.prepared_score is None else self.prepared_score
        fraction = self.budgets[self.budget_index]
        rows = max(1, min(n - 1, int(round(n * fraction))))
        mask, metadata = self.selector.select(
            METHOD_INTERNAL[self.method], score, rows, self.d
        )
        self.collector.append({
            "selected_fraction": float(mask.sum()) / n,
            "fallback": bool(metadata.get("fallback_used", False)),
        })
        return self.selection_type(mask, float(score[mask].sum()), None)


def run_end_to_end(model, tokenizer, model_key, weight_norms, selector,
                   prompts, args, vlmflash):
    dummy = lambda importance, num_load_rows, row_size_kib: vlmflash.Selection(
        torch.ones_like(importance, dtype=torch.bool), float(importance.sum()), None
    )
    handle = vlmflash.attach(
        model, policy=dummy, include=vlmflash.DEFAULT_INCLUDE, sparsity=0.5
    )
    collector, policies, hooks = [], [], []
    try:
        for name in handle.names:
            module = model.get_submodule(name)
            policy = EndToEndPolicy(
                selector, vlmflash.Selection, weight_norms[name],
                module.out_features, args.budgets, collector,
            )
            module.nc_policy = policy
            policies.append(policy)
            hooks.append(module.register_forward_pre_hook(policy.prepare))

        rows = []
        for prompt in prompts:
            if prompt["split"] != "holdout":
                continue
            inputs = EXP32.tokenize(tokenizer, prompt["text"], args.max_input_tokens)
            with torch.inference_mode():
                dense = model(**inputs, use_cache=False).logits.detach()
            for method in METHODS:
                for budget_index, fraction in enumerate(args.budgets):
                    collector.clear()
                    for policy in policies:
                        policy.configure(method, budget_index)
                    with torch.inference_mode(), vlmflash.enabled():
                        sparse = model(**inputs, use_cache=False).logits.detach()
                    metrics = EXP32.logits_error(dense, sparse, inputs["input_ids"])
                    rows.append({
                        "model": model_key, "prompt": prompt["name"],
                        "situation": prompt["situation"], "method": method,
                        "method_label": METHOD_LABELS[method],
                        "score_kind": METHOD_SCORE[method],
                        "budget_index": budget_index,
                        "budget_fraction": fraction, **metrics,
                        "projection_calls": len(collector),
                        "selected_fraction_mean": float(np.mean([
                            item["selected_fraction"] for item in collector
                        ])),
                        "fallback_calls": int(sum(
                            item["fallback"] for item in collector
                        )),
                    })
            print(f"  e2e {model_key} {prompt['name']} complete", flush=True)
            del dense, inputs
        return rows
    finally:
        for hook in hooks:
            hook.remove()
        handle.detach()


def _interpolate(curve: pd.DataFrame, column: str, error: float) -> float:
    points = (
        curve[["relative_l2_error", column]].dropna()
        .groupby("relative_l2_error", as_index=False)[column].min()
        .sort_values("relative_l2_error")
    )
    errors = points.relative_l2_error.to_numpy(dtype=float)
    values = points[column].to_numpy(dtype=float)
    if not len(errors) or error < errors[0] or error > errors[-1]:
        return float("nan")
    return float(np.interp(error, errors, values))


def _best_at_ceiling(curve: pd.DataFrame, error_column: str, ceiling: float,
                     time_column: str):
    allowed = curve[curve[error_column] <= ceiling]
    if not len(allowed):
        return None
    return allowed.sort_values(time_column).iloc[0]


def _pct_gain(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate) or baseline == 0:
        return float("nan")
    return 100.0 * (baseline - candidate) / baseline


def analyze(args: argparse.Namespace) -> None:
    output = args.output_dir
    candidates = pd.read_csv(output / "candidates.csv")
    holdout = candidates[candidates.split == "holdout"]
    fixed = holdout.groupby(
        ["method", "method_label", "score_kind", "budget_index", "budget_fraction"],
        as_index=False,
    ).agg(
        cases=("case_key", "size"),
        actual_total_ms=("actual_total_ms", "mean"),
        legacy_total_ms=("legacy_total_ms", "mean"),
        score_median_ms=("score_median_ms", "mean"),
        selector_median_ms=("selector_median_ms", "mean"),
        selector_total_ms=("selector_total_ms", "mean"),
        read_wall_median_ms=("read_wall_median_ms", "mean"),
        io_median_ms=("io_median_ms", "mean"),
        upload_median_ms=("upload_median_ms", "mean"),
        gather_median_ms=("gather_median_ms", "mean"),
        gemm_median_ms=("gemm_median_ms", "mean"),
        relative_l2_error=("relative_l2_error", "mean"),
        cosine_error=("cosine_error", "mean"),
        selected_fraction=("selected_fraction", "mean"),
        chunks=("chunks", "mean"),
        row_match_rate=("row_match", "mean"),
        fallback_rate=("fallback_used", "mean"),
    )
    fixed.to_csv(output / "fixed_frontiers.csv", index=False)

    columns = (
        "actual_total_ms", "legacy_total_ms", *COMPONENTS,
        "selector_total_ms", "io_median_ms", "upload_median_ms",
        "selected_fraction", "chunks", "row_match_rate",
    )
    rows = []
    for ceiling in ERROR_CEILINGS:
        for method in METHODS:
            curve = fixed[fixed.method == method]
            row = {
                "error_ceiling": ceiling, "method": method,
                "method_label": METHOD_LABELS[method],
            }
            for column in columns:
                row[column] = _interpolate(curve, column, ceiling)
            rows.append(row)
    interpolated = pd.DataFrame(rows)
    interpolated.to_csv(output / "same_error_components.csv", index=False)

    comparisons = []
    for ceiling in ERROR_CEILINGS:
        group = interpolated[interpolated.error_ceiling == ceiling].set_index("method")
        baseline = group.loc["cell1_abs", "actual_total_ms"]
        row = {"error_ceiling": ceiling}
        for method in METHODS:
            value = group.loc[method, "actual_total_ms"]
            row[f"{method}_total_ms"] = value
            row[f"{method}_gain_vs_cell1_abs_pct"] = _pct_gain(baseline, value)
        comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "same_error_comparison.csv", index=False)

    per_model_rows = []
    for model in sorted(holdout.model.unique()):
        model_fixed = holdout[holdout.model == model].groupby(
            ["method", "budget_index", "budget_fraction"], as_index=False
        ).agg(
            actual_total_ms=("actual_total_ms", "mean"),
            relative_l2_error=("relative_l2_error", "mean"),
        )
        for ceiling in (0.42, 0.25, 0.15):
            for method in METHODS:
                best = _best_at_ceiling(
                    model_fixed[model_fixed.method == method],
                    "relative_l2_error", ceiling, "actual_total_ms",
                )
                per_model_rows.append({
                    "model": model, "error_ceiling": ceiling,
                    "method": method, "feasible": best is not None,
                    "actual_total_ms": (
                        float(best.actual_total_ms) if best is not None else np.nan
                    ),
                    "achieved_error": (
                        float(best.relative_l2_error) if best is not None else np.nan
                    ),
                    "budget_fraction": (
                        float(best.budget_fraction) if best is not None else np.nan
                    ),
                })
    per_model = pd.DataFrame(per_model_rows)
    per_model.to_csv(output / "per_model_quality_constrained.csv", index=False)

    e2e_summary = pd.DataFrame()
    e2e_comparison = pd.DataFrame()
    e2e_path = output / "end_to_end.csv"
    if e2e_path.is_file():
        e2e = pd.read_csv(e2e_path)
        e2e_summary = e2e.groupby(
            ["method", "method_label", "score_kind", "budget_index", "budget_fraction"],
            as_index=False,
        ).agg(
            dense_to_sparse_kl_mean=("dense_to_sparse_kl", "mean"),
            top1_agreement_mean=("top1_agreement", "mean"),
            nll_delta_mean=("nll_delta", "mean"),
            selected_fraction_mean=("selected_fraction_mean", "mean"),
            fallback_calls=("fallback_calls", "sum"),
        )
        latency = fixed.set_index(["method", "budget_index"]).actual_total_ms
        e2e_summary["actual_total_ms"] = [
            float(latency.loc[(row.method, int(row.budget_index))])
            for row in e2e_summary.itertuples(index=False)
        ]
        e2e_summary.to_csv(output / "end_to_end_summary.csv", index=False)
        rows = []
        for ceiling in KL_CEILINGS:
            for method in METHODS:
                best = _best_at_ceiling(
                    e2e_summary[e2e_summary.method == method],
                    "dense_to_sparse_kl_mean", ceiling, "actual_total_ms",
                )
                rows.append({
                    "kl_ceiling": ceiling, "method": method,
                    "method_label": METHOD_LABELS[method],
                    "feasible": best is not None,
                    "actual_total_ms": (
                        float(best.actual_total_ms) if best is not None else np.nan
                    ),
                    "achieved_kl": (
                        float(best.dense_to_sparse_kl_mean) if best is not None else np.nan
                    ),
                    "budget_fraction": (
                        float(best.budget_fraction) if best is not None else np.nan
                    ),
                })
        e2e_comparison = pd.DataFrame(rows)
        e2e_comparison.to_csv(output / "end_to_end_comparison.csv", index=False)

    _plot(output, fixed, interpolated, e2e_summary)
    write_report(args, candidates, fixed, interpolated, comparison, per_model,
                 e2e_comparison)


def _plot(output: Path, fixed: pd.DataFrame, interpolated: pd.DataFrame,
          e2e_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    for method in METHODS:
        group = fixed[fixed.method == method].sort_values("budget_fraction")
        axes[0].plot(
            group.actual_total_ms, group.relative_l2_error,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linewidth=1.8, markersize=4, label=METHOD_LABELS[method],
        )
    axes[0].set_xlabel("Measured latency including score construction (ms)")
    axes[0].set_ylabel("Projection relative L2 error")
    axes[0].set_title("Weight-aware projection frontier")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    if len(e2e_summary):
        for method in METHODS:
            group = e2e_summary[e2e_summary.method == method].sort_values(
                "budget_fraction"
            )
            axes[1].plot(
                group.actual_total_ms, group.dense_to_sparse_kl_mean,
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                linewidth=1.8, markersize=4, label=METHOD_LABELS[method],
            )
        axes[1].set_xlabel("Measured latency including score construction (ms)")
        axes[1].set_ylabel("End-to-end logit KL")
        axes[1].set_title("Weight-aware end-to-end frontier")
        axes[1].invert_yaxis()
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output / "weight_aware_frontiers.png", dpi=200)
    fig.savefig(output / "weight_aware_frontiers.pdf")
    plt.close(fig)

    common = interpolated.dropna(subset=["actual_total_ms"])
    counts = common.groupby("error_ceiling").method.nunique()
    common = common[common.error_ceiling.isin(counts[counts == len(METHODS)].index)]
    means = common.groupby("method", as_index=False)[list(COMPONENTS)].mean()
    means["order"] = means.method.map({m: i for i, m in enumerate(METHODS)})
    means = means.sort_values("order")
    means.to_csv(output / "same_error_component_means.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(means))
    bottoms = np.zeros(len(means))
    labels = {
        "score_median_ms": "Score construction",
        "selector_median_ms": "Structured selector",
        "read_wall_median_ms": "SSD read/upload wall",
        "gather_median_ms": "Activation gather",
        "gemm_median_ms": "Compact GEMM",
    }
    colors = {
        "score_median_ms": "#DB2777", "selector_median_ms": "#7C3AED",
        "read_wall_median_ms": "#2563EB", "gather_median_ms": "#D97706",
        "gemm_median_ms": "#0F766E",
    }
    for column in COMPONENTS:
        values = means[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottoms, color=colors[column], label=labels[column])
        bottoms += values
    for index, total in enumerate(bottoms):
        ax.text(index, total + 0.012, f"{total:.3f} ms", ha="center", fontsize=9)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in means.method], rotation=8)
    ax.set_ylabel("Mean latency at common equal-error points (ms)")
    ax.set_title("Weight-aware scoring: latency decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "weight_aware_components.png", dpi=200)
    fig.savefig(output / "weight_aware_components.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for method in ("cell1_x2", "wcell1"):
        group = interpolated[interpolated.method == method].sort_values("error_ceiling")
        baseline = interpolated[interpolated.method == "cell1_abs"].set_index(
            "error_ceiling"
        )
        gains = [
            _pct_gain(baseline.loc[row.error_ceiling, "actual_total_ms"],
                      row.actual_total_ms)
            for row in group.itertuples(index=False)
        ]
        ax.plot(
            group.error_ceiling, gains, color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], label=METHOD_LABELS[method],
        )
    ax.axhline(0.0, color="#6B7280", linewidth=1)
    ax.set_xlabel("Projection error ceiling")
    ax.set_ylabel("Latency gain vs Cell-1 |x| (%)")
    ax.set_title("Does the error proxy move the frontier?")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "weight_aware_gain.png", dpi=200)
    fig.savefig(output / "weight_aware_gain.pdf")
    plt.close(fig)


def _fmt(value: float, suffix: str = "") -> str:
    return "—" if not np.isfinite(value) else f"{value:.3f}{suffix}"


def write_report(args, candidates, fixed, interpolated, comparison, per_model,
                 e2e) -> None:
    lines = [
        "# Experiment 39 보고서: weight-aware Cell-1", "",
        "기존 Cell-1의 `mean(|X_i|)` 점수를 실제 projection squared-error의 "
        "대각 근사 `q_i = mean(X_i^2) * ||W[:,i]||_2^2`로 교체했다. "
        "`Cell-1 X²`를 추가해 activation-energy 변경과 weight norm 효과를 "
        "분리했다. 모든 latency에는 온라인 score 생성 시간이 포함되고, weight "
        "norm은 모델 로딩 때 한 번 사전 계산하므로 포함하지 않는다.", "",
        "## 동일 projection error 보간", "",
        "5% R grid 사이의 선형 보간 진단이다. 양수 gain은 기존 Cell-1보다 "
        "빠르다는 뜻이다.", "",
        "| error | Paper | Cell-1 | Cell-1 X² | gain | WCell-1 | gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.error_ceiling:.2f} | {_fmt(row.paper_total_ms, ' ms')} "
            f"| {_fmt(row.cell1_abs_total_ms, ' ms')} "
            f"| {_fmt(row.cell1_x2_total_ms, ' ms')} "
            f"| {_fmt(row.cell1_x2_gain_vs_cell1_abs_pct, '%')} "
            f"| {_fmt(row.wcell1_total_ms, ' ms')} "
            f"| {_fmt(row.wcell1_gain_vs_cell1_abs_pct, '%')} |"
        )

    common = interpolated.dropna(subset=["actual_total_ms"])
    counts = common.groupby("error_ceiling").method.nunique()
    common = common[common.error_ceiling.isin(counts[counts == len(METHODS)].index)]
    means = common.groupby("method")[["actual_total_ms", *COMPONENTS]].mean()
    lines.extend([
        "", "## 동일-error 구성요소 평균", "",
        "| method | score | selector | SSD/upload | gather | GEMM | total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        if method not in means.index:
            lines.append(f"| {METHOD_LABELS[method]} | — | — | — | — | — | — |")
            continue
        row = means.loc[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row.score_median_ms:.4f} ms "
            f"| {row.selector_median_ms:.4f} ms "
            f"| {row.read_wall_median_ms:.4f} ms "
            f"| {row.gather_median_ms:.4f} ms "
            f"| {row.gemm_median_ms:.4f} ms "
            f"| {row.actual_total_ms:.4f} ms |"
        )

    lines.extend([
        "", "## 모델별 실측-grid 최적점", "",
        "| model | error | Cell-1 | Cell-1 X² | WCell-1 |",
        "|---|---:|---:|---:|---:|",
    ])
    if len(per_model):
        for (model, ceiling), group in per_model.groupby(["model", "error_ceiling"]):
            indexed = group.set_index("method")
            cells = []
            for method in ("cell1_abs", "cell1_x2", "wcell1"):
                row = indexed.loc[method]
                cells.append(
                    "—" if not row.feasible else
                    f"{row.actual_total_ms:.3f} ms (R {100*row.budget_fraction:.0f}%)"
                )
            lines.append(
                f"| {model} | {ceiling:.2f} | " + " | ".join(cells) + " |"
            )

    if len(e2e):
        lines.extend([
            "", "## End-to-end logit KL", "",
            "| KL ceiling | Paper | Cell-1 | Cell-1 X² | WCell-1 |",
            "|---:|---:|---:|---:|---:|",
        ])
        for ceiling in KL_CEILINGS:
            group = e2e[e2e.kl_ceiling == ceiling].set_index("method")
            cells = []
            for method in METHODS:
                row = group.loc[method]
                cells.append(
                    "—" if not row.feasible else
                    f"{row.actual_total_ms:.3f} ms (KL {row.achieved_kl:.3f})"
                )
            lines.append(f"| {ceiling:g} | " + " | ".join(cells) + " |")

    weighted_gains = comparison.wcell1_gain_vs_cell1_abs_pct.dropna()
    x2_gains = comparison.cell1_x2_gain_vs_cell1_abs_pct.dropna()
    mask_table = candidates.pivot(
        index=["case_key", "budget_index"], columns="method",
        values=["mask_sha256", "relative_l2_error"],
    )
    weighted_same_mask = (
        mask_table["mask_sha256"]["cell1_x2"]
        == mask_table["mask_sha256"]["wcell1"]
    )
    weighted_error_delta = (
        mask_table["relative_l2_error"]["wcell1"]
        - mask_table["relative_l2_error"]["cell1_x2"]
    )
    winner_counts = {}
    for ceiling, group in interpolated.groupby("error_ceiling"):
        valid = group.dropna(subset=["actual_total_ms"])
        if len(valid):
            winner = valid.sort_values("actual_total_ms").iloc[0].method
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
    lines.extend([
        "", "## 판정", "",
        "**projection frontier에서는 weight norm 가설이 지지되지 않았다.** "
        f"WCell-1의 기존 Cell-1 대비 동일-error gain 범위는 "
        f"`{weighted_gains.min():.2f}%..{weighted_gains.max():.2f}%`, 평균 "
        f"`{weighted_gains.mean():.2f}%`다. X²-only gain 평균은 "
        f"`{x2_gains.mean():.2f}%`다. ceiling별 전체 winner 횟수는 "
        f"`{winner_counts}`다.", "",
        f"W²를 곱하면 X²-only와 다른 mask를 "
        f"`{100 * (1 - weighted_same_mask.mean()):.1f}%` 선택했지만, 같은 R에서 "
        f"projection error 변화는 평균 `{weighted_error_delta.mean():+.5f}`에 "
        "그쳤다. 즉 weight norm의 영향이 너무 작아서가 아니라, mask 순서를 "
        "상당히 바꾸고도 실제 오차를 거의 줄이지 못했다. 대각 점수는 "
        "neuron 간 Gram off-diagonal 항, 부호와 cancellation을 무시하며 Cell-1의 "
        "s-row 집계가 개별 row 점수의 이득도 평균화한다.", "",
        "End-to-end에서는 WCell-1이 엄격한 KL ceiling 6/2/1에서 가장 빨랐지만, "
        "R 간격이 5%이고 holdout prompt가 모델당 3개라 grid 선택 효과일 수 있다. "
        "따라서 현재의 안정적인 후속 기본값은 **Cell-1 X²**이며, WCell-1의 "
        "end-to-end 이득은 더 촘촘한 R과 추가 prompt로 재검증해야 한다.", "",
        "## 측정 범위", "",
        f"- projection 후보 `{len(candidates)}`개.",
        f"- non-empty O_DIRECT fallback "
        f"`{int(((candidates.selected_rows > 0) & (candidates.direct_rate < 1)).sum())}`건.",
        "- 모델 3개, prompt 6개, 모델당 표본 layer 3개, projection 7종.",
        "- R=10%..95%, 5% 간격; actual total은 score + selector + "
        "O_DIRECT/upload wall + gather + compact GEMM이다.",
        "- weight norm metadata는 GPU에 상주하며 사전 계산 시간은 online "
        "latency에서 제외했다.",
        "- laptop-safe 실행은 모델당 별도 프로세스, CUDA allocator 55%, "
        "128-output-row FP32 norm slab, CPU/O_DIRECT thread 2개, projection 사이 "
        "25 ms 양보를 사용했다.",
        "", f"![Frontiers]({args.output_dir.name}/weight_aware_frontiers.png)", "",
        f"![Components]({args.output_dir.name}/weight_aware_components.png)", "",
        f"![Gain]({args.output_dir.name}/weight_aware_gain.png)", "",
    ])
    report = args.report_output or HERE / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return

    if args.process_nice:
        os.nice(args.process_nice)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)

    prompts = list(PROMPTS[:args.prompt_limit] if args.prompt_limit else PROMPTS)
    lookup = EXP32.BASE.LatencyTable.load(args.profile)
    selector = EXP37.SuperTileSelector(lookup, args.saturation_kib)
    vlmflash = EXP32.EXP22.load_vlmflash()
    from vlmflash._native import native, unavailable_reason
    native_reader = native()
    if native_reader is None:
        raise SystemExit(f"native reader unavailable: {unavailable_reason()}")

    from transformers import AutoConfig
    max_required = 0
    for key in args.models:
        config = AutoConfig.from_pretrained(
            EXP32.cached_snapshot(MODEL_SPECS[key]["repo"]), local_files_only=True
        )
        max_required = max(
            max_required,
            int(config.intermediate_size) * int(config.hidden_size) * 2,
        )
    if args.io_blob.stat().st_size < max_required:
        raise SystemExit(f"I/O blob needs at least {max_required} bytes")

    all_candidates, all_selectors, all_scores, all_io, all_gemm = [], [], [], [], []
    all_e2e, models = [], []
    for model_key in args.models:
        print(f"loading and measuring {MODEL_SPECS[model_key]['label']}", flush=True)
        result = run_model(
            model_key, selector, native_reader, prompts, args, vlmflash
        )
        (
            model, tokenizer, weight_norms, candidates, selectors, scores,
            io_rows, gemm_rows, metadata,
        ) = result
        all_candidates.extend(candidates)
        all_selectors.extend(selectors)
        all_scores.extend(scores)
        all_io.extend(io_rows)
        all_gemm.extend(gemm_rows)
        models.append(metadata)
        pd.DataFrame(all_candidates).to_csv(
            args.output_dir / "candidates.partial.csv", index=False
        )
        if not args.skip_end_to_end and any(
            prompt["split"] == "holdout" for prompt in prompts
        ):
            all_e2e.extend(run_end_to_end(
                model, tokenizer, model_key, weight_norms, selector,
                prompts, args, vlmflash,
            ))
        del model, tokenizer, weight_norms
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(all_candidates).to_csv(args.output_dir / "candidates.csv", index=False)
    pd.DataFrame(all_selectors).to_csv(
        args.output_dir / "selector_samples.csv", index=False
    )
    pd.DataFrame(all_scores).to_csv(args.output_dir / "score_samples.csv", index=False)
    pd.DataFrame(all_io).to_csv(args.output_dir / "io_aggregates.csv", index=False)
    pd.DataFrame(all_gemm).to_csv(args.output_dir / "gemm_aggregates.csv", index=False)
    if all_e2e:
        pd.DataFrame(all_e2e).to_csv(args.output_dir / "end_to_end.csv", index=False)
    partial = args.output_dir / "candidates.partial.csv"
    if partial.exists():
        partial.unlink()

    profile_meta = lookup.meta
    metadata = {
        "format": "experiment-39-weight-aware-cell1-v1",
        "score_definitions": {
            "abs": "mean_t(abs(X_ti))",
            "x2": "mean_t(X_ti^2)",
            "weighted": "mean_t(X_ti^2) * sum_o(W_oi^2)",
        },
        "weight_norm_precompute": (
            "once at model load in bounded GPU row slabs; excluded from online latency"
        ),
        "models": models,
        "prompts": [
            {key: value for key, value in prompt.items() if key != "text"} | {
                "sha256": EXP32.sha256_text(prompt["text"])
            }
            for prompt in prompts
        ],
        "methods": list(METHODS), "budgets": args.budgets,
        "saturation_kib": args.saturation_kib,
        "score_repetitions": args.score_repetitions,
        "score_warmup": args.score_warmup,
        "selector_repetitions": args.selector_repetitions,
        "io_repetitions": args.io_repetitions, "io_warmup": args.io_warmup,
        "gemm_repetitions": args.gemm_repetitions,
        "gemm_warmup": args.gemm_warmup,
        "io_threads": args.io_threads, "io_max_read_kib": args.io_max_read_kib,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "cpu_threads": args.cpu_threads, "process_nice": args.process_nice,
        "norm_chunk_rows": args.norm_chunk_rows,
        "projection_throttle_ms": args.projection_throttle_ms,
        "profile": str(args.profile), "profile_metadata": profile_meta,
        "gpu": torch.cuda.get_device_name(0),
        "flash": profile_meta.get("flash", "unknown"),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    analyze(args)


if __name__ == "__main__":
    main()
