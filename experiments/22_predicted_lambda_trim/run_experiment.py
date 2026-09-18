#!/usr/bin/env python3
"""Experiment 22: compare selectors on importance captured from a real LM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import sys
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
VLMFLASH_ROOT = PROJECT_ROOT / "preliminary_research" / "vlm-flash"
VLMFLASH_SRC = VLMFLASH_ROOT / "src"
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

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPTS = (
    "Explain why contiguous flash reads can be faster than scattered reads.",
    "Summarize the trade-off between neural-network sparsity and accuracy.",
    "Write a short Python function that computes a moving average.",
)
TRACE_FORMAT = "experiment-22-vlm-activation-traces-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+", default=None,
        help="optional subset of captured Table-2 shapes, written as NxD",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt", action="append", default=None,
        help="text prompt to trace; repeat the flag for multiple prompts",
    )
    parser.add_argument(
        "--prompt-file", type=Path,
        help="UTF-8 text file containing one non-empty prompt per line",
    )
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="device used only for the untimed activation-trace forward",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"),
        default="auto", help="model dtype used for activation tracing",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="do not download the model or tokenizer",
    )
    parser.add_argument(
        "--trace-input", type=Path,
        help="reuse a previously captured activation_traces.npz instead of loading a model",
    )
    parser.add_argument(
        "--trace-output", type=Path,
        help="where to save newly captured traces (default: OUTPUT_DIR/activation_traces.npz)",
    )
    parser.add_argument(
        "--max-traces-per-shape", type=int, default=32,
        help="deterministic cap after real activation capture; 0 keeps every trace",
    )
    parser.add_argument(
        "--collect-traces-only", action="store_true",
        help="capture/save real model traces and stop before selector benchmarking",
    )
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
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[str]:
    by_name = {item["shape"]: item for item in SHAPES}
    unknown = sorted(set(args.shapes or ()) - set(by_name))
    if unknown:
        raise SystemExit(f"unknown --shapes: {unknown}; available: {sorted(by_name)}")
    if args.shapes and len(args.shapes) != len(set(args.shapes)):
        raise SystemExit("--shapes must not contain duplicates")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.max_input_tokens < 1:
        raise SystemExit("--max-input-tokens must be positive")
    if args.max_traces_per_shape < 0:
        raise SystemExit("--max-traces-per-shape must be nonnegative")
    if args.prompt and args.prompt_file:
        raise SystemExit("use either --prompt or --prompt-file, not both")
    if args.trace_input and (args.prompt or args.prompt_file):
        raise SystemExit("--trace-input cannot be combined with prompt options")
    if args.collect_traces_only and args.analyze_only:
        raise SystemExit("--collect-traces-only and --analyze-only are mutually exclusive")
    if not args.row_budget_fractions or any(
        not 0.0 < value <= 1.0 for value in args.row_budget_fractions
    ):
        raise SystemExit("--row-budget-fractions must lie in (0, 1]")
    if args.saturation_kib <= 0 or args.deadline_ms <= 0:
        raise SystemExit("saturation and deadline must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is unavailable")
    tracks = list(dict.fromkeys(args.tracks))
    if "cuda" in tracks and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; omitting the cuda-resident round-trip track")
        tracks.remove("cuda")
    if not tracks:
        raise SystemExit("no runnable timing track remains")
    return tracks


def load_vlmflash():
    """Import the checked-out implementation rather than an unrelated install."""
    if not (VLMFLASH_SRC / "vlmflash" / "__init__.py").is_file():
        raise SystemExit(
            f"vlm-flash submodule is missing at {VLMFLASH_ROOT}; "
            "run `git submodule update --init --recursive`"
        )
    source = str(VLMFLASH_SRC)
    if source not in sys.path:
        sys.path.insert(0, source)
    import vlmflash
    return vlmflash


def read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = [
            line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.prompt:
        prompts = [prompt.strip() for prompt in args.prompt if prompt.strip()]
    else:
        prompts = list(DEFAULT_PROMPTS)
    if not prompts:
        raise SystemExit("activation tracing needs at least one non-empty prompt")
    return prompts


def resolve_trace_device(args: argparse.Namespace) -> str:
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def load_language_model(args: argparse.Namespace, device: str):
    """Load one real causal LM without device-map hooks that attach() cannot wrap."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exception:
        raise SystemExit(
            "transformers is required to capture real activation traces; "
            "install preliminary_research/vlm-flash first"
        ) from exception

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    common = {"local_files_only": bool(args.local_files_only)}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **common)
    except TypeError:
        # Compatibility with transformers releases predating the dtype rename.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, **common
        )
    return model.to(device).eval(), tokenizer


class ActivationCapturePolicy:
    """VLMFlash policy that records real importance and keeps the forward dense."""

    def __init__(self, module: str, n: int, d: int, sink: list[dict],
                 state: dict, selection_type):
        self.module = module
        self.n = int(n)
        self.d = int(d)
        self.sink = sink
        self.state = state
        self.selection_type = selection_type

    def __call__(self, importance: torch.Tensor, _num_load_rows: int,
                 _row_size_kib: float):
        call_index = self.state["call_counts"].get(self.module, 0)
        self.state["call_counts"][self.module] = call_index + 1
        values = importance.detach().to("cpu", torch.float32).numpy().copy()
        self.sink.append({
            "trace_id": len(self.sink),
            "input_index": int(self.state["input_index"]),
            "prompt_sha256": self.state["prompt_sha256"],
            "num_tokens": int(self.state["num_tokens"]),
            "module": self.module,
            "call_index": int(call_index),
            "shape": f"{self.n}x{self.d}",
            "n": self.n,
            "d": self.d,
            "values": values,
        })
        # A full mask makes the attached projection exactly dense. Therefore
        # every later layer is traced from the unperturbed model, not from one
        # of the competing selectors.
        mask = torch.ones_like(importance, dtype=torch.bool)
        return self.selection_type(mask, float(values.astype(np.float64).sum()), 0.0)


def tokenize_prompt(tokenizer, prompt: str, max_input_tokens: int, device: str) -> dict:
    text = prompt
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_input_tokens
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def capture_activation_traces(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Run a real LM and capture VLMFlash's per-projection importance vectors."""
    vlmflash = load_vlmflash()
    prompts = read_prompts(args)
    device = resolve_trace_device(args)
    print(f"loading {args.model} on {device} for real activation tracing", flush=True)
    model, tokenizer = load_language_model(args, device)
    traces: list[dict] = []
    state = {
        "input_index": -1, "prompt_sha256": "", "num_tokens": 0,
        "call_counts": {},
    }

    # attach() initially needs a policy. It is replaced by a named capture
    # policy on every wrapped module before the first forward.
    placeholder = ActivationCapturePolicy(
        "<unbound>", 1, 1, traces, state, vlmflash.Selection
    )
    handle = vlmflash.attach(
        model, policy=placeholder, include=vlmflash.DEFAULT_INCLUDE, sparsity=0.5
    )
    for name in handle.names:
        module = model.get_submodule(name)
        module.nc_policy = ActivationCapturePolicy(
            name, module.in_features, module.out_features,
            traces, state, vlmflash.Selection,
        )

    try:
        with torch.inference_mode():
            for input_index, prompt in enumerate(prompts):
                inputs = tokenize_prompt(tokenizer, prompt, args.max_input_tokens, device)
                state.update({
                    "input_index": input_index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "num_tokens": int(inputs["input_ids"].numel()),
                    "call_counts": {},
                })
                with vlmflash.enabled():
                    model(**inputs, use_cache=False)
                if device == "cuda":
                    torch.cuda.synchronize()
                print(
                    f"traced prompt {input_index + 1}/{len(prompts)}: "
                    f"{state['num_tokens']} tokens, {len(traces)} total projection calls",
                    flush=True,
                )
    finally:
        handle.detach()

    if not traces:
        raise RuntimeError("the real model forward produced no VLMFlash projection traces")
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    metadata = {
        "format": TRACE_FORMAT,
        "model": args.model,
        "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "device": device,
        "model_dtype": dtype,
        "prompt_count": len(prompts),
        "prompt_sha256": [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts],
        "max_input_tokens": int(args.max_input_tokens),
        "projection_include": vlmflash.DEFAULT_INCLUDE,
        "importance": "mean(abs(projection_input), over batch and token axes)",
        "forward_mode": "dense via an all-true VLMFlash selection mask",
    }
    return traces, metadata


def save_activation_traces(path: Path, traces: list[dict], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    arrays = {}
    for index, trace in enumerate(traces):
        key = f"values_{index:06d}"
        entry = {name: value for name, value in trace.items() if name != "values"}
        entry["array"] = key
        entries.append(entry)
        arrays[key] = np.asarray(trace["values"], dtype=np.float32)
    manifest = {**metadata, "trace_count": len(entries), "traces": entries}
    np.savez_compressed(
        path,
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
        **arrays,
    )


def load_activation_traces(path: Path) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise SystemExit(f"activation trace file does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if manifest.get("format") != TRACE_FORMAT:
            raise SystemExit(
                f"unsupported trace format in {path}: {manifest.get('format')!r}"
            )
        traces = []
        for entry in manifest["traces"]:
            trace = {name: value for name, value in entry.items() if name != "array"}
            trace["values"] = np.asarray(archive[entry["array"]], dtype=np.float32).copy()
            traces.append(trace)
    if int(manifest.get("trace_count", -1)) != len(traces):
        raise SystemExit(f"trace count mismatch in {path}")
    metadata = {name: value for name, value in manifest.items() if name != "traces"}
    return traces, metadata


def select_activation_traces(traces: list[dict], requested_shapes: list[str] | None,
                             max_per_shape: int) -> tuple[list[dict], list[str]]:
    required = {
        "trace_id", "input_index", "prompt_sha256", "num_tokens", "module",
        "call_index", "shape", "n", "d", "values",
    }
    for index, trace in enumerate(traces):
        missing = sorted(required - set(trace))
        if missing:
            raise SystemExit(f"trace {index} is missing fields: {missing}")
        if trace["shape"] != f"{int(trace['n'])}x{int(trace['d'])}":
            raise SystemExit(f"trace {trace['trace_id']} has inconsistent shape metadata")
    trace_ids = [int(trace["trace_id"]) for trace in traces]
    if len(trace_ids) != len(set(trace_ids)):
        raise SystemExit("activation trace ids must be unique")

    table_shapes = {item["shape"] for item in SHAPES}
    captured_shapes = {trace["shape"] for trace in traces}
    wanted = set(requested_shapes) if requested_shapes else captured_shapes & table_shapes
    unavailable = sorted(wanted - captured_shapes)
    if unavailable:
        raise SystemExit(
            f"requested shapes were not produced by the model: {unavailable}; "
            f"captured: {sorted(captured_shapes)}"
        )
    unsupported = sorted(wanted - table_shapes)
    if unsupported:
        raise SystemExit(
            f"shapes lack the paper's Table-2 chunk parameters: {unsupported}"
        )
    skipped_unsupported = sorted(captured_shapes - table_shapes)

    selected = []
    for shape in sorted(wanted):
        candidates = [trace for trace in traces if trace["shape"] == shape]
        if max_per_shape and len(candidates) > max_per_shape:
            indices = np.linspace(0, len(candidates) - 1, max_per_shape, dtype=int)
            candidates = [candidates[index] for index in indices]
        selected.extend(candidates)
    selected.sort(key=lambda trace: int(trace["trace_id"]))
    if not selected:
        raise SystemExit(
            "the model produced no supported Table-2 projection shape; "
            f"captured: {sorted(captured_shapes)}"
        )
    return selected, skipped_unsupported


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def collect(args: argparse.Namespace, traces: list[dict], tracks: list[str],
            lookup_table) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    timing_rows: list[dict] = []
    model_rows: list[dict] = []
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
        model_rows.append({
            **shape, "shape_index": shape_index, "row_size_kib": row_kib,
            "activation_trace_count": sum(t["shape"] == shape_name for t in traces),
            **model,
        })
        shape_context[shape_name] = (shape_index, shape, row_kib, params, model)

    for trace_index, trace in enumerate(traces):
        shape_index, shape, row_kib, params, model = shape_context[trace["shape"]]
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
        # Both methods see precisely the same float32-representable, normalized
        # scores captured by VLMFlash during the dense model forward.
        values32 = raw_values.copy()
        values32 /= np.float32(total)
        values = values32.astype(np.float64)
        activation_cv = float(
            raw_values.astype(np.float64).std() / raw_values.astype(np.float64).mean()
        )
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
                    "trace_id": int(trace["trace_id"]),
                    "input_index": int(trace["input_index"]),
                    "prompt_sha256": trace["prompt_sha256"],
                    "num_tokens": int(trace["num_tokens"]),
                    "module": trace["module"],
                    "module_call_index": int(trace["call_index"]),
                    "activation_cv": activation_cv,
                    "shape": shape["shape"], "shape_index": shape_index,
                    "n": n, "d": int(shape["d"]), "row_size_kib": row_kib,
                    "paper_start_kib": float(shape["start_kib"]),
                    "paper_jump_kib": float(shape["jump_kib"]),
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
            f"trace={trace_index + 1}/{len(traces)} shape={shape['shape']} "
            f"module={trace['module']} prompt={int(trace['input_index']) + 1}",
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
        "trace_id", "budget_index", "track",
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
        "format": "experiment-22-predicted-lambda-trim-v2-real-activations",
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
            "excluded": (
                "the real LM forward used to produce activation importance, "
                "actual storage I/O, and model compute"
            ),
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
            "activation_trace_count": int(frame["trace_id"].nunique()),
            "input_count": int(frame["input_index"].nunique()),
            "model": args.model,
            "trace_input": str(args.trace_input) if args.trace_input else None,
            "max_traces_per_shape": int(args.max_traces_per_shape),
            "repetitions": int(args.repetitions),
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
    metadata_path = args.output_dir / "metadata.json"
    run_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    trace_metadata = run_metadata.get("activation_trace", {})
    traced_model = trace_metadata.get("model", args.model)
    trace_count = int(run_metadata.get("selected_trace_count", 0))
    input_count = int(trace_metadata.get("prompt_count", 0))
    lines = [
        "# Experiment 22 보고서: Paper vs Predicted-lambda TD-2L(8) + Endpoint Trim",
        "",
        "Paper는 논문의 고정 row-budget 알고리즘을 그대로 실행했다. 각 paired input에서 "
        "Paper가 달성한 importance를 `Q = I(M_paper)`로 두고 proposed method가 최소한 같은 "
        "importance를 유지하도록 했다. 따라서 아래 lookup 비교는 importance-matched 비교다.",
        "",
        f"importance 입력은 `{traced_model}`의 dense forward에서 VLMFlash와 같은 정의인 "
        "`mean(abs(projection input))`로 직접 수집했다. "
        f"{input_count}개 실제 prompt에서 수집한 projection call 중 {trace_count}개 trace, "
        f"{shape_summary['shape'].nunique()}개 Table-2 shape를 평가했다. 모델 forward는 경쟁 "
        "selector가 뒤 레이어 activation을 오염시키지 않도록 all-true mask로 실행했다. "
        f"lookup latency는 실제 NVMe 측정이 아니라 `{args.profile}` profile의 예측값이다.",
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

    default_output = (HERE / "results").resolve()
    report_path = (
        HERE / "report.md" if args.output_dir.resolve() == default_output
        else args.output_dir / "report.md"
    )
    plot_prefix = "results/" if report_path.parent == HERE else ""
    lines.extend([
        "## 측정 범위와 한계", "",
        "- `host`: host float32 importance에서 CPU bool mask까지 측정했다. CUDA가 있으면 Paper native 구현의 GPU sort도 포함된다.",
        "- `cuda`: 미리 만들어 둔 CUDA float32 importance에서 시작해 CPU proposal의 D2H, float64 변환, 8회 이하 DP, mask 복원, trim, H2D 및 synchronization을 모두 포함했다.",
        "- importance는 실제 모델 activation이지만, trace를 만드는 dense LM forward는 selector timing에서 제외했다. 따라서 이 수치는 end-to-end token latency가 아니라 online selector latency다.",
        "- 현재 결과는 Qwen2.5-0.5B와 짧은 text prompt 3개에 한정된다. 여러 모델, 실제 VLM frame-append workload, 대표 데이터셋으로의 일반화는 아직 검증하지 않았다.",
        "- 실제 NVMe I/O와 sparse model accuracy/perplexity는 이 실험의 측정 범위가 아니다.",
        "- lookup latency는 Orin AGX profile 기반이므로 RTX 3050 노트북 selector 시간과 서로 다른 축이다. Jetson의 2 ms 충족 여부는 Jetson에서 다시 측정해야 한다.",
        "- endpoint trim은 two-line 목적을 단조 감소시키지만 측정 잡음이 있는 lookup table의 각 개별 점까지 단조 감소한다고 보장하지는 않는다.",
        "",
        f"![Overall runtime-quality]({plot_prefix}runtime_quality.png)", "",
        f"![Shape comparison]({plot_prefix}shape_comparison.png)", "",
    ])
    report_path.write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.output_dir / "trials.csv")
    if "trace_id" not in frame:
        raise SystemExit(
            "these are Experiment 22's retired synthetic results; rerun the experiment "
            "to create real-model activation results before using --analyze-only"
        )
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
    tracks = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args)
        return

    if args.trace_input:
        traces, trace_metadata = load_activation_traces(args.trace_input)
        trace_path = args.trace_input
        args.model = trace_metadata.get("model", args.model)
        print(
            f"loaded {len(traces)} real activation traces from {trace_path}",
            flush=True,
        )
    else:
        traces, trace_metadata = capture_activation_traces(args)
        trace_path = args.trace_output or (args.output_dir / "activation_traces.npz")
        save_activation_traces(trace_path, traces, trace_metadata)
        print(f"wrote {len(traces)} real activation traces to {trace_path}", flush=True)

    trace_metadata = {**trace_metadata, "trace_count": len(traces)}

    selected_traces, skipped_shapes = select_activation_traces(
        traces, args.shapes, args.max_traces_per_shape
    )
    selected_shape_names = list(dict.fromkeys(t["shape"] for t in selected_traces))
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
        "trace_sha256": sha256_file(trace_path),
        "captured_trace_count": len(traces),
        "selected_trace_count": len(selected_traces),
        "selected_trace_ids": [int(trace["trace_id"]) for trace in selected_traces],
        "max_traces_per_shape": int(args.max_traces_per_shape),
        "skipped_non_table2_shapes": skipped_shapes,
        "shape_specs": shape_specs,
        "method_labels": METHOD_LABELS,
        "requested_tracks": args.tracks,
        "executed_tracks": tracks,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.collect_traces_only:
        print(
            f"selected {len(selected_traces)} traces across "
            f"{len(selected_shape_names)} Table-2 shapes; stopping as requested"
        )
        return

    lookup_table = BASE.LatencyTable.load(args.profile)
    if not args.skip_self_check:
        self_check(lookup_table, args.saturation_kib)
    rows, timing_rows, model_rows = collect(
        args, selected_traces, tracks, lookup_table
    )
    write_csv(args.output_dir / "trials.csv", rows)
    write_csv(args.output_dir / "timing_samples.csv", timing_rows)
    write_csv(args.output_dir / "models.csv", model_rows)
    analyze(args)


if __name__ == "__main__":
    main()
