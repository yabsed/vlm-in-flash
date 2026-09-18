#!/usr/bin/env python3
"""Merge per-model Experiment 43 runs and regenerate joint analysis."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_experiment.py"
TABLES = (
    "candidates.csv", "selector_samples.csv", "score_samples.csv",
    "io_aggregates.csv", "gemm_aggregates.csv",
)


def load_runner():
    spec = importlib.util.spec_from_file_location("experiment_43_merge", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for directory in args.inputs:
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise SystemExit(f"missing metadata: {metadata_path}")
        run = json.loads(metadata_path.read_text())
        if len(run.get("models", [])) != 1:
            raise SystemExit(f"expected one model in {metadata_path}")
        metadata.append(run)
    model_keys = [run["models"][0]["model"] for run in metadata]
    if len(model_keys) != len(set(model_keys)):
        raise SystemExit(f"duplicate model inputs: {model_keys}")
    compatibility = (
        "budgets", "profile", "saturation_kib", "offset_count",
        "boundary_cells", "refine_divisor",
    )
    for key in compatibility:
        values = [run.get(key) for run in metadata]
        if any(value != values[0] for value in values[1:]):
            raise SystemExit(f"incompatible {key}: {values}")

    for table in TABLES:
        paths = [directory / table for directory in args.inputs]
        frames = [pd.read_csv(path) for path in paths]
        pd.concat(frames, ignore_index=True).to_csv(
            args.output_dir / table, index=False
        )

    base = dict(metadata[0])
    base["format"] = "experiment-43-adaptive-shifted-cell1-merged-v1"
    base["models"] = [item for run in metadata for item in run["models"]]
    base["merged_inputs"] = [str(path) for path in args.inputs]
    (args.output_dir / "metadata.json").write_text(
        json.dumps(base, indent=2) + "\n"
    )

    runner = load_runner()
    runner.analyze(SimpleNamespace(
        output_dir=args.output_dir,
        report_output=args.report_output,
        offset_count=base["offset_count"],
        boundary_cells=base["boundary_cells"],
        refine_divisor=base["refine_divisor"],
    ))


if __name__ == "__main__":
    main()
