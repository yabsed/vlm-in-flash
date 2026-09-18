#!/usr/bin/env python3
"""Merge per-model Experiment 45 runs and regenerate the joint report."""

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
    spec = importlib.util.spec_from_file_location("experiment_45_merge", RUNNER)
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
    metadata = [json.loads((path / "metadata.json").read_text()) for path in args.inputs]
    models = [run["models"][0]["model"] for run in metadata]
    if len(models) != len(set(models)):
        raise SystemExit(f"duplicate model inputs: {models}")
    for key in ("controls", "profile", "saturation_kib"):
        values = [run.get(key) for run in metadata]
        if any(value != values[0] for value in values[1:]):
            raise SystemExit(f"incompatible {key}: {values}")
    for table in TABLES:
        pd.concat(
            [pd.read_csv(path / table) for path in args.inputs], ignore_index=True
        ).to_csv(args.output_dir / table, index=False)
    merged = dict(metadata[0])
    merged["format"] = "experiment-45-local-importance-bound-merged-v1"
    merged["models"] = [item for run in metadata for item in run["models"]]
    merged["merged_inputs"] = [str(path) for path in args.inputs]
    (args.output_dir / "metadata.json").write_text(
        json.dumps(merged, indent=2) + "\n"
    )
    runner = load_runner()
    runner.analyze(SimpleNamespace(
        output_dir=args.output_dir, report_output=args.report_output,
    ))


if __name__ == "__main__":
    main()
