#!/usr/bin/env python3
"""Merge safe per-model Experiment 39 runs and regenerate the joint report."""

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
    "io_aggregates.csv", "gemm_aggregates.csv", "end_to_end.csv",
)


def load_runner():
    spec = importlib.util.spec_from_file_location("experiment_39_merge", RUNNER)
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
        if not directory.is_dir():
            raise SystemExit(f"missing input directory: {directory}")
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise SystemExit(f"missing metadata: {metadata_path}")
        run_metadata = json.loads(metadata_path.read_text())
        if len(run_metadata.get("models", [])) != 1:
            raise SystemExit(f"expected one model in {metadata_path}")
        metadata.append(run_metadata)
    model_keys = [run["models"][0]["model"] for run in metadata]
    if len(model_keys) != len(set(model_keys)):
        raise SystemExit(f"duplicate per-model inputs: {model_keys}")
    for key in ("budgets", "profile", "saturation_kib"):
        values = [run.get(key) for run in metadata]
        if any(value != values[0] for value in values[1:]):
            raise SystemExit(f"incompatible {key} across inputs: {values}")
    for table in TABLES:
        paths = [directory / table for directory in args.inputs]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit(f"missing {table}: {missing}")
        frames = [pd.read_csv(path) for path in paths]
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / table, index=False)

    base = dict(metadata[0])
    base["format"] = "experiment-39-weight-aware-cell1-merged-v1"
    base["models"] = [item for run in metadata for item in run["models"]]
    base["merged_inputs"] = [str(path) for path in args.inputs]
    (args.output_dir / "metadata.json").write_text(
        json.dumps(base, indent=2) + "\n"
    )

    runner = load_runner()
    runner.analyze(SimpleNamespace(
        output_dir=args.output_dir,
        report_output=args.report_output,
    ))


if __name__ == "__main__":
    main()
