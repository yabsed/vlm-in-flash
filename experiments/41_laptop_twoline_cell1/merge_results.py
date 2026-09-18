#!/usr/bin/env python3
"""Merge safe per-model Experiment 41 runs and regenerate the report."""

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
    spec = importlib.util.spec_from_file_location("experiment_41_merge", RUNNER)
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
        if run.get("format") != "experiment-41-laptop-twoline-cell1-v1":
            raise SystemExit(f"wrong experiment format: {metadata_path}")
        if len(run.get("models", [])) != 1:
            raise SystemExit(f"expected one model in {metadata_path}")
        metadata.append(run)
    model_keys = [run["models"][0]["model"] for run in metadata]
    if len(model_keys) != len(set(model_keys)):
        raise SystemExit(f"duplicate per-model inputs: {model_keys}")
    for key in ("budgets", "profile", "saturation_kib", "two_line_kib"):
        values = [run.get(key) for run in metadata]
        if any(value != values[0] for value in values[1:]):
            raise SystemExit(f"incompatible {key} across inputs")

    for table in TABLES:
        paths = [directory / table for directory in args.inputs]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit(f"missing {table}: {missing}")
        pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
            args.output_dir / table, index=False
        )

    merged = dict(metadata[0])
    merged["format"] = "experiment-41-laptop-twoline-cell1-merged-v1"
    merged["models"] = [item for run in metadata for item in run["models"]]
    merged["merged_inputs"] = [str(path) for path in args.inputs]
    (args.output_dir / "metadata.json").write_text(
        json.dumps(merged, indent=2) + "\n"
    )

    runner = load_runner()
    runner.analyze(SimpleNamespace(
        output_dir=args.output_dir,
        report_output=args.report_output,
        profile=Path(merged["profile"]),
        saturation_kib=float(merged["saturation_kib"]),
    ))


if __name__ == "__main__":
    main()
