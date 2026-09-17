# Experiment 20: remaining sub-2 ms candidates

Experiment 18 implemented the first four candidate families. This experiment
implements the six remaining families and screens them under the same timing
and validity contract.

The twenty configurations are:

- two references: Paper and the Experiment 18 incumbent
  `TD-2L (16) + trim 256`;
- small lambda banks with 4, 8, or 16 scalarized solves;
- piecewise-linear lookup DP with `(J, calls)` equal to `(4,4)`, `(4,8)`,
  `(8,4)`, or `(8,8)`;
- Paper plus at most 64 or 256 lookup-aware local edits;
- target-directed exact lookup DP with 2, 4, or 8 scalarized solves;
- lambda prediction plus correction with 2, 4, or 8 solves;
- capped-label lookup DP with 2, 4, or 8 labels per run-length state.

The tracks remain separate:

1. coverage-only, with external target `Q`;
2. coverage plus a paired row cap `R`.

The common fallback, validation, and CPU boolean-mask handoff are timed. Each
case is warmed up and repeated 30 times. A case passes the host screen only if
it is valid and its within-case p95 is at most 2 ms. Actual storage I/O and
model compute are not in the selector timer, so this does not establish a
Jetson 2 ms result.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python3 experiments/20_remaining_sub2ms_candidates/run_experiment.py
```

Small smoke test:

```bash
python3 experiments/20_remaining_sub2ms_candidates/run_experiment.py \
  --n-values 97 --trials 1 --repetitions 2 \
  --target-fractions 0.5 --row-budget-fractions 0.6
```

Outputs:

- `results/trials.csv` and `results/timing_samples.csv`;
- `results/summary.csv` and `results/summary.json`;
- `results/runtime_quality.{png,pdf}`;
- `results/deadline_pass_rate.{png,pdf}`;
- `report.md`.

See [`algorithm.md`](algorithm.md) for the executable specification and
limitations of each candidate.

