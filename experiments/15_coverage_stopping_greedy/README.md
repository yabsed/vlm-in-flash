# Experiment 15: Paper greedy with coverage stopping

This experiment changes only the stopping condition of Paper greedy:

- original: stop under a fixed row budget `R`;
- modified: stop when cumulative selected importance reaches `B`.

Candidate windows, lookup-based importance/latency scores, stable score order,
and non-overlap checks remain the same. The coverage target `B` is the
importance achieved by original Paper greedy in each Experiment 14 fixed-R
case.

Experiment 14's original Paper and high-q Quant results are reused. The 900
deterministic inputs are regenerated, but the expensive q-grid is not rerun.

Run:

```bash
python3 experiments/15_coverage_stopping_greedy/run_experiment.py
```

Regenerate analysis from saved CSVs:

```bash
python3 experiments/15_coverage_stopping_greedy/run_experiment.py --analyze-only
```

Outputs:

- `paired_trials.csv`: 17,100 paired cases;
- `input_trials.csv`: coverage-prefix build statistics for 900 inputs;
- `summary.json`: paired clustered statistics;
- `latency_saving_ci.{png,pdf}`;
- `importance_latency_frontiers.{png,pdf}`;
- `latency_ratio.{png,pdf}`; and
- `importance_overshoot.{png,pdf}`.

See `report.md` for results.
