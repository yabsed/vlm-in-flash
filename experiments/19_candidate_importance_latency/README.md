# Experiment 19: candidate importance-latency curves

This experiment plots Experiment 18's fourteen configurations with latency on
the x-axis and retained importance on the y-axis.

It keeps two meanings of latency explicitly separate:

1. `selector_latency_importance`: measured warm host selector latency from
   Experiment 18, including mask recovery, postprocessing, fallback, and CPU
   mask handoff. The 2 ms selector deadline is shown here only.
2. `lookup_latency_importance` and `two_line_latency_importance`: predicted
   I/O latency of the selected masks under the released lookup rule and fitted
   two-line model. No 2 ms selector line is drawn on these I/O plots.

The dense I/O curves use 20 external coverage targets from 0.05 through 0.99,
`N=4,864`, exact CV values `1.25`, `3.30`, and `4.55`, three trials, and
random/local/hot-cold orderings. The main frontier plots follow the earlier
`importance_latency_frontiers` layout: CV on rows and spatial ordering on
columns. The selector-runtime curve reuses Experiment 18's 30-repeat
measurements at targets 0.50, 0.70, and 0.90; it is therefore not split by CV.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python3 experiments/19_candidate_importance_latency/run_experiment.py
```

Outputs:

- `results/curve_trials.csv`: dense coverage-only mask results;
- `results/summary.csv`, `results/condition_summary.csv`, and `results/summary.json`;
- `results/selector_latency_importance.{png,pdf}`;
- `results/lookup_latency_importance.{png,pdf}`;
- `results/two_line_latency_importance.{png,pdf}`;
- `results/{lookup,two_line}_importance_latency_frontiers.{png,pdf}`: six
  representative methods in the CV-by-ordering layout;
- `results/conditioned/*_frontiers.{png,pdf}`: all fourteen configurations,
  separated into the four candidate families;
- `report.md`.
