# Experiment 10: lookup-table recheck of Experiment 07

This experiment reconstructs the masks behind Experiment 07's exact affine
DP-Cover states and evaluates all four methods with the released Orin AGX
latency lookup table:

1. affine DP-Cover,
2. Paper greedy,
3. exact fixed-R DP (DP-R), and
4. Top-R.

The saved Experiment 07 `(R, K, importance)` states are reused. The full
`O(N^3)` frontier is **not** recomputed. Instead, a backtracking DP truncated
to the largest required `K` reconstructs the selected masks in
`O(N R K_max)` time. Selected intervals are permanently stored as zero-based
`start:length` run-length encodings, so another lookup table can later be
applied without rerunning any selector or DP.

Run from the repository root:

```bash
python3 experiments/10_exact_lookup_recheck/run_experiment.py
```

Outputs:

- `importance_lookup_frontiers.{png,pdf}`
- `matched_lookup_ratio.{png,pdf}`
- `affine_lookup_comparison.{png,pdf}`
- `coverage_lookup_trials.csv`
- `fixed_r_lookup_trials.csv`
- `reconstruction_trials.csv`
- `summary.json`

DP-Cover remains exact for the fitted affine objective `aK+cR`. Its lookup
latency is a post-hoc evaluation, not the optimum of a lookup-aware exact
problem.

See `report.md` for the measured results and interpretation.
