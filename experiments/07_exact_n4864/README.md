# Experiment 07: exact DP comparison at N=4,864

This experiment compares exactly four methods at the paper's realistic
`4,864 × 896` down-projection shape:

1. exact coverage DP (`DP-Cover`),
2. Paper greedy,
3. exact fixed-R ratio DP (`DP-R`), and
4. Top-R.

Single interval and Quantized Pareto are excluded. DP-Cover uses an in-place
rolling recurrence with `O(N^2)` memory and `O(N^3)` time. It returns exact
importance, latency, R, and chunk count K, but does not reconstruct the mask.

Run from the repository root:

```bash
python3 experiments/07_exact_n4864/run_experiment.py
```

The default run uses three independent multisets per CV, seven CV values,
three spatial orderings, seven fixed-R budgets, and seven coverage targets.

Generated outputs include:

- `importance_latency_frontiers.{png,pdf}`
- `r_importance.{png,pdf}`
- `r_latency.{png,pdf}`
- `coverage_optimality_ratio.{png,pdf}`
- `chunk_count_histograms.{png,pdf}`
- `coverage_trials.csv`, `fixed_r_trials.csv`, `input_trials.csv`, and
  `summary.json`

See `report.md` for the measured results and interpretation.
