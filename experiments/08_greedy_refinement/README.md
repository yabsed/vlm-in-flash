# Experiment 08: coverage-preserving Paper refinement

This experiment adds one method to Experiment 07 at `N=4,864`:
`Paper + merge + trim`.

The method starts from the Paper greedy mask, merges every internal gap whose
row-transfer cost is smaller than one chunk-opening cost, then removes cheap
chunk-boundary rows while preserving the importance attained by Paper.

Experiment 07's expensive exact DP-Cover, DP-R, and Top-R results are read
directly from `../07_exact_n4864/results/`. The deterministic inputs and Paper
masks are replayed only because Experiment 07 intentionally stored metrics
rather than masks. Exact DP-Cover is never rebuilt.

Run from the repository root:

```bash
python3 experiments/08_greedy_refinement/run_experiment.py
```

The output contains the same five figures as Experiment 07, with the refined
method added, along with an enriched `fixed_r_trials.csv` and `summary.json`.
Measured results and interpretation are in `report.md`.
