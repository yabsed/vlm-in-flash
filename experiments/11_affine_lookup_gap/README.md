# Experiment 11: why the affine gap shrinks under lookup evaluation

This experiment explains the change observed between Experiments 07 and 10.
It reuses Experiment 10's persisted mask run-length encodings and performs no
selection or exact-DP recomputation.

It measures:

1. the chunk-length distributions of Paper, DP-R, Top-R, and their matched
   affine DP-Cover masks;
2. the exact factorization from affine matched ratio to lookup matched ratio;
3. how much of each mask lies beyond the released table's measured 255-KiB
   range; and
4. sensitivity to three beyond-range policies: the released proportional
   extrapolation, an anchored tail-linear extrapolation fitted over
   128--255 KiB, and explicit 255-KiB block splitting.

Run from the repository root:

```bash
python3 experiments/11_affine_lookup_gap/run_experiment.py
```

Outputs:

- `latency_model_and_runs.{png,pdf}`
- `paper_gap_decomposition.{png,pdf}`
- `extrapolation_sensitivity.{png,pdf}`
- `run_length_summary.csv`
- `ratio_decomposition.csv`
- `extrapolation_sensitivity.csv`
- `summary.json`

See `report.md` for results and interpretation.

