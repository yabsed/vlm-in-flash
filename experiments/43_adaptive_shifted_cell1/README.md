# Experiment 43: adaptive shifted Cell-1

This experiment tests whether Cell-1 is only locally optimal because its
saturation-length grid starts at row zero.

Methods:

- `Paper |x|`: released paper selector and score.
- `Cell-1 X²`: one fixed saturation-length grid.
- `Shifted-8 X²`: eight grid origins evaluated in one compiled selector.
- `Adaptive-8 X²`: Shifted-8 followed by `s/2` refinement of only the four
  lowest-score selected and four highest-score unselected boundary cells.

Every frontier includes score construction, selector, real O_DIRECT read and
upload wall time, activation gather, and compact GEMM. Projection relative L2
error is measured against the dense projection.

The analysis also removes fixed `R` post hoc in two distinct ways:

- `importance_bound` shares one mean-absolute-activation loss budget across all
  sampled projections in a workload. It uses robust module-level median
  calibration latency rather than held-out timing, but remains a post-hoc
  measurable-proxy upper bound;
  deployment would need a learned per-layer budget controller.
- `error_oracle` shares the measured projection-error budget. This is an upper
  bound and is not deployable because it observes the dense reference.

Run one laptop-safe model process:

```bash
python experiments/43_adaptive_shifted_cell1/run_experiment.py \
  --models qwen05 \
  --io-blob experiments/35_full_frontier/results_laptop/io_blob.dat \
  --skip-end-to-end
```

The output directory intentionally does not contain an I/O blob, so the result
can be committed without crossing GitHub's 100 MiB file limit.
