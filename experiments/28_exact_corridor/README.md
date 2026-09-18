# Experiment 28: exact-R oracle and sparse corridor DP

This experiment answers two separate questions on the local laptop:

1. Is the Paper mask optimal under our fixed-R two-line latency objective?
2. Can a Paper-independent online algorithm recover the better mask cheaply
   enough to reduce measured selector + O_DIRECT read + GPU-upload time?

The full oracle solves every fixed-Dinkelbach-ratio subproblem exactly over the
entire `(selected rows, skipped rows)` state rectangle.  It is run on a
configurable subset of real Qwen activation traces.  The online candidates use
the same recurrence only inside a prefix-cardinality corridor of width
`W = 4, 8, 16, 32`, centered on a one-price Lagrangian solution repaired to
exactly `R` rows.  Neither path receives the Paper mask or `I(M_paper)`.

For a two-line run cost, every transition range is affine.  Along a fixed
zero-count diagonal its best predecessor is therefore a sliding-window or
prefix maximum.  The full solver costs `O(R(N-R))`; a width-W corridor stores
at most `O(NW)` states.

Smoke test:

```bash
python experiments/28_exact_corridor/run_experiment.py \
  --shapes 896x128 --max-traces-per-shape 1 --tracks host \
  --row-budget-fractions 0.5 --repetitions 2 --oracle-trials 1 \
  --real-oracle-traces-per-shape 1 \
  --output-dir /tmp/experiment28-smoke
```

Laptop run with native reads:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/28_exact_corridor/run_experiment.py \
  --tracks cuda \
  --profile experiments/26_frontier_adaptive_trim/results_laptop/laptop_sn850x_profile.json \
  --saturation-kib 240 \
  --measure-io --io-blob experiments/28_exact_corridor/results_laptop/io_blob.dat \
  --output-dir experiments/28_exact_corridor/results_laptop \
  --report-output experiments/28_exact_corridor/report_laptop.md
```

The blob must reside on the target SSD, be at least as large as the largest
matrix replay, and contain allocated, non-compressible data.
