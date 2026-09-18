# Experiment 41: laptop Cell-1 lookup vs 2-line

This experiment changes exactly one part of Experiment 39: the run-cost oracle
inside Cell-1. It compares the measured laptop SSD lookup table with the fitted
continuous model

```text
T(r) = a + c1*r  for r <= s
     = c2*r      for r > s.
```

The final latency is not the fitted value. Both variants remeasure score
construction, selector runtime, O_DIRECT SSD read/upload, activation gather,
and compact GEMM on the laptop.

Both variants also use the same local run-cost-array cache and the same Cell-1
kernel. This removes the unrelated overhead of rebuilding and hashing the
lookup table on every selector call. The stored laptop profile is trimmed at
240 KiB, so the pre-s branch is fitted to measured points while the post-s
branch is a continuity-constrained linear extrapolation, just like the lookup
table's own out-of-range behavior.

Run each model in a separate process:

```bash
for model in qwen05 smol360 tiny11; do
  python experiments/41_laptop_twoline_cell1/run_experiment.py \
    --models "$model" \
    --io-blob experiments/35_full_frontier/results_laptop/io_blob.dat \
    --output-dir "experiments/41_laptop_twoline_cell1/runs_laptop_safe/$model" \
    --report-output "experiments/41_laptop_twoline_cell1/runs_laptop_safe/$model/report.md"
done

python experiments/41_laptop_twoline_cell1/merge_results.py \
  --inputs \
    experiments/41_laptop_twoline_cell1/runs_laptop_safe/qwen05 \
    experiments/41_laptop_twoline_cell1/runs_laptop_safe/smol360 \
    experiments/41_laptop_twoline_cell1/runs_laptop_safe/tiny11 \
  --output-dir experiments/41_laptop_twoline_cell1/results_laptop_safe \
  --report-output experiments/41_laptop_twoline_cell1/report_laptop_safe.md
```

The runner rejects more than one model per process and retains Experiment 39's
55% CUDA allocator cap, two O_DIRECT/CPU threads, and projection throttling.
