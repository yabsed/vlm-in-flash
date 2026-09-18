# Experiment 29: saturation tiles

Experiment 29 partitions the neuron axis into cells of approximately `s/8`
rows, where `s` is the profiled saturation length.  It considers only sliding
chunks that span eight cells.

For cell-window importance `w[j]`, exactly `k` non-overlapping chunks are
selected by

```text
D[k,j] = max(D[k,j-1], D[k-1,j-8] + w[j]).
```

This removes the Paper selector's multi-scale candidate generation and global
sort.  To return exactly `R` rows, the implementation compares:

- `floor(R/L)` full tiles followed by lookup-aware endpoint expansion;
- `ceil(R/L)` full tiles followed by lookup-aware endpoint trim.

Here `L = 8 * cell_rows`.  The selector evaluates floor, round, and ceil
quantizations of `s/8`.  The report also preserves the 384-case Python
reference probe that motivated this experiment.

Smoke test:

```bash
python experiments/29_saturation_tiles/run_experiment.py \
  --shapes 896x896 --max-traces-per-shape 1 --tracks host \
  --row-budget-fractions 0.5 --repetitions 3 --skip-oracle \
  --output-dir /tmp/experiment29-smoke
```

Laptop run:

```bash
dd if=/dev/urandom \
  of=experiments/29_saturation_tiles/results_laptop/io_blob.dat \
  bs=1M count=64
python experiments/29_saturation_tiles/run_experiment.py \
  --tracks cuda \
  --profile experiments/26_frontier_adaptive_trim/results_laptop/laptop_sn850x_profile.json \
  --saturation-kib 240 --measure-io \
  --io-blob experiments/29_saturation_tiles/results_laptop/io_blob.dat \
  --output-dir experiments/29_saturation_tiles/results_laptop \
  --report-output experiments/29_saturation_tiles/report_laptop.md
```

The measured laptop result is in `report_laptop.md`.  The best single method,
`ceil(s/8)`, reduced mean selector-plus-read time from 1.147 ms to 1.002 ms
(12.69%).  A held-out per-shape dispatch reduced it by 13.06%.
