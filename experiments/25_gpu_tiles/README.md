# Experiment 25: GPU saturation tiles

This experiment ports Experiment 18's three saturation-tile selectors to
CUDA-resident PyTorch operations:

- `Tiles (s/2)`
- `Tiles (s)`
- `Tiles (2s)`

For each selector, both offset grids (`0` and `floor(L/2)`) are evaluated on
the GPU. Tile sums, utility sorting, coverage stopping, run-cost evaluation,
grid choice, and the final boolean mask stay on the GPU. Grid boundaries and
row-to-tile mappings depend only on the matrix shape and are precomputed.

The default run covers all 16 Table-2 matrix shapes, three trials, three
spatial orderings, and Experiment 18's three paired coverage/row-budget
scenarios. Each case has 30 timed repetitions after warm-up. It records:

- CUDA-event device time;
- synchronized wall time from CUDA-resident importance to CUDA mask;
- paired wall time of Experiment 18's original CPU implementation;
- coverage and paired row-budget validity;
- released-lookup and two-line quality against Paper's coverage-stopping
  selector;
- parity with Experiment 18's original CPU Tiles implementation.

The 2 ms decision uses synchronized wall p95. Importance generation, static
layout construction, Paper reference construction, storage I/O, and model
compute are excluded.

Run the full experiment on a CUDA host:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/25_gpu_tiles/run_experiment.py
```

Run one-shape smoke test:

```bash
python experiments/25_gpu_tiles/run_experiment.py \
  --shapes 4864x896 --trials 1 --repetitions 3 --warmup 2 \
  --output-dir /tmp/experiment25-smoke
```

See [`algorithm.md`](algorithm.md) for the GPU dataflow and timing contract.
