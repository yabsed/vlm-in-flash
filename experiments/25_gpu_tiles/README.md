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

The benchmark no longer generates synthetic lognormal vectors. By default it
reuses Experiment 22's Qwen2.5-0.5B-Instruct activation archive, captured from
a real dense forward through the checked-out
`preliminary_research/vlm-flash` attachment path. The deterministic cap keeps
32 projection calls per supported Table-2 shape (128 traces over four Qwen
shapes). Each trace is evaluated on Experiment 18's three paired
coverage/row-budget scenarios, with 30 timed repetitions after warm-up. This
also makes Experiments 22, 24, and 25 directly comparable on identical
activation inputs. It records:

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

To collect a fresh real-model archive first:

```bash
python experiments/25_gpu_tiles/run_experiment.py --capture-traces
```

Run one-shape smoke test:

```bash
python experiments/25_gpu_tiles/run_experiment.py \
  --shapes 4864x896 --max-traces-per-shape 1 --repetitions 3 --warmup 2 \
  --output-dir /tmp/experiment25-smoke
```

See [`algorithm.md`](algorithm.md) for the GPU dataflow and timing contract.
