# Experiment 05: Realistic channel counts

This experiment repeats the coverage comparison at the paper's actual
down-projection row counts:

| Model family | Shape (rows × cols) | N | FP16 row size | AGX `(start, jump)` |
|---|---:|---:|---:|---:|
| LLaVA-OneVision-0.5B | 4864 × 896 | 4,864 | 1.75 KiB | (12, 16) KiB |
| NVILA-Lite-2B | 8960 × 1536 | 8,960 | 3 KiB | (16, 16) KiB |
| VILA1.5-8B | 14336 × 4096 | 14,336 | 8 KiB | (32, 32) KiB |
| LLaVA-OneVision-7B / LongVA-7B | 18944 × 3584 | 18,944 | 7 KiB | (32, 32) KiB |

Exact Coverage DP is removed because the current oracle is `O(N^3)`. The
three methods are Paper greedy, the `O(N)` single interval, and Quantized
Pareto with `q=1024`. The q-Pareto recurrence is unchanged but its build loop
is compiled with Numba so these dimensions are tractable.

Each `results/n_<N>/` directory contains its own CSVs, summary, and four
figures. `results/summary.json` and `results/n_scaling.{png,pdf}` compare all
four channel counts. `quant_reliability.{png,pdf}` reports full-mask fallback,
coverage overshoot, and how often q=1024 beats the single interval. The default
run uses 20 independent multisets per CV.

Run from the repository root:

```bash
python3 experiments/05_realistic_n/run_experiment.py
```
