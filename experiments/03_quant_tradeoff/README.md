# Experiment 03: Quantized Pareto cost–accuracy trade-off

This experiment sweeps the Quantized Pareto importance resolution
`q = 8, 16, ..., 1024` on the paper-calibrated CV and ordering conditions from
Experiment 02. Exact Coverage DP is the accuracy oracle.

Two accuracy gaps are kept separate:

- requested-target gap: quantized latency versus the exact minimum at the
  requested coverage threshold;
- achieved-target gap: quantized latency versus the exact minimum at the
  importance actually achieved after quantization overshoot.

Paper greedy is run on the same inputs. Quantized Pareto is additionally
queried at every importance value achieved by Paper greedy, enabling a paired
quality comparison. Reference CPU wall-clock time is measured, but is not a
hardware-independent complexity result: the Quantized Pareto implementation
is Python-heavy while Paper greedy uses vectorized PyTorch operations.

Run from the repository root:

```bash
python3 experiments/03_quant_tradeoff/run_experiment.py
```

The default run uses 30 independent multisets per CV, all seven Experiment 02
CV targets, all three spatial orderings, seven coverage targets, and seven
Paper-greedy row budgets.

The completed default run and interpretation are in [report.md](report.md).
