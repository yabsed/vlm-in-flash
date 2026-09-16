# Experiment 13: high-q saturation-aware Lagrangian

This experiment implements the two-line latency model and `O(qN)` Lagrangian
heuristic proposed in `experiments/12_note/note2.md`.

The continuous model is fitted with the breakpoint fixed at the paper's
`236 KiB` saturation point:

```text
T(r) = a + c1*r,  r <= r_sat
T(r) = c2*r,      r >  r_sat
```

The default run evaluates `q=131,072` logarithmically spaced multipliers. Each
penalized problem is solved exactly in `O(N)` using a sliding maximum for short
chunks and a prefix maximum for saturated chunks. The final coverage solution
is still a supported-point heuristic rather than an exact constrained optimum.

Here `q` means the number of lambda multipliers. It is unrelated to the
importance-bucket count in the earlier Quantized-Pareto experiments.

Experiment 10's masks are reused for Paper and affine DP-Cover, and Experiment
11's released/tail latency policies are applied to every reconstructed mask.
No `O(N^3)` DP is rerun.

Run:

```bash
python3 experiments/13_two_line_lagrangian/run_experiment.py
```

Outputs include paired Paper-target trials, common coverage frontiers,
q-convergence results, plots, and `summary.json`. Plot outputs are separated by
latency evaluator:

- `results/two_line/`: only the two diagnostics retained for the surrogate
  model, `q_convergence` and `run_distributions`.
- `results/lookup/`: released-lookup comparisons containing only high-q Quant,
  Paper greedy, and Top-R.

The lookup directory contains:

- `importance_latency_frontiers.{png,pdf}`
- `r_importance.{png,pdf}`
- `r_latency.{png,pdf}`
- `coverage_opt_ratio.{png,pdf}`
- `chunk_count.{png,pdf}`

`coverage_opt_ratio` uses the high-q Quant solution at the same Paper
importance as its denominator. It is a useful common-target reference, but it
is not a certificate of the exact lookup-aware coverage optimum. The lookup
figures can be regenerated without rerunning the dense lambda grid:

```bash
python3 experiments/13_two_line_lagrangian/run_experiment.py --plots-only
```

See `report.md` for the measured results and interpretation.
