# Random-input comparison under a bounded row budget

This experiment compares conventional Top-`R`, the paper's released greedy
Neuron Chunking policy, and the two exact results proved in
`research/final_email/idea.md`.

The two rows constraints are intentionally reported separately:

1. **Fixed budget, `R(M)=R`.** The exact chain DP plus Dinkelbach maximizes
   `I(M) / (a K(M) + c R)` under the fitted affine latency model. The greedy
   mask is compared at its realized row count if it underfills the requested
   budget.
2. **Upper budget, `1 <= R(M) <= R`.** The best single interval of length at
   most `R` is the exact optimum for the released lookup-table objective.
   This problem is mathematically valid but can retain far fewer than `R`
   rows, so it should not be interpreted as a quality-matched replacement.

Default inputs are 200 independent nonnegative vectors generated as
`abs(N(0,1))` and normalized to total importance 1. The setup uses `N=256`,
1 KiB per row, the bundled Jetson Orin AGX profile, and the paper heuristic's
8 KiB multiscale sweep. Budgets are multiples of 8 so the greedy policy can
fill them exactly, and every possible selected interval is at most 256 KiB,
the profiled range.

Run from the repository root:

```bash
python3 research/experiments/random_r_bound/run_experiment.py
```

The script first checks both exact solvers against exhaustive enumeration for
all masks of several small random instances. It then writes per-trial CSV,
JSON summaries, and PNG/PDF plots to `results/`. `comparison.png` contains
the original bounded-`R` four-panel comparison. `latency_importance.png`
separately plots retained importance against latency for the three fixed-`R`
methods, under both affine and lookup-table latency.

Distribution sensitivity can be reproduced with:

```bash
python3 research/experiments/random_r_bound/run_experiment.py \
  --distribution lognormal --output-dir /tmp/random-r-lognormal
python3 research/experiments/random_r_bound/run_experiment.py \
  --distribution correlated --output-dir /tmp/random-r-correlated
```

The checked-in aggregate sensitivity values are in
`results/sensitivity_summary.json`.

These are synthetic importance-vector results. They measure optimization
quality for `I/L`; they are not VLM accuracy or physical SSD measurements.
