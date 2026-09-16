# Random-input comparison under a bounded row budget

This experiment compares conventional Top-`R`, the paper's released greedy
Neuron Chunking policy, and the exact results proved in
`research/final_email/idea.md`.

The three formulations are intentionally reported separately:

1. **Fixed budget, `R(M)=R`.** The exact chain DP plus Dinkelbach maximizes
   `I(M) / (a K(M) + c R)` under the fitted affine latency model. The greedy
   mask is compared at its realized row count if it underfills the requested
   budget.
2. **Upper budget, `1 <= R(M) <= R`.** The best single interval of length at
   most `R` is the exact optimum for the released lookup-table objective.
   This problem is mathematically valid but can retain far fewer than `R`
   rows, so it should not be interpreted as a quality-matched replacement.
3. **Importance coverage, `I(M) >= alpha I_total`.** The exact `(r,k)` chain
   DP minimizes `a K(M) + c R(M)` while allowing both row and chunk counts to
   vary. Two `O(qN)` approximations are compared: a Lagrangian multiplier
   grid, and a quantized Pareto DP that retains the least-cost representative
   in each of `q=256` coverage buckets and each chain-ending state. The
   default sweep uses `alpha=0.10,...,0.90,0.95,0.97,0.99`.

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

The script first checks all three exact solvers and every fixed-multiplier
Lagrangian inner solve against exhaustive enumeration on small random
instances. It also checks that every quantized-Pareto result is feasible and
cannot beat the exhaustive optimum. It then writes per-trial CSV, JSON
summaries, and PNG/PDF plots to `results/`. `comparison.png` remains the
original bounded-`R` four-panel comparison. `latency_importance.png` plots
retained importance against latency for the three fixed-`R` methods, the
exact coverage frontier, and both approximations, under affine and
lookup-table latency. `coverage.png` contains standalone coverage diagnostics,
and
`coverage_gap.png` compares every fixed-`R` policy with the exact coverage
latency at that policy's actually achieved importance.
`r_importance.png` and `r_latency.png` directly plot retained importance and
latency, respectively, against the fixed row budget `R/N`.

On the default i.i.d. inputs, the Lagrangian scalarization exposes an
important limitation: 256 multiplier evaluations produce only 3.75 distinct
masks on average. For requested coverage from 10% through 80%, it jumps to a
mask retaining 98.15% importance on average. Its latency is exact at that
*achieved* coverage, but is 17.4--236.7% above the exact DP at the *requested*
coverage. Increasing `q` alone cannot recover Pareto points that are not
supported by a linear scalarization.

The quantized Pareto DP does preserve those intermediate points. With
`q=256`, its mean affine-latency gap to the exact requested-coverage optimum
is 0.083% across the 13 targets (0.165% at worst when target means are
compared); the largest target-wise 95th percentile is 1.034%. Final
feasibility always uses unquantized importance. The implementation uses at
most `2(q+1)` live representatives and `O(qN)` backtracking storage.

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
