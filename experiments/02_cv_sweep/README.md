# Experiment 02: paper-calibrated CV sweep

This experiment replaces Experiment 01's arbitrary half-normal importance
distribution with positive lognormal-shaped multisets whose *sample* CV is
calibrated exactly to values reported in Appendix C, Table 1 of
VLM-in-a-Flash.

Every generated multiset is evaluated under three orderings while preserving
exactly the same values and CV:

- `random`: an independent random permutation;
- `local`: ranks induced by an AR(1) score with default correlation 0.95;
- `hot-cold`: ranks induced by a persistent neuron-hotness profile plus
  per-input noise, approximating offline hot-cold reordering.

The compared policies are Paper greedy, the exact fixed-`R` DP, and the
`O(qN)` Quantized Pareto solver. Fixed-`R` policies are compared with Pareto
at their individually achieved importance, rather than at unmatched quality.
The resulting latency difference is signed: a negative value means that the
fixed-`R` solution beat the quantized approximation at that particular
importance target.

Run from the repository root:

```bash
python3 experiments/02_cv_sweep/run_experiment.py
```

Default CV targets are `1.07, 1.25, 1.44, 2.48, 3.30, 4.55, 9.19`. The first
six span the paper's VLM range; `9.19` is a ReLU OPT reference. The synthetic
hot-cold model is only a controlled proxy—real activation vectors remain the
preferred validation data.

The completed default run and its interpretation are in [report.md](report.md).
