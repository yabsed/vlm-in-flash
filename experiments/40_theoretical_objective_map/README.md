# Experiment 40: theoretical objective map

This CPU-only experiment removes selector runtime, GPU upload, activation
gather, and GEMM. It places four policies on the same retained-importance versus
predicted-I/O-latency plane:

- Paper with fixed R;
- Cell-1 with fixed R;
- Cell-1 with R selected by an importance lower bound; and
- Experiment 16's exact global solution of
  `min L_two_line(M) subject to I(M) >= Q`.

Every plotted point is annotated with its actual selected-row fraction `R/N`.
Cell-1's importance-bound policy searches R at 1% resolution. The exact global
claim applies only to the two-line latency model; the released-lookup plot is a
sensitivity re-evaluation of the same masks.

The supplementary solver map also replots Experiment 16's Paper, dense Quant,
complete two-line-supported, lookup-supported, and exact-coverage points with
the same `R/N` annotations.

Run:

```bash
python experiments/40_theoretical_objective_map/run_experiment.py
```

Regenerate tables and figures from saved points:

```bash
python experiments/40_theoretical_objective_map/run_experiment.py --analyze-only
```
