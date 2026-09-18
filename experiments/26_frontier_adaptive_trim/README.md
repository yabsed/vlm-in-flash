# Experiment 26: frontier search with adaptive fixed-R trim

Experiment 24 kept only the row-price solution whose cardinality was closest
to `R` from above, then repaired that single mask with a fixed-rho endpoint
heap. More row-price calls could therefore replace a good earlier mask with a
worse one.

This experiment evaluates every distinct eligible cardinality encountered on
the row-price frontier. Each overfilled mask is trimmed to exactly `R` rows;
after every deletion, the endpoint yielding the best resulting exact `I/L` is
chosen. The best repaired ratio seen across every mu and rho iteration is
retained, so the selector uses only:

- the activation-importance vector;
- the requested row count `R`;
- the two-line latency model.

It does not receive the Paper mask or `I(M_paper)`. Paper is run only as a
paired benchmark reference, and its returned row count defines the common `R`
for an exact same-cardinality comparison.

Compared methods:

- Paper and Top-R;
- Experiment 24's `rho2/mu4` implementation;
- 16 mu calls, two rho iterations, adaptive trim capped at 64 rows;
- 16 mu calls, two rho iterations, adaptive trim capped at 256 rows;
- 24 mu calls, three rho iterations, adaptive trim capped at 256 rows.

The production benchmark reuses Experiment 22's real
Qwen2.5-0.5B-Instruct activation archive: 128 projection calls over four
supported Table-2 shapes. A separate synthetic `N=18` exhaustive oracle is
used only to measure solver optimality.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/26_frontier_adaptive_trim/run_experiment.py
```

Small smoke test:

```bash
python experiments/26_frontier_adaptive_trim/run_experiment.py \
  --shapes 896x128 --max-traces-per-shape 1 --tracks host \
  --repetitions 2 --oracle-trials 1 --output-dir /tmp/experiment26-smoke
```

See [`algorithm.md`](algorithm.md) for the selection and timing contract.

