# Experiment 44: Paper vs globally allocated Cell-1 R

This experiment overlays Experiment 43's two global Cell-1 allocation
strategies with the Paper and fixed-R Cell-1 frontiers:

- global measurable importance lower bound;
- global measured projection-error oracle.

No masks are remeasured. The source contains real laptop score, selector,
O_DIRECT/upload, gather, GEMM, and dense-reference projection-error
measurements. Reusing those measurements is necessary for a matched overlay.

Run:

```bash
python experiments/44_global_r_vs_paper/run_experiment.py
```

For every global point, the script interpolates Paper and fixed-R Cell-1
latency at the global strategy's achieved mean projection error. This yields a
same-error comparison instead of comparing points that merely share a nominal
Paper target R.
