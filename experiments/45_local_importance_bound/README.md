# Experiment 45: causal local importance-bound Cell-1

This experiment removes every global/post-hoc allocation policy. At each
projection call, the selector sees only the current activation and solves

```text
min predicted read latency(mask)
subject to retained X² importance(mask) >= tau.
```

`tau` is the only quality control. The selected fraction `R` is measured as an
output. Paper and ordinary Cell-1 use fixed `R` controls; their frontiers are
compared with the `tau` frontier at equal measured projection error.

The selector uses fixed-origin saturation-length cells, chooses the minimum
number of high-importance cells reaching `tau`, and trims run endpoints while
preserving the lower bound. It uses neither future projection information nor
dense projection error.

Run one model per process with `--skip-end-to-end`, then merge the three output
directories with `merge_results.py`. No I/O blob is copied into this experiment.
