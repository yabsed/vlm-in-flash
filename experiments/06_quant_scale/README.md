# Experiment 06: realistic-N q scale, Top-R, and chunk histograms

This experiment extends Experiment 05 in three ways:

1. it plots actual chunk-count histograms (`x=K`, `y=fraction`),
2. it adds Top-R in both fixed-R and matched-importance comparisons, and
3. it sweeps Quantized Pareto `q` from `1024` up to `32768 ... 131072`
   (depending on N) to search for the
   smallest setting that is stably better than Paper greedy.

“Stably better” means 100% non-worse latency, at least 95% strict wins, and a
nonnegative fifth-percentile gain on Paper-matched VLM-CV cases. Exact Coverage
is excluded at these channel counts.

The q-sweep computes the same recurrence as Experiment 05 but omits mask
backtracking and retains only importance, latency, row count, and chunk count.
This reduces memory from `O(Nq)` to `O(q)`; the time complexity remains
`O(Nq)`.

Run from the repository root:

```bash
python3 experiments/06_quant_scale/run_experiment.py
```

Results are separated under `results/n_<N>/`.

The default q grids are larger for larger matrices; the two largest matrices
are tested through `q=131072`. Use `--q-values` to override the grid.

The checked-in results use one multiset per CV for the full q sweep, then two
additional independently seeded multisets per CV at the selected threshold q.
The latter are stored under `results/threshold_validation/`; together they
give 378 VLM-CV Paper-matched cases per N at the selected q.
