# Experiment 16: Saturation-aware global chain versus Paper greedy

This experiment compares the saturation-aware binary-chain solution with the
Neuron Chunking Paper greedy baseline at matched importance targets.

The chain method reuses Experiment 13's dense `q=131,072` multiplier sweep.
Every mask in that sweep is a global optimum of its scalarized objective

```text
maximize lambda * importance - two_line_latency,
```

but choosing a feasible point from the sweep is only a supported-point
heuristic for the constrained coverage problem. It is not labeled as an exact
coverage oracle.

Run:

```bash
python3 experiments/16_saturation_global_chain/run_experiment.py
```

Outputs:

- `paired_trials.csv`: matched Paper and global-chain operating points;
- `chunk_lengths.csv`: one row per true maximal selected run;
- `summary.json`: latency-policy sensitivity and structural summaries;
- `importance_latency.{png,pdf}`: I-L curves;
- `importance_rows.{png,pdf}`: I-R curves;
- `latency_rows.{png,pdf}`: L-R curves; and
- `chunk_length_histogram.{png,pdf}`: true run-length distributions.

See `report.md` for the findings and scope of the optimality claim.
