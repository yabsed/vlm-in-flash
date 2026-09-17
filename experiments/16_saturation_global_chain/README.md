# Experiment 16: Exact coverage and direct lookup optimizers

This experiment compares five selectors at Paper-greedy achieved-importance
targets:

- **Exact coverage**: the globally optimal constrained-coverage solution under
  the two-line model, found by an untruncated Pareto-label binary-chain DP;
- **Lookup supported**: an exact `O(Nm)` fixed-lambda DP over released lookup
  increments, followed by adaptive enumeration of the supported frontier;
- **Two-line supported**: adaptive enumeration of every strongly supported
  optimum of the scalarized two-line binary-chain objective;
- **Quant**: the existing fixed `q=131,072` lambda grid; and
- **Paper greedy**: Neuron Chunking under its fixed-R stopping rule.

The primary latency evaluator is the same as Experiment 15: released Orin AGX
lookup values through 255 KiB and endpoint-proportional, constant-throughput
scaling afterwards. Lookup supported also uses this exact function for mask
selection, eliminating the two-line-selection versus lookup-evaluation mismatch.

The adaptive lambda methods are exact for their respective supported
scalarized frontiers, but they miss unsupported points. Exact coverage solves
the original two-line problem `min L(M) subject to I(M) >= Q` and supplies the
missing global-optimality certificate. It is not an exact optimizer for the
released lookup evaluator.

Run:

```bash
python3 experiments/16_saturation_global_chain/run_experiment.py
```

The exact oracle uses four workers by default; change this with
`--exact-workers`. No label cap, beam, importance binning, or epsilon-dominance
is used.

Outputs:

- `paired_trials.csv`: 378 matched five-method comparisons and exact-DP stats;
- `input_trials.csv`: supported-frontier sizes and per-input exact-DP totals;
- `chunk_lengths.csv`: one row per true maximal selected run;
- `summary.json`: primary lookup results and tail sensitivity;
- `importance_latency.{png,pdf}`: released-lookup I-L curves;
- `importance_rows.{png,pdf}`: I-R curves;
- `latency_rows.{png,pdf}`: released-lookup L-R curves; and
- `chunk_length_histogram.{png,pdf}`: true run-length distributions.

See `report.md` for the results and optimality scope.
