# Experiment 16: Direct lookup supported optimizer

This experiment compares four selectors at Paper-greedy achieved-importance
targets:

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

Both adaptive methods are exact for their respective supported scalarized
frontiers. They are not exact constrained-coverage oracles because unsupported
Pareto points remain outside any weighted-sum method.

Run:

```bash
python3 experiments/16_saturation_global_chain/run_experiment.py
```

Outputs:

- `paired_trials.csv`: 378 matched method comparisons;
- `input_trials.csv`: supported-frontier size and build runtime;
- `chunk_lengths.csv`: one row per true maximal selected run;
- `summary.json`: primary lookup results and tail sensitivity;
- `importance_latency.{png,pdf}`: released-lookup I-L curves;
- `importance_rows.{png,pdf}`: I-R curves;
- `latency_rows.{png,pdf}`: released-lookup L-R curves; and
- `chunk_length_histogram.{png,pdf}`: true run-length distributions.

See `report.md` for the results and optimality scope.
