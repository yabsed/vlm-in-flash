# Experiment 16: Global supported frontier versus Quant and Paper

This experiment compares three selectors at Paper-greedy achieved-importance
targets:

- **Global supported**: adaptive enumeration of every strongly supported
  optimum of the scalarized two-line binary-chain objective;
- **Quant**: the existing fixed `q=131,072` lambda grid; and
- **Paper greedy**: Neuron Chunking under its fixed-R stopping rule.

The selection surrogate is the continuous two-line model. The primary latency
evaluator is the same as Experiment 15: released Orin AGX lookup values through
255 KiB and endpoint-proportional, constant-throughput scaling afterwards.

Global supported is exact for the supported scalarized frontier. It is not an
exact constrained-coverage oracle because unsupported Pareto points remain
outside any weighted-sum method.

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
