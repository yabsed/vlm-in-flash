# Experiment 14: dense released-lookup validation

This experiment estimates the lookup-latency advantage of the Experiment 13
high-q Quant method more precisely.

Default design:

- `N=4,864`;
- six paper-reported VLM CV values;
- 50 independent trials per CV;
- random, locally clustered, and persistent hot-cold orderings;
- 19 fixed-R budgets from `0.05N` through `0.95N`;
- `q=131,072` Lagrange multipliers;
- paired Paper-target comparisons; and
- cluster-bootstrap confidence intervals with `(trial, CV)` as the independent
  resampling unit.

The primary evaluator is the released Orin AGX lookup table. Beyond the final
255-KiB table entry, latency is scaled linearly from that endpoint. This is the
constant-throughput-after-saturation assumption requested for this experiment.

Run the full experiment:

```bash
python3 experiments/14_dense_lookup_validation/run_experiment.py
```

Regenerate summaries and figures from the saved CSV:

```bash
python3 experiments/14_dense_lookup_validation/run_experiment.py --analyze-only
```

Outputs:

- `paired_trials.csv`: all 17,100 Paper-matched cases;
- `input_trials.csv`: all 900 spatial inputs and solver build times;
- `summary.json`: clustered confidence intervals and stratified results;
- `latency_saving_ci.{png,pdf}`: the primary saving-versus-budget result;
- `importance_latency_frontiers.{png,pdf}`;
- `r_importance.{png,pdf}`;
- `r_latency.{png,pdf}`;
- `coverage_opt_ratio.{png,pdf}`; and
- `chunk_count.{png,pdf}`.

See `report.md` for results and interpretation.
