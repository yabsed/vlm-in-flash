# Experiment 17: importance-latency curves across N

This experiment compares two selectors:

- **Paper greedy** with a sweep of fixed row budgets;
- **Supported** with the complete strongly-supported frontier and common
  retained-importance targets.

Results are separated by latency model:

```text
results/
  two_line/
  lookup/
```

Within each directory, Paper candidate utilities and supported optimization
use the same chunk-cost model. `N` changes while FP16 row size and Paper's
physical-KiB window parameters stay fixed.

Default selectable lengths:

```text
N = 256, 512, 1,024, 2,048, 4,096, 4,864, 8,192
```

For every `N`, three exact-CV (`CV=3.30`) lognormal multisets are evaluated
under random, locally clustered, and persistent hot-cold orderings.

Paper is evaluated at 19 row-budget fractions from 0.05 to 0.95. Supported is
queried at the same common retained-importance fractions plus 0.99. The main
figure uses predicted I/O latency on the x-axis and retained importance on the
y-axis, averaged over all nine inputs per `N`; a second figure shows the
complete supported frontier for one deterministic example.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python3 experiments/17_online_selection_runtime/run_experiment.py
```

Outputs in each latency-model directory:

- `curve_trials.csv`: Paper and common-target supported curve points;
- `frontier_nodes.csv`: every strongly-supported point;
- `paper_matched_trials.csv`: supported cost at each Paper-attained importance;
- `input_trials.csv`: supported frontier sizes and solve counts;
- `summary.json`;
- `importance_latency.{png,pdf}`: average curves by `N`;
- `importance_latency_examples.{png,pdf}`: complete example frontiers;
- `paper_matched_saving.{png,pdf}`.

See `report.md` for aggregate results. These are model-predicted I/O latency
curves, not selector wall-clock measurements.
