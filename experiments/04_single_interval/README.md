# Experiment 04: O(N) single-interval solver

This experiment tests the simplest solver for the newly defined coverage
problem:

```text
minimize latency L = aK + cR subject to retained importance I >= alpha.
```

For nonnegative importance, the best one-chunk solution is the shortest
interval reaching the target. A two-pointer scan finds it exactly in `O(N)`
time and `O(1)` auxiliary space.

The experiment compares three methods on the CV-calibrated inputs from
Experiments 02–03:

- Paper greedy;
- single interval `O(N)`;
- Exact Coverage DP.

It also records the exact optimum's number of chunks for every CV, spatial
ordering, and coverage target. The output figures are:

- `importance_latency_frontiers.{png,pdf}`;
- `r_importance.{png,pdf}`;
- `r_latency.{png,pdf}`;
- `optimal_chunk_distribution.{png,pdf}`.

Run from the repository root:

```bash
python3 experiments/04_single_interval/run_experiment.py
```

The completed default run and interpretation are in [report.md](report.md).
