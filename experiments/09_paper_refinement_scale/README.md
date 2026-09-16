# Experiment 09: Paper versus merge-and-trim at realistic scales

This is a dedicated paired comparison of Paper greedy and the
coverage-preserving `Paper + merge + trim` refinement from Experiment 08.

The default run covers four realistic down-projection sizes, seven CV values,
three spatial orderings, seven Paper row budgets, and twenty independent
multisets per CV. This produces 11,760 paired cases, of which 10,080 use the
six VLM CV values and form the primary analysis.

Both fitted affine latency and the released Orin AGX lookup-table estimate are
recorded. Confidence intervals use a cluster bootstrap over independently
generated value multisets; the three orderings and seven budgets derived from
one multiset remain in the same resampled cluster.

Run from the repository root:

```bash
python3 experiments/09_paper_refinement_scale/run_experiment.py
```

Results are written both to the aggregate `results/` directory and to
`results/n_<N>/` for each matrix size.

Aggregate figures:

- `comparison_overview.{png,pdf}`
- `latency_ratio_ecdf.{png,pdf}`
- `lookup_ratio_ecdf.{png,pdf}`
- `saving_heatmaps.{png,pdf}`

Raw paired observations are in `paired_trials.csv`; `grouped_summary.csv` and
`summary.json` contain the primary aggregations.
