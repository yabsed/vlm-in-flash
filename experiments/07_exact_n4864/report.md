# Experiment 07 report: exact DP at N=4,864

## Question

At the realistic `4,864 x 896` down-projection size, compare exactly four
methods:

- `DP-Cover exact`: minimize affine latency while retaining at least a target
  importance;
- `Paper greedy`: the paper's chunk-selection heuristic;
- `DP-R exact`: maximize importance/latency at a fixed row count `R`;
- `Top-R`: retain the `R` individually largest channels.

Single-interval and Quantized Pareto methods are not part of this experiment.

## Setup

- `N = 4,864`, FP16 rows of `896` values (`1.75 KiB/row`)
- Orin AGX latency table, approximated as
  `L(R,K) = 0.000244672 R + 0.0110009 K` ms (`R^2 = 0.9867`)
- CV values: `1.07, 1.25, 1.44, 2.48, 3.30, 4.55, 9.19`
- spatial orderings: random, locally clustered, hot-cold reordered
- three independent trials per CV and ordering
- seven fixed-R budgets and seven coverage targets per input
- 63 input vectors and 441 points per sweep

Unless stated otherwise, aggregate results below use the six VLM CVs
(`1.07` through `4.55`); `9.19` is retained in the CSVs as an OPT-style
stress case.

## Exact DP-Cover implementation

For each prefix, selected-row count `R`, chunk count `K`, and ending bit, the
DP stores the maximum attainable importance. The final scan chooses the
minimum `aK+cR` state meeting the requested importance bound.

The implementation updates two `(N+1) x (ceil(N/2)+1)` metric tables in place.
It therefore has exact `O(N^3)` time and `O(N^2)` memory, but deliberately does
not store backtracking parents or reconstruct a mask. It returns the exact
importance, `R`, `K`, and latency needed for all plots.

The rolling recurrence was checked against the previous full exact oracle at
`N = 8, 17, 32, 65`, including complete state tables and solutions at four
coverage targets.

## Main result

The most direct comparison is the latency required to attain each method's
own importance:

`coverage ratio = method latency / exact DP-Cover latency at the same importance`.

| Method | Mean ratio | Median | 95th percentile | Interpretation |
| --- | ---: | ---: | ---: | --- |
| DP-Cover exact | 1.000 | 1.000 | 1.000 | exact lower envelope |
| Paper greedy | 1.300 | 1.285 | 1.479 | 30.0% more latency on average |
| DP-R exact | 1.016 | 1.000 | 1.083 | nearly coverage-optimal |
| Top-R | 8.602 | 8.494 | 16.001 | high importance, severe fragmentation |

`DP-R` landed exactly on the DP-Cover latency in 51.9% of VLM-CV cases. The
ordering breakdown explains its remaining gap: its mean ratio was `1.0005`
for hot-cold, `1.0030` for random, and `1.0435` for local ordering. A fixed-R
optimum need not be the global coverage optimum because DP-Cover may use a
different `R` to attain the same importance.

Paper greedy was never exactly coverage-optimal in the tested cases. Its mean
coverage ratios were `1.253` for random, `1.284` for local, and `1.363` for
hot-cold ordering.

## Fixed-R view

At the same row-budget settings, exact `DP-R` improved importance per latency
over Paper greedy by:

- mean: `31.31%`
- median: `30.61%`
- 5th--95th percentile: `17.34%--49.67%`
- observed range: `7.59%--60.48%`

Top-R retained `25.11%` more importance than Paper greedy on average, but this
did not translate into good latency. It ignored spatial contiguity and opened
hundreds of chunks.

## Chunk counts

| Method | Mean K | Median K | 5th--95th percentile |
| --- | ---: | ---: | ---: |
| DP-Cover at Paper's importance | 4.72 | 3 | 1--13 |
| Paper greedy | 25.24 | 24 | 7--46 |
| DP-R exact | 2.98 | 3 | 1--7 |
| Top-R | 557.75 | 535 | 130--1,163 |

The dominant phenomenon is therefore not merely which rows are important,
but whether the selected rows can be represented by a few contiguous chunks.

## Computational cost

| Operation | Mean wall time |
| --- | ---: |
| Build exact DP-Cover table for one input | 14.13 s |
| Query an already-built DP-Cover table | about 10 ms |
| Paper greedy query | 7.66 ms |
| DP-R exact query | 321.42 ms |
| Top-R preprocessing | 0.345 ms |
| Top-R query | 0.034 ms |

The exact rolling buffers used `0.176 GiB`. Thus the `O(N^3)` exact computation
at `N=4,864` is practical as an offline benchmark on this machine: a table
takes roughly 14 seconds and can answer many targets cheaply. It is not a
competitive online selector, but it is fully usable as the ground-truth oracle
for experiments.

## Figures

- `importance_latency_frontiers.pdf`: the four methods in importance-latency
  space; DP-Cover is the exact lower envelope.
- `r_importance.pdf`: retained importance against selected rows.
- `r_latency.pdf`: latency against selected rows.
- `coverage_optimality_ratio.pdf`: latency gap to DP-Cover at matched
  importance.
- `chunk_count_histograms.pdf`: the full distribution of selected chunk counts.

## Conclusion

The experiment rejects the idea that the paper heuristic is already close to
optimal for the newly defined importance-latency problem. It is fast, but at
matched importance it spends about 30% more latency than the exact solution.
`DP-R` captures almost all of the exact coverage frontier while using only
about three chunks, whereas Top-R shows why maximizing importance alone is the
wrong objective under chunk startup cost.

Most importantly, exact `O(N^3)` DP-Cover is feasible at `N=4,864` once its
memory is reduced to `O(N^2)`: it provides the requested non-quantized,
non-single-interval ground truth directly.
