# Experiment 18 algorithm specification

## Common contract

Input:

- nonnegative importance vector `v[0:N]`, normalized to sum to one;
- external coverage target `Q`;
- optional row cap `R` in the row-budget track;
- the fitted two-line model and released latency lookup table.

Output is a materialized CPU boolean mask. Coverage is successful when
`sum(v[mask]) >= Q - 1e-11`. A row-budget result is valid only when coverage
is successful and `mask.sum() <= R`.

If a method returns a coverage shortfall, the common fallback appends the
highest-importance unselected rows. Coverage-only can use all remaining rows;
the row-budget track can add only up to `R`. The fallback, validation, and
final CPU mask copy are inside the timed region. Exceptions use the same
fallback starting from an empty mask and remain recorded in the CSV.

## Target-directed two-line DP

The exact scalarized solver maximizes

```text
lambda * importance(mask) - two_line_cost(mask)
```

in `O(N)`. Target-directed search keeps one infeasible envelope point `A`
and one feasible point `B`. Its next multiplier is their intersection:

```text
lambda = (cost(B) - cost(A)) / (importance(B) - importance(A)).
```

The returned point replaces `A` or `B` according to whether it reaches `Q`.
Search stops at the configured metric-solve limit or when the solver returns
an endpoint again. Calls at `lambda=0` and at the deterministic full-mask
upper bound count toward the limit. Final mask recovery is one additional
`O(N)` pass and is included in wall time but not in the reported
`scalarized_calls` count.

If the upper endpoint unexpectedly misses `Q`, its multiplier is doubled
within the same call budget. If it still misses, the full mask is returned and
the common fallback/validation path records the outcome.

Configurations: 4, 8, 16, and 32 metric solves.

## Endpoint trim

Trim starts from the feasible target-directed mask. Only the two exposed
endpoints of each selected run are candidates. For run length `L`, removing
one endpoint saves

```text
two_line_cost(L) - two_line_cost(L - 1).
```

A heap orders endpoints by lost importance per saved millisecond. A deletion
is accepted only if its importance does not exceed the current surplus above
`Q`. The newly exposed endpoint is then inserted. The procedure stops at an
empty heap or the deletion limit and verifies coverage before returning.

Configurations are paired rather than crossed:

- 8 scalarized calls and at most 64 deletions;
- 16 scalarized calls and at most 256 deletions.

## Saturation tiles

For tile length `L`, the method constructs two complete, non-overlapping
partitions of the neuron axis:

- grid origin `0`;
- grid origin `floor(L/2)`, with clipped prefix and suffix tiles.

Each partition therefore covers every row, including its edges. Tiles are
ordered by `importance(tile) / two_line_cost(tile)` and selected until `Q` is
reached. Both offsets are run inside one timed query; the feasible result with
the smaller two-line cost is returned. Ties prefer smaller overshoot.

Configurations use `L = round(s/2)`, `round(s)`, and `round(2s)`, where `s`
is the fitted saturation length in rows.

## Paper-bucket

Candidate windows, strides, lookup-table costs, overlap rejection, and the
row-budget behavior follow Paper. Only the global utility ordering changes.
For candidate utility `u`, the stable bucket index is

```text
floor((u - u_min) / (u_max - u_min) * (B - 1)).
```

Candidates are visited from the highest bucket to the lowest and retain their
generation order within a bucket. A compiled counting pass makes generation
plus ordering `O(M+B)` for `M` candidates. Coverage-only stops after the first
accepted window that reaches `Q`; the row-budget track uses Paper's original
fixed-`R` acceptance rule and is checked against `Q` afterwards.

Configurations: `B = 64`, `256`, and `1,024`.

## References and the fourteen configurations

The references are:

- Paper: exact global score sorting; coverage stopping in coverage-only and
  the repository's native fixed-row implementation in the row-budget track;
- Full supported: build the complete strongly-supported two-line frontier and
  replay the selected mask.

The four TD limits, two paired trim settings, three tile lengths, three bucket
counts, and two references make exactly fourteen configurations.

## Timing and pass criteria

Every query is repeated thirty times after JIT/native warm-up, matching the
repeat count in the paper's Appendix H runtime sweep. A method-case
passes the selector deadline only when its within-case p95 is at most 2 ms.
The reported overall pass rate retains all coverage failures, row-cap
failures, fallback cases, and deadline misses in the denominator.

The timed region includes candidate construction, scalarized passes, mask
recovery, postprocessing, fallback, validation path, and CPU mask handoff. It
excludes initial compilation, importance production, storage I/O, and model
compute. Consequently the current host results are a candidate screen, not a
Jetson 2 ms confirmation. The CSV separately reports predicted two-line and
released-lookup I/O latency so a later Jetson run can add measured selector,
I/O, upload, and compute time without changing the algorithm contract.
