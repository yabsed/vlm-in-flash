# Experiment 20 algorithm specification

## Common contract

The input is a nonnegative importance vector `v`, an externally supplied
coverage target `Q`, and, in the row-budget track, a row cap `R`. The output is
a materialized CPU boolean mask. Coverage requires

```text
sum(v[mask]) >= Q - 1e-11.
```

The row-budget track additionally requires `mask.sum() <= R`. A coverage
shortfall invokes the same top-importance fallback used in Experiment 18. The
fallback may use every unselected row in coverage-only, but no row beyond `R`
in the row-budget track. Exceptions, fallback, validation, and CPU mask
handoff stay inside the timed region and remain in all denominators.

`Q` is an external query input. Importance generation, JIT compilation,
latency-model fitting, actual I/O, and model compute are outside the selector
timer.

## References

- **Paper:** exact Paper utility sorting and overlap rejection. Coverage-only
  stops at `Q`; row-budget uses the repository's native fixed-row selector.
- **Incumbent:** Experiment 18's `TD-2L (16) + endpoint trim 256`.

## Small lambda bank

Let `c2` be the saturated two-line marginal cost. Sort importance in descending
order and let `tau_Q` be the last importance value needed by the independent
linear-cost relaxation to reach `Q`. The center prediction is

```text
lambda_0 = c2 / tau_Q.
```

Each bank includes `lambda=0`, a deterministic full-selection upper bound,
and powers-of-two around `lambda_0`. The 4-point bank is nested in the 8-point
bank, which is nested in the 16-point bank. All points are solved independently
with the exact O(N) two-line scalarized solver. The cheapest feasible result
under the two-line objective is replayed once to recover its mask.

Configurations: 4, 8, and 16 metric solves.

## Piecewise-linear lookup DP

The released integer run-cost curve is divided into `J-1` geometrically spaced
pre-tail ranges. Least squares fits one affine line

```text
T_j(l) = a_j + c_j l,  lo_j <= l <= hi_j
```

in each range. The final range is the released lookup evaluator's exact
proportional tail.

For fixed lambda, the prefix recurrence considers a chunk `[i,k]`:

```text
DP[k+1] = max(
    DP[k],
    DP[i-1] + lambda * (P[k+1]-P[i]) - a_j - c_j*(k-i+1)
).
```

For each segment, eligible starts form a sliding length interval. A monotone
deque maintains

```text
DP[i-1] - lambda*P[i] + c_j*i,
```

so one scalarized solve costs `O(JN)` time and `O(JN)` temporary deque space.
Target-directed endpoint intersections select the feasible mask. The
implementation has a brute-force fixed-lambda self-check on a small problem.

Configurations: `(J,calls) = (4,4), (4,8), (8,4), (8,8)`.

This optimizes the fitted piecewise surrogate. All returned masks are evaluated
with the original released lookup table.

## Paper plus local edits

Start from the Paper mask and apply at most `E` accepted operations. Every
operation uses the released lookup run cost directly.

1. Merge the profitable internal gap with the largest positive saving, subject
   to the row cap when present.
2. If no profitable merge exists, remove the exposed endpoint with the lowest
   importance loss per saved millisecond, provided coverage remains feasible.
3. If neither applies, shift a run boundary by one row when run length and
   lookup cost stay fixed and importance strictly increases.

The loop repeats because a merge, trim, or shift may expose another operation.
Accepted operations never increase lookup latency; trims and shifts preserve
coverage, and merges only add nonnegative importance. Row-budget merges are
rejected when they would exceed `R`.

Configurations: at most 64 or 256 accepted edits.

## Target-directed lookup DP

Experiment 16's exact fixed-lambda released-lookup solver maintains the current
run length capped at the lookup tail boundary. One solve costs `O(Nm)`, where
`m` is the number of integer run lengths before the proportional tail.

As in target-directed two-line search, one infeasible and one feasible endpoint
are maintained. Their line intersection supplies the next lambda. The final
feasible mask is recovered by one parent-tracking lookup-DP pass.

Configurations: 2, 4, and 8 metric solves. The two-call configuration consists
only of the empty/full endpoint solves and is intentionally retained as a
lower-bound runtime control.

## Lambda prediction plus correction

The first lambda is the independent linear-relaxation estimate
`c2/tau_Q`. Empty and full solutions provide analytic bracket endpoints without
spending scalarized calls. The observed solution replaces the corresponding
endpoint, after which two-line intersections refine only the target bracket.
Mask replay is included in runtime but not in the metric-solve count.

Configurations: 2, 4, and 8 metric solves.

This is a per-query analytic predictor, not a trained model. Consequently it
does not hide calibration inputs or training time.

## Capped-label lookup DP

The exact lookup chain state is the current selected-run length, capped at `m`.
Instead of retaining one scalarized optimum, every state retains at most `K`
target-directed labels. Label slot `r` asks for the cheapest candidate reaching

```text
Q * r / (K-1),  r=0,...,K-1.
```

If a threshold is unreachable in the current prefix, the highest-importance
candidate is retained. Transitions either skip the current row, start a run,
continue a short run, cross the lookup boundary, or continue in the proportional
tail. The row-budget track rejects transitions beyond `R`. Parents are stored
for exact replay of the pruned label graph.

Configurations: 2, 4, and 8 labels per state.

This method can retain unsupported constrained solutions, but the cap makes it
heuristic: it is not the full Pareto-label DP and has no exactness claim.

## Measurement and decision rule

Defaults match Experiment 18: `N=4,864`, exact `CV=3.30`, three spatial
orderings, three trials, three `(Q,R)` scenarios, and 30 warmed repetitions.
Candidate construction, all solver passes, mask recovery, local edits,
fallback, validation, and CPU handoff are timed. A method-case passes only if
its p95 is at most 2 ms and its output is valid for its track. No failed case is
removed from an average or pass-rate denominator.

