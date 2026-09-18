# Experiment 27 algorithm specification

## What changes

Only two integer arguments to Experiment 24's existing `fixed_r_ratio()` are
changed:

- `rho_iterations`: the number of outer ratio updates;
- `mu_calls`: the maximum number of row-price bisection calls per outer update.

No new trim, frontier retention, stopping rule, or Paper-derived target is
introduced.

## Unchanged objective and solver

For nonnegative activation importance `v`, exact row budget `R`, and two-line
run cost `L`, the objective remains

\[
\max_{|M|=R}\frac{I(M)}{L(M)}.
\]

At ratio `rho` and row price `mu`, the existing exact scalarized DP solves

\[
\max_M\{I(M)-\rho L(M)-\mu |M|\}.
\]

Experiment 24 bisects `mu`, retains its closest supported over-`R` solution,
repairs that solution to exactly `R` rows with its existing fixed-rho endpoint
heap, and keeps Top-R whenever the repaired candidate does not improve the
ratio. The returned ratio seeds the next outer iteration. Experiment 27 uses
this procedure verbatim.

The tested `(rho_iterations, mu_calls)` pairs are `(2,4)`, `(2,8)`, `(2,16)`,
`(3,8)`, `(3,16)`, `(4,16)`, and `(4,32)`. Their maximum scalarized-DP call
counts are therefore 8, 16, 32, 24, 48, 64, and 128.

## Evaluation contract

The real activation traces and paired-cardinality protocol are identical to
Experiment 24. Selector timing includes selection, repair, mask
materialization, and—on the CUDA track—D2H/H2D transfer and synchronization.
LM forward time, model compute, and physical storage I/O are excluded.

The small-N exhaustive oracle is synthetic and reported separately from the
real-activation production-size results.
