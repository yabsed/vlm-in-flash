# Experiment 26 algorithm specification

## Objective

For a nonnegative importance vector `v`, exact row count `R`, and two-line run
cost `L`, optimize

\[
\max_{|M|=R}\frac{I(M)}{L(M)}.
\]

No Paper-derived importance target is used.

## Row-price frontier

At the current ratio `rho`, a row price `mu` gives the unconstrained problem

\[
\max_M\{I(M)-\rho L(M)-\mu |M|\}.
\]

Replacing each `v_i` by `v_i-mu` reduces this to the existing exact `O(N)`
two-line scalarized DP. Bisection produces a deterministic sequence of at most
16 or 24 prices. Unlike Experiment 24, every distinct supported cardinality in
the allowed overfill range is reconstructed and evaluated. Solutions with the
same cardinality need only be reconstructed once: the `-mu|M|` term is a
constant among masks of that cardinality.

## Adaptive exact-R trim

For every supported mask with

\[
R\le |M|\le R+B,
\]

where `B` is 64 or 256, trim endpoints until exactly `R` rows remain. Given
current importance `I` and cost `L`, deleting endpoint `e` saves `Delta L_e`.
The selected deletion maximizes the actual next ratio

\[
\frac{I-v_e}{L-\Delta L_e}.
\]

The ratio is therefore updated after every deletion rather than frozen at the
outer Dinkelbach value. Run boundaries and exact two-line cost changes are
maintained in a Numba kernel.

Top-R is always included as a valid exact-R candidate. After evaluating every
eligible frontier mask, the candidate with the highest measured two-line
`I/L` is retained. The winning ratio seeds the next outer iteration. The three
tested settings are `(mu,rho,B)=(16,2,64)`, `(16,2,256)`, and `(24,3,256)`.

## Evaluation

The same real activation traces and paired cardinality protocol as Experiment
24 are used. Selector time includes selection, every frontier solve and trim,
mask materialization, and, on the CUDA track, D2H/H2D transfer and
synchronization. The real-model trace forward, model compute, and physical
storage I/O are excluded.

The small-N oracle exhaustively enumerates all masks for `N=18`. Its synthetic
inputs are an algorithmic check and are kept separate from the production-N
real-activation results.

