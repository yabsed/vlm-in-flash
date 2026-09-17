# Executable specification

## Paired comparison

For every importance vector and row-budget fraction, run the paper's original
fixed-\(R\) greedy selector and set

\[
Q=I(M_{\mathrm{Paper}}).
\]

The proposed methods are evaluated against this same target without imposing a
row cap. This compares lookup latency at no lower retained importance rather
than conflating selector quality with different coverage.

## Predicted-lambda TD-2L(8)

1. Sort importance values and find the threshold value \(\tau_Q\) whose
   upper tail first reaches \(Q\).
2. Predict \(\lambda_0=c_2/\tau_Q\).
3. Solve the fixed-\(\lambda\) two-line scalarized problem exactly in \(O(N)\).
4. Maintain the infeasible and feasible endpoints and query their score-line
   intersection. Stop after at most eight exact scalarized solves.
5. Reconstruct the final feasible endpoint's mask once.

The empty and full masks form the initial endpoints without DP calls.

## Endpoint trim

Starting from the feasible mask, put each selected run's exposed endpoints in
a heap. For a run of length \(\ell\), an endpoint is eligible only if

\[
v_e \le I(M)-Q,
\qquad
T(\ell)-T(\ell-1)>0.
\]

The heap key is

\[
\rho_e=\frac{v_e}{T(\ell)-T(\ell-1)}.
\]

Delete the lowest-ratio eligible endpoint, update the surplus, and expose the
new boundary. The experiment tests limits 64 and 256. Every accepted deletion
preserves coverage and strictly reduces the two-line objective. Because the
released lookup table contains measurement noise, this monotonic guarantee
does not extend to every individual lookup-table step.

## Timing contract

Compilation and first-touch effects are warmed out. Selection, reconstruction,
trim, fallback/validation, output materialization, and track-specific transfer
and synchronization costs are included. Importance production, real NVMe I/O,
and model compute are excluded.
