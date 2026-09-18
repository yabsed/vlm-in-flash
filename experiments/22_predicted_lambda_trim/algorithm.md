# Executable specification

## Real-model trace rule

Load a causal language model and attach the checked-out `vlm-flash` submodule
to its default decoder projection set. For every wrapped `nn.Linear`, compute
the same online importance as `SparseLinear._select`:

\[
v_i=\frac{1}{BT}\sum_{b,t}|x_{b,t,i}|.
\]

The capture policy returns an all-true mask. The forward is therefore dense,
and every recorded layer sees activations from the original model rather than
activations perturbed by Paper or the proposed policy. Store the raw float32
vector together with model, prompt hash, token count, module name, call index,
and `(in_features, out_features)`.

Normalize each captured vector to sum to one only when replaying it through
the selector benchmark. No random spatial permutation, prescribed CV, or
synthetic distribution is introduced. When a shape has more traces than
`--max-traces-per-shape`, retain evenly spaced calls in deterministic capture
order. A zero cap keeps all calls.

Only captured shapes that have the paper's Table-2 candidate-window
hyperparameters are eligible. By default the selected model is
`Qwen/Qwen2.5-0.5B-Instruct`, whose decoder projections supply the 0.5B
Table-2 shapes.

## Paired comparison

For every captured projection trace and row-budget fraction, run the paper's
original fixed-\(R\) greedy selector and set

\[
Q=I(M_{\mathrm{Paper}}).
\]

The proposed methods are evaluated on the identical float32 trace and against
this same target without imposing a row cap. Pair results by trace id, row
budget, and timing track. This compares lookup latency at no lower retained
importance rather than conflating selector quality with different coverage.

## Predicted-lambda TD-2L(8)

1. Sort importance values and find the threshold value \(\tau_Q\) whose
   upper tail first reaches \(Q\).
2. Predict \(\lambda_0=c_2/\tau_Q\).
3. Solve the fixed-\(\lambda\) two-line scalarized problem exactly in \(O(N)\).
4. Maintain infeasible and feasible endpoints and query their score-line
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
preserves coverage and strictly reduces the two-line objective. Measurement
noise in the released lookup table means this guarantee need not hold for
every individual lookup-table step.

## Timing contract

Trace generation is an untimed setup phase. Compilation and first-touch
effects are warmed out. Selection, reconstruction, trim, fallback/validation,
output materialization, and track-specific transfer and synchronization costs
are included. The dense model forward used to produce importance, real NVMe
I/O, sparse projection compute, and task-level quality evaluation are
excluded.
