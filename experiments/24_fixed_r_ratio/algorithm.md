# Executable specification

## Same-row comparison

The paper's selector can underfill its nominal row cap because it selects whole
windows. For every paired case, define

\[
R_{\mathrm{eff}}=|M_{\mathrm{Paper}}|.
\]

Top-R and all proposed masks contain exactly `R_eff` rows. The proposed method
uses only the scalar row count, not the paper mask or its importance values.

## Objective

For two-line run cost `T`, optimize

\[
\max_{|M|=R}\frac{I(M)}{L_{2L}(M)}.
\]

At ratio `rho`, the Dinkelbach inner objective is

\[
I(M)-\rho L_{2L}(M).
\]

Exact cardinality would require an `O(NR)` state dimension. The online
approximation instead introduces a row price:

\[
\max_M\left\{I(M)-\rho L_{2L}(M)-\mu|M|\right\}.
\]

Setting `lambda=1/rho` and replacing every importance by `v_i-mu` makes this
one call to the existing exact `O(N)` two-line scalarized DP.

## Row-price prediction and correction

The initial ratio is the Top-R mask's two-line `I/L`. Ignoring run starts gives
the row-price estimate

\[
\mu_0=\tau_R-\rho c_2,
\]

where `tau_R` is the R-th largest importance. Bounds that yield the full and
empty masks bracket `mu_0`; bisection then makes 4 or 8 exact scalarized calls.
The closest supported solution with at least R rows is reconstructed once.

## Exact-cardinality repair

If the supported mask has more than R rows, selected-run endpoints are peeled
in increasing order of the loss in the current Dinkelbach objective:

\[
v_e-\rho\left[T(\ell)-T(\ell-1)\right].
\]

The dynamic heap and run boundaries are compiled with Numba. Repair continues
until exactly R rows remain. The implementation also has a defensive Top-R
fill path for an under-budget mask, though it was unused in the full run.

The returned mask is the better of the repaired mask and the initial Top-R
mask under the two-line ratio. A second variant updates `rho=I/L` once and
repeats the four-call row-price search.

This repair is heuristic: scalarization may skip an unsupported exact-R mask,
and endpoint-only removal cannot represent every exact-cardinality optimum.

## Exact oracle and timing

For `N=18`, all `2^N` masks are enumerated to obtain the exact fixed-R
two-line ratio optimum at three row budgets. This validates feasibility and
quantifies approximation error without claiming an oracle at production N.

Compilation is warmed out. The CUDA track includes GPU-to-CPU importance
transfer, the CPU selector and repair, CPU-to-GPU mask transfer, and
synchronization. Actual storage I/O and model compute are excluded.
