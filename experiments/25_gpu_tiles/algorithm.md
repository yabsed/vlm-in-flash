# Experiment 25 algorithm specification

## Contract

The input is a CUDA-resident nonnegative `float32` importance vector `v` from a
real LM projection and a coverage target `Q`. The trace is captured by a
`vlm-flash` all-true policy during a dense Qwen forward as
`mean(abs(projection_input))` over batch and token axes, then normalized in
float32. The output is a CUDA-resident boolean mask. The three tile lengths are

\[
L\in\{\operatorname{round}(s/2),\operatorname{round}(s),
       \operatorname{round}(2s)\},
\]

where `s = 236 KiB / row_size_KiB`, clipped to `[1,N]`.

## Static layout

For each `L`, two complete non-overlapping partitions are built at shape setup
time: origin `0` and origin `floor(L/2)`. The second partition includes clipped
prefix and suffix tiles. Tile boundaries, two-line tile costs, and the tile id
of every row are transferred to the GPU once and excluded from online timing.
They contain no importance-dependent information.

## Online CUDA selection

One GPU prefix sum supplies the importance of every tile. Within each grid,
tiles are stably sorted by

\[
\frac{I(\text{tile})}{T(|\text{tile}|)}.
\]

The first sorted prefix reaching `Q` is selected. A rank scatter followed by a
row-to-tile gather materializes the row mask without a CPU scalar readback.

Adjacent selected tiles merge into a single run. The implementation computes
the exact resulting two-line cost on the tile-selection vector using run-start
propagation and selects the cheaper of the two offset grids. A cost tie uses
smaller coverage overshoot, then origin `0`, matching Experiment 18.

All data-dependent operations remain CUDA tensor operations. There is no D2H
copy or CPU decision in the timed selector.

## Evaluation

The default cases are the supported Table-2 shapes present in Experiment 22's
real Qwen archive, deterministically capped at 32 projection calls per shape,
and the paired scenarios

\[
(Q,R/N)\in\{(0.50,0.25),(0.70,0.50),(0.90,0.75)\}.
\]

Paper's coverage-stopping mask is an untimed quality reference. Paper's
fixed-row mask supplies the row-budget validity baseline. The GPU mask is also
compared with Experiment 18's CPU Tiles mask. The original CPU implementation
is timed separately on the same values and target; this CPU measurement never
enters the GPU timed region.

Each case is warmed up and measured 30 times. Two clocks are recorded:

- CUDA events: device execution between stream events;
- synchronized wall clock: Python dispatch, CUDA execution, and final event
  synchronization.

The 2 ms pass condition is the within-case synchronized wall p95. Static layout
construction, the untimed real-model trace forward, Paper/reference
computation, initial CUDA context setup, storage I/O, and model compute are
excluded.
