# Algorithm

For saturation length `s`, placement parameter `C`, and integer length factor
`a`, define

```text
cell_rows = ceil(s / C)
cells_per_tile = a C
tile_rows = a C cell_rows ≈ a s.
```

Candidate starts remain aligned to `cell_rows`; only the tile length changes.
The generalized exact-K interval DP therefore skips `aC` candidate positions
when taking a tile. Endpoint expansion or trimming returns exact R.

The final comparison eliminates both R and C and asks whether any `a>1` point
beats the best `a=1` curve at the same measured error. A separate piecewise-
linear diagnostic removes gaps due to the 5%-spaced R sweep.
