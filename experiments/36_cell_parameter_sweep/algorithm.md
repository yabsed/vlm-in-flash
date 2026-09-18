# Algorithm

For a measured saturation length `s` and parameter `C`, define

```text
cell_rows = ceil(s / C)
tile_rows = C * cell_rows.
```

Candidate tiles begin only on cell boundaries. An exact-K weighted-interval DP
selects non-overlapping saturation-length tiles, after which endpoint expansion
or trimming returns exactly the requested row budget R. The DP predecessor is
`C` candidate positions behind the current tile, which is the generalized form
of Cell-8's hard-coded offset of eight.

The experiment measures

```text
(T_C(R), E_C(R))
```

for every C and R. It then eliminates R and C jointly:

```text
V(epsilon) = min_{C,R} T_C(R)  subject to E_C(R) <= epsilon.
```

Thus the reported winner is a quality-constrained Cell-C frontier, not the
fastest method at one arbitrarily fixed R.

Measured points remain the primary result. Since R is sampled every five
percentage points, the analysis also linearly interpolates each Cell-C curve at
the same error ceilings. This is explicitly a grid-gap diagnostic and not an
additional hardware measurement.
