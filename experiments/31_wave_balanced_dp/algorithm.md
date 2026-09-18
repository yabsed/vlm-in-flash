# Algorithm

Let `A=512` bytes be the direct-I/O alignment, `b` the byte size of one row,
and `a=A/gcd(A,b)` the number of rows in an aligned cell.  Every run starts on
a cell boundary and contains a whole number of cells.

For an exact row budget `R` and run count `K`, write

```text
R/a = K*l + q,  0 <= q < K.
```

Exactly `q` runs have length `l+1` cells and the others have length `l` cells.
Runs are separated by at least one cell, so the native reader observes exactly
K runs rather than merging adjacent intervals.

With prefix importance `P`, `D[i,k,q]` is the maximum importance in the first
`i` cells using `k` runs, of which `q` are long.  Its transitions skip the last
cell, append a short run, or append a long run.  The predecessor prefix ends
one cell before the new run, enforcing separation.  This is an exact optimizer
within the K-run layout class and always returns exactly R rows.

The experiment measures K values 1, 6, 12, 18, and 24 independently.  The
held-out dispatcher chooses a fixed K per `(shape, budget)` either by minimum
training total time or maximum training `importance / total_time`.
