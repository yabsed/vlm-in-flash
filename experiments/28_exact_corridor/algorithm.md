# Algorithm

For importance vector `v`, exact row budget `R`, and runs `C` of a mask `M`,
Experiment 28 maximizes

```text
sum(i in M) v[i] / sum(C in runs(M)) T(|C|).
```

Dinkelbach iteration `q` changes the inner objective to

```text
sum(i in M) v[i] - q * sum(C) T(|C|), subject to |M| = R.
```

Let a prefix contain `r` selected rows and `z` zeros.  Its length is `r+z`.
Every state on diagonal `z` depends only on diagonal `z-1`: an off transition
keeps `r`, while a final run changes the predecessor count from `s` to `r`.
For affine `T(k)=alpha+beta*k`, the run transition is

```text
prefix[r+z] - q*alpha - q*beta*r
+ max_s(dp[z-1,s] - prefix[s+z] + q*beta*s).
```

The short affine range uses a sliding maximum and the saturated range uses a
prefix maximum.  This makes the full exact inner solve `O(R(N-R))`, rather
than scanning every possible run length at every state.

The online solver retains only states satisfying

```text
abs(r - center_prefix_selected[r+z]) <= W.
```

Each prefix participates in at most `2W+1` diagonals, so storage and work are
`O(NW)`.  The center is a one-price Lagrangian mask repaired to exact R.  A
new corridor is centered on each improved Dinkelbach mask.  Finite W is an
approximation to the full exact state space, but every returned mask is exact
R and its objective cannot be below its center mask.
