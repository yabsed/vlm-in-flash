# Algorithm

Let `b` be `floor(s/8)`, `round(s/8)`, or `ceil(s/8)` rows, and let a tile
contain `L=8b` rows.  Candidate tile starts are `0,b,2b,...`; therefore tile
candidate `j` is compatible with candidate `j-8` and earlier candidates.

Prefix sums produce every tile importance in linear time.  The exact-K
maximum-weight independent set on this fixed-length interval graph is

```text
D[k,j] = max(D[k,j-1], D[k-1,j-8] + w[j]).
```

The DP work and memory are `O(BK)`, where `B` is the number of cells and `K`
the number of tiles.  There is no candidate sort.

For an exact row budget R, the implementation constructs two masks:

1. select `floor(R/L)` full tiles and greedily add at most `L-1` run-end rows;
2. select `ceil(R/L)` full tiles and greedily remove at most `L-1` run-end rows.

Every endpoint decision evaluates the laptop lookup table after accounting
for extension or merging of adjacent runs.  The higher lookup-I/L exact-R mask
is returned.  If `L>N` or `R<L`, the method becomes the maximum-importance
contiguous window of exactly R rows.

Endpoint repair adds at most `L-1` operations.  The current direct
implementation costs `O(LK^2)` for expansion and `O(LK)` for trimming; these
terms are small for the measured shapes but are included in selector timing.
