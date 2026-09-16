# Experiment 10 report: Experiment 07 under the released lookup table

## Question

Experiment 07 optimized and compared masks with the fitted affine latency

\[
L_{\mathrm{aff}}(M)=aK(M)+cR(M).
\]

This experiment asks how its four methods look when the same selected masks
are evaluated with the released Orin AGX table instead:

\[
L_{\mathrm{lookup}}(M)=\sum_{C\in\mathcal C(M)}T(|C|s),
\]

where `C` is a maximal selected chunk and `s=1.75 KiB` is one matrix row.

The compared methods are affine DP-Cover, Paper greedy, exact fixed-R DP
(`DP-R`), and Top-R. Single interval, Quantized Pareto, and the Experiment 08
refinement are intentionally excluded so this remains a direct recheck of
Experiment 07.

## Reusing Experiment 07 without rebuilding its O(N^3) frontier

Experiment 07 saved the exact DP-Cover terminal `(R, K, importance)` for every
query but did not save masks. A lookup latency cannot be recovered from only
`R` and `K`, because different chunk-length compositions can have different
table costs.

For each of the 63 original input vectors, Experiment 10 therefore:

1. deterministically regenerates the same importance values;
2. reads the required exact `(R,K,I)` states from Experiment 07's CSVs;
3. runs an exact chain DP only through the maximum required `R` and `K`;
4. backtracks every saved terminal state and checks its importance, `R`, and
   `K` against Experiment 07; and
5. stores every selected interval as zero-based `start:length` RLE.

The reconstruction costs `O(N R K_max)`, not `O(N^3)`. Here `K_max=20`.
It took a mean of `0.280 s` per input (median `0.235 s`) and at most
`0.464 GiB`. Thus the entire exact mask reconstruction was much cheaper than
rebuilding the original full frontier.

All 441 coverage masks and all matched DP-Cover masks passed the saved
importance/row/chunk checks. The regenerated Paper, DP-R, and Top-R masks also
matched their Experiment 07 metrics. The stored RLEs were independently
checked to reproduce every row and chunk count.

## Important interpretation rule

DP-Cover is exact for `aK+cR`, but it is **not** exact for the lookup table.
The comparisons below therefore use

`method lookup latency / lookup latency of the affine-optimal DP-Cover mask`

at the same retained importance. A ratio below one does not beat a true
lookup optimum; it reveals disagreement between the affine model and lookup
table.

## Main result

The following aggregates use the six VLM CVs (`1.07` through `4.55`) and 378
matched fixed-R cases. CV `9.19` remains in the CSV as a stress case.

| Method | Affine mean ratio | Lookup mean ratio | Lookup median | Lookup 5th--95th percentile |
| --- | ---: | ---: | ---: | ---: |
| Paper greedy | 1.300 | **1.053** | 1.038 | 0.943--1.212 |
| DP-R exact | 1.016 | **1.048** | 1.000 | 1.000--1.220 |
| Top-R | 8.602 | **6.604** | 6.295 | 2.211--11.595 |

The affine model made Paper appear 30.0% more expensive than matched
DP-Cover on average. Under the lookup table, that difference shrinks to 5.35%.
Paper even has lower lookup latency than the affine-optimal DP-Cover mask in
25.4% of cases. This is not a contradiction: the two masks can share similar
`R` and `K` while having different chunk lengths, which the affine model
deliberately discards.

DP-R remains the closest method overall. Its median lookup ratio is exactly
1.0 and it has the same lookup latency as matched DP-Cover in 51.9% of cases.
Its mean rises to 1.048 because the locally clustered inputs contain a smaller
number of substantial disagreements.

Top-R remains clearly unsuitable for the latency objective. Lookup-table
nonlinearity reduces its apparent penalty from 8.60x to 6.60x on average, but
it is still fragmented into hundreds of chunks and never approaches the other
methods.

## Spatial-ordering breakdown

| Ordering | Paper mean lookup ratio | Paper below affine DP-Cover | DP-R mean lookup ratio | Top-R mean lookup ratio |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.984 | 65.1% | 1.020 | 8.243 |
| Locally clustered | 1.098 | 7.9% | 1.121 | 3.262 |
| Persistent hot-cold | 1.078 | 3.2% | 1.002 | 8.308 |

Random ordering is the clearest warning against treating affine DP-Cover as a
lookup oracle: Paper's mask is cheaper under the table in almost two thirds of
these cases. Local clustering is the opposite regime; DP-R's affine-optimal
few-chunk layout is not always the best composition of chunk lengths for the
table.

## Figures and reusable data

- `importance_lookup_frontiers.pdf` evaluates the four Experiment 07 curves
  with released table latency.
- `matched_lookup_ratio.pdf` shows ordering- and budget-dependent ratios to
  the matched affine-DP-Cover masks.
- `affine_lookup_comparison.pdf` contrasts the original affine ratios with the
  lookup re-evaluation.
- `coverage_lookup_trials.csv` and `fixed_r_lookup_trials.csv` contain lookup
  latency and selected-run RLE for every mask.
- `reconstruction_trials.csv` records reconstruction time and memory.

## Conclusion

Experiment 07's qualitative conclusions survive only in part. Top-R is still
far too fragmented, and DP-R remains close to the affine DP-Cover frontier.
However, Paper's reported 30% matched-latency gap is mostly an artifact of the
affine approximation: it is only 5.35% under the released table, and the
affine-exact mask is sometimes slower there.

Consequently, these results should be described as a **lookup sensitivity
analysis**, not lookup optimality. A definitive lookup-table comparison needs
a new exact solver whose objective directly sums `T(chunk length)`.

