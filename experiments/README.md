# Experiments

This directory contains the project's reproducible experiments. Background
reading, paper notes, and preliminary derivations live separately in
`preliminary_research/`.

1. `01_random_r_bound`: baseline random-input comparison of Paper greedy,
   Top-`R`, exact fixed-`R`, exact coverage, and quantized-Pareto methods.
2. `02_cv_sweep`: paper-calibrated CV and spatial-ordering sweep comparing
   Paper greedy, exact fixed-`R`, Exact Coverage, and Quantized Pareto.
3. `03_quant_tradeoff`: Quantized Pareto `q` sweep measuring the
   computation–accuracy trade-off against Exact Coverage and Paper greedy.
4. `04_single_interval`: `O(N)` single-interval coverage solver versus Paper
   greedy and Exact Coverage, including the exact optimum's chunk distribution.
5. `05_realistic_n`: model-scale down-projection channel counts comparing
   Paper greedy, single interval, and Quantized Pareto `q=1024` without the
   unscalable Exact Coverage oracle.
6. `06_quant_scale`: realistic-N Top-R comparison, true chunk-count
   histograms, and a large-q sweep testing how much quantization is required
   to beat Paper greedy reliably.
7. `07_exact_n4864`: exact `O(N^3)` DP-Cover with `O(N^2)` memory at the
   realistic `N=4,864`, compared with Paper greedy, exact DP-R, and Top-R.
8. `08_greedy_refinement`: reuses Experiment 07's exact results and adds a
   coverage-preserving gap-merge and boundary-trim refinement to Paper greedy.
9. `09_paper_refinement_scale`: large paired Paper-versus-refinement benchmark
   across four realistic channel counts, with clustered confidence intervals
   and affine/lookup latency sensitivity analysis.
10. `10_exact_lookup_recheck`: reconstructs Experiment 07's affine-exact masks
    from its saved states and re-evaluates DP-Cover, Paper, DP-R, and Top-R
    with the released latency lookup table without rebuilding the `O(N^3)`
    frontier.
11. `11_affine_lookup_gap`: decomposes why the Experiment 07 Paper gap shrinks
    under Experiment 10's lookup evaluation, quantifies out-of-profile chunk
    lengths, and tests sensitivity to alternative tail extrapolations without
    rerunning any selector or DP.

Future experiments should use the next numbered directory.
