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
12. `12_note`: working notes that replace the single affine model with a
    saturation-aware two-line cost and derive the resulting algorithms.
13. `13_two_line_lagrangian`: fits that two-line model and evaluates a dense
    `q=131,072` lambda-grid `O(qN)` coverage heuristic against Experiment 11's
    Paper, affine-DP, and lookup-sensitivity results.
14. `14_dense_lookup_validation`: repeats the high-q Quant versus Paper lookup
    comparison with 50 trials per paper CV, 19 fixed-R budgets, and paired
    cluster-bootstrap confidence intervals under the constant-throughput tail.
15. `15_coverage_stopping_greedy`: keeps Paper greedy's candidates and utility
    ordering but replaces its fixed-R stop with cumulative-importance coverage,
    reusing Experiment 14's expensive high-q results.
16. `16_saturation_global_chain`: adds an untruncated exact constrained-coverage
    Pareto-label DP for the two-line model and an exact fixed-lambda `O(Nm)`
    released-lookup optimizer, then compares the true coverage optimum with
    both supported frontiers, `q=131,072` Quant, and Paper greedy.

Future experiments should use the next numbered directory.
