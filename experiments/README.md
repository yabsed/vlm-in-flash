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

Future experiments should use the next numbered directory.
