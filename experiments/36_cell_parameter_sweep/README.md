# Experiment 36: Cell-C parameter sweep

Experiment 36 generalizes Cell-8 to Cell-C. A saturation-length tile is split
into `C` aligned cells, so its start position has a resolution of approximately
`s/C` rows. The experiment sweeps `C={1,2,4,8,16,32}` and `R=10%..95%` on the
same three cached language models and laptop hardware as Experiment 35.

Larger C offers finer placement and can retain more activation importance, but
it increases the selector search space. All reported latency includes selector,
native O_DIRECT/upload, activation gather, and compact GEMM. End-to-end sparse
forwards provide dense-to-sparse logit KL as a second quality measurement.

The primary comparison uses measured points. A separately labeled piecewise-
linear diagnostic compares C values at exactly the same error and checks whether
an apparent winner is only a gap in the 5%-spaced R grid.

```bash
python experiments/36_cell_parameter_sweep/run_experiment.py \
  --io-blob experiments/35_full_frontier/results_laptop/io_blob.dat \
  --output-dir experiments/36_cell_parameter_sweep/results_laptop \
  --report-output experiments/36_cell_parameter_sweep/report_laptop.md
```
