# Experiment 35: full quality-latency frontier

Experiment 35 removes retained-row ratio `R` from the comparison axis. Paper
and Cell-8 are both measured from 10% through 95% in 5-point increments; `R`
only parameterizes each method's curve. The final comparison asks for the
minimum measured latency at a given measured-error ceiling. A piecewise-linear
diagnostic checks whether an apparent adaptive win is merely a gap between the
measured fixed-R points; it is explicitly not reported as another measurement.

The Experiment 34 calibration-only adaptive activation strategy is plotted on
the same axes. This separates two claims:

- whether Cell-8's mask geometry beats Paper's algorithm;
- whether adaptive `R` beats the full fixed-`R` Cell-8 envelope.

Both projection relative-L2 and end-to-end KL figures invert the error axis, so
smaller errors such as 0.1 appear above larger errors such as 0.45.

```bash
fallocate -l 64M experiments/35_full_frontier/results_laptop/io_blob.dat
python experiments/35_full_frontier/run_experiment.py \
  --io-blob experiments/35_full_frontier/results_laptop/io_blob.dat \
  --output-dir experiments/35_full_frontier/results_laptop \
  --report-output experiments/35_full_frontier/report_laptop.md
```
