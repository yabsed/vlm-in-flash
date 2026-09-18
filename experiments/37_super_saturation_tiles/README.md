# Experiment 37: tiles longer than saturation

Experiment 37 separates placement resolution from tile length. For
`C={1,2,4,8}`, placement cells remain approximately `s/C` rows wide while the
selected tile length is swept over `L={s,2s,3s,4s}`. This directly tests
whether native-read saturation makes runs longer than `s` beneficial after
selector cost and measured output error are included.

Paper and every `(C,L)` pair are measured over `R=10%..95%` on the same three
cached models and laptop hardware as Experiments 35-36.

```bash
python experiments/37_super_saturation_tiles/run_experiment.py \
  --io-blob experiments/35_full_frontier/results_laptop/io_blob.dat \
  --output-dir experiments/37_super_saturation_tiles/results_laptop \
  --report-output experiments/37_super_saturation_tiles/report_laptop.md
```
