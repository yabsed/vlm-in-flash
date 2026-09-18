# Experiment 32: multi-model measured error

Experiment 32 evaluates Paper, Experiment 29 cell-8, and Experiment 31's
wave-balanced candidates on three real checkpoints and six calibration/holdout
prompt situations.  It measures the actual error of each selected mask rather
than treating retained activation importance as accuracy:

- projection relative L2, relative MAE, and cosine error against the dense
  projection using the checkpoint's real weights;
- held-out full-forward logit KL, relative L2, next-token top-1 agreement, and
  teacher-forced NLL change;
- selector latency plus native O_DIRECT read and GPU-upload latency on the
  laptop.

The sampled projection test uses early, middle, and late layers.  Dispatch is
fit only on calibration prompts and then frozen for held-out prompts.

```bash
dd if=/dev/urandom \
  of=experiments/32_multimodel_measured_error/results_laptop/io_blob.dat \
  bs=1M count=64
python experiments/32_multimodel_measured_error/run_experiment.py \
  --io-blob experiments/32_multimodel_measured_error/results_laptop/io_blob.dat \
  --output-dir experiments/32_multimodel_measured_error/results_laptop \
  --report-output experiments/32_multimodel_measured_error/report_laptop.md
```

The model checkpoints must already be present in the local Hugging Face cache.
