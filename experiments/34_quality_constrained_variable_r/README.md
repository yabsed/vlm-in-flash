# Experiment 34: quality-constrained variable R

Experiment 34 tests the claim that the retained-row ratio should be an output
of optimization rather than a fixed input.  It evaluates 18 candidate budgets
from 10% through 95% on three real checkpoints and separates calibration and
held-out prompts.

The deployable policies never use Paper's online importance or held-out error.
They predict a starting budget from calibration, repair only when necessary,
and enforce one of three measurable score-retention constraints:

- activation L1;
- `||X_i||_2 ||W_:,i||_2`, whose omitted sum upper-bounds projection error;
- the squared diagonal approximation
  `||X_i||_2^2 ||W_:,i||_2^2`.

The measured total includes selector search, native O_DIRECT read and GPU
upload, activation gather, and compact GEMM.  A calibration-only policy chooses
the score threshold that meets a configured projection relative-L2 ceiling.
Actual held-out projection error and sparse full-forward logit KL/NLL are
reported separately.

The online search sorts the score once to obtain an unstructured lower bound
on `R`, adds a structured-fragmentation margin learned only on calibration
prompts, and normally evaluates one cell-8 mask. It repairs upward only when
that mask misses the requested score retention.

Laptop run:

```bash
fallocate -l 64M experiments/34_quality_constrained_variable_r/results_laptop/io_blob.dat
python experiments/34_quality_constrained_variable_r/run_experiment.py \
  --io-blob experiments/34_quality_constrained_variable_r/results_laptop/io_blob.dat \
  --output-dir experiments/34_quality_constrained_variable_r/results_laptop \
  --report-output experiments/34_quality_constrained_variable_r/report_laptop.md
```

The three checkpoints must already exist in the local Hugging Face cache.
