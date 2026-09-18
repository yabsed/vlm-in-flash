# Algorithm

For a projection `Y=XW^T+b`, Experiment 34 defines three per-input-row scores:

```text
raw_i   = mean_t |X[t,i]|
bound_i = ||X[:,i]||_2 ||W[:,i]||_2
diag_i  = bound_i^2.
```

The weight-aware bound is justified by

```text
||X_omitted W_omitted^T||_F
  <= sum_{i omitted} ||X[:,i]||_2 ||W[:,i]||_2.
```

For every score and candidate ratio, the Experiment 29 cell-8 structured
selector constructs a mask. Calibration measures the mean non-selector cost
for each `(model, shape, score, ratio)`. It also measures the largest gap
between the unstructured minimum ratio and the structured minimum ratio on
calibration inputs. At inference, one score sort gives the unstructured lower
bound, the frozen structured margin predicts a starting ratio, and only ratios
at or above that point are tried in calibrated-cost order. A failed constraint
check triggers an upward repair. Thus selection uses neither `I(M_paper)` nor
test error, and its cost includes the score sort and every attempted selector.

The calibration-only adaptive policy chooses `tau` per `(model, shape, score)`
from `{.50, .55, ..., .90, .925, .95, .975}`. Among thresholds whose calibration
projection relative-L2 is at most the configured ceiling, it chooses the one
with minimum calibration total time.  If none meets the ceiling, it chooses the
lowest-error threshold.  The threshold and latency ordering are then frozen for
holdout prompts.

For a selected mask `M`, total projection-path time is

```text
score-sort predictor time
+ sum(selector time for every attempted ratio)
+ native O_DIRECT/read/upload wall time for M
+ activation index_select time
+ compact F.linear(X[:,M], W[:,M]) time.
```

The held-out error oracle is deliberately labeled non-deployable: it sees the
actual held-out projection error and only estimates the remaining headroom.
