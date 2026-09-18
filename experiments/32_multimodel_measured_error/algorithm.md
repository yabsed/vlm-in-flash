# Algorithm

For projection input `X`, checkpoint weight `W`, bias `b`, and selector mask
`M`, Experiment 32 computes

```text
Y_dense  = X W^T + b
Y_sparse = X[:, M] W[:, M]^T + b.
```

The primary local error is `||Y_sparse-Y_dense||_2 / ||Y_dense||_2`.  Cosine
error and relative MAE are also recorded.  This is an exact projection-level
measurement under the original dense activation, not an importance proxy.

For end-to-end evaluation, every decoder projection is wrapped by `vlm-flash`
and the sparse forward is actually executed.  Its logits are compared with a
dense forward using dense-to-sparse KL, relative L2, tokenwise argmax
agreement, and teacher-forced NLL change.

Three dispatchers are calibrated per `(model, shape, budget)`:

- wave dispatch maximizes the wave candidate's own importance/actual-total;
- min-total dispatch picks the smallest measured actual-total across methods;
- error guard picks the fastest method whose calibration projection error is
  within 1% of Paper's calibration error.

All choices are frozen before evaluating the held-out prompts.
