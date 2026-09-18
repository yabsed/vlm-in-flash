# Experiment 42: SSD latency after s

This experiment directly measures the laptop SN850X beyond the previously
trimmed saturation point `s = 240 KiB`. It uses the same native O_DIRECT reader
and saturated-throughput conversion as `profile_flash.py`.

To expose time and thermal drift, sizes from 192 through 768 KiB are measured
once in ascending order and once in descending order. The primary diagnostic
plots `r` on the x-axis and `L(r)/r` on the y-axis: a true post-s model
`L(r) = c2*r` must be horizontal.

```bash
python experiments/42_post_s_ssd_linearity/run_experiment.py
```

The benchmark holds the standard VLMFlash SSD lock so another drive benchmark
cannot silently contend with it.
