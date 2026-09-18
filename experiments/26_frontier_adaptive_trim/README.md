# Experiment 26: frontier search with adaptive fixed-R trim

Experiment 24 kept only the row-price solution whose cardinality was closest
to `R` from above, then repaired that single mask with a fixed-rho endpoint
heap. More row-price calls could therefore replace a good earlier mask with a
worse one.

This experiment evaluates every distinct eligible cardinality encountered on
the row-price frontier. Each overfilled mask is trimmed to exactly `R` rows;
after every deletion, the endpoint yielding the best resulting exact `I/L` is
chosen. The best repaired ratio seen across every mu and rho iteration is
retained, so the selector uses only:

- the activation-importance vector;
- the requested row count `R`;
- the two-line latency model.

It does not receive the Paper mask or `I(M_paper)`. Paper is run only as a
paired benchmark reference, and its returned row count defines the common `R`
for an exact same-cardinality comparison.

Compared methods:

- Paper and Top-R;
- Experiment 24's `rho2/mu4` implementation;
- 16 mu calls, two rho iterations, adaptive trim capped at 64 rows;
- 16 mu calls, two rho iterations, adaptive trim capped at 256 rows;
- 24 mu calls, three rho iterations, adaptive trim capped at 256 rows.

The production benchmark reuses Experiment 22's real
Qwen2.5-0.5B-Instruct activation archive: 128 projection calls over four
supported Table-2 shapes. A separate synthetic `N=18` exhaustive oracle is
used only to measure solver optimality.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/26_frontier_adaptive_trim/run_experiment.py
```

Small smoke test:

```bash
python experiments/26_frontier_adaptive_trim/run_experiment.py \
  --shapes 896x128 --max-traces-per-shape 1 --tracks host \
  --repetitions 2 --oracle-trials 1 --output-dir /tmp/experiment26-smoke
```

See [`algorithm.md`](algorithm.md) for the selection and timing contract.

## Laptop-native rerun

`results_laptop` profiles and validates the actual drive holding this checkout
(`WD_BLACK SN850X 1000GB`) and reruns the CUDA track on the laptop's
`NVIDIA GeForce RTX 3050 6GB Laptop GPU`. The measured profile saturates at
240 KiB with 5864 MiB/s peak logical throughput. Its 21-pattern O_DIRECT
validation has Pearson `r=0.9992` (`R²=0.9983`).

Unlike the original Orin-profile result, this rerun also replays every selected
mask through the native O_DIRECT reader 30 times and uploads the returned rows
to CUDA. It reports actual SSD I/O, upload, read-call wall time, and the paired
`selector + read` total. See [`report_laptop.md`](report_laptop.md).

The device profile and validation can be reproduced with:

```bash
python preliminary_research/vlm-flash/scripts/profile_flash.py \
  --blob experiments/26_frontier_adaptive_trim/results_laptop/profile_blob.dat \
  --output experiments/26_frontier_adaptive_trim/results_laptop/laptop_sn850x_profile.json \
  --threads 6 --blob-mb 128 --step-kb 1 --iters 10 --max-kb 512

python preliminary_research/vlm-flash/scripts/validate_latency_model.py \
  --blob experiments/26_frontier_adaptive_trim/results_laptop/profile_blob.dat \
  --profile experiments/26_frontier_adaptive_trim/results_laptop/laptop_sn850x_profile.json \
  --threads 6 --iters 30 \
  --output experiments/26_frontier_adaptive_trim/results_laptop/laptop_sn850x_validation.json
```
