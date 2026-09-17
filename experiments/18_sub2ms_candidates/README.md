# Experiment 18: sub-2 ms selector candidates

This experiment is an initial screen of fourteen selector configurations:

- Paper and the complete strongly-supported frontier as references;
- target-directed two-line DP with 4, 8, 16, or 32 scalarized solves;
- target-directed DP plus endpoint trimming at `(8, 64)` and `(16, 256)`
  solve/deletion limits;
- saturation tiles of length `s/2`, `s`, or `2s`, each trying offsets `0`
  and `L/2` internally;
- Paper candidate ordering quantized into 64, 256, or 1,024 buckets.

The two evaluation tracks are deliberately separate:

1. **coverage-only:** reach an externally supplied importance target `Q`;
2. **row-budget audit:** reach the same external `Q` while also satisfying a
   paired row cap `R`.

Defaults use `N=4,864`, exact `CV=3.30` inputs, the three spatial orderings,
the released Orin AGX lookup table, FP16 rows of 1.75 KiB, and a two-line fit
with 236 KiB saturation. The three `(Q / total, R / N)` scenarios are
`(0.50, 0.25)`, `(0.70, 0.50)`, and `(0.90, 0.75)`. Each timed query is
repeated 30 times after warm-up, following Appendix H's repeat count.

Runtime includes candidate construction, scalarized solves, final mask
recovery, trim/postprocessing, fallback, and CPU mask materialization. Numba
and native-extension compilation are warmed up and excluded. Measurements in
this checkout are host CPU measurements, not Jetson measurements; therefore
they screen implementations but cannot establish the Jetson 2 ms claim.
Predicted I/O latency is reported separately under both the two-line fit and
released lookup rule.

See [`algorithm.md`](algorithm.md) for the exact recurrence search, endpoint
ranking, tile construction, bucket mapping, fallback, and timing contract.

Run the default experiment:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python3 experiments/18_sub2ms_candidates/run_experiment.py
```

Run a small smoke test:

```bash
python3 experiments/18_sub2ms_candidates/run_experiment.py \
  --n-values 257 --trials 1 --repetitions 2 --skip-self-check
```

Outputs:

- `results/trials.csv`: every method/input/scenario/track result;
- `results/timing_samples.csv`: every timed repetition, including failures;
- `results/summary.csv` and `results/summary.json`;
- `results/runtime_quality.{png,pdf}`;
- `results/deadline_pass_rate.{png,pdf}`;
- `report.md`.
