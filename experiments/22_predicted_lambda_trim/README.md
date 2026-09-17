# Experiment 22: Paper vs Predicted-lambda TD-2L(8) + Endpoint Trim

This experiment compares the paper's fixed-row-budget selector with the new
coverage-directed method at paired retained importance. For each input and row
budget, the target is

\[
Q = I(M_{\mathrm{Paper}}).
\]

The proposed selector must retain at least this importance. The main variants
are endpoint-trim limits 64 and 256; the untrimmed method is retained as an
ablation.

All 16 matrix shapes and AGX hyperparameters from the paper's Table 2 are
included. The default benchmark uses three synthetic inputs, three spatial
orders, three row-budget fractions, and 30 warm repetitions per case. It runs
two timing tracks when CUDA is available:

- `host`: host float32 importance to CPU boolean mask;
- `cuda`: CUDA-resident float32 importance to CUDA boolean mask, including the
  proposal's D2H copy, CPU solve, H2D mask copy, and synchronization.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/22_predicted_lambda_trim/run_experiment.py
```

Small smoke test:

```bash
python experiments/22_predicted_lambda_trim/run_experiment.py \
  --shapes 4864x896 --trials 1 --repetitions 2 \
  --row-budget-fractions 0.5
```

Key outputs:

- `results/trials.csv`: one row per paired case and method;
- `results/timing_samples.csv`: every warm timing repetition;
- `results/summary.csv` and `results/shape_summary.csv`;
- `results/runtime_quality.{png,pdf}`;
- `results/shape_comparison.{png,pdf}`;
- `report.md`.

The selector timing is a laptop screening result. Predicted lookup latency is
computed from the released Orin AGX profile; actual storage I/O and model
compute are not executed, and a Jetson 2 ms claim requires a Jetson rerun.
