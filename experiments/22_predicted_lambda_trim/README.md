# Experiment 22: real-activation predicted-lambda + endpoint trim

This experiment compares the paper's fixed-row-budget selector with the
coverage-directed selector at paired retained importance. Unlike the original
version of Experiment 22, its input is not a synthetic one-dimensional
lognormal vector. The default run loads
`Qwen/Qwen2.5-0.5B-Instruct`, performs real dense language-model forwards, and
captures exactly the importance signal used by the `vlm-flash` submodule:

\[
v_i=\operatorname{mean}_{\text{batch,tokens}} |x_i|.
\]

`vlmflash.attach()` wraps the language model's q/k/v/o/gate/up/down
projections. A capture policy returns an all-true mask, so the traced forward
is still dense and neither competing selector can alter a later layer's
activation. The saved activation traces are then replayed through both
selectors.

For every real projection trace and row budget, the paired target is

\[
Q = I(M_{\mathrm{Paper}}).
\]

The proposed selector must retain at least this importance. Endpoint-trim
limits 64 and 256 are compared with the untrimmed method.

## Run

The first run downloads the model if it is not already in the Hugging Face
cache. By default it traces three prompts, retains up to 32 projection calls
per supported shape, evaluates row-budget fractions 0.25/0.50/0.75, and uses
30 warm selector repetitions per case.

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/22_predicted_lambda_trim/run_experiment.py
```

A small real-model smoke test is:

```bash
python experiments/22_predicted_lambda_trim/run_experiment.py \
  --prompt "Explain contiguous flash reads." \
  --shapes 896x128 --max-traces-per-shape 1 \
  --tracks host --paper-impl torch --repetitions 2 \
  --row-budget-fractions 0.5 --output-dir /tmp/experiment22-smoke
```

Trace collection can be separated from selector benchmarking. This is useful
when model inference and timing happen on different machines:

```bash
# Machine with enough memory for the model.
python experiments/22_predicted_lambda_trim/run_experiment.py \
  --collect-traces-only --trace-output /tmp/qwen-traces.npz \
  --output-dir /tmp/experiment22-trace

# Timing machine; no model load or download occurs.
python experiments/22_predicted_lambda_trim/run_experiment.py \
  --trace-input /tmp/qwen-traces.npz --output-dir /tmp/experiment22-replay
```

Use repeated `--prompt` flags or a one-prompt-per-line `--prompt-file` for a
representative workload. `--max-traces-per-shape 0` keeps every captured
projection call. Only shapes with the paper's Table-2 chunk parameters are
benchmarked; unsupported shapes are recorded in `metadata.json` and skipped.

## Outputs

- `results/activation_traces.npz`: raw float32 activation-importance vectors
  plus model, prompt-hash, module, shape, and token-count metadata;
- `results/trials.csv`: one row per real trace, row budget, timing track, and
  method;
- `results/timing_samples.csv`: every warm selector timing repetition;
- `results/summary.csv`, `results/shape_summary.csv`, and plots;
- `results/metadata.json` and `report.md`.

The language-model forward is real but deliberately untimed: this experiment
isolates online mask-selection latency and predicted I/O quality. It still
does not measure sparse-model accuracy, actual NVMe reads, or end-to-end token
latency. Lookup latency comes from the released Orin AGX profile, so a Jetson
2 ms claim requires a Jetson rerun.

The checked-in `results/` and `report.md` use the v2 real-activation protocol.
`--analyze-only` explicitly rejects retired synthetic-v1 CSVs so old and new
results cannot be mixed accidentally.
