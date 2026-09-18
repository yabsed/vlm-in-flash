# Experiment 39: weight-aware Cell-1

This experiment tests the diagonal approximation to projection squared error:

`q_i = mean_t(X_ti^2) * ||W[:, i]||_2^2`.

It compares Paper, the existing activation-only Cell-1, an activation-energy
ablation, and weight-aware Cell-1. Weight norms are precomputed once at model
load. Online score construction is measured and included in every frontier.

The laptop-safe protocol runs one model per process, computes weight norms in
bounded 128-row GPU slabs, caps the CUDA allocator at 55% of VRAM, uses two
CPU/O_DIRECT threads, pauses briefly between projections, and increases process
niceness. Per-model CSVs are combined with `merge_results.py` only after all
CUDA processes have exited.

The runner enforces one model per process. The following loop is sequential,
so it never keeps two model processes resident at once:

```bash
for model in qwen05 smol360 tiny11; do
  python experiments/39_weight_aware_cell1/run_experiment.py \
    --models "$model" \
    --io-blob experiments/35_full_frontier/results_laptop/io_blob.dat \
    --output-dir "experiments/39_weight_aware_cell1/runs_laptop_safe/$model" \
    --report-output "experiments/39_weight_aware_cell1/runs_laptop_safe/$model/report.md"
done

python experiments/39_weight_aware_cell1/merge_results.py \
  --inputs \
    experiments/39_weight_aware_cell1/runs_laptop_safe/qwen05 \
    experiments/39_weight_aware_cell1/runs_laptop_safe/smol360 \
    experiments/39_weight_aware_cell1/runs_laptop_safe/tiny11 \
  --output-dir experiments/39_weight_aware_cell1/results_laptop_safe \
  --report-output experiments/39_weight_aware_cell1/report_laptop_safe.md
```

Do not pass multiple names to `--models`; validation rejects that configuration
before loading a model.
