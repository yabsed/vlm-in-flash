# Experiment 31: wave-balanced exact-R DP

Experiment 31 evaluates exact-R masks made of `K in {1,6,12,18,24}` aligned,
near-equal, separated runs.  Six and its multiples match the laptop reader's
six static worker lanes.  For each K, dynamic programming maximizes retained
importance within that hardware layout class.

Paper and Experiment 29's `ceil(s/8)` method are measured baselines.  A
held-out dispatcher chooses K for each shape and row budget from the first half
of real Qwen traces, then evaluates that fixed choice on the second half.  The
max-efficiency dispatcher uses only each wave candidate's own importance and
measured selector-plus-read time; it never receives the Paper mask or Paper
importance.

Laptop run:

```bash
dd if=/dev/urandom \
  of=experiments/31_wave_balanced_dp/results_laptop/io_blob.dat \
  bs=1M count=64
python experiments/31_wave_balanced_dp/run_experiment.py \
  --tracks cuda --saturation-kib 240 --measure-io \
  --io-blob experiments/31_wave_balanced_dp/results_laptop/io_blob.dat \
  --output-dir experiments/31_wave_balanced_dp/results_laptop \
  --report-output experiments/31_wave_balanced_dp/report_laptop.md
```
