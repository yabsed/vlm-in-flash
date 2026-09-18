# Experiment 38: Paper vs Paper-s vs Cell-1

This experiment isolates the effect of fixing the paper's candidate chunk
length to the laptop SSD saturation length `s`.

- `Paper`: released multi-scale Algorithm 1.
- `Paper-s`: the same greedy utility selector, stride cap, overlap rule, and
  whole-window budget rule, with exactly one candidate length `s`.
- `Cell-1`: one `s`-sized cell/tile with exact-R endpoint repair.

The laptop run uses the measured RTX 3050 + WD_BLACK SN850X latency profile,
real cached language models, native O_DIRECT reads, GPU upload, activation
gather, and compact GEMM.
