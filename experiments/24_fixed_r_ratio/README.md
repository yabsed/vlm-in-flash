# Experiment 24: fixed-R two-line ratio selection

This experiment addresses Experiment 22's row-budget confound. The paper is
run with its nominal row budget, then every comparison mask is required to
select exactly the number of rows the paper actually returned:

\[
R_{\mathrm{eff}}=|M_{\mathrm{Paper}}|.
\]

The proposed online approximation maximizes two-line importance efficiency
under that fixed cardinality. It adds a row price `mu` to the existing exact
fixed-scalarization DP, performs four or eight row-price corrections, repairs
the closest over-budget mask by peeling endpoints, and optionally updates the
Dinkelbach ratio once.

Compared methods:

- Paper;
- Top-R;
- one ratio iteration with 4 row-price solves;
- one ratio iteration with 8 row-price solves;
- two ratio iterations with 4 row-price solves each.

The default run covers all 16 Table-2 shapes, three inputs, three spatial
orders, three row-budget fractions, 30 warm repetitions, and host/CUDA timing
tracks. A separate exhaustive `N=18` oracle measures approximation quality.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/24_fixed_r_ratio/run_experiment.py
```

Smoke test:

```bash
python experiments/24_fixed_r_ratio/run_experiment.py \
  --shapes 4864x896 --trials 1 --repetitions 2 \
  --row-budget-fractions 0.5 --oracle-n 12 --oracle-trials 1
```

Outputs include raw trials/timings, small-N oracle trials, overall and
shape-level summaries, static-hybrid results, plots, and `report.md`.

Lookup latency is predicted with the released Orin AGX profile. The laptop
selector measurements are not a Jetson end-to-end result.
