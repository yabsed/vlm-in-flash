# Experiment 27: scale Experiment 24's rho/mu budgets

This experiment changes no selection rule from Experiment 24. It calls the
same `fixed_r_ratio()` implementation and varies only its compute budgets:

- `rho2/mu4` (the Experiment 24 incumbent);
- `rho2/mu8` and `rho2/mu16`;
- `rho3/mu8` and `rho3/mu16`;
- `rho4/mu16` and `rho4/mu32`.

Paper and Top-R are retained as references. Paper's returned row count defines
the common `R`, but neither the Paper mask nor `I(M_paper)` is passed to the
fixed-R selector.

The production benchmark reuses Experiment 22's real
Qwen2.5-0.5B-Instruct activation archive: 128 projection calls across four
supported Table-2 shapes. It measures both host and CUDA round-trip selector
latency. The separate synthetic `N=18` exhaustive oracle measures how close
each compute budget comes to the exact fixed-R ratio optimum.

Run:

```bash
TORCH_EXTENSIONS_DIR=/tmp/vlmflash_torch_extensions \
python experiments/27_fixed_r_scale/run_experiment.py
```

Small smoke test:

```bash
python experiments/27_fixed_r_scale/run_experiment.py \
  --shapes 896x128 --max-traces-per-shape 1 --tracks host \
  --repetitions 2 --oracle-trials 1 --output-dir /tmp/experiment27-smoke
```

See [`algorithm.md`](algorithm.md) for the unchanged selector and evaluation
contract.
