"""Check the notebook's DP and ratio bounds against exhaustive enumeration."""

from itertools import product
from pathlib import Path

import nbformat
import numpy as np
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "conclusion/python/results/02/02_global_oracle.ipynb"
FUNCTION_CELLS = {
    "global-dp-03", "global-dp-06", "global-dp-07",
    "global-dp-09", "global-dp-11", "global-dp-12",
}
notebook = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.id in FUNCTION_CELLS:
        exec(compile(cell.source, cell.id, "exec"))

def enumerate_small(v, T):
    masks = np.array(list(product([False, True], repeat=len(v))))
    latency = np.zeros(len(masks))
    length = np.zeros(len(masks), int)
    for column in masks.T:
        latency[~column] += T[length[~column]]
        length = np.where(column, length + 1, 0)
    return masks, masks @ v, latency + T[length]


checks = 0
for n in (1, 5, 9):
    for v in (np.zeros(n), np.random.default_rng(n).uniform(0, 4, n)):
        for boundary in (0.5, 1, 2.5, 4, n + 2):
            for short_slope in (0., 0.25, 1.):
                T = costs(n, boundary, short_slope, 1.)
                masks, imp, delay = enumerate_small(v, T)
                for R in range(1, n + 1):
                    allowed = masks.sum(axis=1) == R
                    for q in (0., 0.1, 1., 10.):
                        D = bellman(v, R, q, boundary, short_slope, 1.)
                        mask = recover(v, R, q, T, D)
                        I, L = metrics(v, mask, T)
                        exact = np.max(imp[allowed] - q * delay[allowed])
                        assert mask.sum() == R
                        np.testing.assert_allclose([D[-1, -1], I - q * L], exact, atol=1e-9)
                        checks += 1
                    mask, lower, upper, calls = solve(v, R, boundary, short_slope, 1.)
                    np.testing.assert_allclose(delay[allowed].min(), T[R])
                    assert mask.sum() == R
                    exact = np.max(imp[allowed] / delay[allowed])
                    assert lower - 1e-9 <= exact <= upper + 1e-9
                    np.testing.assert_allclose(metrics(v, mask, T)[0] / metrics(v, mask, T)[1], lower)
                    assert upper - lower <= 1e-6 * lower + 1e-9
print(f"{checks} exhaustive additive checks passed, including recovery and ratio bounds.")

for name in ("02_capture_traces.ipynb", "02_analyze_oracle.ipynb"):
    notebook = nbformat.read(NOTEBOOK.with_name(name), as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
print("Capture and analysis notebook structure and syntax passed.")
