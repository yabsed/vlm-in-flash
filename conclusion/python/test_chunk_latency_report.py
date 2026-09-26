"""The delivered notebook is an executed, numerically grounded report."""
from pathlib import Path

import nbformat
import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent


def rendered_text(nb):
    parts = [c.source for c in nb.cells if c.cell_type == 'markdown']
    parts += [o.get('data', {}).get('text/markdown', '') for c in nb.cells
              if c.cell_type == 'code' for o in c.outputs]
    return '\n'.join(parts)


def test_report_has_embedded_figures_without_logs_or_stale_labels():
    nb = nbformat.read(HERE / '02_chunk_latency.ipynb', as_version=4)
    outputs = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs]
    assert sum('image/png' in o.get('data', {}) for o in outputs) == 4
    assert not any(o.output_type in {'stream', 'error'} for o in outputs)
    text = rendered_text(nb)
    for log_term in ['Ts learned', 'RERUN_MEASUREMENTS', 'input_sha256', 'Run All', 'Execution summary']:
        assert log_term not in text
    assert 'Every model below has fitted coefficients' in text
    assert 'retrospective' in text and 'not a confidence interval' in text
    visible = [c.source for c in nb.cells if c.cell_type == 'code'
               and not c.metadata.get('jupyter', {}).get('source_hidden')]
    assert len(visible) == 3
    assert all(len(s.splitlines()) <= 4 for s in visible)


def test_report_numbers_are_outputs_of_the_saved_curve():
    import chunk_latency as cl
    _, estimates, _ = cl.load_cached(HERE / 'runs/02_chunk_latency')
    curve, _ = cl.split_curve(estimates)
    x, train, holdout = curve.size_kib.to_numpy(), curve.train_ms.to_numpy(), curve.holdout_ms.to_numpy()
    losses = []
    for s in x[3:-3]:
        X = np.column_stack((x, np.maximum(x, s)))
        b = nnls(X, train)[0]
        losses.append(np.mean((X @ b - train)**2))
    s_star = x[3:-3][int(np.argmin(losses))]
    band = x[3:-3][np.asarray(losses) <= min(losses) * 1.01**2]
    parameters, _ = cl.train_models(x, train)
    fits, _ = cl.compare_models(x, holdout, parameters)
    nb = nbformat.read(HERE / '02_chunk_latency.ipynb', as_version=4)
    text = rendered_text(nb)
    assert s_star == 211
    assert f'{s_star:g}' in text
    assert f'{band.min():g}–{band.max():g} KiB' in text
    for _, r in fits.iterrows():
        assert f'{r.holdout_rmse_us:.3f}' in text
        assert f'{r.holdout_mape_pct:.2f}%' in text


def test_builder_is_cache_only_and_notebook_keeps_the_opt_in_switch():
    from build_chunk_latency_report import notebook
    nb = notebook()
    sources = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'RERUN_MEASUREMENTS = False' in sources
    assert 'reporting.begin' not in sources
    assert 'prepare_blob' not in sources and 'profiler.measure' not in sources
    assert 'curve.to_csv' not in sources
