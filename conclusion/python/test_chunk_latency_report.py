"""The delivered notebook is an executed, numerically grounded report."""
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs/02_chunk_latency/breakpoint_comparison'


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
    x, y = curve.size_kib.to_numpy(), curve.train_ms.to_numpy()
    losses = []
    for s in x[3:-3]:
        X = np.column_stack((x, np.maximum(x, s)))
        b = nnls(X, y)[0]
        losses.append(np.mean((X @ b - y)**2))
    summary = json.loads((OUT / 'findings.json').read_text())
    assert summary['ts_free_breakpoint_kib'] == x[3:-3][np.argmin(losses)]
    band = x[3:-3][np.asarray(losses) <= min(losses) * 1.01**2]
    assert summary['ts_1pct_rmse_band_kib'] == [float(band.min()), float(band.max())]
    table = pd.read_csv(OUT / 'model_comparison.csv')
    nb = nbformat.read(HERE / '02_chunk_latency.ipynb', as_version=4)
    text = rendered_text(nb)
    for _, r in table.iterrows():
        assert f'{r.holdout_rmse_us:.3f}' in text
        assert f'{r.holdout_mape_pct:.2f}%' in text
    assert len(table) == 6 and not table.model.str.contains('learned', case=False).any()
    assert (table.boundary == 'free').sum() == 2
    assert summary['ts_free_breakpoint_kib'] == 211


def test_report_tracks_original_measurement_provenance():
    import chunk_latency as cl
    data = HERE / 'runs/02_chunk_latency'
    provenance = json.loads((OUT / 'input_provenance.json').read_text())
    assert provenance['input_sha256'] == cl.input_hashes(data)
    assert provenance['native_measurements_repeated'] is False
    assert provenance['train_blocks'] == [0, 1, 2, 3]
    assert provenance['holdout_blocks'] == [4, 5, 6]
    assert not list(data.rglob('*.dat'))


def test_builder_is_cache_only_and_notebook_keeps_the_opt_in_switch():
    from build_chunk_latency_report import notebook
    nb = notebook()
    sources = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'RERUN_MEASUREMENTS = False' in sources
    assert 'reporting.begin' not in sources
    assert 'prepare_blob' not in sources and 'profiler.measure' not in sources
    assert 'curve.to_csv' not in sources
