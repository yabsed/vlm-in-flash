"""Native delegation and saved-CSV analysis; never run an SSD benchmark."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import sys

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / '01_chunk_latency.ipynb'


def cells():
    return {c.id: c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells}


def namespace():
    source = cells()
    ns = {}
    exec(source['setup'], ns)
    exec(source['native-acquisition'], ns)
    assert ns['RERUN_MEASUREMENTS'] is False
    return ns


def test_notebook_has_no_reimplemented_reader_or_estimator():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    source = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    for removed in ('preadv', 'ThreadPoolExecutor', 'mmap', 'ALIGNMENT',
                    'def batch_seconds', 'def saturated_throughput'):
        assert removed not in source
    for delegated in ('profiler.native()', 'profiler.prepare_blob(',
                      'profiler.measure(', 'profiler.saturation_throughput(',
                      'profiler.exclusive('):
        assert delegated in source


def test_cached_read_is_read_only_and_missing_csv_never_measures(tmp_path):
    ns = namespace()
    path = tmp_path / 'block_estimates.csv'
    shutil.copy2(HERE / 'block_estimates.csv', path)
    before = path.read_bytes()
    ns['CSV_PATH'] = path
    def forbidden(*args):
        pytest.fail('Cached analysis attempted native acquisition')
    ns['measure'] = forbidden
    exec(cells()['measurement-load'], ns)
    assert path.read_bytes() == before
    pd.testing.assert_frame_equal(ns['data'], pd.read_csv(path))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        exec(cells()['measurement-load'], ns)
    assert not path.exists()


def fake_profiler(monkeypatch, *, direct=True, available=True):
    state = SimpleNamespace(locked=False, args=[], estimates=[], blobs=[])
    reader = SimpleNamespace(read_rows=lambda *a, **k: (None, 123.0, 0.0, direct))
    @contextmanager
    def exclusive(label):
        assert label == '01_chunk_latency' and not state.locked
        state.locked = True
        try:
            yield
        finally:
            state.locked = False
    def prepare_blob(blob, num_rows):
        assert state.locked and num_rows == 128 * 1024
        state.blobs.append(Path(blob))
        Path(blob).write_bytes(b'unit-test fixture, not a storage profile')
    def measure(ext, blob, num_rows, sizes, iters, warmup, threads):
        assert state.locked and Path(blob).is_file()
        assert (num_rows, iters, warmup, threads) == (128 * 1024, 10, 3, 6)
        assert len(sizes) == 1
        assert ext.read_rows('sentinel')[1] == 123.0
        runs = [(1, [101., 102.]), (4, [201., 202.]), (28, [301., 302.])]
        state.args.append((sizes[0], runs))
        return {sizes[0]: runs}
    def saturation_throughput(runs, size):
        assert runs is state.args[-1][1] and size == state.args[-1][0]
        state.estimates.append(size)
        return 1000. + size
    module = SimpleNamespace(ROW_KB=1, native=lambda: reader if available else None,
        unavailable_reason=lambda: 'test native unavailable', exclusive=exclusive,
        prepare_blob=prepare_blob, measure=measure, saturation_throughput=saturation_throughput)
    monkeypatch.setitem(sys.modules, 'profile_flash', module)
    # measure prepends paths; do not leak that mutation to other tests.
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    return state


def test_acquisition_delegates_native_calls_and_keeps_csv_schema(tmp_path, monkeypatch):
    ns = namespace()
    ns.update(BLOCKS=2, SIZES_KIB=np.array([37, 94, 240]), CSV_PATH=tmp_path / 'block_estimates.csv')
    state = fake_profiler(monkeypatch)
    ns['RERUN_MEASUREMENTS'] = True
    exec(cells()['measurement-load'], ns)
    data = pd.read_csv(ns['CSV_PATH'])
    assert list(data) == ['block', 'order', 'size_kib', 'latency_ms',
                          'throughput_mib_s', 'count_points', 'max_count']
    rng = np.random.default_rng(0)
    expected = np.concatenate([rng.permutation(ns['SIZES_KIB']) for _ in range(2)])
    np.testing.assert_array_equal(data.size_kib, expected)
    np.testing.assert_array_equal(data.block, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(data.order, [0, 1, 2, 0, 1, 2])
    np.testing.assert_allclose(data.throughput_mib_s, 1000 + expected)
    np.testing.assert_allclose(data.latency_ms, 1000 * expected / (1024 * (1000 + expected)))
    assert data.count_points.eq(3).all() and data.max_count.eq(28).all()
    assert len(state.args) == len(state.estimates) == 6
    assert not state.locked and all(not p.exists() for p in state.blobs)


@pytest.mark.parametrize('direct,available', [(False, True), (True, False)])
def test_failed_native_acquisition_preserves_saved_csv(tmp_path, monkeypatch, direct, available):
    ns = namespace()
    path = tmp_path / 'block_estimates.csv'
    path.write_text('old reference must survive\n')
    before = path.read_bytes()
    ns.update(BLOCKS=1, SIZES_KIB=np.array([240]), CSV_PATH=path, RERUN_MEASUREMENTS=True)
    state = fake_profiler(monkeypatch, direct=direct, available=available)
    with pytest.raises(RuntimeError, match='O_DIRECT|native unavailable'):
        exec(cells()['measurement-load'], ns)
    assert path.read_bytes() == before
    assert not state.locked and all(not p.exists() for p in state.blobs)


def test_cached_report_without_native_dependencies(tmp_path):
    shutil.copy2(HERE / 'block_estimates.csv', tmp_path)
    before = (tmp_path / 'block_estimates.csv').read_bytes()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    # The kernel cannot import a real profiler, torch or native reader, even if
    # notebook control flow regresses. No submodule is copied into the workspace.
    guard = nbformat.v4.new_code_cell('''
get_ipython().run_line_magic('matplotlib', 'inline')
import importlib.abc, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'profile_flash', 'vlmflash'}:
            raise AssertionError('Native import in cached report: ' + fullname)
sys.meta_path.insert(0, NoNative())
''')
    check = nbformat.v4.new_code_cell('''
assert not RERUN_MEASUREMENTS
assert (s99, z_rho, h_star, s_star) == (256, 240, 230, 211)
np.testing.assert_allclose(comparison['Holdout MAPE (%)'],
    [3.803290879, 4.266998332, 3.592304784, 3.332244385, 2.918659884, 10.003808636], rtol=1e-7)
assert not any(m in sys.modules for m in ('torch', 'profile_flash', 'vlmflash'))
''')
    nb.cells.insert(0, guard)
    nb.cells.append(check)
    NotebookClient(nb, timeout=90, kernel_name='python3',
        resources={'metadata': {'path': str(tmp_path)}}).execute()
    nb.cells = nb.cells[1:-1]
    outputs = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs]
    assert sum('image/png' in o.get('data', {}) for o in outputs) == 4
    assert not any(o.output_type in {'stream', 'error'} for o in outputs)
    assert (tmp_path / 'block_estimates.csv').read_bytes() == before
    assert not list(tmp_path.glob('*.dat'))
    for i, cell in enumerate((c for c in nb.cells if c.cell_type == 'code'), 1):
        cell.execution_count = i
        cell.metadata.pop('execution', None)
        for output in cell.outputs:
            if 'execution_count' in output:
                output.execution_count = i
    if os.environ.get('NOTEBOOK_ARTIFACT_DIR'):
        out = Path(os.environ['NOTEBOOK_ARTIFACT_DIR'])
        out.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out / NOTEBOOK.name)
        from nbconvert import HTMLExporter
        from traitlets.config import Config
        config = Config()
        config.TagRemovePreprocessor.remove_input_tags = {'hide-input'}
        config.HTMLExporter.exclude_input_prompt = True
        config.HTMLExporter.exclude_output_prompt = True
        html, _ = HTMLExporter(config=config).from_notebook_node(nb)
        (out / '01_chunk_latency.html').write_text(html, encoding='utf-8')
