"""CSV reuse, model results, and a CPU-only notebook run."""
from pathlib import Path
import os
import shutil

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pytest

import chunk_latency as cl

HERE = Path(__file__).resolve().parent


def test_read_existing_csv(tmp_path, monkeypatch):
    expected = pd.DataFrame({"block": [0], "size_kib": [240], "latency_ms": [.04]})
    expected.to_csv(tmp_path / "block_estimates.csv", index=False)
    monkeypatch.setattr(cl, "measure", lambda run: pytest.fail("Unexpected SSD measurement"))
    pd.testing.assert_frame_equal(cl.acquire(tmp_path), expected)
    with pytest.raises(FileNotFoundError):
        cl.acquire(tmp_path / "missing")


def test_remeasurement_is_explicit(tmp_path, monkeypatch):
    calls = []
    def measure(run):
        calls.append(run)
        pd.DataFrame({"latency_ms": [.04]}).to_csv(run / "block_estimates.csv", index=False)
    monkeypatch.setattr(cl, "measure", measure)
    assert cl.acquire(tmp_path, rerun=True).latency_ms.iloc[0] == .04
    assert calls == [tmp_path]


def test_two_line_fit():
    x = np.asarray(cl.SIZES_KIB, float)
    y = cl.design(x, "Ts", 200) @ [.00012, .00006]
    models, _ = cl.train_models(x, y)
    model = models["Ts learned"]
    assert model["s_kib"] == 200
    prediction = cl.design(x, "Ts", 200) @ model["coefficients"]
    np.testing.assert_allclose(prediction, y)
    np.testing.assert_allclose(prediction[x >= 200] / x[x >= 200], .00018)


def test_saved_results():
    curve = cl.split_curve(cl.acquire(HERE / "runs/02_chunk_latency"))
    models, _ = cl.train_models(curve.size_kib, curve.train_ms)
    fits, _ = cl.compare_models(curve.size_kib, curve.holdout_ms, models)
    assert models["Ts learned"]["s_kib"] == 211
    assert models["Hinge"]["s_kib"] == 230
    assert cl.throughput_threshold(curve.size_kib, curve.train_ms)[0] == 256
    assert curve.size_kib.iloc[(curve.median_ms / curve.size_kib).argmin()] == 240
    expected = [3.803290879, 4.266998332, 3.592304784, 3.332244385, 2.918659884, 10.003808636]
    order = ["Ts@240", "Ts@256", "Ts@230", "Ts learned", "Hinge", "Affine"]
    np.testing.assert_allclose(fits.set_index("model").loc[order].holdout_mape_pct, expected)


def test_notebook_with_only_estimates_csv(tmp_path):
    run = tmp_path / "runs/02_chunk_latency"
    run.mkdir(parents=True)
    shutil.copy2(HERE / "runs/02_chunk_latency/block_estimates.csv", run)
    shutil.copy2(HERE / "chunk_latency.py", tmp_path)
    nb = nbformat.read(HERE / "02_chunk_latency.ipynb", as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(
        "assert 'profile_flash' not in sys.modules\nassert 'torch' not in sys.modules"))
    executed = NotebookClient(nb, timeout=120, kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}}).execute()
    executed.cells.pop()
    outputs = [o for c in executed.cells if c.cell_type == "code" for o in c.outputs]
    assert sum("image/png" in o.get("data", {}) for o in outputs) == 4
    if os.environ.get("NOTEBOOK_ARTIFACT_DIR"):
        out = Path(os.environ["NOTEBOOK_ARTIFACT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, out / "02_chunk_latency.ipynb")
        shutil.copytree(run / "breakpoint_comparison", out / "breakpoint_comparison", dirs_exist_ok=True)
