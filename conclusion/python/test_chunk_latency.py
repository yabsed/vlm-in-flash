"""CPU regression tests: cached Run All must never launch the native benchmark."""
import builtins
import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pytest

import chunk_latency as cl

HERE = Path(__file__).resolve().parent


def make_cache(path, config=None):
    config = copy.deepcopy(config or dict(cl.DEFAULT_CONFIG,
        sizes_kib=[16, 64, 128, 160, 200, 220, 230, 240, 248, 256, 280, 320, 512, 768],
        chunk_counts=[1, 2, 4], iterations=2))
    path.mkdir(parents=True)
    estimates, raw = [], []
    rng = np.random.default_rng(0)
    for block in range(config["blocks"]):
        for order, size in enumerate(rng.permutation(config["sizes_kib"])):
            size = int(size)
            latency = (.00012 * size + .00006 * max(220, size)) * (1 + .001 * block)
            counts = [c for c in config["chunk_counts"] if c * (size + config["gap_kib"]) <= 1024 * config["blob_mib"]]
            estimates.append(dict(block=block, order=order, size_kib=size, latency_ms=latency,
                throughput_mib_s=1000 * size / (1024 * latency), count_points=len(counts), max_count=max(counts)))
            for count in counts:
                for repeat in range(config["iterations"]):
                    raw.append(dict(block=block, order=order, size_kib=size, size_bytes=size * 1024,
                        size_bits=size * 8192, count=count, repeat=repeat, io_us=1000 * latency * count))
    (path / "hardware.json").write_text(json.dumps(dict(config, mount={"fstype": "btrfs"})))
    pd.DataFrame(estimates).to_csv(path / "block_estimates.csv", index=False)
    pd.DataFrame(raw).to_csv(path / "io_raw.csv", index=False)
    return config


def test_cache_is_read_only_and_needs_no_native_module(tmp_path, monkeypatch):
    config = make_cache(tmp_path / "run")
    before = cl.input_hashes(tmp_path / "run")
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "profile_flash", "vlmflash", "_lock"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(cl, "_measure_new", lambda *a: pytest.fail("Unexpected native sweep"))
    run, raw, estimates, metadata = cl.acquire(tmp_path / "run", config=config)
    curve, _ = cl.split_curve(estimates)
    models, _ = cl.train_models(curve.size_kib, curve.train_ms)
    fits, _ = cl.compare_models(curve.size_kib, curve.holdout_ms, models)
    assert len(fits) == 6 and len(raw) > 0
    assert metadata["blocks"] == 7
    assert cl.input_hashes(run) == before
    assert not list(run.glob("*.dat"))


def test_missing_cache_never_starts_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "_measure_new", lambda *a: pytest.fail("Unexpected native sweep"))
    with pytest.raises(FileNotFoundError, match="RERUN_MEASUREMENTS=True"):
        cl.acquire(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("damage", ["missing_block", "duplicate", "nan", "raw_repeat", "units", "settings", "incomplete"])
def test_invalid_cache_is_rejected(tmp_path, damage):
    run = tmp_path / "run"
    config = make_cache(run)
    estimates = pd.read_csv(run / "block_estimates.csv")
    raw = pd.read_csv(run / "io_raw.csv")
    if damage == "missing_block":
        estimates = estimates[estimates.block != 6]
    elif damage == "duplicate":
        estimates = pd.concat([estimates, estimates.iloc[:1]])
    elif damage == "nan":
        estimates.loc[0, "latency_ms"] = np.nan
    elif damage == "raw_repeat":
        raw = raw.iloc[1:]
    elif damage == "units":
        raw.loc[0, "size_bits"] += 1
    elif damage == "settings":
        config["threads"] += 1
    elif damage == "incomplete":
        (run / ".incomplete").touch()
    estimates.to_csv(run / "block_estimates.csv", index=False)
    raw.to_csv(run / "io_raw.csv", index=False)
    with pytest.raises(ValueError):
        cl.load_cached(run, config)


def test_ts_formula_continuity_nonnegative_and_flat_tail():
    s, b, d = 240., .00012, .00006
    x = np.array([1., s - 1e-6, s, s + 1e-6, 512., 768.])
    y = cl.design(x, "Ts", s) @ [b, d]
    np.testing.assert_allclose(y, np.where(x <= s, d * s + b * x, (b + d) * x))
    np.testing.assert_allclose(y[x >= s] / x[x >= s], b + d)
    assert np.all(y > 0) and np.all(np.diff(y) >= 0)
    assert abs(y[1] - y[3]) < 1e-8


def test_ts_learns_its_own_breakpoint_and_beats_fixed_training_loss():
    x = np.asarray(cl.SIZES_KIB, float)
    train = cl.design(x, "Ts", 200) @ [.00012, .00006]
    models, search = cl.train_models(x, train)
    assert models["Ts learned"]["s_kib"] == 200
    assert models["Ts learned"]["breakpoint_identified"]
    for name in ("Ts@240", "Ts@256", "Ts@230"):
        assert models["Ts learned"]["train_mse_ms2"] <= models[name]["train_mse_ms2"]
        assert min(models[name]["coefficients"]) >= 0
    assert set(search.kind) == {"Ts", "Hinge"}
    # A free hinge with a nonzero tail intercept is not the constrained Ts model.
    other = cl.design(x, "Hinge", 230) @ [.01235, .000124, .0000652]
    models, _ = cl.train_models(x, other)
    assert models["Hinge"]["s_kib"] == 230
    assert models["Ts learned"]["s_kib"] != models["Hinge"]["s_kib"]


def test_holdout_does_not_change_parameters_or_search():
    x = np.asarray(cl.SIZES_KIB, float)
    train = cl.design(x, "Ts", 220) @ [.00012, .00006]
    models, search = cl.train_models(x, train)
    before = copy.deepcopy(models)
    first, _ = cl.compare_models(x, train, models)
    second, _ = cl.compare_models(x, train * 3, models)
    assert models == before
    assert not np.allclose(first.holdout_rmse_us, second.holdout_rmse_us)
    after, after_search = cl.train_models(x, train)
    assert after == before
    pd.testing.assert_frame_equal(search, after_search)


def test_unidentified_ts_breakpoint_is_reported():
    x = np.asarray(cl.SIZES_KIB, float)
    models, _ = cl.train_models(x, .00018 * x)
    assert not models["Ts learned"]["breakpoint_identified"]


def test_explicit_measurement_failure_preserves_old_cache(tmp_path, monkeypatch):
    import importlib
    import subprocess
    import sys
    from contextlib import nullcontext
    run = tmp_path / "run"
    config = make_cache(run)
    before = cl.input_hashes(run)
    module = tmp_path / "submodule"
    (module / "scripts").mkdir(parents=True)
    (module / "scripts/profile_flash.py").touch()
    fake = SimpleNamespace(ROW_KB=1, NUM_CHUNKS_SCHED=config["chunk_counts"], GAP_ROWS=config["gap_kib"],
        native=lambda: object(), prepare_blob=lambda p, n: Path(p).write_bytes(b"test"),
        measure=lambda *a: (_ for _ in ()).throw(RuntimeError("injected acquisition failure")))
    old_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **kw:
        fake if name == "profile_flash" else SimpleNamespace(exclusive=lambda *a: nullcontext())
        if name == "_lock" else old_import(name, *a, **kw))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: '{"filesystems":[{"fstype":"ext4"}]}')
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(RuntimeError, match="injected acquisition failure"):
        cl.acquire(run, rerun=True, submodule=module, config=config)
    assert cl.input_hashes(run) == before
    fresh = list(tmp_path.glob("run_measurement_*"))
    assert len(fresh) == 1 and (fresh[0] / ".incomplete").exists()
    assert not (fresh[0] / "profile_blob.dat").exists()


def execute_notebook(root, config, *, artifacts=False):
    shutil.copy2(HERE / "chunk_latency.py", root / "chunk_latency.py")
    nb = nbformat.read(HERE / "02_chunk_latency.ipynb", as_version=4)
    # Synthetic smoke cases use their own declared acquisition configuration.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("MEASUREMENT_CONFIG = dict(chunk_latency.DEFAULT_CONFIG, seed=SEED)",
                                               "MEASUREMENT_CONFIG = " + repr(config))
    nb.cells.append(nbformat.v4.new_code_cell("assert 'torch' not in sys.modules\nassert 'profile_flash' not in sys.modules"))
    executed = NotebookClient(nb, timeout=120, kernel_name="python3",
        resources={"metadata": {"path": str(root)}}).execute()
    assert not list(root.rglob("*.dat"))
    if artifacts and os.environ.get("NOTEBOOK_ARTIFACT_DIR"):
        out = Path(os.environ["NOTEBOOK_ARTIFACT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, out / "02_chunk_latency.executed.ipynb")
        shutil.copytree(root / "runs/02_chunk_latency/breakpoint_comparison", out / "breakpoint_comparison", dirs_exist_ok=True)
    _, estimates, _ = cl.load_cached(root / "runs/02_chunk_latency", config)
    curve, _ = cl.split_curve(estimates)
    x = curve.size_kib.to_numpy()
    parameters, _ = cl.train_models(x, curve.train_ms)
    fits, _ = cl.compare_models(x, curve.holdout_ms, parameters)
    s99 = cl.throughput_threshold(x, curve.train_ms)[0]
    rho = curve.median_ms.to_numpy() / x
    return dict(s_star=parameters["Ts learned"]["s_kib"], hinge_breakpoint_kib=parameters["Hinge"]["s_kib"],
                argmin_kib=float(x[rho.argmin()]), saturation_99pct_kib=s99, n_models=len(fits))


def test_notebook_run_all_with_synthetic_cache(tmp_path):
    config = make_cache(tmp_path / "runs/02_chunk_latency")
    summary = execute_notebook(tmp_path, config)
    assert summary["s_star"] == 220


def test_notebook_run_all_with_committed_cache(tmp_path):
    source = HERE / "runs/02_chunk_latency"
    if not all((source / name).is_file() for name in cl.INPUT_FILES):
        pytest.skip("Committed measurement CSVs are not present in this checkout")
    destination = tmp_path / "runs/02_chunk_latency"
    destination.mkdir(parents=True)
    before = cl.input_hashes(source)
    for name in cl.INPUT_FILES:
        shutil.copy2(source / name, destination / name)
    summary = execute_notebook(tmp_path, cl.DEFAULT_CONFIG, artifacts=True)
    assert cl.input_hashes(source) == before
    assert cl.input_hashes(destination) == before
    assert summary["n_models"] == 6
    assert summary["argmin_kib"] == 240
    assert summary["saturation_99pct_kib"] == 256
    assert summary["hinge_breakpoint_kib"] == 230
    print(json.dumps({k: summary[k] for k in ("argmin_kib", "saturation_99pct_kib", "hinge_breakpoint_kib", "s_star")}))
