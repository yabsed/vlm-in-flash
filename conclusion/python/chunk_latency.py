"""Read the saved latency CSV and fit the report's models."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

SIZES_KIB = np.r_[np.arange(1, 257), np.arange(260, 769, 4)].tolist()
BLOCKS, ITERS, WARMUP, THREADS, BLOB_MIB = 7, 10, 3, 6, 128


def acquire(run, rerun=False):
    run = Path(run)
    if rerun:
        measure(run)
    return pd.read_csv(run / "block_estimates.csv")


def measure(run):
    """Explicitly repeat the original SSD sweep; never called by default."""
    submodule = Path(__file__).resolve().parents[2] / "preliminary_research/vlm-flash"
    sys.path[:0] = [str(submodule / "scripts"), str(submodule / "src")]
    import profile_flash as profiler
    from _lock import exclusive

    reader = profiler.native()
    if reader is None:
        raise RuntimeError(profiler.unavailable_reason())
    run.mkdir(parents=True, exist_ok=True)
    blob = run / "profile_blob.dat"
    rng = np.random.default_rng(0)
    raw, estimates = [], []
    with exclusive("02_chunk_latency"):
        try:
            profiler.prepare_blob(str(blob), BLOB_MIB * 1024)
            for block in range(BLOCKS):
                for order, size in enumerate(rng.permutation(SIZES_KIB)):
                    size = int(size)
                    runs = profiler.measure(reader, str(blob), BLOB_MIB * 1024,
                                            [size], ITERS, WARMUP, THREADS)[size]
                    throughput = profiler.saturation_throughput(runs, size)
                    estimates.append(dict(block=block, order=order, size_kib=size,
                        latency_ms=1000 * size / (1024 * throughput),
                        throughput_mib_s=throughput, count_points=len(runs), max_count=runs[-1][0]))
                    for count, times in runs:
                        for repeat, time in enumerate(times):
                            raw.append(dict(block=block, order=order, size_kib=size,
                                size_bytes=size * 1024, size_bits=size * 8192,
                                count=count, repeat=repeat, io_us=float(time)))
            pd.DataFrame(raw).to_csv(run / "io_raw.csv", index=False)
            pd.DataFrame(estimates).to_csv(run / "block_estimates.csv", index=False)
        finally:
            blob.unlink(missing_ok=True)


def split_curve(estimates, train_blocks=4):
    wide = estimates.pivot(index="size_kib", columns="block", values="latency_ms")
    wide = wide.sort_index().sort_index(axis=1)
    x, samples = wide.index.to_numpy(float), wide.to_numpy(float)
    return pd.DataFrame(dict(size_kib=x, median_ms=np.median(samples, axis=1),
        train_ms=np.median(samples[:, :train_blocks], axis=1),
        holdout_ms=np.median(samples[:, train_blocks:], axis=1)))


def throughput_threshold(x, latency_ms):
    throughput = 1000 * np.asarray(x) / (1024 * np.asarray(latency_ms))
    smooth = pd.Series(throughput).rolling(3, center=True, min_periods=1).median().to_numpy()
    peak = float(smooth.max())
    s = float(np.asarray(x)[np.flatnonzero(smooth >= .99 * peak)[0]])
    return s, throughput, smooth, peak


def design(x, kind, s=None):
    x = np.asarray(x, float)
    if kind == "Affine":
        return np.column_stack([np.ones_like(x), x])
    if kind == "Hinge":
        return np.column_stack([np.ones_like(x), x, np.maximum(x - s, 0)])
    if kind == "Ts":
        return np.column_stack([x, np.maximum(s, x)])
    raise ValueError(f"Unknown model: {kind}")


def fit_at(x, train, kind, s=None):
    matrix = design(x, kind, s)
    if kind == "Ts":
        coefficients = nnls(matrix, train)[0]
    else:
        coefficients = np.linalg.lstsq(matrix, train, rcond=None)[0]
    return dict(kind=kind, s_kib=s, coefficients=coefficients.tolist(),
                train_mse_ms2=float(np.mean((train - matrix @ coefficients) ** 2)))


def train_models(x, train):
    """Fit coefficients and free breakpoints on training data only."""
    x, train = np.asarray(x, float), np.asarray(train, float)
    parameters = {"Affine": fit_at(x, train, "Affine")}
    search = []
    for kind, name in (("Hinge", "Hinge"), ("Ts", "Ts learned")):
        trials = [fit_at(x, train, kind, float(s)) for s in x[3:-3]]
        parameters[name] = min(trials, key=lambda row: row["train_mse_ms2"])
        for row in trials:
            search.append(dict(kind=kind, s_kib=row["s_kib"],
                train_mse_ms2=row["train_mse_ms2"],
                train_rmse_us=1000 * np.sqrt(row["train_mse_ms2"])))
    for s in (240., 256., 230.):
        parameters[f"Ts@{s:g}"] = fit_at(x, train, "Ts", s)
    return parameters, pd.DataFrame(search)


def compare_models(x, holdout, parameters):
    """Evaluate fitted models without changing their parameters."""
    x, holdout = np.asarray(x, float), np.asarray(holdout, float)
    rows, predictions = [], {}
    for name, model in parameters.items():
        prediction = design(x, model["kind"], model["s_kib"]) @ model["coefficients"]
        predictions[name] = prediction
        tail = np.ones(len(x), bool) if model["s_kib"] is None else x >= model["s_kib"]
        rows.append(dict(model=name, kind=model["kind"], s_kib=model["s_kib"],
            train_rmse_us=1000 * np.sqrt(model["train_mse_ms2"]),
            holdout_rmse_us=1000 * np.sqrt(np.mean((holdout - prediction) ** 2)),
            holdout_mape_pct=100 * np.mean(np.abs(holdout - prediction) / holdout),
            tail_holdout_mape_pct=100 * np.mean(np.abs(holdout[tail] - prediction[tail]) / holdout[tail])))
    return pd.DataFrame(rows), predictions
