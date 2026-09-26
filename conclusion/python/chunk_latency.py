"""CPU-only cache analysis for notebook 02; native I/O is explicitly opt-in."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# 1KiB to 256KiB in 1KiB increments
# then 260KiB to 768KiB in 4KiB increments.
SIZES_KIB = np.r_[np.arange(1, 257), np.arange(260, 769, 4)].tolist()

# default config 
DEFAULT_CONFIG = dict(
    sizes_kib=SIZES_KIB, blocks=7, iterations=10, warmup=3, threads=6,
    blob_mib=128, chunk_counts=[1, 2, 4, 8, 16, 28, 32, 48, 64, 96, 128, 192, 256, 384, 512],
    gap_kib=32, max_native_read_kib=768,
    size_order="randomized within each block", logical_request_bytes=True,
    seed=0, blob_seed=0, estimator="profile_flash.saturation_throughput",
)

INPUT_FILES = ("hardware.json", "block_estimates.csv", "io_raw.csv")


def inputhashes_(run: Path) -> dict[str, str]:
    result = {}
    for name in INPUT_FILES:
        with (Path(run) / name).open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def load_cached(run: Path, expected: dict | None = None):
    """Read and validate a complete run without importing torch or touching disk.

    A missing, partial, or incompatible cache never starts a hardware sweep.
    """
    run = Path(run)
    expected = DEFAULT_CONFIG if expected is None else expected
    missing = [name for name in INPUT_FILES if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing cached files in {run}: {missing}. Restore the CSVs or explicitly "
            "set RERUN_MEASUREMENTS=True on the target SSD; no sweep was started."
        )
    if (run / ".incomplete").exists():
        raise ValueError(f"Incomplete measurement in {run}; restore a complete cache or rerun explicitly.")
    hardware = json.loads((run / "hardware.json").read_text(encoding="utf-8"))
    mismatches = {k: (hardware.get(k), v) for k, v in expected.items() if hardware.get(k) != v}
    if mismatches:
        raise ValueError(f"Cached measurement settings differ (stored, requested): {mismatches}")
    estimates = pd.read_csv(run / "block_estimates.csv")
    raw = pd.read_csv(run / "io_raw.csv")
    ecols = {"block", "order", "size_kib", "latency_ms", "throughput_mib_s", "count_points", "max_count"}
    rcols = {"block", "order", "size_kib", "size_bytes", "size_bits", "count", "repeat", "io_us"}
    if not ecols.issubset(estimates) or not rcols.issubset(raw):
        raise ValueError("Cached CSV schema is incomplete.")
    for frame, columns in ((estimates, ecols), (raw, rcols)):
        values = frame[sorted(columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Cached CSV contains nonfinite values.")
        integer_cols = columns - {"latency_ms", "throughput_mib_s", "io_us"}
        values = frame[sorted(integer_cols)].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError("Cached identifiers and byte counts must be integers.")
    blocks, sizes, iters = hardware["blocks"], hardware["sizes_kib"], hardware["iterations"]
    expected_pairs = pd.MultiIndex.from_product([range(blocks), sizes], names=["block", "size_kib"])
    actual_pairs = pd.MultiIndex.from_frame(estimates[["block", "size_kib"]])
    if actual_pairs.has_duplicates or set(actual_pairs) != set(expected_pairs):
        raise ValueError("Cached estimates have missing, duplicate, or unexpected block/size pairs.")
    for _, group in estimates.groupby("block"):
        if sorted(group.order.tolist()) != list(range(len(sizes))):
            raise ValueError("Cached within-block order is not a permutation.")
    if (estimates[["latency_ms", "throughput_mib_s"]] <= 0).any().any() or (raw.io_us <= 0).any():
        raise ValueError("Cached timings and throughput must be positive.")
    derived = 1000 * estimates.size_kib / (1024 * estimates.throughput_mib_s)
    if not np.allclose(derived, estimates.latency_ms, rtol=1e-9, atol=1e-12):
        raise ValueError("Latency/throughput units or estimates are inconsistent.")
    if not ((raw.size_bytes == raw.size_kib * 1024) & (raw.size_bits == raw.size_kib * 8192)).all():
        raise ValueError("Raw logical-byte/bit units are inconsistent.")
    keys = ["block", "size_kib", "count", "repeat"]
    if raw.duplicated(keys).any() or not raw.repeat.between(0, iters - 1).all():
        raise ValueError("Duplicate or invalid raw repeat identifiers.")
    expected_groups = {
        (b, size, count) for b in range(blocks) for size in sizes
        for count in hardware["chunk_counts"]
        if count * (size + hardware["gap_kib"]) <= hardware["blob_mib"] * 1024
    }
    groups = raw.groupby(["block", "size_kib", "count"]).size()
    if set(groups.index) != expected_groups or not groups.eq(iters).all():
        raise ValueError("Raw timings have missing or unexpected chunk-count/repeat groups.")
    orders = raw.groupby(["block", "size_kib"]).order.agg(["first", "nunique"])
    expected_order = estimates.set_index(["block", "size_kib"]).order.reindex(orders.index)
    if not orders["nunique"].eq(1).all() or not orders["first"].eq(expected_order).all():
        raise ValueError("Raw and estimated acquisition orders disagree.")
    counts = raw.groupby(["block", "size_kib"])["count"].agg(["nunique", "max"])
    estimate_counts = estimates.set_index(["block", "size_kib"]).reindex(counts.index)
    if not (counts["nunique"].eq(estimate_counts.count_points) & counts["max"].eq(estimate_counts.max_count)).all():
        raise ValueError("Raw chunk counts disagree with the estimates.")
    return raw, estimates, hardware


def acquire(run: Path, *, rerun: bool = False, submodule: Path | None = None, config: dict | None = None):
    """Cache by default; opt-in measurements use a NEW sibling directory.

    Old CSVs/metadata remain intact even if a new native run fails halfway.
    """
    config = DEFAULT_CONFIG if config is None else config
    if not rerun:
        return Path(run), *load_cached(run, config)
    return _measure_new(Path(run), submodule, config)


def _measure_new(run: Path, submodule: Path | None, config: dict):
    # These imports, native initialization, findmnt, and blob writes are unreachable
    # in the default cache path. In particular, no submodule is needed for analysis.
    import importlib
    import os
    import subprocess
    import sys
    from contextlib import redirect_stdout
    from datetime import datetime, timezone
    from io import StringIO
    from tempfile import gettempdir
    from uuid import uuid4
    from tqdm.auto import tqdm

    if submodule is None or not (Path(submodule) / "scripts/profile_flash.py").is_file():
        raise FileNotFoundError("Explicit remeasurement requires the vlm-flash submodule.")
    sys.path[:0] = [str(Path(submodule) / "scripts"), str(Path(submodule) / "src")]
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path(gettempdir()) / "vlmflash-torch-extensions"))
    profiler = importlib.import_module("profile_flash")
    exclusive = importlib.import_module("_lock").exclusive
    if profiler.ROW_KB != 1 or list(profiler.NUM_CHUNKS_SCHED) != config["chunk_counts"] or profiler.GAP_ROWS != config["gap_kib"]:
        raise ValueError("Profiler schedule/units changed; update configuration explicitly.")
    extension = profiler.native()
    if extension is None:
        raise RuntimeError(profiler.unavailable_reason())

    class DirectOnly:
        def read_rows(self, *args, **kwargs):
            result = extension.read_rows(*args, **kwargs)
            if not result[3]:
                raise RuntimeError("Native reader did not use O_DIRECT")
            return result

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fresh = run.parent / f"{run.name}_measurement_{stamp}_{uuid4().hex[:8]}"
    fresh.mkdir(parents=True, exist_ok=False)
    (fresh / ".incomplete").write_text("Native acquisition has not completed.\n")
    mount = json.loads(subprocess.check_output(
        ["findmnt", "-T", str(fresh), "-J", "-o", "SOURCE,FSTYPE,TARGET"], text=True,
    ))["filesystems"][0]
    if mount["fstype"] in {"tmpfs", "ramfs"}:
        raise RuntimeError("Place the experiment on the target storage filesystem")
    hardware = dict(config, mount=mount, direct_io_required=True, timestamp_utc=stamp)
    (fresh / "hardware.json").write_text(json.dumps(hardware, indent=2))
    blob = fresh / "profile_blob.dat"
    num_rows = config["blob_mib"] * 1024
    rng = np.random.default_rng(config["seed"])
    raw_records, estimates = [], []
    with exclusive("02_chunk_latency: explicit native remeasurement"):
        try:
            profiler.prepare_blob(str(blob), num_rows)
            with tqdm(total=config["blocks"] * len(config["sizes_kib"]), desc="SSD sweep", unit="size") as bar:
                for block in range(config["blocks"]):
                    bar.set_postfix(block=f"{block + 1}/{config['blocks']}", refresh=False)
                    for order, kib in enumerate(rng.permutation(config["sizes_kib"])):
                        kib = int(kib)
                        with redirect_stdout(StringIO()):
                            runs = profiler.measure(DirectOnly(), str(blob), num_rows, [kib],
                                config["iterations"], config["warmup"], config["threads"])[kib]
                        throughput = profiler.saturation_throughput(runs, kib)
                        estimates.append(dict(block=block, order=order, size_kib=kib,
                            latency_ms=1000 * kib / (1024 * throughput), throughput_mib_s=throughput,
                            count_points=len(runs), max_count=runs[-1][0]))
                        for count, times in runs:
                            raw_records.extend(dict(block=block, order=order, size_kib=kib,
                                size_bytes=kib * 1024, size_bits=kib * 8192, count=count,
                                repeat=i, io_us=float(t)) for i, t in enumerate(times))
                        bar.update()
                    for name, records in (("io_raw.csv", raw_records), ("block_estimates.csv", estimates)):
                        temp = fresh / (name + ".tmp")
                        pd.DataFrame(records).to_csv(temp, index=False)
                        temp.replace(fresh / name)
        finally:
            blob.unlink(missing_ok=True)
    (fresh / ".incomplete").unlink()
    try:
        cached = load_cached(fresh, config)
    except Exception:
        (fresh / ".incomplete").write_text("Post-acquisition validation failed.\n")
        raise
    return fresh, *cached


def split_curve(estimates: pd.DataFrame, train_blocks: int = 4):
    wide = estimates.pivot(index="size_kib", columns="block", values="latency_ms").sort_index().sort_index(axis=1)
    if not 0 < train_blocks < wide.shape[1] or wide.isna().any().any():
        raise ValueError("A complete train/holdout block split is required.")
    x, samples = wide.index.to_numpy(float), wide.to_numpy(float)
    median = np.median(samples, axis=1)
    lo, hi = np.quantile(samples, [.1, .9], axis=1)
    curve = pd.DataFrame(dict(size_kib=x, median_ms=median, p10_ms=lo, p90_ms=hi,
        train_ms=np.median(samples[:, :train_blocks], axis=1),
        holdout_ms=np.median(samples[:, train_blocks:], axis=1), ms_per_kib=median / x))
    return curve, samples


def throughput_threshold(x, latency_ms):
    x, latency_ms = np.asarray(x, float), np.asarray(latency_ms, float)
    throughput = 1000 * x / (1024 * latency_ms)
    smooth = np.median(np.vstack([np.r_[throughput[0], throughput[:-1]], throughput,
                                  np.r_[throughput[1:], throughput[-1]]]), axis=0)
    smooth[0], smooth[-1] = np.median(throughput[:2]), np.median(throughput[-2:])
    peak = float(smooth.max())
    s = float(x[np.flatnonzero(smooth >= .99 * peak)[0]])
    return s, throughput, smooth, peak


def design(x, kind, s=None):
    x = np.asarray(x, float)
    if kind == "Affine":
        return np.column_stack([np.ones_like(x), x])
    if s is None or s <= 0:
        raise ValueError("A positive breakpoint is required.")
    if kind == "Hinge":
        return np.column_stack([np.ones_like(x), x, np.maximum(x - s, 0)])
    if kind == "Ts":
        return np.column_stack([x, np.maximum(s, x)])
    raise ValueError(f"Unknown model family: {kind}")


def fit_at(x, train, kind, s=None):
    matrix = design(x, kind, s)
    coefficients = nnls(matrix, train)[0] if kind == "Ts" else np.linalg.lstsq(matrix, train, rcond=None)[0]
    prediction = matrix @ coefficients
    return dict(kind=kind, s_kib=None if s is None else float(s), coefficients=coefficients.tolist(),
                train_mse_ms2=float(np.mean((train - prediction) ** 2)))


def train_models(x, train):
    """Select coefficients AND s using only train; holdout is not an argument.

    Ts forces a constant T/x tail. A free hinge does not impose this constraint.
    Both free-breakpoint models search the same interior measured grid, x[3:-3].
    """
    x, train = np.asarray(x, float), np.asarray(train, float)
    if x.ndim != 1 or train.shape != x.shape or len(x) < 8:
        raise ValueError("Need matching one-dimensional arrays and at least eight measured sizes.")
    if not np.isfinite(x).all() or not np.isfinite(train).all() or np.any(x <= 0) or np.any(train <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Sizes must be sorted/unique/positive and training latencies finite/positive.")
    candidates = x[3:-3]
    parameters = {"Affine": fit_at(x, train, "Affine")}
    search = []
    for kind, name in (("Hinge", "Hinge"), ("Ts", "Ts learned")):
        trials = [fit_at(x, train, kind, float(s)) for s in candidates]
        best = min(trials, key=lambda row: row["train_mse_ms2"])
        best["breakpoint_at_search_edge"] = best["s_kib"] in (candidates[0], candidates[-1])
        # If the hinge coefficient vanishes, the breakpoint is not identified.
        best["breakpoint_identified"] = abs(best["coefficients"][-1]) > 1e-12
        parameters[name] = best
        search.extend(dict(kind=kind, s_kib=row["s_kib"], train_mse_ms2=row["train_mse_ms2"],
                           train_rmse_us=1000 * np.sqrt(row["train_mse_ms2"])) for row in trials)
    for s in (240., 256., 230.):
        parameters[f"Ts@{s:g}"] = fit_at(x, train, "Ts", s)
    return parameters, pd.DataFrame(search)


def compare_models(x, holdout, parameters):
    """Evaluate frozen fits; never choose a breakpoint from holdout errors."""
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
            tail_holdout_mape_pct=100 * np.mean(np.abs(holdout[tail] - prediction[tail]) / holdout[tail]),
            nonpositive_predictions=int(np.count_nonzero(prediction <= 0)),
            breakpoint_at_search_edge=model.get("breakpoint_at_search_edge", False)))
    return pd.DataFrame(rows), predictions
