"""Fresh Linux direct-I/O measurements. No saved profiles or buffered fallbacks."""
import json
import mmap
import os
import platform
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd

PAGE = 4096


class DirectReader:
    def __init__(self, directory, workers=4, mib=128, seed=0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        mount = subprocess.check_output(["findmnt", "-T", str(self.directory), "-J", "-o", "SOURCE,FSTYPE,TARGET"], text=True)
        self.mount = json.loads(mount)["filesystems"][0]
        if self.mount["fstype"] in {"tmpfs", "ramfs"}:
            raise RuntimeError("Choose a flash-backed directory; RAM filesystems are not SSDs")
        self.size = mib * 2**20
        self.workers = workers
        self.local = threading.local()
        self.buffers = []
        self.pool = ThreadPoolExecutor(max_workers=workers)
        handle, self.path = tempfile.mkstemp(prefix="fresh-", suffix=".bin", dir=self.directory)
        rng = np.random.default_rng(seed)
        with os.fdopen(handle, "wb") as output:
            for _ in range(mib):
                output.write(rng.bytes(2**20))
            output.flush()
            os.fsync(output.fileno())
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            self.read([(0, PAGE)])
        except Exception:
            self.close()
            raise RuntimeError("O_DIRECT failed; no cached-I/O substitute is permitted")

    def _one(self, request):
        offset, size = request
        if offset % PAGE or size % PAGE or size <= 0 or offset + size > self.size:
            raise ValueError("Requests must be positive, in range, and 4096-byte aligned")
        if not hasattr(self.local, "buffer") or len(self.local.buffer) < size:
            buffer = mmap.mmap(-1, size)
            self.local.buffer = buffer
            self.buffers.append(buffer)
        view = memoryview(self.local.buffer)[:size]
        try:
            actual = os.preadv(self.fd, [view], offset)
        finally:
            view.release()
        if actual != size:
            raise IOError(f"Short direct read: {actual} != {size}")

    def read(self, requests):
        started = perf_counter_ns()
        if self.workers == 1:
            for request in requests:
                self._one(request)
        else:
            list(self.pool.map(self._one, requests))
        return (perf_counter_ns() - started) / 1e6

    def close(self):
        self.pool.shutdown(wait=True)
        if hasattr(self, "fd"):
            os.close(self.fd)
        for buffer in self.buffers:
            buffer.close()
        Path(self.path).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def profile(directory, lengths=None, repeats=7, counts=(1, 8, 32, 64), workers=4, seed=20260926):
    """Randomized paired rounds; each batch uses distinct random aligned offsets."""
    if lengths is None:
        lengths = np.unique(np.r_[np.arange(1, 17), [20, 24, 32, 40, 48, 64, 80, 96,
                             128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048]])
    rng = np.random.default_rng(seed)
    records = []
    mib = max(128, int(np.ceil(max(lengths) * PAGE * max(counts) / 2**20)))
    with DirectReader(directory, workers=workers, mib=mib, seed=seed) as reader:
        # Preallocate/touch each worker's largest buffer before recording samples.
        reader.read([(i * int(max(lengths)) * PAGE, int(max(lengths)) * PAGE) for i in range(64)])
        jobs = [(int(length), count) for length in lengths for count in counts]
        for repeat in range(repeats):
            for index in rng.permutation(len(jobs)):
                length, count = jobs[index]
                slots = reader.size // (length * PAGE)
                offsets = rng.choice(slots, count, replace=False) * length * PAGE
                requests = [(int(offset), length * PAGE) for offset in offsets]
                ms = reader.read(requests)
                records.append(dict(repeat=repeat, length=length, kib=length * 4, count=count,
                                    workers=workers, batch_ms=ms, per_chunk_ms=ms / count))
        meta = dict(mount=reader.mount, direct=True, page_bytes=PAGE, blob_mib=mib,
                    workers=workers, seed=seed, repeats=repeats, counts=list(counts),
                    platform=platform.platform(), timer="perf_counter_ns; dispatch+completion",
                    regime="largest measured batch; saturation must be checked")
    return pd.DataFrame(records), meta


def cost_table(raw, n):
    group = raw[raw["count"] == raw["count"].max()].groupby("length").per_chunk_ms.median()
    if n > group.index.max():
        raise ValueError("No latency extrapolation: profile longer chunks first")
    return np.r_[0., np.interp(np.arange(1, n + 1), group.index, group.values)]


def near_saturation(raw, tolerance=.05):
    group = raw[raw["count"] == raw["count"].max()].groupby("length").per_chunk_ms.median()
    ratio = group.to_numpy() / group.index.to_numpy()
    return int(group.index[np.flatnonzero(ratio <= (1 + tolerance) * ratio.min())[0]])
