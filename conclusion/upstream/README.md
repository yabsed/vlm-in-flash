# Reference implementation

`vlm-flash/` is the unchanged Git submodule from `snuhcs/vlm-flash`. Its
revision is recorded by the parent repository, not selected at runtime.
The latency notebook calls its profiler and native reader rather than
maintaining another I/O implementation.

From the repository root:

```sh
git submodule sync --recursive
git submodule update --init --recursive conclusion/upstream/vlm-flash
```

[Notebook 01](../python/results/01/01_chunk_latency.ipynb) contains the
acquisition definition, data checks, model fits and figures. With
`RERUN_MEASUREMENTS = False`, it reads the existing CSV and never starts
acquisition merely because the CSV is missing. Native acquisition is
explicit and requires the submodule's native build dependencies.

The notebook CI checks the saved report without initializing or importing
this submodule. It does not claim to reproduce hardware timings.
