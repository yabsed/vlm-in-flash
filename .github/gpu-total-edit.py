from pathlib import Path
import json
path = Path('conclusion/python/results/03/03_selection_latency.ipynb')
nb = json.loads(path.read_text())
cells = {c['id']: c for c in nb['cells']}
def source(key):
    s = cells[key]['source']
    return ''.join(s) if isinstance(s, list) else s
def put(key, text):
    cells[key]['source'] = text.splitlines(keepends=True)
    if cells[key]['cell_type'] == 'code':
        cells[key]['execution_count'] = None
        cells[key]['outputs'] = []
        cells[key]['metadata'].pop('execution', None)
def replace(key, old, new):
    s = source(key)
    assert old in s, (key, old)
    put(key, s.replace(old, new))
replace('setup', 'RUN_BENCHMARK = True', '''RUN_BENCHMARK = False
RUN_CUDA = False  # True: also measure Paper GPU sort and packed CUDA matmul.
if RUN_CUDA:
    RESULTS, ENVIRONMENT = Path("selection_cuda_results.csv"), Path("environment_cuda.json")''')
replace('title', 'This is a **CPU-resident selection microbenchmark**, not an SSD or inference benchmark. The input importance vector is already available on the CPU. The output is a CPU mask, selected importance, and a cost diagnostic. The oracle is not timed.', '''The selection boundary is a prepared CPU importance vector to a CPU mask. With `RUN_CUDA = True`, compare CPU sorting with Paper's hybrid CUDA-sort path and measure packed GPU matmul on the same machine. The final section adds these measurements to Notebook 02's modeled read cost. This is a projection-level composed estimate, not an observed inference wall time. The oracle is not timed.''')
replace('definition', 'The timed Paper entry point is the **unchanged native kernel**, with CPU sorting explicitly selected.', '''The timed Paper entry point is the **unchanged native kernel**. `Paper CPU` passes `cuda_sort=False`; `Paper GPU sort` passes `True`. In the latter, CPU prefix sums and candidates precede score upload, CUDA stable sort, index download, and CPU greedy selection, as in [Appendix E](https://openreview.net/pdf?id=3xwsD68F2K) and the checked-out native source.''')
replace('definition', 'The public Python policy wrapper, importance extraction, compilation, disk reads, and GPU transfers are excluded. Thus this compares prepared native CPU calls, not the GPU-to-GPU online path. Python binding and timer overhead are included for all three methods.', '''The public Python policy wrapper, importance extraction, compilation, and disk reads are excluded. Score/index transfers inside the hybrid kernel are included; the initial GPU-activation transfer and final GPU-mask transfer are not. Both Paper variants return a CPU mask, as do the unchanged C++ tiles and Top-$R$. Binding and timer overhead are included.''')
replace('native-build', '    from torch.utils.cpp_extension import load', '''    if RUN_CUDA:
        assert torch.cuda.is_available(), "RUN_CUDA requires CUDA PyTorch and a visible GPU; no CPU fallback."
        torch.cuda.init()
        torch.cuda.synchronize()
    from torch.utils.cpp_extension import load''')
replace('inputs', '    quality = pd.read_csv(QUALITY)', '''    if RUN_CUDA:
        assert all(t["num_tokens"] > 0 for t in traces), "Captured token counts are required."
        assert manifest["model_dtype"] in ("float16", "bfloat16", "float32")
        assert manifest["model_dtype"] != "bfloat16" or torch.cuda.is_bf16_supported()
    quality = pd.read_csv(QUALITY)''')
replace('timing', 'No forced GPU synchronization is needed because every timed kernel is CPU-only.', '''In CUDA mode, synchronize before each call, outside the timer. The hybrid kernel's blocking index copy back to CPU completes its GPU sort before return; the timer includes that round trip. Warm-up includes the actual sort path.''')
replace('timing', 'Set `RUN_BENCHMARK = True` to replace the CSV, or leave it false to read the report without compilation.', '''Set both `RUN_BENCHMARK = True` and `RUN_CUDA = True` for the new experiment. CPU and hybrid selection are remeasured together, then GEMM is timed locally; earlier CPU timings are not mixed into this run. CUDA mode writes `selection_cuda_results.csv` and `environment_cuda.json`, leaving the legacy files untouched. To replay, keep `RUN_CUDA = True` and set `RUN_BENCHMARK = False`. CUDA absence fails before measurement, not under a GPU label.''')
replace('measurement', '"Paper greedy": lambda: paper_native.select_chunks(values, R, sizes, delays, stride, True)', '"Paper CPU": lambda: paper_native.select_chunks(values, R, sizes, delays, stride, False)')
replace('measurement', '            checked = {}', '''            if RUN_CUDA:
                functions["Paper GPU sort"] = lambda: paper_native.select_chunks(values, R, sizes, delays, stride, True)
            checked = {}''')
replace('measurement', 'old = quality.loc[(trace["trace_id"], round(float(fraction),6), name)]', '''reference = "Paper greedy" if name.startswith("Paper") else name
                old = quality.loc[(trace["trace_id"], round(float(fraction),6), reference)]''')
replace('measurement', 'if name == "Paper greedy" else', 'if name.startswith("Paper") else')
replace('measurement', '                    begin = perf_counter_ns()', '''                    if RUN_CUDA: torch.cuda.synchronize()
                    begin = perf_counter_ns()''')
replace('measurement', 'method=name,\n                    median_us=', '''method=name,
                    cuda_sort=name == "Paper GPU sort", num_tokens=int(trace["num_tokens"]),
                    d=int(trace["d"]), dtype=manifest["model_dtype"],
                    median_us=''')
mm_definition = r'''## Measure the multiplication used by the flash path.

The checked-out [`SparseLinear.forward`](../../../upstream/vlm-flash/src/vlmflash/linear.py) uses `x[..., mask] @ w_rows.to(x.dtype)` after reading the selected, transposed weight rows. Its in-memory path instead masks activations before a full dense linear operation. We time the **packed flash-path multiplication**, not that dense path:

$$A\in\mathbb R^{p\times K},\quad B\in\mathbb R^{K\times d},\quad Y=AB,\qquad K=|M|,\quad \mathrm{FLOPs}\simeq2pKd.$$

Here $p$ is the trace's archived token count and $d$ its output width; dtype also comes from the archive. Paper's underfilled mask uses its own $K$, not nominal $R$. Random dense operands stand in for already packed activations and resident weight rows. Their values are synthetic; the dimensions are not. No checkpoint is loaded.

One warmed measurement per distinct $(p,K,d,\mathrm{dtype})$ is shared across matching cases. Each sample is a synchronized host-wall interval around `A @ B`, including launch and output allocation. Operand preparation is outside the timer. These are isolated packed-GEMM costs, not complete forward times; equal shapes share the same estimate by construction. [CUDA synchronization](https://docs.pytorch.org/docs/2.10/notes/cuda.html#asynchronous-execution) is necessary because GPU launches are asynchronous.'''
mm_code = '''def matmul_samples(tokens, selected, width, dtype, device="cuda"):
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    with torch.inference_mode():
        a = torch.randn(tokens, selected, dtype=dtype, device=device)
        b = torch.randn(selected, width, dtype=dtype, device=device)
        samples = []
        for repeat in range(WARMUP + REPEATS):
            sync()
            begin = perf_counter_ns()
            y = a @ b
            sync()
            elapsed = (perf_counter_ns() - begin) / 1e6
            if repeat >= WARMUP: samples.append(elapsed)
    return samples


if RUN_BENCHMARK and RUN_CUDA:
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    measured = pd.DataFrame(records)
    dimensions = ["num_tokens", "R_selected", "d", "dtype"]
    compute = []
    for key in measured[dimensions].drop_duplicates().itertuples(index=False, name=None):
        p, k, d, dtype = key
        samples = matmul_samples(p, k, d, getattr(torch, dtype))
        compute.append(dict(zip(dimensions, key), matmul_ms=np.median(samples),
            matmul_p95_ms=np.percentile(samples, 95), matmul_samples_ms=json.dumps(samples)))
    measured = measured.merge(pd.DataFrame(compute), on=dimensions, how="left", validate="many_to_one")
    assert measured.matmul_ms.notna().all() and (measured.matmul_ms > 0).all()
    measured.to_csv(RESULTS, index=False)'''
idx = next(i for i,c in enumerate(nb['cells']) if c['id']=='environment')
nb['cells'][idx:idx] = [dict(cell_type='markdown',id='matmul-definition',metadata={},source=mm_definition.splitlines(keepends=True)), dict(cell_type='code',id='matmul-measurement',metadata={},execution_count=None,outputs=[],source=mm_code.splitlines(keepends=True))]
replace('environment', 'cuda_sort=False,', '''cuda_sort=RUN_CUDA,
        cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name() if RUN_CUDA else None,
        matmul_dtype=manifest["model_dtype"] if RUN_CUDA else None,
        matmul_tf32=False if RUN_CUDA else None,''')
replace('environment', 'source_commit="cc4f8c589716fbcaee186cc9f2b928193d25a65e"', 'notebook_base="78c1bc3074f1147da29aabc8b4427f450260265f"')
put('environment', source('environment') + '''
if not frame.empty and "cuda_sort" not in frame:
    display(Markdown("Legacy run: sort mode was not recorded per method; its old `cuda_sort` metadata is not reliable. Rerun to identify CPU and GPU-sort timings."))
paper_methods = [name for name in ("Paper CPU", "Paper GPU sort", "Paper greedy") if not frame.empty and name in set(frame.method)]''')
replace('paired-summary', 'A value above one favors tiles.', 'A value above one favors tiles; compute it separately for each Paper variant.')
put('summary', '''if not frame.empty:
    times = frame.pivot(index=["shape","trace_id","fraction"],columns="method",values="median_us")
    speedup = times[paper_methods].div(times["Saturation tiles"], axis=0)
    summary = frame.groupby(["shape","method"]).median_us.median().unstack()
    display(summary.round(3))
    display(speedup.groupby("shape").median().round(3).rename_axis(columns="Paper / Tiles"))
    ax = summary.plot.bar(figsize=(8,3.5),rot=0)
    ax.set(ylabel="Median selection time (µs; log scale)", xlabel="Projection shape", yscale="log")
    plt.tight_layout()
    plt.show()''')
replace('quality-caption', 'all three methods', 'all measured variants')
replace('quality-plot', 'frame.method == "Paper greedy"', 'frame.method == paper_methods[0]')
put('conclusion', '''if not frame.empty:
    for name in paper_methods:
        low, high = speedup[name].quantile([.05,.95])
        display(Markdown(
            f"Across **{len(times):,}** cases, **{name} / Tiles** has median **{speedup[name].median():.2f}×** "
            f"(5th–95th case percentiles **{low:.2f}–{high:.2f}×**)."))
    display(Markdown(f"**{int(frame.mask_matches_02.sum()):,}/{len(frame):,}** masks match Notebook 02 exactly; "
        "all passed selected-count, importance, and merged-cost checks."))''')
put('combined-definition', r'''## Compose selection, read, and multiplication costs.

Join by the same trace, requested budget, actual selected count, and reference method. Both Paper variants use Paper's checked mask cost from Notebook 02. In milliseconds,

$$\widehat C_{j,m}=\widehat t_{j,m}/1000+\widehat L(M_{j,m}),\qquad
\boxed{\widehat T_{j,m}^{\rm projection}=\widehat C_{j,m}+\widehat G(p_j,|M_{j,m}|,d_j,\mathrm{dtype}).}$$

$\widehat G$ is the packed-GEMM median measured above. It is added only when present in the same run's CSV; legacy selection samples are never combined with a later GPU measurement. Without CUDA results, the report displays selection plus modeled I/O only.

[Figure 8](https://openreview.net/pdf?id=3xwsD68F2K) separates compute, I/O and selection, and the paper states that its evaluation does not overlap I/O and compute. This sum adopts a serial composition, **not** a reproduction of that full-model experiment. The submodule's `run.py` stores summed read timers as `measured_total`, not complete wall time; `native.cc` times weight upload separately from `io_us`. Here I/O remains Notebook 01's fitted profile, possibly from another device. Importance extraction, activation gathering, initial/final selector transfers, weight upload, bias, attention and other layers are absent. Therefore $\widehat T^{\rm projection}$ is not measured $T_{\rm wall}$.

On equal-count cases, identical packed-GEMM shapes share $G$, so adding it dilutes a partial speedup: $(C_P+G)/(C_T+G)$ approaches one as $G$ grows. Equal counts still do not establish equal importance or complete forward cost.''')
put('combined-cost', '''if not frame.empty:
    keys = ["shape", "trace_id", "fraction", "R_nominal", "R_selected", "reference_method"]
    io = pd.read_csv(QUALITY).rename(columns={"method": "reference_method", "latency": "modeled_io_ms"})
    io["fraction"] = io.fraction.round(6)
    reference = frame.method.where(~frame.method.str.startswith("Paper"), "Paper greedy")
    online = frame.assign(fraction=frame.fraction.round(6), reference_method=reference).merge(
        io[keys + ["modeled_io_ms"]], on=keys, how="left", validate="many_to_one")
    assert online.modeled_io_ms.notna().all(), "02 must contain the same cases and selected counts."
    online["selection_ms"] = online.median_us / 1000
    online["combined_ms"] = online.selection_ms + online.modeled_io_ms
    components = ["selection_ms", "modeled_io_ms"]
    metric = "combined_ms"
    if "matmul_ms" in online:
        assert online.matmul_ms.notna().all() and (online.matmul_ms > 0).all()
        online["projection_estimate_ms"] = online.combined_ms + online.matmul_ms
        components += ["matmul_ms"]
        metric = "projection_estimate_ms"
    label = "Selection + modeled I/O + packed GEMM" if "matmul_ms" in online else "Selection + modeled I/O"
    for shape, data in online.groupby("shape"):
        curves = data.groupby(["fraction", "method"])[metric].median().unstack()
        ax = curves.plot(marker="o", figsize=(7, 3.5), logy=True)
        ax.set(title=shape, xlabel="Requested fraction R/N", ylabel=label + " (ms; log scale)")
        plt.tight_layout()
        plt.show()''')
replace('combined-pairing', 'Each column is a median across those same cases; the combined column is the median of per-case sums, not the sum of the other two medians.', '''Each column is a median across those same cases; the final column is the median of per-case sums, not a sum of marginal medians. CPU and GPU-sort Paper are reported separately. This is a same-count comparison, not a matched-accuracy speedup.''')
replace('combined-summary', 'online.method == "Paper greedy"', 'online.method == paper_methods[0]')
replace('combined-summary', '[["selection_ms", "modeled_io_ms", "combined_ms"]]', '[components + [metric]]')
replace('combined-summary', 'values="combined_ms"', 'values=metric')
a = source('combined-summary')
start = a.index('        ratio = paired["Paper greedy"]')
end = a.index('    else:', start)
a = a[:start] + '''        for name in paper_methods:
            ratio = paired[name] / paired["Saturation tiles"]
            saving = paired[name] - paired["Saturation tiles"]
            display(Markdown(
                f"On **{len(paired):,}/{len(paper_rows):,}** exact-fill cases, Tiles cost less than **{name}** in "
                f"**{int((saving > 0).sum()):,}/{len(paired):,}** pairs. Median paired ratio: "
                f"**{ratio.median():.2f}×**; saving: **{saving.median():.3f} ms**. "
                f"Metric: {label}. This is a composed estimate, not measured wall time or equal-quality gain."))
''' + a[end:]
put('combined-summary', a)
put('scope', '''The timed selector calls have CPU-resident input and output. Only Paper's score sorting optionally uses CUDA; tiles remain C++ CPU. The CUDA experiment also measures a shape-equivalent packed GEMM, using archived token counts and actual selected rows. No extra VLM runtime or SSD benchmark is introduced. The saved I/O model is still an assumption when forming the projection estimate, and omitted costs must not be called zero.

The legacy CSV predates explicit per-method sort labels. Its environment file contains a hard-coded `cuda_sort=False` although the HEAD call passed `True`; that run's actual sort device cannot be established from this metadata. The file is preserved, not relabelled as verified CPU or GPU evidence. New runs record both variants explicitly, the CUDA runtime, GPU, dtype, and raw GEMM samples.

Sources: [Notebook 02](../02/02_global_oracle.ipynb), [reference policy](../../../upstream/vlm-flash/src/vlmflash/policy.py), [native selector and read timers](../../../upstream/vlm-flash/src/vlmflash/csrc/native.cc), [flash-path multiplication](../../../upstream/vlm-flash/src/vlmflash/linear.py), [I/O-only run aggregation](../../../upstream/vlm-flash/scripts/run.py), and [PyTorch CUDA timing](https://docs.pytorch.org/docs/2.10/notes/cuda.html#asynchronous-execution).''')
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print('Python lines:', sum(len(''.join(c['source']).splitlines()) for c in nb['cells'] if c['cell_type']=='code'))
