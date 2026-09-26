"""One-time notebook editor; removed before the pull request is finalized."""
from pathlib import Path
from io import StringIO
import json
import nbformat as nbf
import pandas as pd

ROOT = Path.cwd()
P1 = ROOT / 'conclusion/python/results/01/01_chunk_latency.ipynb'
P2 = ROOT / 'conclusion/python/results/02/02_global_oracle.ipynb'
nb1 = nbf.read(P1, as_version=4)
old2 = nbf.read(P2, as_version=4)
old = {c.id: c for c in old2.cells}

def md(id, text):
    return nbf.v4.new_markdown_cell(text.strip(), id=id)

def code(id, text, *, folded=False, recipe=False):
    metadata = {}
    if folded:
        metadata = {'jupyter': {'source_hidden': True}, 'tags': ['hide-input']}
    if recipe:
        metadata.setdefault('tags', []).append('recipe')
    return nbf.v4.new_code_cell(text.strip(), id=id, metadata=metadata)

# 01: keep the native acquisition cell and all fitting definitions unchanged.
by1 = {c.id: c for c in nb1.cells}
by1['setup'].source = by1['setup'].source.replace('RERUN_MEASUREMENTS = True', 'RERUN_MEASUREMENTS = False')
by1['title'].source = by1['title'].source.split('The figures use the previously recorded')[0] + (
    'The figures and numerical equations are computed from the current `block_estimates.csv`. '
    'The saved native measurements are reused by default; evaluating this report does not rerun the SSD experiment. '
    'The fitted training model is exported for the next notebook with the SHA-256 of its source CSV.')
new1 = [md('parameters-definition', r'''
## The fitted coefficients specify the model, not just its error.

The following equations substitute the fitted coefficients into the three model families. All coefficients use the first four blocks; the held-out blocks do not refit them. Displayed numbers are rounded, while predictions and the model exported to notebook 02 retain full precision. The parameter table separates the short and long intercepts, so a hinge is not mistaken for a zero-intercept tail.
'''), code('numerical-models', r'''
c1, delta = nnls(X_s(s_star), T_train)[0]
ha, hb, hd = np.linalg.lstsq(X_h(h_star), T_train, rcond=None)[0]
aa, ab = np.linalg.lstsq(X_a, T_train, rcond=None)[0]
np.testing.assert_allclose(X_s(s_star) @ [c1, delta], predictions[free_line])
np.testing.assert_allclose(X_h(h_star) @ [ha, hb, hd], predictions[free_hinge])
np.testing.assert_allclose(X_a @ [aa, ab], predictions['Affine'])
display(Markdown(fr'''
The fitted two-line model, with $z$ in KiB, is

$$\boxed{{\widehat T_{{2L}}(z)=\begin{{cases}}
{1000*delta*s_star:.3f}+{1000*c1:.6f}z,&0<z\le {s_star:g},\\
{1000*(c1+delta):.6f}z,&z>{s_star:g},
\end{{cases}}\qquad\mu\mathrm{{s}}.}}$$

The other fitted families are

$$\widehat T_H(z)={1000*ha:.3f}{1000*hb:+.6f}z{1000*hd:+.6f}(z-{h_star:g})_+\quad\mu\mathrm{{s}},$$
$$\widehat T_A(z)={1000*aa:.3f}{1000*ab:+.6f}z\quad\mu\mathrm{{s}}.$$
'''))
'''.replace("r'''", "r\"\"\"") if False else '''
c1, delta = nnls(X_s(s_star), T_train)[0]
ha, hb, hd = np.linalg.lstsq(X_h(h_star), T_train, rcond=None)[0]
aa, ab = np.linalg.lstsq(X_a, T_train, rcond=None)[0]
np.testing.assert_allclose(X_s(s_star) @ [c1, delta], predictions[free_line])
np.testing.assert_allclose(X_h(h_star) @ [ha, hb, hd], predictions[free_hinge])
np.testing.assert_allclose(X_a @ [aa, ab], predictions["Affine"])
display(Markdown(fr"""
The fitted two-line model, with $z$ in KiB, is

$$\\boxed{{\\widehat T_{{2L}}(z)=\\begin{{cases}}
{1000*delta*s_star:.3f}+{1000*c1:.6f}z,&0<z\\le {s_star:g},\\\\
{1000*(c1+delta):.6f}z,&z>{s_star:g},
\\end{{cases}}\\qquad\\mu\\mathrm{{s}}.}}$$

The other fitted families are

$$\\widehat T_H(z)={1000*ha:.3f}{1000*hb:+.6f}z{1000*hd:+.6f}(z-{h_star:g})_+\\quad\\mu\\mathrm{{s}},$$
$$\\widehat T_A(z)={1000*aa:.3f}{1000*ab:+.6f}z\\quad\\mu\\mathrm{{s}}.$$
"""))
''', folded=True), code('parameter-table', '''
parameter_rows = []
for boundary, name in [(240., "Two-line (240 KiB)"), (256., "Two-line (256 KiB)"),
                       (230., "Two-line (230 KiB)"), (float(s_star), free_line)]:
    c, d = nnls(X_s(boundary), T_train)[0]
    parameter_rows.append([name, boundary, 1000*d*boundary, 1000*c, 0., 1000*(c+d)])
parameter_rows += [[free_hinge, float(h_star), 1000*ha, 1000*hb,
                   1000*(ha-hd*h_star), 1000*(hb+hd)],
                  ["Affine", np.nan, 1000*aa, 1000*ab, 1000*aa, 1000*ab]]
parameters = pd.DataFrame(parameter_rows, columns=["Model", "Boundary (KiB)",
    "Short intercept (µs)", "Short slope (µs/KiB)", "Tail intercept (µs)",
    "Tail slope (µs/KiB)"]).set_index("Model")
parameters.round(6)
''', folded=True), code('export-model', '''
import hashlib, json
from tempfile import NamedTemporaryFile

model = dict(format="vlm-in-flash-two-line-v1", unit=dict(length="KiB", time="ms"),
    csv_sha256=hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
    fit_blocks=wide.columns[:4].tolist(), holdout_blocks=wide.columns[4:].tolist(),
    s_kib=float(s_star), c1_ms_per_kib=float(c1), delta_ms_per_kib=float(delta),
    s99_kib=float(s99), measured_grid_kib=z.tolist(),
    fit="unweighted NNLS; free boundary on measured grid [3:-3]",
    acquisition_reference="9023679ebfd1900fa2912c4098f51bfb9efa7bb8")
model_path = CSV_PATH.with_name("latency_model.json")
with NamedTemporaryFile(mode="w", dir=model_path.parent, delete=False, encoding="utf-8") as f:
    json.dump(model, f, indent=2, allow_nan=False)
    f.write("\\n")
Path(f.name).replace(model_path)
''', folded=True)]
insert = next(i for i,c in enumerate(nb1.cells) if c.id == 'predictions') + 1
nb1.cells[insert:insert] = new1
nbf.write(nb1, P1)

# Keep the old host timings as an explicitly historical planning reference.
historical = None
for output in old['global-dp-32'].get('outputs', []):
    html = output.get('data', {}).get('text/html')
    if html:
        if isinstance(html, list):
            html = ''.join(html)
        historical = pd.read_html(StringIO(html))[0].to_dict('records')
        break
if historical is None:
    raise ValueError('The old scaling table is needed to label the planning estimate honestly.')
for row in historical:
    for key in list(row):
        if str(key).startswith('Unnamed'):
            del row[key]

cells = []
def M(id, text): cells.append(md(id, text))
def C(id, text, **kw): cells.append(code(id, text, **kw))
M('title', r'''
# A measured cost makes the global fixed-row comparison concrete.

Seol Handong

The fixed-budget question in [idea.md](../../../../preliminary_research/idea.md) is

$$\boxed{\max_{\|M\|_1=R}\frac{I(M)}{\sum_{C\in\mathcal C(M)}T(|C|)}.}$$

We retain the last-zero dynamic program, but replace its illustrative cost and synthetic main inputs. Notebook 01 supplies a fitted native-SSD profile; Experiment 22 supplies dense-forward activation importance. The released selector supplies the comparison mask. Every competing mask is evaluated at its **realized** row count.

This is an offline optimum under an additive fitted model, not a measured end-to-end speedup. The archive contains text-model activations, not a new VLM evaluation. All algorithms and report calculations remain in this notebook. A normal evaluation loads available results; the full experiment and the large scaling study require separate explicit switches.
''')
C('setup', '''
from pathlib import Path
from itertools import product
from time import perf_counter
from tempfile import NamedTemporaryFile
import hashlib, json, platform, sys
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d
from IPython.display import Markdown, display

HERE = Path.cwd()
ROOT = next(p for p in (HERE, *HERE.parents) if (p / ".gitmodules").is_file())
PROFILE = ROOT / "conclusion/python/results/01/block_estimates.csv"
MODEL = PROFILE.with_name("latency_model.json")
TRACE = ROOT / "experiments/22_predicted_lambda_trim/results/activation_traces.npz"
UPSTREAM = ROOT / "conclusion/upstream/vlm-flash"
UPSTREAM_SHA = "9023679ebfd1900fa2912c4098f51bfb9efa7bb8"
RUN_ORACLE = False
RUN_SCALING = False
MAX_TRACES_PER_SHAPE = 0  # 0: every archived trace; 32: the earlier 128-trace subset.
BUDGETS = tuple(np.arange(1, 10) / 10)
RTOL, MAX_TABLE_MIB = 1e-6, 1024
PAPER_IMPL = "native"  # Explicit; no silent downgrade to another implementation.
SCALING_SIZES = (128, 256, 512, 896, 2048, 4864, 8960, 14336, 18944)
SCALING_REPEATS = 3
''', folded=True)
M('cost-definition', r'''
## The row cost inherits the fitted KiB boundary without rounding it.

Read the full-precision model exported by [notebook 01](../01/01_chunk_latency.ipynb), rather than transcribing its displayed decimals or fitting a second model. The export is rejected if its CSV fingerprint is stale.

For a projection with $d$ output columns and $w$ bytes per stored weight, one input-channel row occupies $q=wd/1024$ KiB. We assume storage in the archive's floating-point dtype; the archive contains importance vectors, not serialized weight files. This assumption does not cover quantized or compressed storage.

$$T_q(r)=c_1qr+\delta\max(s_{\rm KiB},qr),\quad r>0,\qquad T_q(0)=0.$$

Equivalently, put $s=s_{\rm KiB}/q$, $b_1=qc_1$, $b_2=q(c_1+\delta)$ and $a=(b_2-b_1)s$. Then

$$T_q(r)=\begin{cases}a+b_1r,&1\le r\le s,\\b_2r,&r>s.\end{cases}$$

The real boundary $s$ is generally fractional. Only the short-run **index range** uses $h=\lfloor s\rfloor$; the intercept remains $a=\delta s_{\rm KiB}$. Rounding $s$ in the cost would define a different model.
''')
C('model-input', '''
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

model = json.loads(MODEL.read_text())
assert model["format"] == "vlm-in-flash-two-line-v1"
assert model["unit"] == {"length": "KiB", "time": "ms"}
if model["csv_sha256"] != sha256(PROFILE):
    raise ValueError("Stale latency_model.json: evaluate notebook 01 with RERUN_MEASUREMENTS=False.")
s_kib = float(model["s_kib"])
c1 = float(model["c1_ms_per_kib"])
delta = float(model["delta_ms_per_kib"])
s99 = float(model["s99_kib"])
assert np.isfinite([s_kib, c1, delta, s99]).all()
assert s_kib > 0 and c1 >= 0 and delta >= 0 and c1 + delta > 0
profile = pd.read_csv(PROFILE)
wide = profile.pivot(index="size_kib", columns="block", values="latency_ms").sort_index()
z_grid = wide.index.to_numpy(float)
np.testing.assert_array_equal(z_grid, model["measured_grid_kib"])
T_holdout = np.median(wide[model["holdout_blocks"]].to_numpy(), axis=1)
assert np.isfinite(T_holdout).all() and (T_holdout > 0).all()
display(Markdown(f"The inherited boundary is **{s_kib:g} KiB**; the short and tail slopes are "
    f"**{1000*c1:.6f}** and **{1000*(c1+delta):.6f} µs/KiB**. "
    "The same full-precision coefficients generate both the greedy candidate costs and the oracle objective."))
''', folded=True, recipe=True)
C('costs', '''
def costs(n, s, b1, b2):
    r = np.arange(n + 1, dtype=float)
    T = b1 * r + (b2 - b1) * np.maximum(s, r)
    T[0] = 0.
    return T


def runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def metrics(v, mask, T):
    intervals = runs(mask)
    return float(v[mask].sum()), float(T[np.diff(intervals, axis=1).ravel()].sum())


def top_r(v, R):
    mask = np.zeros(len(v), bool)
    mask[np.argsort(-v, kind="stable")[:R]] = True
    return mask
''', recipe=True)
cells.append(old['global-dp-04'])
M('range-recurrence', r'''
## Two range maxima remove the scan over run lengths.

For the fixed zero count $z$, define $A_t=D_{z-1,t}-P_{z+t}+\lambda b_1t$ and $B_t=D_{z-1,t}-P_{z+t}+\lambda b_2t$. Here $\lambda$ is the trial ratio called $q$ in the preceding recurrence; it is unrelated to the row size $q$ above. With $h=\lfloor s\rfloor$,

$$D_{z,r}=\max\left\{\begin{aligned}
&D_{z-1,r},\\
&P_{z+r}-\lambda(a+b_1r)+\max_{\max(0,r-h)\le t<r}A_t,\\
&P_{z+r}-\lambda b_2r+\max_{0\le t<r-h}B_t.
\end{aligned}\right.$$

An empty maximum is $-\infty$. The short range is a trailing maximum; the long range is a prefix maximum. When $s<1$, there are no positive short runs. The [SciPy moving maximum](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter1d.html) is linear in its input length. Thus one value-only solve takes $O((N-R+1)(R+1))$ time and $O(R+1)$ working memory.

The ratio search needs values, not a traceback at every step. Only its final feasible boundary retains the full table for recovery. This saves memory traffic without restricting the mask family.
''')
C('bellman', '''
def bellman(v, R, q, s, b1, b2, *, retain=False):
    Z = len(v) - R
    P = np.r_[0., np.cumsum(v)]
    r = np.arange(R + 1)
    previous = P[:R + 1] - q * costs(R, s, b1, b2)
    D = np.empty((Z + 1, R + 1)) if retain else None
    if retain:
        D[0] = previous
    h = min(int(np.floor(s)), R)
    for z in range(1, Z + 1):
        prefix = P[z:z + R + 1]
        base = previous - prefix
        short = np.full(R + 1, -np.inf)
        if h:
            short[1:] = maximum_filter1d(base[:-1] + q * b1 * r[:-1],
                size=h, origin=(h - 1) // 2, mode="constant", cval=-np.inf)
        long = np.full(R + 1, -np.inf)
        long[h + 1:] = np.maximum.accumulate(base + q * b2 * r)[:R - h]
        previous = np.maximum.reduce([previous,
            prefix - q * ((b2 - b1) * s + b1 * r) + short,
            prefix - q * b2 * r + long])
        if retain:
            D[z] = previous
    return D if retain else float(previous[-1])
''', recipe=True)
M('recovery-definition', r'''
### Recover a mask at the final feasible ratio.

At $(z,r)$, rescan the original recurrence for a maximizing $t$, select $[z+t,z+r)$, and continue at $(z-1,t)$. At $z=0$, the remaining prefix consists of ones. Ties may produce different optimal masks. The final recovery takes another $O((N-R+1)(R+1))$ operations and retains one float64 value table. Its memory is not the total process memory.
''')
cells.append(old['global-dp-09'])
cells[-1].metadata.setdefault('tags', []).append('recipe')
M('ratio-definition', r'''
## A bracket bounds the remaining optimization error.

Top-$R$ supplies a feasible lower bound $\ell_0$ and the upper bound

$$u_0=\frac{I(M_{\rm Top-R})}{b_2R},\qquad T_q(r)\ge b_2r.$$

Bisect $[\ell,u]$. A nonnegative additive optimum moves the feasible boundary to the midpoint; a negative value moves the upper boundary. Stop at $u-\ell\le\epsilon\ell_0$, then recover a mask at $\ell$. In exact arithmetic its actual ratio $u_M$ satisfies

$$\boxed{0\le1-u_M/\eta_R^\star\le\epsilon.}$$

For fractional $s>0$, use $\kappa=\max(1,T_q(1)/b_2)$ rather than an integer-boundary shortcut. At most $\lceil\log_2(\kappa/\epsilon)\rceil+1$ value calls suffice, followed by one retained-table solve. Zero importance is handled separately. The full arithmetic bound is

$$O\!\left(N\log N+(N-R+1)(R+1)\log(\kappa/\epsilon)\right).$$

The implementation uses float64, checks the recovered additive value and reports the final bracket. These are numerical consistency checks, not an interval-arithmetic certificate against rounding error. The word global refers to the declared fitted objective, not to measured SSD wall time or all possible accuracy objectives.
''')
C('ratio-solve', '''
def solve(v, R, s, b1, b2, rtol=1e-6, max_table_mib=1024):
    v = np.asarray(v, dtype=float)
    assert v.ndim == 1 and np.isfinite(v).all() and (v >= 0).all()
    assert isinstance(R, (int, np.integer)) and 1 <= R <= len(v)
    assert np.isfinite([s, b1, b2]).all() and s > 0 and 0 <= b1 <= b2 and b2 > 0
    assert 0 < rtol < 1
    T = costs(len(v), s, b1, b2)
    initial = top_r(v, R)
    I, L = metrics(v, initial, T)
    lower, upper = I / L, I / (b2 * R)
    if I == 0:
        return initial, 0., 0., 0
    tolerance = rtol * lower
    kappa = max(1., T[1] / b2)
    max_calls = int(np.ceil(np.log2(kappa / rtol))) + 1
    mib = 8 * (len(v) - R + 1) * (R + 1) / 2**20
    if mib > max_table_mib:
        raise MemoryError(f"Traceback table needs {mib:.1f} MiB; limit is {max_table_mib}.")
    calls = 0
    while upper - lower > tolerance:
        q = (lower + upper) / 2
        if q == lower or q == upper or calls >= max_calls:
            raise ArithmeticError("Ratio search did not attain the requested bracket.")
        if bellman(v, R, q, s, b1, b2) >= 0:
            lower = q
        else:
            upper = q
        calls += 1
    D = bellman(v, R, lower, s, b1, b2, retain=True)
    mask = recover(v, R, lower, T, D)
    I, L = metrics(v, mask, T)
    ratio = I / L
    np.testing.assert_allclose(I - lower * L, D[-1, -1], rtol=1e-9,
                               atol=1e-9 * max(1., v.sum()))
    assert mask.sum() == R
    slack = 1e-10 * max(1., upper)
    if ratio + slack < lower or ratio > upper + slack:
        raise ArithmeticError("Recovered ratio conflicts with the numerical bracket.")
    upper = max(upper, ratio)
    if upper - ratio > rtol * ratio + slack:
        raise ArithmeticError("Recovered mask misses the requested ratio tolerance.")
    return mask, ratio, upper, calls + 1
''', recipe=True)
M('verification-definition', '''
### Enumeration checks the recurrence; it is not the main experiment.

At most nine channels are enumerated. Every positive budget is checked against an independent run counter, including zero and tied importance, equal slopes, fractional boundaries, boundaries below one row and boundaries beyond the vector. The two-row and retained-table Bellman paths must agree. No real trace sweep is executed by this check.
''')
cells.append(old['global-dp-14'])
C('verification', '''
checks = 0
for n in (1, 5, 9):
    for v in (np.zeros(n), np.ones(n), np.random.default_rng(n).uniform(0, 4, n)):
        for boundary in (0.5, 1., 1.5, 4.25, n + 2.):
            for short_slope in (0., .25, 1.):
                T = costs(n, boundary, short_slope, 1.)
                masks, imp, delay = enumerate_small(v, T)
                for R in range(1, n + 1):
                    allowed = masks.sum(axis=1) == R
                    for q in (0., .1, 1., 10.):
                        D = bellman(v, R, q, boundary, short_slope, 1., retain=True)
                        mask = recover(v, R, q, T, D)
                        I, L = metrics(v, mask, T)
                        exact = np.max(imp[allowed] - q * delay[allowed])
                        assert mask.sum() == R
                        np.testing.assert_allclose([D[-1, -1], I-q*L,
                            bellman(v,R,q,boundary,short_slope,1.)], exact, atol=1e-9)
                        checks += 1
                    mask, lower, upper, calls = solve(v,R,boundary,short_slope,1.)
                    exact = np.max(imp[allowed] / delay[allowed])
                    assert lower <= exact + 1e-9 and exact <= upper + 1e-9
                    assert upper-lower <= 1e-6*max(lower,1.) + 1e-9
''', folded=True)
M('trace-definition', r'''
## Real activation traces replace the synthetic main inputs.

Read the existing [Experiment 22 archive](../../../../experiments/22_predicted_lambda_trim/results/activation_traces.npz). Its manifest records model revision, prompts, projection names, shapes and dtype. Importance is $v_i=\operatorname{mean}_{\rm batch,token}|x_i|$ from an unperturbed dense forward. No model is downloaded or executed here.

All archived traces are used by default. Setting `MAX_TRACES_PER_SHAPE = 32` selects the same evenly spaced per-shape subset rule used by Experiment 22. We retain every channel; this is not a crop to make the oracle easier. Mean normalization rescales the objective but not its exact optimum; the normalized float32 vector passed to upstream is also evaluated by the float64 oracle.

Traces from one prompt or layer are dependent observations. Shape- and prompt-level summaries are descriptive; thousands of budget cases do not create thousands of independent prompts.
''')
C('trace-input', '''
def load_traces(path, cap=0):
    assert isinstance(cap, int) and cap >= 0
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        assert manifest["format"] == "experiment-22-vlm-activation-traces-v1"
        traces = []
        for entry in manifest["traces"]:
            t = {k: v for k, v in entry.items() if k != "array"}
            v = np.asarray(archive[entry["array"]], dtype=np.float32).copy()
            assert v.shape == (t["n"],) and t["shape"] == f"{t['n']}x{t['d']}"
            assert np.isfinite(v).all() and (v >= 0).all()
            t["values"] = v
            traces.append(t)
    assert len(traces) == manifest["trace_count"]
    assert len({t["trace_id"] for t in traces}) == len(traces)
    chosen = []
    for shape in sorted({t["shape"] for t in traces}):
        group = [t for t in traces if t["shape"] == shape]
        indices = np.linspace(0, len(group)-1, cap, dtype=int) if cap and len(group)>cap else range(len(group))
        chosen.extend(group[i] for i in indices)
    return sorted(chosen, key=lambda t: t["trace_id"]), manifest


def importance_vector(trace):
    v = trace["values"].astype(float)
    return (v / v.mean()).astype(np.float32).astype(float) if v.sum() else v

traces, trace_manifest = load_traces(TRACE, MAX_TRACES_PER_SHAPE)
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
weight_bytes = DTYPE_BYTES[trace_manifest["model_dtype"]]
shape_frame = pd.DataFrame([{k:t[k] for k in ("shape", "n", "d")} for t in traces])
shape_summary = shape_frame.groupby(["shape", "n", "d"]).size().rename("traces").reset_index()
shape_summary["KiB/row"] = weight_bytes * shape_summary.d / 1024
shape_summary["boundary (rows)"] = s_kib / shape_summary["KiB/row"]
shape_summary["cases"] = shape_summary.traces * len(BUDGETS)
display(Markdown(f"**{len(traces)}** selected traces from **{trace_manifest['model']}**, "
    f"revision `{trace_manifest['model_revision']}`, yield **{len(traces)*len(BUDGETS)}** budget cases. "
    f"The archive records **{trace_manifest['prompt_count']}** prompts; dtype is **{trace_manifest['model_dtype']}**."))
shape_summary.round(4)
''', folded=True, recipe=True)
M('baseline-definition', r'''
## The released selector fixes the realized budget.

Call the pinned submodule's [`select_chunks`](../../../upstream/vlm-flash/src/vlmflash/policy.py); do not copy its greedy loop. Use Experiment 22's start/stride settings for each archived shape. Its candidate costs come from the same fitted model as the oracle, and the exclusive upper window bound is $s_{99}+1$ KiB. The upstream window quantization, stable sorting and non-overlap rules remain unchanged. This is the released **algorithm retargeted to the fitted device profile**, not a claim to reproduce its bundled AGX table or GPU runtime.

For nominal $R_0=\lfloor fN\rfloor$, obtain $M_P$ and set $R=\|M_P\|_1$. Solve the oracle, Top-$R$, and saturation tiles at this $R$. Empty upstream selections are recorded as exclusions rather than given an undefined ratio. Recompute every cost over maximal runs; `Selection.est_cost_ms` is not the shared evaluator because adjacent accepted candidates can merge.

Tiles use lengths $\lceil s\rceil$, offsets zero and half a tile, and one residual interval. Below a tile they use the best contiguous length-$R$ interval. They are a declared local baseline, not another upstream implementation.
''')
C('upstream-selector', '''
PAPER_GEOMETRY = {"896x896": (8.,8.), "896x128": (8.,8.),
                  "896x4864": (8.,8.), "4864x896": (12.,16.)}
assert {t["shape"] for t in traces} <= set(PAPER_GEOMETRY), "Declare parameters for new archive shapes."


def load_upstream():
    import subprocess
    if not (UPSTREAM / "src/vlmflash/policy.py").is_file():
        raise FileNotFoundError("Initialize conclusion/upstream/vlm-flash before RUN_ORACLE=True.")
    actual = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_SHA:
        raise ValueError("The checked-out upstream revision differs from this protocol.")
    sys.path.insert(0, str((UPSTREAM / "src").resolve()))
    import torch
    import vlmflash.policy as policy
    from vlmflash.latency import LatencyTable
    assert Path(policy.__file__).resolve().is_relative_to(UPSTREAM.resolve())
    if PAPER_IMPL == "native":
        from vlmflash._native import native, unavailable_reason
        if native() is None:
            raise RuntimeError(unavailable_reason())
    table = LatencyTable({int(z): float(c1*z + delta*max(s_kib,z)) for z in z_grid})
    runtime = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
        cuda_available=torch.cuda.is_available(), paper_impl=PAPER_IMPL)
    return torch, policy, table, runtime


def paper_mask(trace, v, R, backend):
    torch, policy, table, _ = backend
    start, jump = PAPER_GEOMETRY[trace["shape"]]
    params = policy.ChunkParams(start_kb=start, end_kb=s99+1,
                               step_kb=start, jump_cap_kb=jump)
    row_kib = weight_bytes * trace["d"] / 1024
    selection = policy.select_chunks(torch.tensor(v, dtype=torch.float32), R,
        row_kib, table, params, impl=PAPER_IMPL)
    return selection.mask.detach().cpu().numpy().astype(bool)
''', folded=True, recipe=True)
# Preserve the previous local tile geometry; remove the dependency on integer model boundaries.
tile_source = old['global-dp-19'].source
C('tiles', tile_source, folded=True, recipe=True)
M('evaluation-definition', r'''
## A fitted optimum and a measured-lookup check answer different questions.

For a heuristic ratio $u_M$, the numerical oracle bracket $[\ell,u]$ bounds its fitted-objective gap:

$$\max(0,100(1-u_M/\ell))\le g(M)\le\max(0,100(1-u_M/u)).$$

The main comparison uses this common objective and common $R$. Report retained importance, modeled milliseconds, run count, selector time and fill ratio separately.

A second evaluator sums the **held-out** raw profile median at the recovered run lengths, linearly interpolating only inside the measured grid. If any run is below the minimum or above the maximum profiled KiB, its total lookup score is unavailable. We record out-of-range byte fractions rather than clamp or silently extrapolate. Pairwise lookup gains require both masks to be fully supported.

The fitted model itself extrapolates when a run leaves that range. In particular, long runs may exceed the native reader's 768 KiB task cap; the additive objective here does **not** simulate task splitting, six-way scheduling or queueing. A separate hardware experiment is needed for that question. Even an in-range summed lookup score is a modeled cost, not a fresh measurement of the complete mask.
''')
C('evaluate-mask', '''
def evaluate_mask(trace, v, mask, T):
    I, L = metrics(v, mask, T)
    spans = runs(mask)
    lengths = np.diff(spans, axis=1).ravel()
    kib = lengths * weight_bytes * trace["d"] / 1024
    supported = (kib >= z_grid[0]) & (kib <= z_grid[-1])
    lookup = float(np.interp(kib, z_grid, T_holdout).sum()) if supported.all() else None
    return dict(importance=I, retained_pct=100*I/v.sum(), fitted_ms=L,
        ratio=I/L, chunks=len(lengths), max_run_rows=int(lengths.max()),
        lookup_ms=lookup, lookup_ratio=I/lookup if lookup else None,
        out_of_range_byte_pct=100*float(kib[~supported].sum()/kib.sum()),
        lookup_supported=bool(supported.all()))


def one_case(trace, fraction, backend):
    v = importance_vector(trace)
    n = len(v)
    nominal = max(1, int(np.floor(fraction*n)))
    start = perf_counter()
    paper = paper_mask(trace, v, nominal, backend)
    paper_seconds = perf_counter()-start
    R = int(paper.sum())
    base = dict(trace_id=int(trace["trace_id"]), shape=trace["shape"], module=trace["module"],
        prompt=trace["prompt_sha256"], fraction=float(fraction), R_nominal=nominal, R=R)
    if R == 0 or v.sum() == 0:
        return dict(**base, status="empty_selection" if R == 0 else "zero_importance", records=[], masks={})
    q = weight_bytes*trace["d"]/1024
    s, b1, b2 = s_kib/q, q*c1, q*(c1+delta)
    T = costs(n,s,b1,b2)
    start = perf_counter()
    oracle, lower, upper, calls = solve(v,R,s,b1,b2,RTOL,MAX_TABLE_MIB)
    seconds = {"Global DP": perf_counter()-start, "Upstream greedy": paper_seconds}
    masks = {"Global DP": oracle, "Upstream greedy": paper}
    for name, select in (("Top-R", lambda: top_r(v,R)),
                         ("Saturation tiles", lambda: tiles(v,R,T,max(1,int(np.ceil(s)))))):
        start = perf_counter()
        masks[name] = select()
        seconds[name] = perf_counter()-start
    rows = []
    for name, mask in masks.items():
        assert mask.shape == v.shape and mask.sum() == R
        result = evaluate_mask(trace,v,mask,T)
        assert result["ratio"] <= upper + 1e-8*max(1.,upper)
        rows.append(dict(**base, method=name, **result,
            gap_lower_pct=max(0.,100*(1-result["ratio"]/lower)),
            gap_upper_pct=max(0.,100*(1-result["ratio"]/upper)),
            oracle_lower=lower, oracle_upper=upper, oracle_calls=calls,
            selector_seconds=seconds[name], fill_pct=100*R/nominal))
    return dict(**base, status="ok", records=rows,
                masks={name:np.flatnonzero(mask).tolist() for name,mask in masks.items()})
''', folded=True, recipe=True)
M('runtime-definition', r'''
## Plan the full archive, but do not confuse a projection with a timing measurement.

The default experiment evaluates every archived trace at fractions $0.1,\ldots,0.9$. The table below estimates work at the **nominal** budgets; actual $R$ is learned only after upstream selection. It uses the earlier notebook's saved host timings and state counts. No calibration sweep runs while constructing this plan.

The projection scales the old 4,864-channel solve by states and a conservative model-dependent call count. It excludes upstream compilation, greedy selection, cache writes and host differences, and the new rolling-value implementation need not have the old timing constant. It is a planning calculation, not a promised duration or a statistical bound. The separately enabled scaling study measures complete new solves and labels its synthetic inputs explicitly.
''')
C('historical-timing', 'historical_timing = pd.DataFrame(' + repr(historical) + ')\n# Saved outputs of notebook 02 at base f3620ce; not new timings.\nhistorical_timing[["N", "R", "calls", "seconds", "table_MiB"]]', folded=True)
C('runtime-plan', '''
def call_bound(s, b1, b2):
    kappa = max(1., costs(1,s,b1,b2)[1]/b2)
    return int(np.ceil(np.log2(kappa/RTOL))) + 2

reference = historical_timing.set_index("N").loc[4864]
seconds_per_state_call = reference.seconds/(reference.states*reference.calls)
plans = []
for row in shape_summary.to_dict("records"):
    n, q = int(row["n"]), float(row["KiB/row"])
    calls = call_bound(s_kib/q,q*c1,q*(c1+delta))
    states = sum((int(np.floor(f*n))+1)*(n-int(np.floor(f*n))+1) for f in BUDGETS)
    plans.append(dict(shape=row["shape"], cases=int(row["traces"])*len(BUDGETS),
        projected_oracle_minutes=seconds_per_state_call*calls*states*row["traces"]/60,
        max_traceback_MiB=max(8*(int(np.floor(f*n))+1)*(n-int(np.floor(f*n))+1)/2**20 for f in BUDGETS)))
plan = pd.DataFrame(plans)
plan.round(3)
''', folded=True)
M('execution-definition', r'''
## Explicit execution is resumable; ordinary evaluation is read-only.

Set `RUN_ORACLE = True` to compute missing cases. Native selector initialization is outside the per-case timers. Each completed case stores its masks, realized budget, objective bracket, diagnostics and runtime metadata atomically. Interrupting one case does not discard earlier cases.

The cache namespace hashes the CSV, full-precision model, trace archive, upstream revision, numerical recipe, selected traces and experiment settings. A changed input or algorithm starts a different namespace. Existing matching cases are reused; there is no automatic fallback to synthetic data. A normal evaluation reports missing cases without starting the sweep. The progress display is transient and is outside the measured solver calls.
''')
C('case-cache', '''
def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(value, f, allow_nan=False)
    Path(f.name).replace(path)


def check_payload(payload, trace, fraction):
    assert payload["fingerprint"] == run_id
    assert payload["trace_id"] == trace["trace_id"] and payload["fraction"] == float(fraction)
    assert payload["shape"] == trace["shape"]
    if payload["status"] != "ok":
        assert payload["status"] in {"empty_selection", "zero_importance"} and not payload["records"]
        return
    v = importance_vector(trace)
    q = weight_bytes*trace["d"]/1024
    T = costs(len(v),s_kib/q,q*c1,q*(c1+delta))
    assert len(payload["records"]) == 4
    for row in payload["records"]:
        indices = payload["masks"][row["method"]]
        assert len(indices) == len(set(indices)) == payload["R"]
        assert all(isinstance(i,int) and 0<=i<len(v) for i in indices)
        mask = np.zeros(len(v),bool)
        mask[indices] = True
        I,L = metrics(v,mask,T)
        np.testing.assert_allclose([I,L,I/L], [row["importance"],row["fitted_ms"],row["ratio"]], rtol=1e-10)

notebook = json.loads((HERE / "02_global_oracle.ipynb").read_text())
recipe = "\\n".join("".join(c["source"]) for c in notebook["cells"]
    if "recipe" in c.get("metadata",{}).get("tags",[]))
request = dict(version="measured-fixed-r-v1", profile_sha256=sha256(PROFILE),
    model_sha256=sha256(MODEL), trace_sha256=sha256(TRACE), upstream=UPSTREAM_SHA,
    recipe_sha256=hashlib.sha256(recipe.encode()).hexdigest(), budgets=list(BUDGETS),
    selected_trace_ids=[int(t["trace_id"]) for t in traces], rtol=RTOL,
    paper_impl=PAPER_IMPL, geometry=PAPER_GEOMETRY, weight_bytes=weight_bytes,
    max_table_mib=MAX_TABLE_MIB, numpy=np.__version__, scipy=scipy.__version__)
run_id = hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
run_dir = HERE / "oracle_runs" / run_id
cases = [(t,f) for t in traces for f in BUDGETS]


def case_path(trace, fraction):
    return run_dir / f"trace-{trace['trace_id']:06d}-budget-{int(round(100*fraction)):02d}.json"

if RUN_ORACLE:
    backend = load_upstream()
    atomic_json(run_dir / "manifest.json", dict(request=request, runtime=backend[3]))
    progress = display(Markdown("Computing the real-trace comparison…"), display_id=True)
    try:
        for index,(trace,fraction) in enumerate(cases,1):
            path = case_path(trace,fraction)
            if path.is_file():
                check_payload(json.loads(path.read_text()),trace,fraction)
            else:
                payload = dict(one_case(trace,fraction,backend),fingerprint=run_id,runtime=backend[3])
                check_payload(payload,trace,fraction)
                atomic_json(path,payload)
            progress.update(Markdown(f"Global oracle: **{index}/{len(cases)}** cases completed or reused."))
    finally:
        progress.update(Markdown(""))

completed = []
for trace,fraction in cases:
    path = case_path(trace,fraction)
    if path.is_file():
        payload = json.loads(path.read_text())
        check_payload(payload,trace,fraction)
        completed.append(payload)
frame = pd.DataFrame([r for p in completed for r in p["records"]])
exclusions = pd.DataFrame([{k:p[k] for k in ("trace_id","shape","fraction","status")}
                           for p in completed if p["status"] != "ok"])
display(Markdown(f"**{len(completed)}/{len(cases)} cases available.** " +
    ("The real-trace sweep is complete." if len(completed)==len(cases) else
     "The sweep is incomplete; no claim below represents the full archive.")))
if not exclusions.empty:
    display(exclusions)
''', folded=True, recipe=True)
M('results-definition', r'''
## Report shape-specific gaps, not one pooled verdict.

A larger importance/latency ratio need not mean more retained importance. The figures below appear only when matching real results exist. Medians and 90th percentiles are descriptive over the selected trace/budget cases. Prompt-level summaries retain the archive's actual grouping; they are not confidence intervals.
''')
C('gap-results', '''
if not frame.empty:
    summary = frame.groupby(["shape","method"]).agg(
        cases=("trace_id","size"), median_gap_pct=("gap_upper_pct","median"),
        p90_gap_pct=("gap_upper_pct",lambda x: x.quantile(.9)),
        retained_pct=("retained_pct","median"), fitted_ms=("fitted_ms","median"),
        chunks=("chunks","median"), selector_seconds=("selector_seconds","median"),
        lookup_supported_pct=("lookup_supported",lambda x: 100*x.mean()))
    display(summary.round(4))
    gaps = frame[frame.method != "Global DP"].groupby(["shape","method"]).gap_upper_pct.median().unstack()
    ax = gaps.plot.bar(figsize=(9,3.6),rot=0)
    ax.set(xlabel="Projection shape (input rows × output columns)", ylabel="Median fitted-ratio gap upper bound (%)")
    plt.tight_layout(); plt.show()
    display(Markdown("**Figure 1.** Each method uses the upstream mask's realized row count. "
        "Gaps refer to the fitted additive objective, not device latency."))
    by_prompt = frame.groupby(["prompt","shape","method"]).gap_upper_pct.median().unstack()
    display(by_prompt.round(4))
else:
    display(Markdown("No real-trace comparison has been computed for this input fingerprint. The protocol is ready; results are not fabricated from the previous synthetic examples."))
''', folded=True)
C('budget-results', '''
if not frame.empty:
    for shape, group in frame[frame.method != "Global DP"].groupby("shape"):
        curve = group.groupby(["fraction","method"]).gap_upper_pct.median().unstack()
        ax = curve.plot(figsize=(8,3),marker=".")
        ax.set(title=f"{shape}: median fitted-objective gap", xlabel="Nominal retained fraction", ylabel="Gap upper bound (%)")
        plt.tight_layout(); plt.show()
    fill = frame[frame.method == "Upstream greedy"].groupby(["shape","fraction"]).fill_pct.agg(["min","median"])
    display(fill.round(3))
''', folded=True)
C('geometry-results', '''
if not frame.empty:
    example = next((p for p in completed if p["status"]=="ok" and np.isclose(p["fraction"],.5)), None)
    if example is not None:
        rows = pd.DataFrame(example["records"])
        plt.figure(figsize=(8,3.4))
        for _,row in rows.iterrows():
            plt.scatter(row.fitted_ms,row.retained_pct,label=row.method)
        plt.xlabel("Fitted additive latency (ms)"); plt.ylabel("Retained importance (%)")
        plt.legend(fontsize=8); plt.tight_layout(); plt.show()
        display(rows.set_index("method")[["importance","fitted_ms","chunks","gap_upper_pct"]].round(4))
        n = next(t["n"] for t in traces if t["trace_id"]==example["trace_id"])
        geometry = np.zeros((len(example["masks"]),n),bool)
        for i,indices in enumerate(example["masks"].values()):
            geometry[i,indices] = True
        plt.figure(figsize=(9,2.4)); plt.imshow(geometry,aspect="auto",interpolation="nearest")
        plt.yticks(range(len(geometry)),list(example["masks"]))
        plt.xlabel(f"Trace {example['trace_id']}: channel; every mask has R={example['R']}")
        plt.tight_layout(); plt.show()
''', folded=True)
C('lookup-results', '''
if not frame.empty:
    keys = ["trace_id","fraction","shape"]
    oracle = frame[frame.method=="Global DP"][keys+["lookup_ratio"]]
    paper = frame[frame.method=="Upstream greedy"][keys+["lookup_ratio"]]
    pairs = oracle.merge(paper,on=keys,suffixes=("_oracle","_paper"))
    pairs["paired_supported"] = pairs[["lookup_ratio_oracle","lookup_ratio_paper"]].notna().all(axis=1)
    valid = pairs[pairs.paired_supported].copy()
    valid["lookup_ratio_gain_pct"] = 100*(valid.lookup_ratio_oracle/valid.lookup_ratio_paper-1)
    display(pairs.groupby("shape").paired_supported.agg(["sum","count"]))
    if not valid.empty:
        ax = valid.groupby("shape").lookup_ratio_gain_pct.median().plot.bar(figsize=(8,3.2),rot=0)
        ax.set(xlabel="Projection shape (fully supported pairs only)",ylabel="Median held-out lookup ratio gain (%)")
        plt.tight_layout(); plt.show()
    display(Markdown("**Lookup robustness.** Unsupported pairs are excluded and counted, not extrapolated. "
        "A positive gain is not a proof of lookup-optimality or a measured wall-time speedup."))
''', folded=True)
M('scaling-definition', r'''
## Large scaling is a separate, explicitly synthetic timing study.

At $R=N/2$, the largest planned value table has $(R+1)(N-R+1)$ float64 entries. The table below shows its allocation and the old-host state-work projection. Set `RUN_SCALING = True` to measure three complete ratio solves per size, including final recovery, on a seeded lognormal vector. It does not load a model or measure SSD I/O. Its row width is fixed at 1.75 KiB only to give a declared cost; dimensions absent from the archive are never presented as captured activations.

Scaling results have their own fingerprint and resume independently of the real-trace experiment. Timings are for this Python/SciPy oracle, not a sub-two-millisecond online implementation.
''')
C('scaling', '''
q = 1.75
s, b1, b2 = s_kib/q, q*c1, q*(c1+delta)
scaling_plan = pd.DataFrame([dict(N=n,R=n//2,states=(n//2+1)*(n-n//2+1),
    table_MiB=8*(n//2+1)*(n-n//2+1)/2**20,
    projected_seconds=seconds_per_state_call*call_bound(s,b1,b2)*(n//2+1)*(n-n//2+1))
    for n in SCALING_SIZES])
display(scaling_plan.round(3))
scaling_id = hashlib.sha256(json.dumps(dict(run_id=run_id,row_kib=q,repeats=SCALING_REPEATS,
                                          sizes=list(SCALING_SIZES))).encode()).hexdigest()
scaling_dir = HERE / "scaling_runs" / scaling_id
if RUN_SCALING:
    for n in SCALING_SIZES:
        path = scaling_dir / f"{n}.json"
        if path.is_file():
            continue
        v = np.random.default_rng(0).lognormal(0,1,n)
        v /= v.mean()
        seconds, counts = [], []
        for _ in range(SCALING_REPEATS):
            start = perf_counter()
            mask,lower,upper,calls = solve(v,n//2,s,b1,b2,RTOL,MAX_TABLE_MIB)
            seconds.append(perf_counter()-start); counts.append(calls)
        atomic_json(path,dict(fingerprint=scaling_id,N=n,R=n//2,seconds=seconds,calls=counts,
            platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__))
scaling_rows = []
for n in SCALING_SIZES:
    path = scaling_dir / f"{n}.json"
    if path.is_file():
        result = json.loads(path.read_text())
        assert result["fingerprint"] == scaling_id and result["N"] == n
        assert len(result["seconds"]) == SCALING_REPEATS
        scaling_rows.append(dict(N=n,seconds=np.median(result["seconds"]),calls=np.median(result["calls"])))
if scaling_rows:
    timing = pd.DataFrame(scaling_rows)
    plt.figure(figsize=(8,3.5))
    plt.loglog(timing.N,timing.seconds,"o-",label="New measured solves (median)")
    plt.xlabel("Channels N"); plt.ylabel("Complete oracle solve (seconds)")
    plt.legend(); plt.tight_layout(); plt.show()
else:
    display(Markdown("The new large-size timing sweep has not been run. The table above contains projections only."))
''', folded=True)
M('scope', r'''
## The empirical conclusion remains conditional.

The oracle is global over masks of the realized cardinality under the inherited two-line cost. It is not global for an arbitrary lookup table, a coverage constraint, an entire Pareto frontier, or an end-to-end accuracy objective. The archive's mean absolute activation is an importance proxy, not a measurement of output error.

The default use of real traces does not remove the need for controlled examples: zero/tied inputs and short-run counterexamples remain in the exhaustive checks. Nor does a reproduced profile prove the additive model at unmeasured run lengths. Report lookup coverage, underfilling, case completeness and the number of actual prompts alongside any advantage.

The distinction is deliberate: notebook 01 estimates a cost; this notebook measures selection quality **under that cost**. Hardware mask latency and task-level accuracy are subsequent questions.
''')
nb2 = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
    'language_info': {'name':'python'}})
for cell in nb2.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        compile(cell.source, cell.id, 'exec')
nbf.validate(nb2)
nbf.write(nb2, P2)
(P2.parent / '.gitignore').write_text('oracle_runs/\nscaling_runs/\n')
print('Edited notebook sources only; native profile and full trace sweep were not executed.')
