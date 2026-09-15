**Neuron Chunking selects contiguous groups of weight rows according to the activation importance they preserve per unit of flash I/O time.** Its central observation is

$$
\boxed{
\text{less data transferred}
\;\not\Rightarrow\;
\text{less time spent transferring it}.
}
$$

The method combines **offline hardware profiling**, **offline neuron reordering**, and **input-dependent chunk selection**. The following notes cover the paper’s argument and technical appendices, with repeated experimental plots condensed to their findings. 

---

An edge device can have insufficient RAM to store a VLM:

$$
M_{\text{weights}}>M_{\text{RAM}}.
$$

For example, the paper gives approximately \(16\) GB of FP16 weights for LLaVA-OneVision-7B, compared with \(8\) GB of RAM on Jetson Orin Nano.

**Weight offloading** stores weights on flash and loads the required portions into RAM during inference:

$$
\text{flash weights}
\xrightarrow{\text{I/O}}
\text{RAM}
\xrightarrow{\text{matrix multiplication}}
\text{activations}.
$$

Without I/O–compute overlap,

$$
T_{\text{inference}}
=
T_{\text{I/O}}+T_{\text{compute}}+T_{\text{management}}.
$$

When \(T_{\text{I/O}}\) dominates, reducing computation alone has limited effect. Moving weights from GPU memory to CPU memory does not solve the capacity problem on an edge SoC whose CPU and GPU share the same DRAM.

The evaluated system keeps the vision encoder and KV cache resident; Appendix L additionally identifies the LM head as resident. Backbone transformer matrices are loaded on demand.  

A streaming VLM contains

$$
f_{\text{vision}},\qquad f_{\text{proj}},\qquad f_{\text{LLM}}.
$$

For a frame \(F_i\), its visual tokens are

$$
X_i=f_{\text{proj}}\!\left(f_{\text{vision}}(F_i)\right)
\in\mathbb R^{n_i\times d}.
$$

Its inference stages are:

| Stage           | Tokens processed           | Purpose                                      |
| --------------- | -------------------------- | -------------------------------------------- |
| Prefill         | Initial text prompt        | Construct the initial KV cache               |
| Frame appending | New frame’s visual tokens  | Extend the KV cache                          |
| Decoding        | Previously generated token | Extend the cache and generate the next token |

To make the cache operation explicit, consider one attention head at one layer. For \(B\) newly processed tokens,

$$
X\in\mathbb R^{B\times d},
\qquad
Q=XW_Q,\quad K_{\text{new}}=XW_K,\quad V_{\text{new}}=XW_V.
$$

Append the new keys and values:

$$
K^+=
\begin{bmatrix}
K_{\text{past}}\\K_{\text{new}}
\end{bmatrix},
\qquad
V^+=
\begin{bmatrix}
V_{\text{past}}\\V_{\text{new}}
\end{bmatrix}.
$$

Then

$$
\operatorname{Attention}(X)
=
\operatorname{softmax}
\left(
\frac{Q(K^+)^\top}{\sqrt{d_k}}+\mathcal C
\right)V^+,
$$

where \(\mathcal C\) is the causal mask: permitted positions receive \(0\), and future positions receive \(-\infty\).

Thus:

$$
\boxed{
\text{KV cache stores computed }K,V;
\quad W_K,W_V\text{ are learned weight matrices}.
}
$$

Caching avoids recomputing the past tokens’ keys and values. New queries still attend to cached keys and combine cached values.

Prefill and frame appending process multiple tokens together; ordinary decoding processes one token at a time. Token outputs during prefill and frame appending are normally ignored. Decoding begins after an appended user query or a model-generated control signal; in the latter case, the final frame-stage output token is retained to initiate decoding. 

For activation sparsification, write any projection as

$$
A\in\mathbb R^{B\times N},
\qquad
W\in\mathbb R^{N\times d_o},
\qquad
Y=AW.
$$

Here \(N\) is the number of input channels, \(B\) is the number of tokens processed together, and \(W_{i,:}\) is the weight row associated with input channel \(i\).

For one token,

$$
y=\sum_{i=0}^{N-1}a_iW_{i,:}.
$$

A binary mask \(M\in\{0,1\}^N\) gives

$$
\widetilde Y
=
A\operatorname{diag}(M)W
=
A_{:,S}W_{S,:},
\qquad
S=\{i:M_i=1\}.
$$

Only rows \(W_{S,:}\) need to be loaded. For sparsity \(\rho\),

$$
R=\lfloor(1-\rho)N\rfloor,
\qquad
|S|\le R.
$$

The paper measures channel importance using

$$
\boxed{
v_i=\frac1B\sum_{b=1}^{B}|A_{b,i}|.
}
$$

For \(B=1\), this becomes \(v_i=|a_i|\). For multiple tokens, one shared mask is selected from their average absolute activations.

Magnitude-based top-\(R\) selection solves

$$
\max_{\substack{M\in\{0,1\}^N\\\sum_iM_i=R}}
\sum_i v_iM_i.
$$

A threshold alternative uses \(M_i=\mathbf1\{v_i\ge\tau\}\), allowing the selected count to vary. Both methods are input-dependent. 

**Activation magnitude is a proxy for importance, not an exact measure of accuracy.** The approximation error is

$$
Y-\widetilde Y
=
\sum_i(1-M_i)A_{:,i}W_{i,:}.
$$

The triangle inequality gives the useful bound

$$
\frac1B\sum_{b=1}^{B}
\|Y_{b,:}-\widetilde Y_{b,:}\|_2
\le
\sum_i(1-M_i)v_i\|W_{i,:}\|_2.
$$

Therefore, keeping large \(v_i\) is especially reasonable when row norms are comparable. Actual error also depends on weight norms, cancellation, and subsequent layers; task accuracy must be measured separately.

The method applies to attention and MLP projections. A standard gated MLP has the form

$$
G=\operatorname{SiLU}(XW_{\text{gate}}),
\qquad
U=XW_{\text{up}},
\qquad
H=G\odot U,
\qquad
Y=HW_{\text{down}},
$$

where

$$
\operatorname{SiLU}(x)=x\sigma(x),
\qquad
\sigma(x)=\frac1{1+e^{-x}}.
$$

The \(q,k,v\) projections share input activations, as do the gate and up projections. The down projection receives \(H\). Selection uses each matrix’s available input activations, so no separate activation predictor is required.

Modern VLMs generally provide less extreme sparsity than older ReLU-based LLMs:

$$
\operatorname{ReLU}(x)=\max(0,x)
\quad\Rightarrow\quad
\text{many exact zeros},
$$

whereas smooth activations such as SiLU and GeLU generally produce nonzero values:

$$
\operatorname{GeLU}(x)=x\Phi(x).
$$

Averaging over multiple visual tokens further suppresses token-specific fluctuations. For independent, equal-variance samples \(Z_b\),

$$
\operatorname{Var}\left(\frac1B\sum_b Z_b\right)
=
\frac{\operatorname{Var}(Z_1)}B.
$$

Actual visual tokens are correlated, so this identity explains a tendency rather than guaranteeing smoothing. The paper verifies the effect empirically using

$$
\operatorname{CV}(v)
=
\frac{\operatorname{std}_i(v_i)}
{\operatorname{mean}_i(v_i)}.
$$

Before down projection,

$$
\operatorname{CV}_{\text{VLM}}\in[1.07,4.55],
\qquad
\operatorname{CV}_{\text{OPT-6.7B}}\in[8.63,11.65].
$$

Hence, many channels have moderately similar importance. Choosing slightly less important neighbors can save substantial I/O time, while retaining only a tiny set of dominant channels is less successful. The paper contrasts its typical \(40\%-60\%\) sparsity regime with the \(>90\%\) sparsity targeted by LLM in a Flash.   

The hardware problem is that **flash throughput depends on read contiguity**.

Let \(b\) be bytes per weight row and \(\beta(z)\) the steady-state throughput for reads of size \(z\). Typically,

$$
\beta(z)\uparrow\beta_{\max}
\quad\text{as }z\text{ increases}.
$$

Small reads are dominated by request-processing overhead; large reads approach the bandwidth limit.

For mask \(M\),

$$
D(M)=b\sum_iM_i,
\qquad
L(M)=\frac{D(M)}{\beta_{\text{eff}}(M)}.
$$

Consequently, for two access patterns,

$$
\frac{L_2}{L_1}
=
\frac{D_2}{D_1}\frac{\beta_{\text{eff},1}}{\beta_{\text{eff},2}}.
$$

Even when \(D_2<D_1\),

$$
L_2>L_1
\iff
\frac{\beta_{\text{eff},2}}{\beta_{\text{eff},1}}
<
\frac{D_2}{D_1}.
$$

**Sparsification becomes slower when fragmentation reduces throughput proportionally more than it reduces transferred bytes.** Figure 4, on page 4, demonstrates this nonmonotonic latency behavior. Throughput becomes comparatively stable once sufficiently many read requests are available, enabling a compact steady-state model. 

A **chunk** is a maximal contiguous run of selected rows. For example,

$$
S=\{1,2,4,6,7\}
\quad\Rightarrow\quad
\mathcal C(M)=\{\{1,2\},\{4\},\{6,7\}\}.
$$

Define the contiguity histogram

$$
h_M(r)=\#\{C\in\mathcal C(M):|C|=r\}.
$$

Then

$$
\sum_r h_M(r)=\text{number of chunks},
\qquad
\sum_r r\,h_M(r)=|S|.
$$

In the example, \(h_M(1)=1\) and \(h_M(2)=2\).

The histogram retains chunk sizes and counts while discarding exact positions and distances between chunks. It is an approximation of hardware-relevant structure.

The proposed latency model is

$$
\boxed{
\widehat L(M)
=
\sum_{C\in\mathcal C(M)}T(|C|)
=
\sum_r h_M(r)T(r).
}
$$

Here \(T(r)\) is the profiled, amortized cost of an \(r\)-row chunk. With sufficiently many equal-sized requests,

$$
T(r)
\approx
\frac{L_{\text{profile}}(r,q)}q
=
\frac{rb}{\beta(rb)}.
$$

This is a steady-state cost contribution, rather than the latency of one isolated request. Concurrency and amortized fixed overhead are already reflected in profiling. 

Offline profiling measures throughput by read size, using sufficiently many chunks and controlled placement. The paper describes a roughly \(128\) MB dummy file, \(1\) KB increments, and completion within \(20\) minutes per device. Reported throughput standard deviation is below \(1\%\) of the mean. The flash experiments use Linux direct I/O and a six-thread C++ pool.

Define the saturation size by

$$
z_{\max}
=
\min\{z:\beta(z)\ge0.99\beta_{\max}\}.
$$

Appendices D and H give

$$
z_{\max}^{\text{AGX}}=236\text{ KB},
\qquad
z_{\max}^{\text{Nano}}=348\text{ KB}.
$$

The Figure 4 caption instead lists \(328\) KB for AGX and \(236\) KB for Nano, so the PDF is internally inconsistent on these constants. The implementation principle is to use the device’s measured saturation point.   

Measured latency is approximately proportional to estimated latency:

$$
L_{\text{actual}}(M)\approx c\,\widehat L(M),
\qquad c>0.
$$

Figure 5, on page 5, reports:

| Model              | AGX \(R^2\) | Nano \(R^2\) |
| ------------------ | ----------: | -----------: |
| LLaVA-OneVision-7B |       0.995 |        0.950 |
| NVILA-Lite-2B      |       0.929 |        0.832 |

The proportional bias arises because profiling isolates one chunk size with regular strides, whereas inference mixes sizes and gaps, changing controller and queue behavior. Smaller workloads and weaker concurrency reduce averaging and worsen prediction.

If the same multiplicative factor applies to candidate costs,

$$
U_{\text{actual}}(C)
=
\frac{I(C)}{cT(|C|)}
=
\frac1cU_{\text{estimated}}(C),
$$

so utility ordering is unchanged. Approximate proportionality supports this robustness empirically; it does not guarantee identical rankings for every access pattern. 

The paper formulates selection as

$$
\boxed{
\max_{\substack{M\in\{0,1\}^N\\0<\sum_iM_i\le R}}
\frac{\sum_i v_iM_i}{\widehat L(M)}.
}
$$

The numerator is retained activation importance; the denominator is predicted flash latency; \(R\) is a **row-count budget**, not a latency budget. Excluding the empty mask makes the ratio well-defined.

Searching all masks requires considering \(2^N\) possibilities. The implementation instead ranks a restricted collection of contiguous candidates by

$$
I(C)=\sum_{i\in C}v_i,
\qquad
\boxed{
U(C)=\frac{I(C)}{T(|C|)}.
}
$$

The algorithm is as follows.

1. Convert byte-size parameters into row counts. If one row occupies \(b_{\text{KB}}\) KB,

   $$
   r_{\min}=\max\left(1,\left\lfloor\frac{s_{\min}}{b_{\text{KB}}}\right\rfloor\right),
   \qquad
   r_{\max}=\max\left(1,\left\lfloor\frac{s_{\max}}{b_{\text{KB}}}\right\rfloor\right),
   $$

   $$
   \Delta r=\max\left(1,\left\lfloor\frac{\Delta s}{b_{\text{KB}}}\right\rfloor\right),
   \qquad
   j_{\max}=\max\left(1,\left\lfloor\frac{j_{\text{cap}}}{b_{\text{KB}}}\right\rfloor\right).
   $$

2. Compute a CPU prefix sum:

   $$
   P_0=0,\qquad P_j=\sum_{i=0}^{j-1}v_i.
   $$

   An interval \(C_{i,r}=\{i,\ldots,i+r-1\}\) then has

   $$
   I(C_{i,r})=P_{i+r}-P_i
   $$

   in constant time.

3. For every candidate size

   $$
   r=r_{\min},r_{\min}+\Delta r,\ldots,r_{\max},
   $$

   slide windows with stride

   $$
   \delta(r)=\min(r,j_{\max}).
   $$

   Thus small windows are nonoverlapping at a given scale; larger windows overlap when their stride is capped. Score valid windows using

   $$
   U_{i,r}=\frac{P_{i+r}-P_i}{T(r)}.
   $$

4. Sort candidates by decreasing utility on the GPU. Initialize \(M=0\), \(k=0\). Accept candidate \((i,r)\) exactly when

   $$
   \sum_{j=i}^{i+r-1}M_j=0,
   \qquad
   k+r\le R.
   $$

   On acceptance,

   $$
   M_{i:i+r}\leftarrow1,
   \qquad
   k\leftarrow k+r.
   $$

   Stop when \(k=R\) or no candidates remain.

This is a **greedy heuristic**. Candidate scores are evaluated individually, so the method can miss benefits from combining adjacent regions. The restricted candidate sizes can also leave part of the budget unused.

There is an additional mathematical distinction: maximizing a ratio under \(|S|\le R\) does not itself require filling \(R\). Thus the budget-filling procedure is not a proved exact optimizer of the displayed objective. The paper provides empirical effectiveness rather than an optimality guarantee. Adjacent accepted candidates may merge into a maximal run larger than an individual candidate.  

Search granularity must keep selection overhead small:

$$
s_{\min}\downarrow,\quad
\Delta s\downarrow,\quad
j_{\text{cap}}\downarrow
\quad\Rightarrow\quad
\text{more candidates and usually more overhead}.
$$

The paper tunes

$$
\theta=(s_{\min},j_{\text{cap}}),
\qquad
\Delta s=s_{\min},
\qquad
s_{\max}=z_{\max}.
$$

It sweeps \(0\)–\(64\) KB in \(4\) KB increments, with row conversion clamped to at least one row. Configurations are tested separately for matrix shapes and devices, using \(10\%\) sparsity as a conservative high-selection workload.

The selection rule is:

$$
T_{\text{selection}}(\theta)<2\text{ ms},
$$

then prefer fine-grained configurations near the feasible boundary, with a small margin. Each configuration receives \(30\) trials. GPU sorting accounts for over \(80\%\) of runtime; its data-independent radix-sort behavior motivates using random activations for timing.

Final \(s_{\min}\) values range from \(8\) to \(40\) KB, and jump caps from \(8\) to \(36\) KB across the listed matrices. Larger matrices require coarser settings; AGX supports more configurations than Nano. A full sensitivity analysis remains future work. 

**Hot–cold reordering** improves the physical arrangement before online selection. On calibration inputs, define

$$
f_i
=
\frac1C\sum_{c=1}^{C}
\mathbf1
\left\{
i\in\operatorname{Top}_{50\%}(v^{(c)})
\right\}.
$$

Sort channels by decreasing \(f_i\), placing frequently important channels together.

For permutation matrix \(P\), reorder both weights and activation coordinates:

$$
W'=PW,
\qquad
A'=AP^\top.
$$

The dense computation is preserved exactly:

$$
A'W'
=
AP^\top PW
=
AW.
$$

At runtime, chunk selection operates in this reordered coordinate system. The paper reports a mean activation-permutation overhead of \(1.5\) ms for the profiled down projection, with a \(95\%\) upper confidence bound of \(1.8\) ms; its stated comparison is below \(0.02\%\) of total inference latency.

Reordering cannot determine the best mask for every future input:

$$
f_i=\text{historical frequency},
\qquad
v_i(x)=\text{importance for current input}.
$$

The separate activation-frequency analysis labels neurons hot above \(99\%\) activation frequency and cold below \(1\%\). Many neurons lie between these extremes, establishing substantial input dependence. Layer-specific sparsity also varies strongly: at \(40\%\) effective sparsity, one examined \(q\) projection has \(94\%\) sparsity.

Simple hot–cold reordering and Ripple’s coactivation-based reordering yield similar modest contiguity improvements overall; Ripple performs better in one examined early-layer \(o\) projection. Analysis uses layers \(0,13,27\), with \(q,o,\text{gate},\text{down}\) representing the distinct projection inputs. Of the \(25\) analysis videos, \(20\) calibrate reordering and \(5\) validate it.   

Layer-wise sparsity is allocated using TEAL for both the baseline and Neuron Chunking:

$$
R_\ell=\lfloor(1-\rho_\ell)N_\ell\rfloor,
\qquad
\rho_\ell\text{ may differ across matrices}.
$$

The division of responsibility is

$$
\boxed{
\text{TEAL determines how much to sparsify;}
\quad
\text{Neuron Chunking determines which rows to load}.
}
$$

Calibration uses \(25\) of TempCompass’s \(410\) videos, excluded from the main evaluation. The comparison baseline selects the largest activation magnitudes.

The hardware configurations and headline results are:

| Device    |   RAM | SSD; advertised peak sequential read | Mean I/O speedup | Maximum I/O speedup |
| --------- | ----: | ------------------------------------ | ---------------: | ------------------: |
| Orin Nano |  8 GB | SK Hynix Gold P31; 3500 MB/s         |   \(2.19\times\) |      \(4.65\times\) |
| Orin AGX  | 32 GB | Samsung 990 Pro; 7450 MB/s           |   \(2.89\times\) |      \(5.76\times\) |

Five frame-by-frame VLMs are evaluated: LLaVA-OneVision-Qwen2-7B, LLaVA-OneVision-Qwen2-0.5B, Llama-3-VILA1.5-8B, NVILA-Lite-2B, and LongVA-7B. Models requiring the whole video at once are outside the streaming setup.

| Dataset            | Task                                            | Quality metric                 |
| ------------------ | ----------------------------------------------- | ------------------------------ |
| TempCompass        | Multiple-choice video QA                        | Fraction correct               |
| NExT-QA            | Multiple-choice video QA; 3000 sampled examples | Fraction correct               |
| VideoDetailCaption | Video description                               | Model-judged score from 0 to 5 |

Caption judging uses `gpt-4o-mini-2024-07-18`, so those scores are not directly comparable with prior work using a different judge.

Accuracy is evaluated on a server with eight RTX A6000 GPUs at sparsities

$$
\rho\in\{0,0.1,\ldots,0.7\}.
$$

Device latency measurements use \(30\) repeated trials, fixed Jetson clocks, and disabled swap. The text reports medians with \(95\%\) BCa bootstrap confidence intervals from \(10{,}000\) resamples; some figure captions instead specify \(\pm1\) standard deviation.

Speedups compare interpolated points at comparable quality:

$$
S_{\text{I/O}}(a)
=
\frac{L_{\text{baseline}}(a)}
{L_{\text{chunking}}(a)}.
$$

The larger AGX gains reflect its larger throughput gap between scattered and contiguous reads. Small models can suffer especially severe fragmentation because individual weight rows are smaller. Occasional accuracy increases after sparsification are interpreted as possible removal of weak or noisy activations, rather than a guaranteed effect.   

**The headline speedups concern I/O, not the entire inference pipeline.** Let baseline compute and I/O times be \(C\) and \(I\). If chunking introduces additional compute \(\Delta C\) and selection overhead \(H\),

$$
\boxed{
S_{\text{total}}
=
\frac{C+I}
{C+\Delta C+I/S_{\text{I/O}}+H}.
}
$$

At matched accuracy, chunking may retain slightly more rows:

$$
R_{\text{chunking}}>R_{\text{top-}R},
$$

giving a small compute increase, while contiguity makes their loading faster.

Figure 8, on page 8, shows this breakdown at a reported \(5\%\) accuracy drop. Selection costs about \(2\) ms per weight matrix, or approximately

$$
200\times2\text{ ms}\approx400\text{ ms}
$$

for a full model pass. Compute optimization and I/O–compute overlap could improve total speedup; the experiments do not use that overlap.

Ablations show that reordering provides a smaller gain than online chunk selection:

$$
\text{reordering alone: up to }1.23\times,
$$

$$
\text{reordering + chunk selection: up to }2.55\times
$$

in the reported LLaVA-7B comparison.

Figure 10, on page 8, gives representative mean chunk sizes:

$$
1.1
\xrightarrow{\text{reordering}}
1.8
\xrightarrow{\text{chunk selection}}
49.6.
$$

The corresponding dominant chunk size becomes \(48\). Extended visualizations across layers and projections confirm that the online policy supplies most of the contiguity improvement.  

Visual-token reduction changes context capacity. With context limit \(C_{\max}\), \(m\) prompt tokens, \(n\) tokens per frame, and \(r\) tokens reserved for later text,

$$
F_{\max}
=
\left\lfloor
\frac{C_{\max}-m-r}{n}
\right\rfloor.
$$

Reducing \(n\) allows more frames to fit. Spatial pooling experiments use

$$
n\in\{14^2,9^2,7^2,5^2\}
=
\{196,81,49,25\}.
$$

Lower token density causes a modest accuracy decrease, but Neuron Chunking retains its advantage over the baseline throughout these settings. 

The relationship to prior methods follows from what each method optimizes.

Deja Vu exploits dynamic activation sparsity; CATS extends sparsification to gated non-ReLU MLPs; TEAL also sparsifies attention projections and allocates sparsity across matrices. These methods primarily optimize activation selection. Their GPU-resident settings tolerate much smaller read units than flash-offloaded inference.

LLM in a Flash and PowerInfer-2 use **bundling**: store related weights together, such as an up-projection column and its corresponding down-projection row. Bundling improves each request’s size but does not necessarily make selected neurons form long contiguous runs.

For this paper’s predictor-free pipeline, up/down bundling is awkward because the matrices are selected using different input activations. Bundling matrices sharing inputs, such as \(Q/K/V\) or gate/up, is more directly applicable, but the largest examined bundles are approximately \(74\) KB:

$$
74\text{ KB}
<
236\text{--}348\text{ KB}.
$$

They reach only about half the optimal bandwidth on the evaluated hardware. Bundling overlapping selected rows can also fragment the remaining unbundled rows and add processing overhead.

Across the reported model–dataset combinations,

$$
\frac{L_{\text{baseline}}}{L_{\text{ours}}}
\approx1.5\text{--}3.4,
\qquad
\frac{L_{\text{baseline+bundling}}}{L_{\text{ours}}}
\approx1.7\text{--}4.0.
$$

Bundling usually worsens the tested baseline, with LLaVA-0.5B as the exception.

The paper suggests that Jetson’s larger saturation sizes may partly result from NVMe interrupts being concentrated on one CPU core, limiting IOPS; this is a proposed hardware explanation.  

**Caching and chunking are complementary.** Sliding-window caching retains recently used weights; hot-neuron caching retains frequently used weights. Both require additional RAM. Offline reordering requires no additional resident cache of model weights, but is less adaptive.

For cached row set \(\mathcal H\), the paper proposes flash-selection scores

$$
v_i^{\text{flash}}
=
v_i\mathbf1\{i\notin\mathcal H\}.
$$

Zeroing this score means that cached rows provide no reward for another flash load; their contributions remain available from RAM. Conceptually,

$$
\widetilde Y
=
A_{:,\mathcal H}W_{\mathcal H,:}
+
A_{:,S\setminus\mathcal H}W_{S\setminus\mathcal H,:}.
$$

After hot weights are cached, remaining uncached accesses may be more scattered, increasing the value of contiguity-aware selection. If nearly all weights fit in RAM, flash I/O becomes negligible and the benefit diminishes.  

Quantization, weight pruning, and distillation are complementary forms of model compression. Quantization reduces row size,

$$
b=d_o\frac{\text{bits per weight}}8,
$$

while activation sparsification changes the rows selected for the current input.

Training-time regularization provides another route to sparsity:

$$
\mathcal L_{\text{L1}}
=
\mathcal L_{\text{task}}
+\lambda\sum_{i,j}|W_{ij}|,
$$

$$
\mathcal L_{\text{group}}
=
\mathcal L_{\text{task}}
+\lambda\sum_{g}\|W_g\|_2,
$$

where \(g\) can be a row or column.

Element-wise L1 sparsity need not eliminate entire rows, so it need not reduce flash row reads. Group penalties can remove whole channels. In a gated MLP,

$$
W_{\text{gate},:,j}=0
\quad\text{or}\quad
W_{\text{up},:,j}=0
\quad\Rightarrow\quad
H_{:,j}=0,
$$

making down-projection row \(j\) unnecessary. Low-norm rows can also receive low importance under a combined metric such as

$$
v_i\|W_{i,:}\|_2.
$$

Such weight pruning is input-independent, whereas Neuron Chunking changes its mask with the current activations. ReLU-ification can increase exact activation sparsity but requires retraining; the paper cites approaches using at least \(50\) billion training tokens.   

The deployment objective is an improved **accuracy–latency trade-off**. It can be expressed as either

$$
\min_M L(M)
\quad\text{subject to}\quad
\operatorname{Accuracy}(M)\ge a_0,
$$

or

$$
\max_M\operatorname{Accuracy}(M)
\quad\text{subject to}\quad
L(M)\le\ell_0.
$$

The algorithm uses importance-per-latency as a practical surrogate; experiments assess whether it improves these task-level trade-offs.

A configuration dominates another when

$$
a_1\ge a_2,\qquad L_1\le L_2,
$$

with at least one strict inequality. The intended improvement is a better Pareto frontier, rather than guaranteed preservation of dense-model accuracy.

For streaming applications, output delay also matters: an answer about an earlier scene may be less useful after the scene changes. The paper therefore motivates accepting modest accuracy loss for faster object retrieval, temporal localization, and interactive correction. 

Generalization is most promising when

$$
\boxed{
\text{importance is relatively smooth},
\quad
\beta(\text{one row})<\beta_{\max},
\quad
\text{flash I/O is a substantial bottleneck}.
}
$$

Multi-token LLM workloads—speculative decoding, parallel sampling, and batched inference—also aggregate activation importance:

$$
v_i=\frac1B\sum_b|A_{b,i}|,
$$

and can share a mask and weight-loading schedule across tokens. Plain non-ReLU LLMs can benefit with less smoothing; ViTs can benefit when their smaller channels produce inefficient individual reads.

Asynchronous I/O mechanisms such as `io_uring` may reduce scattered-read overhead. The paper expects contiguity to remain useful, but this is a future-looking argument rather than a demonstrated universal result.

Preliminary GSM8K-input experiments report

$$
S_{\text{LLaMA3-8B}}=1.22\times,
\qquad
S_{\text{Qwen2-7B}}=2.09\times.
$$

These compare retained-importance–latency trade-offs in early, middle, and late layers. They use

$$
\sum_i v_iM_i
$$

as an accuracy proxy, so they do **not** establish those speedups at matched measured GSM8K accuracy. Full task-level validation for these broader workloads remains open.  
