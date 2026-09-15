Neuron Chunking selects contiguous weight rows by **activation importance per estimated flash-read time**. The notes below cover the main argument and the substantive content of Appendices A–N. Additional mathematical derivations make the paper’s assumptions explicit. 

1. **The deployment problem is insufficient memory combined with expensive weight loading.**

   A model fits entirely in memory only if

   $$
   M_{\text{weights}}+M_{\text{KV}}+M_{\text{working}}
   \le M_{\text{DRAM}}.
   $$

   The paper’s motivating example is approximately \(16\) GB of FP16 model weights versus \(8\) GB of memory on Jetson Orin Nano.

   With **weight offloading**, most backbone weights remain on flash and are loaded when needed. The evaluated system keeps the vision encoder and KV cache in memory; Appendix L also identifies the LM head as resident.

   On a unified-memory system,

   $$
   \text{CPU memory and GPU memory share the same DRAM pool},
   $$

   so moving weights from GPU memory to CPU memory does not create additional capacity.

   The practical objective is

   $$
   \min T_{\text{inference}}
   \quad\text{subject to}\quad
   \operatorname{Accuracy}\ge a_{\min}.
   $$

   Here \(a_{\min}\) may allow a modest accuracy loss. For streaming video, timeliness itself affects usefulness: an answer about an earlier scene can become stale while inference is running. The intended improvement is therefore a better **accuracy–latency frontier**.  

2. **Streaming VLM inference consists of prefill, frame appending, and decoding.**

   Write the model as

   $$
   f_{\text{vision}},\qquad f_{\text{proj}},\qquad f_{\text{LLM}}.
   $$

   For a prompt \(p=(t_1,\ldots,t_m)\), prefill constructs the initial KV cache:

   $$
   (\text{next-token prediction},\mathcal K_0)
   =f_{\text{LLM}}(p).
   $$

   For incoming frame \(F_i\), the vision encoder and projector produce visual tokens:

   $$
   Z_i=f_{\text{proj}}\!\left(f_{\text{vision}}(F_i)\right)
   \in\mathbb R^{B_i\times d_{\text{model}}}.
   $$

   Frame appending updates the cache:

   $$
   (\text{next-token prediction},\mathcal K_i)
   =f_{\text{LLM}}(Z_i;\mathcal K_{i-1}).
   $$

   At each transformer layer, this appends the new tokens’ keys and values:

   $$
   K_{\text{cache}}\leftarrow[K_{\text{cache}};K_{\text{new}}],
   \qquad
   V_{\text{cache}}\leftarrow[V_{\text{cache}};V_{\text{new}}].
   $$

   Thus, after \(n\) frames, the cache represents

   $$
   m+\sum_{i=1}^{n}B_i
   $$

   processed tokens, before counting additional query or response tokens. Previously computed keys and values are reused.

   Intermediate token predictions are normally discarded during prefill and frame appending. Decoding begins after an explicit query is appended, or after a designated control token triggers a response. It then proceeds autoregressively:

   $$
   (y_{j+1},\mathcal K^{\text{dec}}_{j+1})
   =
   f_{\text{LLM}}(y_j;\mathcal K^{\text{dec}}_j).
   $$

   In the control-token case, the relevant final prediction from frame appending is preserved to initiate decoding. 

3. **Activation sparsification selects which input channels contribute to a matrix multiplication.**

   For one projection, let

   $$
   A\in\mathbb R^{B\times N},
   \qquad
   W\in\mathbb R^{N\times d},
   \qquad
   Y=AW.
   $$

   Here \(B\) is the number of tokens, \(N\) the number of input channels, and \(W_{i,:}\) the weight row associated with channel \(i\). Biases are omitted from the notation.

   The output decomposes into channel contributions:

   $$
   Y=\sum_{i=0}^{N-1}A_{:,i}W_{i,:}.
   $$

   A binary mask \(M\in\{0,1\}^{N}\) gives

   $$
   \widetilde Y
   =
   A\operatorname{diag}(M)W
   =
   \sum_iM_iA_{:,i}W_{i,:}.
   $$

   Only rows with \(M_i=1\) need to be loaded. The same mask is shared across the \(B\) tokens.

   The paper uses mean absolute activation as channel importance:

   $$
   \boxed{
   v_i=\frac1B\sum_{t=1}^{B}|A_{ti}|
   }.
   $$

   For a single token, \(v_i=|a_i|\). Define retained importance and sparsity by

   $$
   I(M)=\sum_i v_iM_i,
   \qquad
   \rho(M)=1-\frac{\|M\|_1}{N}.
   $$

   Given target sparsity \(\rho\), the channel budget is

   $$
   R=\lfloor(1-\rho)N\rfloor.
   $$

   Conventional top-\(R\) sparsification solves

   $$
   \max_{\|M\|_1\le R}I(M),
   $$

   by retaining the \(R\) largest \(v_i\). A fixed magnitude threshold is another possible selection rule.

   **Importance is a proxy for output quality.** The following derived bound explains its rationale:

   $$
   \begin{aligned}
   \frac1B\sum_{t=1}^{B}
   \|Y_{t,:}-\widetilde Y_{t,:}\|_2
   &\le
   \frac1B\sum_t\sum_i
   (1-M_i)|A_{ti}|\|W_{i,:}\|_2\\
   &=
   \sum_i(1-M_i)v_i\|W_{i,:}\|_2.
   \end{aligned}
   $$

   When weight-row norms are comparable, dropping small-\(v_i\) channels tends to limit this bound. Activation magnitude alone does not determine downstream task accuracy.

   Selection uses the current input’s activations and must be repeated during inference; the method requires no learned activation predictor. 

4. **Modern VLM importance distributions are relatively smooth, making storage-aware compromises practical.**

   ReLU produces exact zeros:

   $$
   \operatorname{ReLU}(x)=\max(0,x).
   $$

   Smooth alternatives include

   $$
   \operatorname{SiLU}(x)=\frac{x}{1+e^{-x}},
   \qquad
   \operatorname{GELU}(x)=x\Phi(x),
   $$

   where \(\Phi\) is the standard normal cumulative distribution function.

   For example, a SwiGLU-style MLP computes

   $$
   H=\operatorname{SiLU}(XW_{\text{gate}})
   \odot(XW_{\text{up}}),
   \qquad
   Y=HW_{\text{down}},
   $$

   with \(\odot\) denoting elementwise multiplication. Such activations generally lack ReLU’s large mass of exact zeros.

   VLM channel importance is also averaged over many visual tokens—for example,

   $$
   B=14\times14=196.
   $$

   Averaging reduces token-specific fluctuations in the observed workloads. It does not mathematically guarantee smoothness for every possible activation distribution.

   The paper measures variation across channels using

   $$
   \mu_v=\frac1N\sum_i v_i,
   \qquad
   \sigma_v^2=\frac1N\sum_i(v_i-\mu_v)^2,
   \qquad
   \operatorname{CV}(v)=\frac{\sigma_v}{\mu_v}.
   $$

   Before the down projection, it reports:

   | Model                | First layer | Middle layer | Last layer |
   | -------------------- | ----------: | -----------: | ---------: |
   | LLaVA-OneVision-7B   |        1.44 |         1.25 |       3.30 |
   | LLaVA-OneVision-0.5B |        1.31 |         1.33 |       3.58 |
   | VILA-8B              |        1.25 |         1.38 |       2.48 |
   | NVILA-2B             |        1.07 |         1.32 |       4.55 |
   | LongVA               |        1.20 |         1.34 |       3.01 |
   | ReLU-based OPT-6.7B  |       11.65 |         8.63 |       9.19 |

   Consequently, moderately important neighboring channels can be reasonable substitutes for scattered, slightly more important channels.

   The paper contrasts its typical operating regime,

   $$
   \rho\approx40\%-60\%,
   $$

   with the \(>90\%\) sparsity targeted by *LLM in a Flash*. These characterize the compared workloads; individual layers can have very different sparsities.  

5. **Reducing transferred bytes does not necessarily reduce flash latency.**

   If each weight row occupies \(b\) bytes, then

   $$
   D(M)=b\|M\|_1.
   $$

   For effective throughput \(\beta_{\text{eff}}(M)\),

   $$
   L_{\text{I/O}}(M)
   =
   \frac{D(M)}{\beta_{\text{eff}}(M)}.
   $$

   Flash throughput generally increases with contiguous request size \(z\), eventually approaching a bandwidth ceiling:

   $$
   \beta(z)\uparrow\beta_{\max}.
   $$

   Small reads are dominated by request-processing overhead; sufficiently large reads become bandwidth-limited.

   For two access patterns,

   $$
   \frac{L_2}{L_1}
   =
   \frac{D_2}{D_1}\frac{\beta_1}{\beta_2}.
   $$

   Therefore,

   $$
   \boxed{
   D_2<D_1
   \quad\text{but}\quad
   L_2>L_1
   \iff
   \frac{\beta_2}{\beta_1}<\frac{D_2}{D_1}
   }.
   $$

   The throughput loss can exceed the proportional data saving. Figure 4 on page 4 demonstrates this: scattered sparsification can initially increase latency, whereas sufficiently contiguous reads become faster as fewer bytes are loaded.

   This also explains why chunking may load **more channels at comparable accuracy yet finish sooner**.

   Expressed directly as a utility trade-off, if importance and latency decrease by \(\Delta I\) and \(\Delta L\), with positive remaining values,

   $$
   \frac{I-\Delta I}{L-\Delta L}>\frac IL
   \iff
   \frac{\Delta L}{L}>\frac{\Delta I}{I}.
   $$

   A small proportional importance loss is worthwhile when it buys a larger proportional latency reduction.  

6. **The contiguity distribution summarizes a mask by the lengths of its consecutive selected runs.**

   Let \(\mathcal C(M)\) be the set of **maximal contiguous runs** of selected indices. Define

   $$
   h_M(r)
   =
   \#\{C\in\mathcal C(M):|C|=r\}.
   $$

   Then

   $$
   \|M\|_1=\sum_{r\ge1}r\,h_M(r),
   \qquad
   n_{\text{chunks}}=\sum_{r\ge1}h_M(r),
   $$

   and the mean chunk length is

   $$
   \bar r
   =
   \frac{\sum_r r\,h_M(r)}{\sum_r h_M(r)}.
   $$

   For example,

   $$
   \{1,2,4,6,7\}
   =
   \{1,2\}\cup\{4\}\cup\{6,7\},
   $$

   giving

   $$
   h_M(1)=1,\qquad h_M(2)=2.
   $$

   This representation keeps chunk lengths and counts while discarding their exact locations, intervening gaps, and global arrangement.

   With a lookup table \(T[r]\) of effective per-chunk costs,

   $$
   \boxed{
   \widehat L(M)
   =
   \sum_{C\in\mathcal C(M)}T[|C|]
   =
   \sum_rh_M(r)T[r]
   }.
   $$

   The example therefore has estimated latency

   $$
   \widehat L=T[1]+2T[2].
   $$

   This is an **additive approximation** to hardware behavior. Its purpose is to make runtime estimation cheap enough to use for roughly \(200\) weight matrices per frame in LLaVA-OneVision-7B. 

7. **The latency table is obtained from saturated, steady-state read experiments.**

   For \(Q\) equal-sized chunks, with \(Q\) large enough to reach stable throughput,

   $$
   T[r]
   \approx
   \frac{t_{\text{measured}}(Q,r)}{Q}
   =
   \frac{br}{\beta(br)}.
   $$

   Thus \(T[r]\) represents an amortized contribution under the profiling workload; it is not the latency of an isolated request.

   The profiling procedure uses uniform-size reads, described in the main text as fixed-stride placements, and measures sufficiently many requests to amortize initiation overhead. Figure 3 on page 3 shows why this is practical: throughput stabilizes once the request count exceeds a modest threshold.

   The saturation size is defined approximately by

   $$
   z_{\text{sat}}
   =
   \min\{z:\beta(z)\ge0.99\beta_{\max}\}.
   $$

   Appendices D and H use

   $$
   z_{\text{sat}}=
   \begin{cases}
   236\text{ KB},&\text{AGX},\\
   348\text{ KB},&\text{Nano}.
   \end{cases}
   $$

   They profile in \(1\) KB increments, report throughput standard deviation below \(1\%\) of the mean, and finish profiling within \(20\) minutes per device. The flash microbenchmarks use a roughly \(128\) MB file; Figure 4 specifies Linux direct I/O and a six-thread C++ pool.

   **Source discrepancy:** Figure 4’s caption instead lists \(328\) KB for AGX and \(236\) KB for Nano. The values above follow the repeated settings in Appendices D and H. Saturation size should ultimately come from the target device’s profile.   

8. **The model is useful because its errors are approximately proportional.**

   The observed relationship is roughly

   $$
   L_{\text{measured}}
   =
   \alpha\widehat L+\varepsilon,
   \qquad \alpha>0.
   $$

   Uniform profiling does not reproduce the mixed chunk lengths, strides, controller behavior, and queue interactions of real accesses. These produce a systematic latency bias.

   Figure 5 on page 5 reports:

   | Model              | AGX \(R^2\) | Nano \(R^2\) |
   | ------------------ | ----------: | -----------: |
   | LLaVA-OneVision-7B |       0.995 |        0.950 |
   | NVILA-Lite-2B      |       0.929 |        0.832 |

   Lower concurrency in smaller models or lower-end hardware weakens the averaging effect and increases deviations.

   If all candidate costs receive the same multiplicative correction,

   $$
   T_{\text{actual}}[r]\approx\alpha T[r],
   $$

   their utilities satisfy

   $$
   U_{\text{actual}}(C)
   \approx
   \frac{I(C)}{\alpha T[|C|]}
   =
   \frac1\alpha U(C).
   $$

   Their ranking is therefore unchanged. This argument specifically requires a shared multiplicative factor; arbitrary affine or candidate-dependent errors can change the ordering. 

9. **Offline hot–cold reordering increases the opportunity to form useful contiguous chunks.**

   For calibration inputs \(c=1,\ldots,C\), define

   $$
   f_i
   =
   \frac1C
   \sum_{c=1}^{C}
   \mathbf1
   \left\{
   i\in\operatorname{Top}_{50\%}(v^{(c)})
   \right\}.
   $$

   “Active” here means selected among the most important channels. It does not mean merely having a nonzero activation.

   Sort channels by decreasing \(f_i\). For the corresponding permutation matrix \(P\),

   $$
   W'=PW,\qquad A'=AP^\top.
   $$

   The dense computation is preserved exactly:

   $$
   \boxed{
   A'W'=AP^\top PW=AW
   }.
   $$

   Runtime masks operate in this reordered coordinate system:

   $$
   \widetilde Y=A'\operatorname{diag}(M')W'.
   $$

   Reordering uses marginal activation frequencies. Ripple instead uses coactivation information. Appendix G finds broadly comparable, modest improvements from these two offline approaches.

   Offline ordering cannot determine the best subset for every future input. In the activation-frequency analysis, with frequency \(g_i\) measured under the evaluated selection policy,

   $$
   \text{hot}:g_i>0.99,\qquad
   \text{cold}:g_i<0.01.
   $$

   Many channels lie between these extremes, demonstrating input-dependent selection. Layer-specific sparsity also matters: at \(40\%\) effective sparsity, the first layer’s \(q\) projection is assigned approximately \(94\%\) sparsity.

   For one profiled down-projection permutation on Nano, the paper reports

   $$
   \text{mean}=1.5\text{ ms},
   \qquad
   95\%\text{ upper confidence bound}=1.8\text{ ms}.
   $$

   Its stated comparison is below \(0.02\%\) of total inference latency. This measurement concerns that single permutation operation, rather than the sum of all model-wide permutations.  

10. **The stated selection objective maximizes retained importance per estimated latency.**

    For \(R\ge1\), excluding the undefined empty-mask ratio,

    $$
    \boxed{
    \max_{\substack{M\in\{0,1\}^{N}\\1\le\|M\|_1\le R}}
    \frac{I(M)}{\widehat L(M)}
    =
    \max_M
    \frac{\sum_i v_iM_i}
    {\sum_rh_M(r)T[r]}
    }.
    $$

    For a candidate chunk \(C\),

    $$
    I(C)=\sum_{i\in C}v_i,
    \qquad
    U(C)=\frac{I(C)}{T[|C|]}.
    $$

    The algorithm ranks candidates using \(U(C)\) and greedily accepts compatible chunks.

    **Mathematical clarification:** this greedy procedure is a heuristic, not an exact solver for the displayed objective. Under the additive model,

    $$
    \frac{I(M)}{\widehat L(M)}
    =
    \sum_{C\in\mathcal C(M)}
    \underbrace{
    \frac{T[|C|]}{\sum_DT[|D|]}
    }_{\omega_C}
    U(C),
    \qquad
    \sum_C\omega_C=1.
    $$

    The total ratio is a weighted average of component utilities. An upper-bound budget alone therefore need not encourage using all \(R\) channels, whereas the implementation continues selecting eligible chunks toward that budget.

    Another limitation is that adjacent chunks may satisfy

    $$
    T[r_1+r_2]<T[r_1]+T[r_2].
    $$

    Their merged region can be more attractive than separate scores suggest. The greedy algorithm does not continually recompute these merger benefits, and the paper provides no global optimality guarantee. 

11. **The implementation generates multiscale windows, scores them, sorts them, and accepts nonoverlapping candidates.**

    Let \(b_{\text{KB}}\) be one row’s size in the same KB units as the hyperparameters. Convert the minimum size, maximum size, size increment, and stride cap to row counts:

    $$
    \begin{aligned}
    r_{\min}&=\max\!\left(1,\left\lfloor s_{\min}/b_{\text{KB}}\right\rfloor\right),\\
    r_{\max}&=\max\!\left(1,\left\lfloor s_{\max}/b_{\text{KB}}\right\rfloor\right),\\
    \Delta r&=\max\!\left(1,\left\lfloor\Delta s/b_{\text{KB}}\right\rfloor\right),\\
    J&=\max\!\left(1,\left\lfloor\text{jump\_cap}/b_{\text{KB}}\right\rfloor\right).
    \end{aligned}
    $$

    Build a CPU prefix-sum array:

    $$
    c_0=0,\qquad
    c_j=\sum_{i=0}^{j-1}v_i.
    $$

    For each candidate length

    $$
    r=r_{\min},r_{\min}+\Delta r,\ldots
    \le\min(r_{\max},N),
    $$

    use stride

    $$
    d_r=\min(r,J).
    $$

    Candidate starting indices are

    $$
    i=0,d_r,2d_r,\ldots\le N-r.
    $$

    Each candidate \(C=[i,i+r)\) receives

    $$
    \boxed{
    U(i,r)=\frac{c_{i+r}-c_i}{T[r]}
    }.
    $$

    Prefix sums make each importance calculation \(O(1)\). The candidate count is

    $$
    K_{\text{cand}}
    =
    \sum_r
    \left(
    1+\left\lfloor\frac{N-r}{d_r}\right\rfloor
    \right).
    $$

    When \(r\le J\), same-size windows do not overlap. When \(r>J\), the stride cap introduces overlapping windows and broader search coverage.

    Transfer candidate scores to the GPU and sort them in descending order. Initialize

    $$
    M=0,\qquad q=0.
    $$

    For each sorted candidate \((i,r)\), accept it exactly when

    $$
    q+r\le R
    \quad\text{and}\quad
    \sum_{j=i}^{i+r-1}M_j=0.
    $$

    On acceptance,

    $$
    M_{i:i+r}\leftarrow1,\qquad q\leftarrow q+r.
    $$

    Stop when \(q=R\) or no candidates remain. Overlap checking terminates early when an already selected row is encountered.

    The returned mask always respects the upper budget, but it can underfill it because available chunk sizes, alignment, or overlaps may prevent an exact fit. Adjacent accepted windows form one maximal selected run in the final mask. 

12. **Hyperparameters balance search coverage against an approximately \(2\) ms runtime budget per matrix.**

    The selection procedure is:

    $$
    \text{discard configurations with runtime}>2\text{ ms},
    $$

    then choose feasible settings with relatively small minimum sizes and strides.

    The sweep uses \(0\)–\(64\) KB in \(4\) KB increments, with

    $$
    \Delta s=s_{\min},
    \qquad
    s_{\max}=z_{\text{sat}}.
    $$

    Each configuration is measured \(30\) times at sparsity \(0.1\), a conservative high-retention setting, using random activation magnitudes.

    More than \(80\%\) of runtime is attributed to GPU sorting using data-independent radix sort, supporting the use of random inputs for this overhead measurement. Larger candidate spaces increase overhead; AGX supports more feasible configurations than Nano.

    Table 2 gives the following settings. Each pair is

    $$
    (s_{\min},\text{jump\_cap})\quad\text{in KB}.
    $$

    | Weight shape \(N\times d\) |         AGX |        Nano |
    | -------------------------- | ----------: | ----------: |
    | \(3584\times3584\)         | \((20,20)\) | \((24,36)\) |
    | \(8960\times1536\)         | \((16,16)\) | \((20,20)\) |
    | \(896\times4864\)          |   \((8,8)\) |   \((8,8)\) |
    | \(4096\times1024\)         | \((12,12)\) | \((16,16)\) |
    | \(3584\times18944\)        |   \((8,8)\) |   \((8,8)\) |
    | \(4096\times4096\)         | \((20,20)\) | \((24,24)\) |
    | \(18944\times3584\)        | \((32,32)\) | \((36,36)\) |
    | \(1536\times1536\)         | \((16,12)\) | \((16,12)\) |
    | \(1536\times256\)          |   \((8,8)\) |   \((8,8)\) |
    | \(896\times128\)           |   \((8,8)\) |   \((8,8)\) |
    | \(14336\times4096\)        | \((32,32)\) | \((40,36)\) |
    | \(4864\times896\)          | \((12,16)\) | \((20,16)\) |
    | \(3584\times512\)          |  \((8,12)\) |  \((8,12)\) |
    | \(896\times896\)           |   \((8,8)\) |   \((8,8)\) |
    | \(4096\times14336\)        |   \((8,8)\) |   \((8,8)\) |
    | \(1536\times8960\)         |   \((8,8)\) |   \((8,8)\) |

    These settings are chosen near the feasible boundary with a margin. The paper does not provide a full sensitivity analysis. 

13. **The evaluation compares channel-selection policies under the same layer-specific sparsity allocation.**

    Hardware:

    | Device           | Memory | SSD               | Rated peak sequential read |
    | ---------------- | -----: | ----------------- | -------------------------: |
    | Jetson Orin Nano |   8 GB | SK Hynix Gold P31 |                  3500 MB/s |
    | Jetson AGX Orin  |  32 GB | Samsung 990 Pro   |                  7450 MB/s |

    Models:

    $$
    \begin{gathered}
    \text{LLaVA-OneVision-Qwen2-7B},\\
    \text{LLaVA-OneVision-Qwen2-0.5B},\\
    \text{Llama-3-VILA1.5-8B},\\
    \text{NVILA-Lite-2B},\qquad
    \text{LongVA-7B}.
    \end{gathered}
    $$

    These support frame-by-frame processing. The benchmarks are TempCompass, NExT-QA with \(3000\) randomly sampled examples, and VideoDetailCaption.

    Both methods use TEAL profiling to allocate sparsity across layers and matrices:

    $$
    R_\ell
    =
    \left\lfloor
    (1-\rho_\ell)N_\ell
    \right\rfloor.
    $$

    TEAL determines **how much to retain** in each matrix. Top-\(R_\ell\) selection or Neuron Chunking determines **which channels to retain**.

    Profiling uses \(25\) of TempCompass’s \(410\) videos, excluded from the main evaluation. Reordering analysis splits these into \(20\) calibration videos and \(5\) validation videos.

    Appendix analyses examine layers

    $$
    \ell\in\{0,13,27\}
    $$

    of the \(28\)-layer LLaVA-OneVision-7B backbone, focusing on \(q,o,\text{gate},\text{down}\). The \(k,v,\text{up}\) projections are omitted from these visualizations because their inputs are shared with \(q,q,\text{gate}\), respectively.

    Evaluation sparsities are

    $$
    \rho\in\{0,0.1,\ldots,0.7\}.
    $$

    QA accuracy is the correct-answer fraction. Caption quality is a \(0\)–\(5\) score from `gpt-4o-mini-2024-07-18`; changing the evaluator prevents direct comparison with earlier papers’ caption scores.

    Accuracy is measured on a server with eight RTX A6000 GPUs. Latency is measured on the Jetsons over \(30\) repetitions, with `jetson_clocks` enabled and swap disabled.

    The methods text reports medians and \(95\%\) BCa bootstrap confidence intervals using \(10{,}000\) resamples. Several figure captions instead specify error bars of \(\pm1\) standard deviation, so the PDF is inconsistent about that reporting detail.   

14. **The reported headline improvements are I/O speedups at comparable accuracy.**

    Using linear interpolation to compare the curves at accuracy \(a\),

    $$
    S_{\text{I/O}}(a)
    =
    \frac{L_{\text{baseline}}(a)}
    {L_{\text{chunking}}(a)}.
    $$

    | Device | Average I/O speedup | Maximum I/O speedup |
    | ------ | ------------------: | ------------------: |
    | Nano   |      \(2.19\times\) |      \(4.65\times\) |
    | AGX    |      \(2.89\times\) |      \(5.76\times\) |

    AGX gains more because its throughput gap between contiguous and scattered reads is wider. Smaller models can also benefit strongly because their smaller weight rows make fragmentation more severe.

    Baseline latency can worsen at low or moderate sparsity. Occasional small accuracy improvements after sparsification are attributed to removing weak or noisy activations; this is an empirical possibility rather than a guaranteed effect.

    End-to-end gains are smaller than I/O-only gains. A derived decomposition makes this explicit:

    $$
    T_{\text{base}}=C+I,
    $$

    $$
    T_{\text{ours}}
    =
    C+\Delta C+\frac{I}{S_{\text{I/O}}}+H,
    $$

    hence

    $$
    \boxed{
    S_{\text{end-to-end}}
    =
    \frac{C+I}
    {C+\Delta C+I/S_{\text{I/O}}+H}
    }.
    $$

    Here \(C\) is baseline computation, \(\Delta C\) the additional computation from retaining more channels at matched accuracy, and \(H\) the added selection and related overhead.

    Chunk selection costs approximately

    $$
    2\text{ ms/matrix}\times200\text{ matrices}
    \approx400\text{ ms}
    $$

    for the full model. Figure 8 on page 8 shows the breakdown at a reported \(5\%\) accuracy drop. The evaluation does not apply I/O–compute overlap or the suggested further kernel optimizations.  

15. **Ablations show that online selection produces most of the contiguity improvement.**

    For LLaVA-7B, the component ablation reports up to

    $$
    1.23\times
    \quad\text{with reordering},
    $$

    increasing to

    $$
    2.55\times
    \quad\text{with reordering and chunk selection}.
    $$

    In Figure 10 on page 8,

    $$
    \bar r:
    \quad
    1.1
    \xrightarrow{\text{reordering}}
    1.8
    \xrightarrow{\text{chunking}}
    49.6.
    $$

    The corresponding most frequent chunk lengths are

    $$
    1\rightarrow1\rightarrow48.
    $$

    The plot’s “Max” annotation identifies the histogram peak—the **mode**—rather than the largest observed chunk. Appendix J clarifies this terminology and extends the analysis across layers and projections.

    These values describe one case study. Across settings, reordering yields limited and variable gains, while input-dependent chunk selection consistently shifts access patterns toward longer runs.

    The method also remains effective when visual-token density changes. Spatial pooling tests

    $$
    B\in\{14^2,9^2,7^2,5^2\}
    =
    \{196,81,49,25\}.
    $$

    For context capacity \(C_{\text{ctx}}\), prompt length \(m\), and reserved query/response capacity \(r_{\text{reserve}}\), a derived frame-capacity bound is

    $$
    n_{\text{frames}}
    \le
    \left\lfloor
    \frac{C_{\text{ctx}}-m-r_{\text{reserve}}}{B}
    \right\rfloor.
    $$

    Fewer tokens allow more frames in context, with a modest measured accuracy loss. Figure 16 on page 22 shows that the chunking advantage persists at all tested token densities. More advanced token-reduction methods such as clustering are also compatible in principle.  

16. **Prior sparsification and bundling methods address related parts of the problem.**

    | Method or family              | Main role                                                                              |
    | ----------------------------- | -------------------------------------------------------------------------------------- |
    | Deja Vu                       | Exploits dynamic MLP sparsity                                                          |
    | CATS                          | Extends activation sparsification to gated, non-ReLU MLPs                              |
    | TEAL                          | Extends sparsification to attention and allocates sparsity across matrices             |
    | Ripple                        | Improves locality through offline coactivation-based reordering                        |
    | LLM in a Flash / PowerInfer-2 | Reduce flash traffic using sparsity, weight bundling, and memory-management techniques |
    | Neuron Chunking               | Selects runtime contiguous regions using measured storage costs                        |

    GPU-resident sparsification encounters a different memory hierarchy, where bandwidth can saturate at much smaller access granularity. Its channel-selection policy can therefore perform poorly when transferred directly to flash.

    ReLU-ification increases exact activation sparsity through retraining. The paper characterizes the compared approaches as requiring substantial retraining, on the order of at least \(50\) billion tokens.

    In *LLM in a Flash*, row–column bundling combines an up-projection column with the corresponding down-projection row. Directly applying that scheme conflicts with this paper’s predictor-free selection from each projection’s own input activations.

    Matrices sharing inputs can instead be bundled, such as

    $$
    (Q,K,V),\qquad(\text{up},\text{gate}).
    $$

    However, the largest adapted bundles are approximately

    $$
    74\text{ KB}<236\text{–}348\text{ KB},
    $$

    reaching only about half the optimal bandwidth on the evaluated hardware. Bundling overlapping selected channels may also scatter the remaining unbundled channels.

    The paper contrasts Jetson’s larger saturation sizes with the below-\(100\) KB sizes reported for the MacBook setup in *LLM in a Flash*. It proposes differences in NVMe interrupt distribution as a possible explanation, rather than establishing that cause experimentally.

    Appendix L reports these speedup ratios. Each cell is

    $$
    \left(
    \frac{L_{\text{baseline}}}{L_{\text{ours}}}
    \;,\;
    \frac{L_{\text{baseline+bundling}}}{L_{\text{ours}}}
    \right).
    $$

    | Dataset            |    LLaVA-7B |  LLaVA-0.5B |     VILA-8B |    NVILA-2B |      LongVA |
    | ------------------ | ----------: | ----------: | ----------: | ----------: | ----------: |
    | TempCompass        | 2.06 / 2.41 | 2.05 / 1.94 | 1.60 / 1.83 | 3.24 / 3.76 | 2.15 / 2.50 |
    | VideoDetailCaption | 2.11 / 2.45 | 2.06 / 2.02 | 1.60 / 1.78 | 3.22 / 3.70 | 2.25 / 2.59 |
    | NExT-QA            | 1.76 / 1.98 | 2.12 / 1.99 | 1.50 / 1.70 | 3.44 / 3.96 | 2.04 / 2.34 |

    Bundling worsens latency in most of these cases, with LLaVA-0.5B as the exception. Its benefit depends on the resulting access pattern.  

17. **Caching complements chunking when additional memory is available.**

    Sliding-window caching keeps recently activated weights; hot-neuron caching keeps frequently accessed weights. Both trade additional memory for fewer flash reads.

    Offline reordering improves the layout without retaining an additional cache of weight rows, but it adapts less directly to recent runtime behavior.

    To distinguish cached computation from new I/O, let

    $$
    c_i=\mathbf1\{\text{row }i\text{ is cached}\},
    \qquad
    o_i=\mathbf1\{\text{row }i\text{ is selected for loading}\}.
    $$

    The paper proposes zeroing the I/O-selection importance of cached rows:

    $$
    \bar v_i=(1-c_i)v_i.
    $$

    If cached rows are included in the computation, the retained mask is

    $$
    m_i=c_i\lor o_i,
    \qquad
    \widetilde Y=A\operatorname{diag}(m)W.
    $$

    Thus zero I/O-selection importance does **not** mean discarding a cached channel’s contribution. Actual read-cost accounting must still reflect gaps or any deliberate rereading of cached rows.

    Caching hot channels can make the remaining uncached requests more scattered, increasing the value of chunking. Conversely,

    $$
    \text{most weights cached}
    \;\Longrightarrow\;
    \text{little remaining flash I/O}
    \;\Longrightarrow\;
    \text{smaller potential benefit}.
    $$

    Asynchronous interfaces such as `io_uring` may narrow the scattered-versus-contiguous performance gap. The authors expect read coalescing and prefetching to preserve some advantage for contiguous access, but this is a prospective argument.  

18. **Static compression can be combined with dynamic activation sparsification.**

    Quantization, weight pruning, and distillation reduce the model’s fixed storage or computation requirements. Activation sparsification additionally exploits input-dependent relevance:

    $$
    M=M(A).
    $$

    Unstructured \(L_1\) regularization,

    $$
    \mathcal L_{\text{train}}(W)
    +\lambda\sum_{i,j}|W_{ij}|,
    $$

    encourages individual small or zero weights. A row containing any needed entries may still have to be loaded, so scattered weight zeros need not save row-level I/O.

    Group-Lasso regularization,

    $$
    \mathcal L_{\text{train}}(W)
    +\lambda\sum_g\|W_g\|_2,
    $$

    can encourage entire rows or columns to vanish.

    For a gated MLP, eliminating an output channel of the up or gate projection can produce

    $$
    H_{:,i}=0,
    $$

    making the corresponding down-projection row unnecessary. Row-norm shrinkage can also support a weight-aware importance measure such as

    $$
    v_i^{\text{weighted}}
    =
    v_i\|W_{i,:}\|_2,
    $$

    consistent with the error bound in point 3.

    Static pruning removes the same parameters across inputs and can incur accuracy loss at relatively modest pruning rates; Appendix L cites an example around \(20\%\), which is not a universal threshold.

    Compression and dynamic chunk selection are complementary, although changed weight sizes or architectures require appropriate profiling.  

19. **Generalization depends on both activation statistics and the hardware–model pair.**

    The favorable conditions are

    $$
    \text{relatively smooth channel importance},
    $$

    $$
    \beta(\text{one weight row})<\beta_{\max},
    $$

    together with enough I/O cost to outweigh selection overhead.

    From the end-to-end decomposition, net improvement requires

    $$
    \boxed{
    I\left(1-\frac1{S_{\text{I/O}}}\right)
    >
    \Delta C+H
    }.
    $$

    Proposed extensions include speculative decoding, parallel sampling, and batched LLM inference. Aggregating importance across tokens or samples,

    $$
    v_i=\frac1B\sum_{t=1}^{B}|A_{ti}|,
    $$

    gives a shared mask and shared weight-access pattern for that group.

    Plain LLMs with non-ReLU activations can also benefit, though their single-token importance may be less smooth. ViT backbones are structurally compatible; their smaller weight rows can exacerbate fragmentation on constrained devices.

    Appendix N presents preliminary GSM8K-input experiments using selected importance as the quality proxy:

    $$
    \text{proxy quality}=I(M).
    $$

    Across first, middle, and last layers, the reported average importance–latency speedups are

    $$
    \text{LLaMA3-8B}:1.22\times,
    \qquad
    \text{Qwen2-7B}:2.09\times.
    $$

    These experiments establish preliminary improvements in **importance versus latency**. Full-task accuracy–latency validation for those LLM extensions, quantitative ViT validation, and broader hyperparameter sensitivity remain open. 
