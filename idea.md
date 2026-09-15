# Idea for my bachelor's thesis: *VLM in Flash*

Seol Handong

---

## Under the displayed upper-bound formulation, an exact optimum can always be represented by a single contiguous chunk.

Let $N$ be the number of input channels of a weight matrix and let

$$M\in\{0,1\}^{N},$$

where $M_i=1$ means that channel $i$ and its corresponding weight row are selected.

Let $\mathcal C(M)$ denote the set of maximal contiguous runs of selected channels. Under the paper's additive latency model,

$$\widehat L(M)=\sum_{C\in\mathcal C(M)}T[|C|],$$

where $T[r]$ is the profiled flash-read latency associated with a contiguous chunk of $r$ rows.

Define the total retained activation importance as

$$I(M)=\sum_{i=0}^{N-1}v_iM_i,$$

where $v_i$ is the importance of channel $i$. The paper's displayed selection problem can then be written as

$$\boxed{\max_{\substack{M\in\{0,1\}^{N}\\1\le\|M\|_1\le R}}\frac{I(M)}{\widehat L(M)}},\qquad R=\left\lfloor(1-\rho)N\right\rfloor.$$

The lower bound excludes the empty mask, for which the ratio is undefined.

For a contiguous chunk $C$, define

$$I(C)=\sum_{i\in C}v_i,\qquad U(C)=\frac{I(C)}{T[|C|]}.$$

Here $U(C)$ is the chunk's retained importance per unit of estimated flash-read latency.

Because distinct maximal chunks partition the selected channels,

$$I(M)=\sum_{C\in\mathcal C(M)}I(C).$$

Therefore,

$$\frac{I(M)}{\widehat L(M)}=\frac{\sum_{C\in\mathcal C(M)}I(C)}{\sum_{D\in\mathcal C(M)}T[|D|]}=\sum_{C\in\mathcal C(M)}\underbrace{\frac{T[|C|]}{\sum_{D\in\mathcal C(M)}T[|D|]}}_{w_C}\underbrace{\frac{I(C)}{T[|C|]}}_{U(C)}.$$

The coefficients satisfy

$$w_C\ge0,\qquad \sum_{C\in\mathcal C(M)}w_C=1.$$

Hence the mask-level importance–latency ratio is a latency-weighted average of the utilities of its constituent chunks:

$$\boxed{\frac{I(M)}{\widehat L(M)}\le\max_{C\in\mathcal C(M)}U(C)}.$$

Let $C^\star$ be the highest-utility contiguous interval whose length does not exceed $R$:

$$C^\star\in\arg\max_{\substack{C\text{ contiguous}\\1\le|C|\le R}}\frac{I(C)}{T[|C|]}.$$

The mask selecting only $C^\star$ is feasible and attains

$$\frac{I(C^\star)}{T[|C^\star|]}=U(C^\star).$$

Consequently,

$$\boxed{\max_{\substack{M\in\{0,1\}^{N}\\1\le\|M\|_1\le R}}\frac{I(M)}{\widehat L(M)}=\max_{\substack{C\text{ contiguous}\\1\le|C|\le R}}U(C)}.$$

Thus, under the displayed objective and additive latency assumption, there always exists an exact optimum consisting of only one contiguous chunk. Multiple chunks can also be optimal in the exceptional case where every selected chunk has the same maximum utility.

This optimum can be found without examining all $2^N$ masks. Define the importance prefix sums

$$P_0=0,\qquad P_j=\sum_{i=0}^{j-1}v_i.$$

A chunk

$$C_{i,\ell}=\{i,i+1,\ldots,i+\ell-1\}$$

then has

$$I(C_{i,\ell})=P_{i+\ell}-P_i$$

and utility

$$U(i,\ell)=\frac{P_{i+\ell}-P_i}{T[\ell]}.$$

There are

$$\sum_{\ell=1}^{R}(N-\ell+1)=\frac{R(2N-R+1)}2$$

contiguous intervals of length at most $R$. Without the length restriction, the total number is

$$\binom{N+1}{2}=\frac{N(N+1)}2.$$

Prefix sums require $O(N)$ preprocessing and make each interval score computable in $O(1)$. The exact optimum is therefore obtainable in

$$\boxed{O(NR)}$$

time and $O(N)$ memory, which becomes $O(N^2)$ time in the worst case $R=\Theta(N)$.

This simple exact solution exposes a mismatch between the displayed formulation and the implemented algorithm. The displayed constraint

$$\|M\|_1\le R$$

permits the optimizer to leave most of the budget unused, whereas the paper's greedy procedure continues accepting chunks toward $R$. Its implemented behavior is therefore closer to a fixed-cardinality or cardinality-band problem such as

$$\|M\|_1=R$$

or

$$R_-\le\|M\|_1\le R_+,$$

for which the maximum-utility single-chunk argument no longer provides a general exact solution.

---

## The paper's greedy algorithm approximates a fixed-budget importance-per-latency objective.

The paper's implementation treats

$$R=\left\lfloor(1-\rho)N\right\rfloor$$

as a target number of weight rows to retain. Rather than deciding the sparsity level, the selection algorithm receives $R$ from the sparsity allocation and determines which rows should be loaded.

Let $\mathcal G$ denote the generated set of contiguous candidate chunks. Each candidate $C\in\mathcal G$ is assigned the utility

$$U(C)=\frac{I(C)}{T[|C|]},\qquad I(C)=\sum_{i\in C}v_i.$$

Thus, $U(C)$ measures the activation importance preserved by the chunk per unit of estimated flash-read latency.

The candidates are sorted by decreasing utility:

$$U(C_{(1)})\ge U(C_{(2)})\ge\cdots\ge U(C_{(K)}).$$

The algorithm initializes

$$S\leftarrow\varnothing,\qquad q\leftarrow0,$$

where $S$ is the set of accepted chunks and $q$ is the number of selected rows. It then processes the sorted candidates in order. A candidate $C_{(j)}$ is accepted if

$$C_{(j)}\cap\bigcup_{D\in S}D=\varnothing$$

and

$$q+|C_{(j)}|\le R.$$

Upon acceptance,

$$S\leftarrow S\cup\{C_{(j)}\},\qquad q\leftarrow q+|C_{(j)}|.$$

The procedure continues until $q=R$ or no remaining candidate can be accepted. The resulting mask is

$$\boxed{M_i^{\mathrm{greedy}}=\mathbf1\left\{i\in\bigcup_{C\in S}C\right\}.}$$

The algorithm may return fewer than $R$ rows when the remaining budget cannot be filled by any nonoverlapping candidate. Nevertheless, its operational goal is to select approximately $R$ rows rather than to determine the row count from scratch.

The candidate utility is the chunk-level counterpart of the mask-level objective

$$\frac{I(M)}{\widehat L(M)}=\frac{\sum_{C\in\mathcal C(M)}I(C)}{\sum_{C\in\mathcal C(M)}T[|C|]}.$$

Accordingly, the algorithm gives priority to chunks with high importance per estimated latency and greedily combines them while attempting to fill the prescribed row budget. This is a heuristic construction: ranking chunks individually does not guarantee that their combination maximizes the global ratio.

The position of Neuron Chunking can be understood by comparing it with conventional top-$R$ sparsification. Given the same prescribed row count, top-$R$ selects

$$\boxed{M_{\mathrm{top}\text{-}R}\in\arg\max_{\substack{M\in\{0,1\}^{N}\\\|M\|_1=R}}I(M).}$$

Because

$$I(M)=\sum_i v_iM_i,$$

this selects the $R$ channels with the largest individual importance values. It maximizes retained activation importance without considering the physical arrangement or flash-read latency of the selected rows.

Neuron Chunking instead aims to balance retained importance against flash latency:

$$\boxed{M_{\mathrm{chunk}}\approx\arg\max_{\substack{M\in\{0,1\}^{N}\\\|M\|_1=R}}\frac{I(M)}{\widehat L(M)}.}$$

The symbol $\approx$ indicates that the paper's greedy procedure is intended as a fast heuristic for this objective, not as an exact optimizer.

The methodological difference is therefore

$$\boxed{\begin{aligned}\text{Top-}R&:\quad\max_{\|M\|_1=R} I(M),\\\text{Neuron Chunking}&:\quad\max_{\|M\|_1=R}\frac{I(M)}{\widehat L(M)}.\end{aligned}}$$

Top-$R$ always prefers the individually most important rows, even when they are scattered across storage. Neuron Chunking may instead select contiguous, slightly less important rows when the reduction in flash-read latency compensates for the lost importance.

Section 3.2.1 formally states the constraint as

$$\|M\|_1\le R.$$

The fixed-$R$ expression above is therefore not the literal displayed formulation. It is a reconstruction of the problem suggested by the algorithm and experimental comparison: $R$ is determined by the target sparsity, and the greedy procedure continues selecting chunks toward that prescribed budget.

---

## The profiled chunk latency is closely approximated by an affine function.

The paper represents the latency of a contiguous chunk using the profiled lookup table

$$T[r],$$

where $r$ is the number of weight rows in the chunk. The released profiles instead record chunk size in kilobytes. Let $z$ denote the chunk size in KB. A least-squares fit to these profiles gives

$$\boxed{
T_{\mathrm{KB}}(z)\approx a+c_{\mathrm{KB}}z.
}$$

The fitted models are:

| Device | Profiled range | Affine fit, with \(z\) in KB and \(T\) in ms | \(R^2\) |
|---|---:|---:|---:|
| Jetson Orin AGX | \(1\text{--}255\) KB | \(T(z)\approx0.01100+0.0001398z\) | \(0.9867\) |
| Jetson Orin Nano | \(1\text{--}350\) KB | \(T(z)\approx0.01573+0.0002842z\) | \(0.9940\) |

These $R^2$ values are not reported in the original paper; they are obtained by fitting the latency tables released with its implementation. The mean absolute percentage errors are approximately $4.22\%$ on AGX and $3.21\%$ on Nano.

The latency curves therefore have a nearly linear increasing shape with a positive intercept, together with relatively small local fluctuations. The affine approximation is useful as a structural model rather than as an exact replacement for every lookup-table entry.

The affine model can also be validated in the throughput domain. The released implementation computes logical throughput as

$$
\beta(z)
=
\frac{1000z}{1024T_{\mathrm{KB}}(z)}
\quad\text{MiB/s},
$$

where $z$ is measured in KB and $T_{\mathrm{KB}}(z)$ in milliseconds. Substituting

$$T_{\mathrm{KB}}(z)\approx a+c_{\mathrm{KB}}z$$

gives

$$
\boxed{
\widehat\beta(z)
=
\frac{1000z}
{1024\left(a+c_{\mathrm{KB}}z\right)}
}.
$$

This function increases with chunk size and approaches

$$
\boxed{
\beta_\infty
=
\frac{1000}{1024c_{\mathrm{KB}}}
\quad\text{MiB/s}
}
$$

as $z\to\infty$. It therefore reproduces the qualitative behavior observed in the paper: throughput rises rapidly for small contiguous reads and gradually approaches a bandwidth ceiling.

The throughput curves derived from the affine model closely match the released measurements. The coefficients of determination in the throughput domain are

$$
R^2_{\mathrm{AGX}}=0.9875,
\qquad
R^2_{\mathrm{Nano}}=0.9866.
$$

At the saturation sizes reported in Appendices D and H,

$$
z_{\mathrm{sat}}^{\mathrm{AGX}}=236\text{ KB},
\qquad
z_{\mathrm{sat}}^{\mathrm{Nano}}=348\text{ KB},
$$

the measured and affine-predicted throughputs are approximately

$$
\begin{array}{c|cc}
&\text{Measured}&\text{Affine prediction}\\
\hline
\text{AGX}&5221\text{ MiB/s}&5238\text{ MiB/s}\\
\text{Nano}&3096\text{ MiB/s}&2965\text{ MiB/s}.
\end{array}
$$

The fitted asymptotic throughputs are

$$
\beta_\infty^{\mathrm{AGX}}
\approx6985\text{ MiB/s},
\qquad
\beta_\infty^{\mathrm{Nano}}
\approx3436\text{ MiB/s}.
$$

For comparison, the advertised SSD limits are

$$
7450\text{ MB/s}\approx7105\text{ MiB/s}
$$

for AGX and

$$
3500\text{ MB/s}\approx3338\text{ MiB/s}
$$

for Nano. The affine asymptotes differ from these advertised limits by approximately $1.7\%$ and $2.9\%$, respectively.

This comparison should be interpreted cautiously. The quantities $\beta_\infty$ are extrapolated asymptotes of fitted models, whereas the paper's throughput measurements are obtained at finite chunk sizes. Nevertheless, the agreement in both the latency and throughput domains supports the affine approximation over the profiled operating range.

To express the model in weight-row units, suppose one row occupies $b$ KB. A chunk containing $r$ rows then occupies

$$z=br\text{ KB}.$$

Therefore,

$$
T[r]
=
T_{\mathrm{KB}}(br)
\approx
a+c_{\mathrm{KB}}br.
$$

Defining

$$c=bc_{\mathrm{KB}},$$

the row-based latency model becomes

$$\boxed{
T[r]\approx a+cr.
}$$

Here:

- $a$ is an effective fixed cost incurred once per contiguous chunk;
- $c$ is the marginal latency of adding one more row to that chunk.

The parameter $a$ should be interpreted as an effective per-chunk penalty that summarizes small-read inefficiencies. It is not separately measured by the paper as one particular hardware operation.

Let

$$K(M)=|\mathcal C(M)|$$

be the number of maximal contiguous chunks in mask $M$, and let

$$
R(M)
=
\|M\|_1
=
\sum_{C\in\mathcal C(M)}|C|
$$

be the total number of selected rows. Substituting the affine approximation into the paper's additive latency model gives

$$
\begin{aligned}
\widehat L(M)
&=
\sum_{C\in\mathcal C(M)}T[|C|]\\
&\approx
\sum_{C\in\mathcal C(M)}
\left(a+c|C|\right)\\
&=
a|\mathcal C(M)|
+
c\sum_{C\in\mathcal C(M)}|C|\\
&=
\boxed{
aK(M)+cR(M)
}.
\end{aligned}
$$

Thus, the affine latency depends only on two structural properties of the mask:

- the number of selected rows, $R(M)$;
- the number of separate contiguous chunks, $K(M)$.

When the selected row count is fixed at $R$,

$$R(M)=R,$$

the row-transfer term is constant:

$$
\widehat L(M)
\approx
aK(M)+cR.
$$

For $a>0$, differences in affine latency among masks with the same row count are therefore determined entirely by their numbers of chunks:

$$
\boxed{
R(M)=R
\quad\Longrightarrow\quad
\min_M\widehat L(M)
\iff
\min_MK(M).
}
$$

This explains why the paper emphasizes long contiguous regions. Among masks containing the same number of rows, a mask with fewer and longer chunks pays the effective per-chunk cost $a$ fewer times than a fragmented mask.