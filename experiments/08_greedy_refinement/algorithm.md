# Coverage-preserving refinement of Neuron Chunking

---

## Short-gap merging and boundary trimming improve a feasible coverage mask at low additional cost.

### Problem setup

Let $N$ be the number of input channels and let

$$
M\in\{0,1\}^{N},
$$

where $M_i=1$ means that channel $i$ and its corresponding weight row are
selected. Let $\mathcal C(M)$ be the set of maximal contiguous runs of selected
channels. Define

$$
R(M)=\|M\|_1,
\qquad
K(M)=|\mathcal C(M)|,
$$

and assume nonnegative channel importance,

$$
v_i\ge0.
$$

The retained importance is

$$
I(M)=\sum_{i=0}^{N-1}v_iM_i.
$$

Under the fitted affine latency model,

$$
\boxed{
L_{\mathrm{aff}}(M)=aK(M)+cR(M),
}
$$

where $a>0$ is the cost of opening a chunk and $c>0$ is the marginal cost of
reading one row.

Let $M^{(0)}$ be the mask returned by the paper's greedy selection algorithm.
The refinement uses its attained importance as the coverage lower bound:

$$
\boxed{
B=I(M^{(0)}).
}
$$

The corresponding coverage problem is

$$
\boxed{
\min_M L_{\mathrm{aff}}(M)
\quad\text{subject to}\quad
I(M)\ge B.
}
$$

The refinement is a feasible local improvement method for this problem. It
does not claim to solve the coverage problem exactly.

### Dominance of a short internal gap

Consider two consecutive maximal chunks separated by an unselected internal
gap $G$ of length $g$. Let $M^{+G}$ denote the mask obtained by selecting every
row in $G$. Then

$$
R(M^{+G})=R(M)+g
$$

and the two chunks become one, so

$$
K(M^{+G})=K(M)-1.
$$

The latency change is therefore

$$
\begin{aligned}
L_{\mathrm{aff}}(M^{+G})-L_{\mathrm{aff}}(M)
&=
a\bigl(K(M)-1\bigr)+c\bigl(R(M)+g\bigr)
-aK(M)-cR(M)\\
&=cg-a.
\end{aligned}
$$

Consequently,

$$
\boxed{
g<\frac ac
\quad\Longrightarrow\quad
L_{\mathrm{aff}}(M^{+G})<L_{\mathrm{aff}}(M).
}
$$

Because every importance value is nonnegative,

$$
I(M^{+G})
=
I(M)+\sum_{i\in G}v_i
\ge
I(M).
$$

Thus, a feasible mask containing an internal gap with $g<a/c$ is strictly
dominated: filling the gap does not reduce importance and strictly reduces
latency.

For integer gap lengths, the largest strictly profitable gap is

$$
\boxed{
g_{\max}=\left\lceil\frac ac\right\rceil-1.
}
$$

The merge phase fills every internal gap satisfying

$$
1\le g\le g_{\max}.
$$

Let the resulting mask be $M^{(m)}$. It satisfies

$$
I(M^{(m)})\ge B,
\qquad
L_{\mathrm{aff}}(M^{(m)})\le L_{\mathrm{aff}}(M^{(0)}),
$$

and every remaining internal gap has length at least $\lceil a/c\rceil$.

### Coverage surplus created by merging

Merging can retain additional importance beyond the required bound. Define the
current coverage surplus as

$$
\boxed{
S(M)=I(M)-B.
}
$$

A selected row $i$ is a boundary row if it has at most one selected immediate
neighbor:

$$
M_i=1,
\qquad
M_{i-1}+M_{i+1}\le1,
$$

where out-of-range neighbors are treated as unselected.

Removing a boundary row never splits a chunk. If the row belongs to a chunk of
length at least two, its removal changes the row count but not the chunk count:

$$
\Delta R=-1,
\qquad
\Delta K=0.
$$

The saved latency is then

$$
\delta_i(M)=c.
$$

If the boundary row is a singleton chunk, removing it also closes that chunk:

$$
\Delta R=-1,
\qquad
\Delta K=-1,
$$

and the saved latency is

$$
\delta_i(M)=a+c.
$$

Both cases can be written as

$$
\boxed{
\delta_i(M)
=
c+a\,\mathbf1\{i\text{ is a singleton chunk in }M\}.
}
$$

The local loss per saved latency is

$$
\boxed{
\phi_i(M)=\frac{v_i}{\delta_i(M)}.
}
$$

The trimming phase gives priority to the currently exposed boundary row with
the smallest $\phi_i(M)$.

### Greedy boundary-trimming rule

Initialize

$$
M\leftarrow M^{(m)},
\qquad
S\leftarrow I(M)-B.
$$

Insert every current boundary row into a min-priority queue using
$\phi_i(M)$ as its key. Repeatedly extract the smallest-key candidate $i$.
Because earlier removals may have changed its neighborhood, first recompute
$\delta_i(M)$ and its boundary status. A stale candidate is discarded or
reinserted with its updated key.

If

$$
v_i\le S,
$$

remove row $i$ and update

$$
M_i\leftarrow0,
\qquad
S\leftarrow S-v_i.
$$

The newly exposed neighboring row, if any, is then inserted into the priority
queue. If $v_i>S$, row $i$ cannot be removed. Since $S$ only decreases during
trimming, the same row cannot become feasible later and may be discarded.

The procedure stops when the priority queue is empty. Denote the returned mask
by

$$
M^{(r)}.
$$

An implementation-equivalent description is:

```text
Input: importance v, Paper mask M^(0), affine costs a and c

B <- I(M^(0))
M <- M^(0)

for each internal gap G between consecutive chunks of M:
    if c * |G| < a:
        select every row in G

S <- I(M) - B
H <- min-heap of current boundary rows keyed by v_i / delta_i(M)

while H is not empty:
    i <- extract-min(H)
    refresh the boundary status and delta_i(M)
    if i is stale:
        discard it or reinsert it with the updated key
    else if v_i <= S:
        M_i <- 0
        S <- S - v_i
        insert any newly exposed neighbor into H

return M
```

### Feasibility and monotonic improvement

Every merge preserves feasibility because it only changes zero entries to one
and $v_i\ge0$. Every accepted trimming step removes importance of at most the
current surplus:

$$
v_i\le S(M)=I(M)-B.
$$

Therefore, after removing row $i$,

$$
I(M)-v_i\ge B.
$$

By induction over all trimming steps,

$$
\boxed{
I(M^{(r)})\ge B=I(M^{(0)}).
}
$$

Each accepted merge with $g<a/c$ strictly reduces latency. Each accepted
boundary removal saves either $c$ or $a+c$, both of which are positive. Hence,

$$
\boxed{
L_{\mathrm{aff}}(M^{(r)})
\le
L_{\mathrm{aff}}(M^{(m)})
\le
L_{\mathrm{aff}}(M^{(0)}).
}
$$

The first inequality is strict if at least one boundary row is removed. The
second is strict if at least one gap is merged.

Thus the refinement has a per-input non-worsening guarantee under the affine
latency model:

$$
\boxed{
I(M^{(r)})\ge I(M^{(0)})
\quad\text{and}\quad
L_{\mathrm{aff}}(M^{(r)})\le L_{\mathrm{aff}}(M^{(0)}).
}
$$

### Time and memory complexity

The maximal chunks and their internal gaps can be found in one scan. Each gap
is processed once, so the merge phase takes

$$
O(N)
$$

time. During trimming, each selected row becomes exposed only a constant
number of times. There are therefore $O(N)$ priority-queue insertions and
extractions, each costing $O(\log N)$. The trimming phase takes

$$
O(N\log N)
$$

time and $O(N)$ memory. The complete refinement has complexity

$$
\boxed{
O(N\log N)\text{ time and }O(N)\text{ memory}.
}
$$

The paper's current implementation also sorts a linear number of candidate
chunks by utility. Its comparison-sort complexity is $O(N\log N)$ under a
fixed candidate-window configuration. Adding the refinement therefore does
not change the asymptotic time complexity of the complete selection pipeline.

### Relationship to fixed-budget selection

The refinement does not preserve the Paper row budget. Gap merging can increase
$R(M)$, while boundary trimming can subsequently decrease it. Its invariant is

$$
I(M)\ge B,
$$

rather than

$$
R(M)=R
$$

or

$$
R(M)\le R.
$$

It is therefore an algorithm for the importance-coverage formulation, using
the Paper mask only to define a feasible target and a low-cost initial
solution. It is not a feasible refinement for a strict fixed-$R$ problem when
merging causes the row count to exceed the prescribed budget.

### Optimality limitation

The short-gap merge rule is a dominance rule: every accepted merge is provably
beneficial under the affine model. Boundary trimming is greedy and has no
global optimality guarantee.

In particular, a globally better solution may require:

* removing several rows as one package even when none is the best current
  boundary candidate;
* deleting an entire chunk to recover its opening cost;
* merging a gap with $g\ge a/c$ and compensating by trimming a different
  region; or
* replacing selected rows with a different collection of chunks.

The method never performs these general exchanges. Consequently,

$$
L_{\mathrm{aff}}(M^{(r)})
\ge
L_{\mathrm{aff}}^\star(B),
$$

where

$$
L_{\mathrm{aff}}^\star(B)
=
\min_{M:I(M)\ge B}L_{\mathrm{aff}}(M).
$$

The exact target-relative gap is

$$
\boxed{
\operatorname{Gap}_{B}(M^{(r)})
=
\frac{
L_{\mathrm{aff}}(M^{(r)})-L_{\mathrm{aff}}^\star(B)
}{
L_{\mathrm{aff}}^\star(B)
}.
}
$$

This comparison uses the Paper importance $B$, which is the constraint actually
given to the refinement. The refined mask may retain slightly more than $B$.

### Dependence on the affine latency model

The threshold

$$
g<\frac ac
$$

is specific to

$$
L_{\mathrm{aff}}(M)=aK(M)+cR(M).
$$

For a general chunk-latency table $T[\ell]$, suppose the adjacent chunks have
lengths $\ell_1$ and $\ell_2$. Filling a gap of length $g$ is beneficial only
when

$$
\boxed{
T[\ell_1+g+\ell_2]
<
T[\ell_1]+T[\ell_2].
}
$$

The implemented refinement uses the affine condition and inherits the affine
model's approximation error relative to the original lookup table and physical
flash device.
