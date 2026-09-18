# Algorithm

For method `A` and retained-row ratio `R`, measure the pair

```text
(T_A(R), E_A(R))
```

where `T` includes selector, native O_DIRECT/upload, activation gather, and
compact GEMM, and `E` is checkpoint-weight projection relative L2. The
quality-constrained value function is

```text
V_A(epsilon) = min_R T_A(R)  subject to E_A(R) <= epsilon.
```

This partial minimization eliminates `R` from the final comparison. `R` remains
only as a small annotation on the measured curve.

The primary envelope uses only measured points. Because the sweep is spaced by
5 percentage points, adaptive points are also checked against piecewise-linear
interpolation of the Cell-8 curve. This second check is diagnostic rather than
a measured result: it detects wins caused only by the coarse fixed-R grid.

The adaptive strategy sorts activation importance once, adds a structured
margin fitted only on calibration prompts, selects a Cell-8 mask, and repairs
upward only if the requested retained-importance threshold is missed. Its
threshold is also chosen only from calibration projection error. Holdout error
is never an online policy input.

The same dominance test is repeated with dense-to-sparse logit KL from real
end-to-end sparse forwards, so the strategy conclusion does not depend only on
the local projection relative-L2 proxy.
