결론부터 말하면, **단일 affine 모델 \(T(r)=a+c_1r\)을 모든 chunk 길이에 적용하는 주장은 버려야 합니다.** 하지만 연구의 핵심인 coverage formulation과 빠른 chain optimization은 무너지지 않습니다. latency를 **포화점을 경계로 한 piecewise-affine 모델**로 고치면 됩니다. 오히려 실험 결과를 더 잘 설명합니다.

## 1. 두 기울기의 비교

AGX에서 기존 전체-range affine fit은

$$
a=0.0110009\text{ ms},\qquad
c_1=0.000244672\text{ ms/row}
$$

입니다. 공개 구현의 255 KiB 이후 외삽 기울기는

$$
c_2^{255}=0.0003323\text{ ms/row}
$$

이므로

$$
\frac{c_2^{255}}{c_1}=1.358,
$$

즉 포화 이후 기울기가 약 \(35.8\%\) 큽니다. 

그런데 물리적으로 더 자연스러운 값은 논문이 포화점으로 제시한 \(236\) KiB에서 계산한 기울기입니다. 한 행이 \(1.75\) KiB이므로

$$
r_{\mathrm{sat}}
=
\frac{236}{1.75}
\approx134.86\text{ rows}.
$$

그 지점의 측정 latency와 throughput으로부터

$$
c_2^{\mathrm{sat}}
=
\frac{T(236)}
{236/1.75}
\approx0.000327317\text{ ms/row}
$$

을 얻습니다. 논문 정리문에도 AGX의 포화점은 \(236\) KiB, 측정 throughput은 약 \(5221\) MiB/s로 기록되어 있습니다. 

따라서

$$
\frac{c_2^{\mathrm{sat}}}{c_1}
\approx1.338.
$$

즉, 포화 이후 행당 latency는 기존 affine 기울기보다 약 \(33.8\%\) 큽니다.

더 흥미로운 점은 두 직선의 교점입니다.

$$
r_\times
=
\frac{a}{c_2^{\mathrm{sat}}-c_1}
\approx133.1\text{ rows}
\approx232.9\text{ KiB}.
$$

이는 논문의 포화점 \(236\) KiB와 불과 약 \(1.3\%\) 차이입니다. 우연이라고 보기에는 상당히 잘 맞습니다.

즉, 기존 affine 식

$$
a+c_1r
$$

은 전 범위 모델이라기보다 **포화 전 branch**였고, 포화 후에는

$$
c_2r
$$

로 바뀐다고 해석하는 것이 자연스럽습니다.

## 2. 수정된 latency 모델

가장 깔끔한 모델은 다음과 같습니다.

$$
\boxed{
T_{\mathrm{pw}}(r)
=
\begin{cases}
a+c_1r, & r\le r_{\mathrm{sat}},\\[2mm]
c_2r, & r>r_{\mathrm{sat}}.
\end{cases}
}
$$

연속성을 요구하면

$$
a+c_1r_{\mathrm{sat}}
=
c_2r_{\mathrm{sat}},
$$

따라서

$$
a=(c_2-c_1)r_{\mathrm{sat}}.
$$

동일한 모델을 hinge 형태로 쓰면

$$
\boxed{
T_{\mathrm{pw}}(r)
=
a+c_1r
+
(c_2-c_1)(r-r_{\mathrm{sat}})_+,
}
$$

여기서

$$
(x)_+=\max(x,0).
$$

실제로 breakpoint를 \(236\) KiB로 고정하고 이 물리적 제약을 둬 다시 fitting하면 대략

$$
a=0.01108,\qquad
c_1=0.0002432,\qquad
c_2=0.0003253
$$

이 나옵니다.

| 모델                        |    \(R^2\) |       MAPE |
| ------------------------- | ---------: | ---------: |
| 단일 affine                 | \(0.9867\) | \(4.22\%\) |
| saturation-aware two-line | \(0.9876\) | \(4.11\%\) |

측정 범위 안에서의 개선은 작습니다. 중요한 차이는 **긴 chunk로 외삽했을 때 물리적으로 올바른 형태를 갖는다**는 점입니다.

## 3. 마스크 latency는 세 가지 통계에 의해 결정됨

다음을 정의합시다.

$$
E_{\mathrm{sat}}(M)
=
\sum_{C\in\mathcal C(M)}
\left(|C|-r_{\mathrm{sat}}\right)_+.
$$

이는 각 chunk에서 포화 길이를 초과한 행의 수를 합한 것입니다. 그러면

$$
\begin{aligned}
\widehat L_{\mathrm{pw}}(M)
&=
\sum_{C\in\mathcal C(M)}T_{\mathrm{pw}}(|C|)\\
&=
\boxed{
aK(M)+c_1R(M)
+(c_2-c_1)E_{\mathrm{sat}}(M)
}.
\end{aligned}
$$

따라서 latency를 결정하는 것은 이제

* 선택된 행 수 \(R(M)\)
* chunk 수 \(K(M)\)
* 포화 길이를 초과한 행 수 \(E_{\mathrm{sat}}(M)\)

입니다.

기존 모델

$$
aK(M)+c_1R(M)
$$

은 모든 chunk가 포화점 이하이거나 \(c_1=c_2\)인 특수한 경우입니다.

특히 \(R(M)=R\)이 고정되어도

$$
\widehat L_{\mathrm{pw}}(M)
=
c_1R+aK(M)+(c_2-c_1)E_{\mathrm{sat}}(M)
$$

이므로, 더 이상

$$
\min L(M)\iff\min K(M)
$$

라고 할 수 없습니다. 정확한 목표는

$$
\boxed{
\min_M
\left[
aK(M)+(c_2-c_1)E_{\mathrm{sat}}(M)
\right]
}
$$

입니다.

즉, chunk 수를 줄이는 것은 좋지만, 그 과정에서 지나치게 긴 chunk를 만들면 추가 비용을 지불합니다.

## 4. 왜 affine DP가 지나치게 긴 chunk를 골랐는가

단일 affine 모델에서는 두 chunk를 합치면 크기와 무관하게 항상 고정비 \(a\) 하나를 절약합니다. 따라서 거대한 chunk를 만드는 방향으로 강하게 유도됩니다.

반면 두 chunk가 이미 포화점보다 길다면

$$
T(r_1)+T(r_2)
=
c_2(r_1+r_2).
$$

둘 사이의 \(g\)개 행까지 읽어 하나로 합치면

$$
T(r_1+g+r_2)
=
c_2(r_1+r_2+g).
$$

따라서 \(g>0\)이면 합치는 것이 오히려 더 비쌉니다. 포화 이후에는 이미 최대 throughput이므로, chunk를 더 길게 만들어도 추가적인 고정비 절감 효과가 없는 것입니다.

실제로 Paper Greedy의 중앙 chunk 길이는 \(84\)행이고 측정 범위 밖에 속한 행은 \(20.8\%\)였지만, affine DP-Cover는 중앙값 \(206.5\)행이며 행의 \(90.3\%\)가 측정 범위보다 긴 chunk에 들어갔습니다. 

따라서 affine DP가 잘못된 방향으로 과도하게 merge한 이유가 정확히 설명됩니다.

## 5. 그래도 \(O(qN)\) 휴리스틱은 유지할 수 있음

좋은 소식은 두 직선이라는 구조 때문에 Lagrangian 문제

$$
\max_M
\left\{
\lambda I(M)-\widehat L_{\mathrm{pw}}(M)
\right\}
$$

를 여전히 각 \(\lambda\)마다 \(O(N)\)에 풀 수 있다는 것입니다.

마지막 chunk가 \([i,j]\)라고 하면 prefix DP는

$$
D_j
=
\max\left\{
D_{j-1},
\max_{1\le i\le j}
\left[
D_{i-2}
+\lambda(P_j-P_{i-1})
-T_{\mathrm{pw}}(j-i+1)
\right]
\right\}.
$$

후보 시작점 \(i\)를

* 길이가 \(r_{\mathrm{sat}}\) 이하인 short run
* 길이가 \(r_{\mathrm{sat}}\)보다 긴 saturated run

으로 나누면, 각 집합에서 목적식이 \(i\)에 대한 affine 식이 됩니다. short 후보는 sliding-window maximum, long 후보는 prefix maximum으로 관리할 수 있으므로 각 위치를 amortized \(O(1)\)에 처리할 수 있습니다.

따라서 \(q\)개의 multiplier를 시험하면

$$
\boxed{O(qN)}
$$

이 그대로 가능합니다. 다만 이것은 각 \(\lambda\)의 penalized problem에는 exact이지만, unsupported Pareto point 때문에 원래 coverage-constrained problem 전체에 대해서는 여전히 휴리스틱입니다.

## 최종 판단

우리 연구의 주장은 다음처럼 수정하는 것이 가장 강합니다.

> 단일 affine 모델은 측정 범위 내에서 유용한 1차 근사이지만, 포화 이후까지 그대로 외삽해서는 안 된다. 실제 구조는 pre-saturation affine branch와 bandwidth-saturated linear branch가 결합된 piecewise-affine 형태에 가깝다. 이 수정은 coverage optimization을 무너뜨리지 않으며, 오히려 affine DP가 긴 chunk를 과도하게 선호한 이유를 설명한다. 또한 두 개의 선형 branch라는 구조를 이용하면 \(O(qN)\) Lagrangian solver도 유지할 수 있다.

즉, **기존 \(aK+cR\) 모델 자체는 축소되어야 하지만, 연구 전체는 충분히 살아남습니다. 오히려 “포화점을 무시한 affine 모델의 실패를 발견하고, saturation-aware 모델로 일반화했다”는 쪽이 더 설득력 있는 연구 서사가 됩니다.**
