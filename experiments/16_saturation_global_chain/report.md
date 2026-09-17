# Experiment 16 보고서: Direct lookup supported optimizer

## 질문

Experiment 15와 동일한 released lookup을 평가에만 사용하는 대신, selector도
released lookup 자체를 직접 최적화하면 결과가 개선되는가?

다음 네 방법을 Paper가 달성한 importance target에서 비교한다.

1. **Lookup supported**: released lookup 목적함수의 strongly-supported frontier
2. **Two-line supported**: two-line 목적함수의 strongly-supported frontier
3. **Quant**: two-line 목적함수에 대한 기존 `q=131,072` multiplier grid
4. **Paper greedy**: Neuron Chunking의 fixed-R greedy

모든 headline latency와 I--L/L--R 그래프는 Experiment 15와 동일한 released
lookup evaluator를 사용한다.

## Released lookup을 직접 푸는 DP

Released evaluator는 `255 KiB`까지 실제 table lookup과 선형보간을 사용하고,
그 이후에는 마지막 endpoint를 비례 확장한다.

\[
T_{\mathrm{released}}(z)=
\begin{cases}
T_{\mathrm{table}}(z),&z\le255\text{ KiB},\\
T_{\mathrm{table}}(255)\dfrac{z}{255},&z>255\text{ KiB}.
\end{cases}
\]

한 FP16 row가 `1.75 KiB`이므로

\[
m=\left\lceil\frac{255}{1.75}\right\rceil=146.
\]

상태를 현재 run 길이

\[
h\in\{0,1,\ldots,m\}
\]

로 둔다. 상태 \(m\)은 길이가 \(m\) 이상인 모든 run을 의미한다. 짧은 run을
연장할 때는 실제 lookup의 증분

\[
\Delta T(k)=T_{\mathrm{released}}(k)-T_{\mathrm{released}}(k-1)
\]

을 사용하고, \(m\) 이후에는

\[
c_{\mathrm{released}}
=\frac{T(255)}{255}\times1.75
\approx0.00033232\text{ ms/row}
\]

를 사용한다. 따라서 고정된 \(\lambda\)에서

\[
\boxed{
\max_M\{\lambda I(M)-L_{\mathrm{released}}(M)\}
}
\]

을 시간 \(O(Nm)\)에 정확히 푼다. Frontier의 score/importance/latency만 구하는
pass는 rolling memory \(O(m)\)이고, 최종 mask를 복원하는 pass는 현재 구현에서
parent table 때문에 \(O(Nm)\) memory를 쓴다. Adaptive intersection search로 이
scalarized problem의 strongly-supported frontier를 전부 열거한다.

작은 \(N\)에서 모든 mask를 열거하여 fixed-\(\lambda\) DP의 score, importance,
latency와 adaptive convex frontier가 brute force와 일치하는지 검증했다.

## 최적성 범위

Lookup supported는 다음 집합에서는 exact다.

\[
\left\{
\arg\max_M\bigl[\lambda I(M)-L_{\mathrm{released}}(M)\bigr]
:\lambda\ge0
\right\}.
\]

그러나 target \(Q\)에 대한 전체 문제

\[
\min_{I(M)\ge Q}L_{\mathrm{released}}(M)
\]

의 exact oracle은 아니다. Lookup frontier에도 unsupported Pareto point가 있을
수 있다. 각 target에서는 열거된 supported point 중 \(I\ge Q\)인 최소-latency
point를 선택한다.

## 설정

- `N=4,864`, matrix `4864x896`
- 6 CV, CV당 3 trials
- random, locally clustered, persistent hot-cold ordering
- 54 spatial inputs, 378 Paper-matched targets
- Paper budgets: `R/N=0.125, 0.25, ..., 0.875`
- Lookup cap: `m=146 rows`
- Two-line breakpoint: `s=134.86 rows`
- Quant: `q=131,072`

## 주 결과

| 방법 | Paper 대비 평균 released-latency 절감 | 중앙값 | 승률 |
| --- | ---: | ---: | ---: |
| **Lookup supported** | **8.563%** | **7.639%** | **92.59%** |
| Two-line supported | 7.328% | 6.687% | 87.83% |
| Quant (`q=131,072`) | 7.322% | 6.658% | 87.83% |

Lookup objective를 직접 사용하면 two-line supported보다 released latency를 평균
**1.247%** 추가로 줄였다. Paper 대비 평균 절감도 `7.328% -> 8.563%`로
증가하고 승률은 `87.83% -> 92.59%`로 높아졌다.

Lookup supported가 two-line supported보다 빠른 비율은 `76.72%`, 같거나 빠른
비율은 `76.98%`였다. 나머지 87개 사례에서 lookup supported가 더 느린 것은
모순이 아니다. Two-line이 생성한 mask가 lookup 공간에서는 unsupported
Pareto point일 수 있기 때문이다. Lookup **supported** optimizer는 그런 점을
선택할 수 없다. 완전한 lookup coverage oracle이라면 모든 feasible mask보다
느릴 수 없지만, 현재 방법의 최적성 범위는 supported frontier다.

## I--L 곡선

[`importance_latency.pdf`](results/importance_latency.pdf)는 네 방법을 released
lookup latency로 평가한다. 보라색 Lookup supported curve가 새로 추가됐다.
대부분의 패널에서 보라색 곡선은 초록색 Two-line supported와 Quant보다 조금
왼쪽에 있으며, 작은 차이는 평균 curve에서 잘 보이지 않을 수도 있다.

Lookup supported의 ordering별 Paper 대비 절감은 다음과 같다.

| ordering | 평균 절감 |
| --- | ---: |
| Random | 5.32% |
| Locally clustered | **13.18%** |
| Persistent hot-cold | 7.20% |

## I--R 및 L--R 곡선

- [`importance_rows.pdf`](results/importance_rows.pdf)
- [`latency_rows.pdf`](results/latency_rows.pdf)

| 구조 지표 | Lookup supported | Two-line supported | Quant | Paper |
| --- | ---: | ---: | ---: | ---: |
| 평균 rows | **2,347.62** | 2,374.47 | 2,374.65 | 2,430.00 |
| 평균 chunks | 18.30 | **16.24** | **16.24** | 25.24 |
| 평균 shortfall rows | 347.85 | 192.73 | 192.72 | 1,270.04 |
| pooled chunk 중앙값 | 135 | 143 | 143 | 84 |
| two-line saturation 이상 chunk | 58.05% | 68.16% | 68.16% | 18.46% |

Lookup supported는 two-line 방법보다 평균 26.85행을 덜 선택하지만 chunk는 약
2.06개 더 만든다. Released table의 실제 증분비용은 단순 shortfall penalty와
같지 않으므로, 더 적은 행과 조금 더 많은 run을 선택하는 것이 유리해진다.

## 청크 길이 히스토그램

[`chunk_length_histogram.pdf`](results/chunk_length_histogram.pdf)는 네 방법의
maximal run 분포를 보여준다. 다음 두 기준선을 함께 표시했다.

- two-line breakpoint: `s=134.86 rows`
- released proportional-tail cap: `m=146 rows`

Two-line supported와 Quant는 거의 완전히 겹친다. Lookup supported는 별도의
보라색 분포로 나타나며 pooled 중앙값이 `135 rows`다. Lookup 최적화가 단순히
모든 chunk를 146행으로 맞추는 것은 아니다. 146행 전에는 실제 table의 불규칙한
증분비용과 importance 위치를 함께 고려한다.

## Budget별 결과

| Paper R/N | Lookup supported 절감 | Lookup 승률 |
| ---: | ---: | ---: |
| 0.125 | 3.36% | 74.07% |
| 0.250 | 6.57% | 83.33% |
| 0.375 | 7.55% | 92.59% |
| 0.500 | 8.62% | 98.15% |
| 0.625 | 10.02% | 100.00% |
| 0.750 | 11.35% | 100.00% |
| 0.875 | 12.48% | 100.00% |

작은 coverage에서는 supported-point overshoot 때문에 Paper보다 느린 사례가
남아 있다. Lookup supported의 평균 importance overshoot는 `0.00603`으로,
Two-line supported의 `0.00731`보다 작았다.

## Tail 민감도

| mask selector | Two-line 평가 | Released 평가 | Anchored tail | Block split |
| --- | ---: | ---: | ---: | ---: |
| Lookup supported | 9.640% | **8.563%** | 8.245% | **6.323%** |
| Two-line supported | **10.034%** | 7.328% | 7.549% | 1.912% |
| Quant | 10.029% | 7.322% | 7.545% | 1.907% |

각 값은 Paper 대비 평균 latency 절감이다. 목적함수와 evaluator를 일치시키면
released 기준뿐 아니라 block-splitting sensitivity에서도 더 견고한 mask가
나왔다. 반대로 two-line evaluator에서는 당연히 Two-line supported가 더 좋다.

## 계산 비용

| solver | 평균 supported points | 평균 scalarized solves | 평균 build time |
| --- | ---: | ---: | ---: |
| Lookup supported, \(O(Nm)\) inner DP | 295.7 | 590.4 | 1,361.7 ms |
| Two-line supported, optimized \(O(N)\) inner DP | 370.8 | 740.6 | 약 44 ms |

Lookup DP는 입력당 평균 약 `1.36 s`로, two-line 특수구조를 이용한 solver보다
약 30배 느리다. 현재 구현은 offline oracle 용도이며 Paper의 online selector를
직접 대체하지 않는다.

## 제한

이 실험은 Experiment 13의 54개 입력을 사용한다. Evaluator는 Experiment 15와
같지만 Experiment 15의 900개 입력보다 표본이 작다. 평균 효과의 정밀한
신뢰구간을 얻으려면 lookup optimizer를 900개 입력으로 확장해야 한다.

또한 activation importance는 downstream accuracy의 surrogate이며, Lookup
supported도 unsupported coverage point를 놓친다.

## 결론

Released lookup의 proportional tail을 이용하면 two-line surrogate 없이도
fixed-\(\lambda\) problem을 \(O(Nm)\)에 정확히 풀 수 있다. 이 direct lookup
supported optimizer는 Paper 대비 released latency를 평균 **8.563%** 줄였고,
Two-line supported보다 평균 **1.247%** 추가 개선했다.

따라서 `two-line으로 선택, lookup으로 평가`에서 생기는 surrogate mismatch는
실제로 측정 가능한 손실을 만든다. 다만 남은 문제는 여전히 unsupported
coverage point다. 다음 exactness 개선은 multiplier 해상도가 아니라 lookup
coverage Pareto set을 직접 표현하는 방향이어야 한다.
