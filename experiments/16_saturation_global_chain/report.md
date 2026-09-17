# Experiment 16 보고서: saturation-aware global chain versus Paper greedy

## 질문

constant-throughput two-line 모델을

\[
L(M)=c_2R(M)+\delta\sum_{C\in\mathcal C(M)}(s-|C|)_+
\]

로 정확히 다시 쓰면, 이 비용을 전역적으로 다루는 binary-chain 해는 Neuron
Chunking의 Paper greedy보다 어떤 importance--latency--row 구조를 만드는가?
그리고 그 차이는 saturation 이후 tail 외삽을 바꿔도 유지되는가?

## 비교 방법

Experiment 13의 동일한 54개 spatial input과 378개 Paper-matched target을
재사용했다.

- `N=4,864`
- CV: `1.07, 1.25, 1.44, 2.48, 3.30, 4.55`
- CV당 3 trials
- random, locally clustered, persistent hot-cold ordering
- Paper fixed-R fraction: `0.125, 0.25, ..., 0.875`
- two-line breakpoint: `s=134.857` rows
- dense multiplier grid: `q=131,072`

Paper가 각 fixed-R 지점에서 얻은 importance를 coverage lower bound로 사용했다.
Global-chain은 각 \(\lambda\)에 대해

\[
\max_M\{\lambda I(M)-L_{\mathrm{2line}}(M)\}
\]

를 전역 최적으로 푼 mask 중 target을 만족하는 최소-cost supported point다.

여기서 최적성의 범위를 구분해야 한다. 각 fixed-\(\lambda\) mask는 scalarized
objective의 전역해이지만, 유한한 \(\lambda\)-grid가 고른 mask는 coverage 문제
전체의 exact oracle이 아니다. Unsupported Pareto point를 놓칠 수 있으므로
그래프에는 `Global-chain DP (supported)`라고 표기했다.

모든 저장 mask에 대해 run length를 다시 계산하여

\[
c_2R+\delta\sum_C(s-|C|)_+
\]

가 저장된 two-line latency와 일치함을 검증했다.
또한 명시적인 \(O(Ns)\) capped-run recurrence를 작은 무작위 입력에서
Experiment 13의 최적화된 exact \(O(N)\) interval solver와 대조하여 score,
importance, latency가 일치하는지 확인했다.

## 주 결과

Global-chain은 Paper와 같거나 더 큰 importance를 보존하면서 two-line latency를
평균 **10.03%** 줄였고, 378개 사례 중 **94.97%**에서 더 낮은 latency를 얻었다.

| evaluator | 평균 latency 절감 | 중앙값 절감 | Global-chain 승률 |
| --- | ---: | ---: | ---: |
| Two-line constant-throughput | **10.03%** | 9.30% | **94.97%** |
| Released lookup + proportional tail | **7.32%** | 6.66% | 87.83% |
| Anchored tail-linear | **7.54%** | 7.13% | 86.51% |
| 255-KiB block splitting | **1.91%** | 1.84% | 56.08% |

따라서 global-chain의 구조적 이점은 released와 anchored-tail 평가에서도
유지되지만, 효과 크기는 tail 가정에 민감하다. 특히 block splitting에서는
평균 이점이 1.91%까지 줄어든다. 이는 \(T(r)=c_2r\)가 saturation에서 필연적으로
나오는 법칙이 아니라는 이론적 주의를 실험적으로 뒷받침한다.

Global-chain의 평균 importance overshoot는 `0.00733`이었다. 작은 budget의
일부 사례에서 Global-chain이 Paper보다 느린 주된 이유도 exact coverage point가
아니라 다음 supported point까지 넘어가기 때문이다.

## I--L 곡선

[`importance_latency.pdf`](results/importance_latency.pdf)는 representative CV와
세 ordering에서 importance와 two-line latency를 직접 비교한다. 대부분의
영역에서 Global-chain 곡선이 Paper보다 왼쪽 위에 있다. 즉 같거나 더 높은
importance를 더 낮은 latency로 보존한다.

효과는 importance가 공간적으로 모이는 locally clustered 입력에서 가장 크다.

| ordering | 평균 two-line latency 절감 |
| --- | ---: |
| Random | 7.05% |
| Locally clustered | **14.36%** |
| Persistent hot-cold | 8.68% |

CV가 커질수록 평균 절감도 증가했다. CV `1.07`에서는 5.51%, CV `4.55`에서는
16.99%였다. 중요한 행이 불균일하고 공간적으로 모일수록 gap inclusion과 run
extension을 전역적으로 판단할 여지가 커진다.

## I--R 곡선

[`importance_rows.pdf`](results/importance_rows.pdf)는 latency 개선이 단순히 더
적은 행을 읽어서만 생기는 것이 아님을 보여준다.

- Paper: 평균 `2,430.00` rows, importance `0.72775`
- Global-chain: 평균 `2,374.65` rows, importance `0.73508`

전체 평균에서는 Global-chain이 약 55행 적게 읽으면서 importance는 `0.00733`
더 높았다. 그러나 `R/N=0.125--0.50`에서는 supported-point overshoot 때문에
Global-chain이 Paper보다 행을 조금 더 읽는 경우가 있다. 그럼에도 run 구조를
개선하여 latency를 줄일 수 있다. 반대로 큰 budget에서는 Paper보다 행 수도
적다.

## L--R 곡선

[`latency_rows.pdf`](results/latency_rows.pdf)는 같은 행 수 부근에서도
Global-chain의 two-line latency가 더 낮음을 보여준다. 차이는 short-run deficit에서
나온다.

| 구조 지표 | Global-chain | Paper greedy |
| --- | ---: | ---: |
| 평균 chunk 수 | **16.24** | 25.24 |
| solution당 평균 shortfall rows | **192.72** | 1,270.04 |
| pooled chunk-length 중앙값 | **143** | 84 |
| saturation 이상 chunk 비율 | **68.16%** | 18.46% |

Global-chain은 평균 chunk 수를 9.0개 줄이고, 짧은 run의 총 부족분을 약
85% 줄였다. 이 때문에 선택 행 수가 비슷하거나 조금 많아도

\[
\delta\sum_C(s-|C|)_+
\]

항이 크게 감소한다.

## 청크 길이 히스토그램

[`chunk_length_histogram.pdf`](results/chunk_length_histogram.pdf)는 mask를 구성하는
진짜 maximal run을 모은 chunk-weighted histogram이다. 수직선은
`s=134.857` rows를 나타낸다.

Paper의 분포는 saturation 아래에 넓게 퍼져 있다. 반면 Global-chain은 세
ordering 모두에서 \(s\) 부근과 그 위에 강하게 집중된다. 특히 random 및
persistent hot-cold 입력에서 약 135행 부근의 peak가 뚜렷하다. 이는 알고리즘이
고정 길이 135 후보를 기계적으로 고른다는 뜻이 아니라, shortfall penalty가
사라지는 경계까지 run을 연장하거나 인접 중요 영역을 연결하는 것이 자주
최적이라는 뜻이다. Locally clustered 입력에서는 이미 긴 중요 구간이 존재하여
분포가 더 넓게 saturation 위로 이어진다.

## Budget에 따른 차이

| Paper R/N | 평균 two-line 절감 | Global-chain 승률 |
| ---: | ---: | ---: |
| 0.125 | 3.64% | 74.07% |
| 0.250 | 7.86% | 90.74% |
| 0.375 | 9.32% | 100.00% |
| 0.500 | 10.20% | 100.00% |
| 0.625 | 11.89% | 100.00% |
| 0.750 | 13.19% | 100.00% |
| 0.875 | 14.10% | 100.00% |

낮은 budget에서는 supported-point overshoot가 상대적으로 크다. `R/N>=0.375`인
이 표본에서는 Global-chain이 모든 사례에서 Paper보다 낮은 two-line latency를
얻었다.

## 제한

이 실험은 새로운 모델 실행 없이 Experiment 13의 저장 mask를 재평가한
구조 분석이다. CV당 trial이 3개뿐이므로 Experiment 14처럼 정밀한
cluster-bootstrap 추정으로 해석해서는 안 된다. 더 큰 표본에서의 released
lookup 결과는 Experiment 14의 `6.64% [6.16%, 7.12%]`가 더 신뢰할 만하다.

또한 activation importance는 downstream accuracy의 surrogate다. 본 결과는
mask-selection objective에 관한 것이며 VLM task accuracy의 직접 측정이 아니다.

## 결론

constant-throughput two-line 모델에서 핵심 비용은 chunk 개수 자체가 아니라

\[
\boxed{c_2R+\delta\cdot\text{short-run deficit}}
\]

이다. Global-chain은 이 구조를 직접 최적화하여 Paper보다 행을 조금 덜
선택하면서도 더 높은 importance를 보존하고, 무엇보다 saturated run 비율을
`18.46% -> 68.16%`로 높였다. 그 결과 two-line latency는 평균 10.03% 감소했다.

다만 coverage 전체의 exact 최적성은 아직 인증되지 않았고, tail 모델을
block-splitting으로 바꾸면 평균 이점은 1.91%로 축소된다. 논문의 정확한 주장은
“scalarized binary-chain objective의 전역해”와 “coverage에 대한 supported-point
heuristic”을 분리하고, tail sensitivity를 함께 보고하는 것이어야 한다.
