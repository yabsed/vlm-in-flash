# Experiment 16 보고서: Global supported versus Quant and Paper greedy

## 수정된 질문

Experiment 15와 동일한 latency evaluator에서 다음 세 방법을 비교한다.

1. **Global supported**: adaptive \(\lambda\) 탐색으로 scalarized two-line
   objective의 strongly-supported frontier 전체를 열거
2. **Quant**: 기존 `q=131,072` 고정 \(\lambda\)-grid
3. **Paper greedy**: Neuron Chunking의 fixed-R greedy

Global supported와 Quant는 모두 각 multiplier에서

\[
\max_M\{\lambda I(M)-L_{\mathrm{2line}}(M)\}
\]

을 전역 최적으로 푼다. 차이는 Global supported가 인접 frontier point의
교차 multiplier를 재귀적으로 풀어 supported point를 빠짐없이 찾는 반면,
Quant는 미리 정한 유한 grid만 계산한다는 점이다.

두 방법 모두 weighted-sum scalarization이므로 unsupported coverage point까지
포함하는 exact constrained-coverage oracle은 아니다.

## Experiment 15와 동일한 latency 기준

주 평가 기준을 Experiment 15와 똑같이 변경했다.

\[
T_{\mathrm{released}}(z)=
\begin{cases}
T_{\mathrm{table}}(z),&z\le255\text{ KiB},\\
T_{\mathrm{table}}(255)\dfrac{z}{255},&z>255\text{ KiB}.
\end{cases}
\]

FP16 row가 `1.75 KiB`이므로 마지막 측정점은 `145.71 rows`다. 그 이후에는

\[
T_{\mathrm{released}}(r)=c_{\mathrm{released}}r,
\qquad
c_{\mathrm{released}}\approx0.00033232\text{ ms/row}
\]

로 외삽된다.

Two-line 모델은 selector를 만드는 optimization surrogate로만 사용했다. 모든
I--L 및 L--R 주 그래프와 headline 수치는 released lookup latency다.

## 설정

- `N=4,864`, matrix shape `4864x896`
- 6 CV: `1.07, 1.25, 1.44, 2.48, 3.30, 4.55`
- CV당 3 trials
- random, locally clustered, persistent hot-cold ordering
- 54 spatial inputs, 378 Paper-matched targets
- Paper budget: `R/N=0.125, 0.25, ..., 0.875`
- Quant: `q=131,072`
- Global supported: 입력당 평균 370.8 supported points

각 coverage bound는 해당 지점에서 Paper greedy가 실제로 얻은 importance다.

## 주 결과

| 방법 | Paper 대비 평균 released-latency 절감 | 중앙값 | 승률 |
| --- | ---: | ---: | ---: |
| Global supported | **7.3276%** | 6.6873% | 87.83% |
| Quant (`q=131,072`) | **7.3224%** | 6.6579% | 87.83% |

Global supported와 Quant의 결과는 사실상 같다.

- 동일 mask 비율: **99.47%** (`376/378`)
- Global의 Quant 대비 평균 released-latency 절감: **0.00565%**
- 중앙값 및 95 percentile 차이: `0%`
- Global의 최대 추가 절감: `1.0885%`

즉 `q=131,072` Quant는 이 표본에서 exact supported frontier를 거의 완전히
재현했다. 새로운 adaptive enumeration은 Quant가 이미 수렴했다는 것을
확인하는 oracle 역할을 하지만, 평균 latency를 실질적으로 더 낮추지는 않는다.

두 방법이 달랐던 것은 두 사례뿐이다.

| CV | ordering | Paper R/N | Global의 Quant 대비 lookup 절감 |
| ---: | --- | ---: | ---: |
| 1.25 | Local | 0.75 | 1.0885% |
| 2.48 | Persistent hot-cold | 0.50 | 1.0471% |

두 사례 모두 Quant grid가 coverage target 직후의 더 작은 supported point를
건너뛰고 다음 point를 선택한 경우다.

## I--L 곡선

[`importance_latency.pdf`](results/importance_latency.pdf)는 released lookup
latency를 x축으로 사용한다. Global supported와 Quant는 거의 완전히 겹치며,
대부분의 영역에서 Paper보다 왼쪽 위에 있다.

Global supported의 ordering별 평균 절감은 다음과 같다.

| ordering | Paper 대비 released-latency 절감 |
| --- | ---: |
| Random | 3.55% |
| Locally clustered | **11.89%** |
| Persistent hot-cold | 6.55% |

importance가 공간적으로 모인 local ordering에서 gap merge와 saturated-run
구조의 효과가 가장 크다.

## I--R 및 L--R 곡선

- [`importance_rows.pdf`](results/importance_rows.pdf)
- [`latency_rows.pdf`](results/latency_rows.pdf)

| 구조 지표 | Global supported | Quant | Paper greedy |
| --- | ---: | ---: | ---: |
| 평균 선택 행 | 2,374.47 | 2,374.65 | 2,430.00 |
| 평균 chunk 수 | 16.24 | 16.24 | 25.24 |
| 평균 shortfall rows | 192.73 | 192.72 | 1,270.04 |
| pooled chunk 중앙값 | 143 | 143 | 84 |
| saturation 이상 chunk | 68.16% | 68.16% | 18.46% |

Global과 Quant가 겹치는 것은 두 방법이 같은 scalarized objective를 풀기
때문이다. Paper와의 차이는 선택 행 수보다 run 구조에서 더 크게 나타난다.

## 청크 길이 히스토그램

[`chunk_length_histogram.pdf`](results/chunk_length_histogram.pdf)는 세 방법의
진짜 maximal run 분포다. Global supported와 Quant histogram은 겹친다.

Paper는 two-line saturation인 `s=134.86 rows` 아래에 많은 짧은 run을 만들지만,
두 전역 scalarized 방법은 saturation 부근과 그 위에 run을 집중시킨다. 이는
독립 chunk utility ranking보다 전체 mask의 row cost와 shortfall을 함께 보는
효과다.

## Tail 민감도

| evaluator | Global 절감 | Quant 절감 |
| --- | ---: | ---: |
| Two-line surrogate | 10.034% | 10.029% |
| **Released lookup (primary)** | **7.328%** | **7.322%** |
| Anchored tail-linear | 7.549% | 7.545% |
| 255-KiB block splitting | 1.912% | 1.907% |

Experiment 15와 직접 비교할 값은 굵게 표시한 released lookup 결과다. Two-line
`10.03%`는 이제 주 결과가 아니라 surrogate sensitivity다.

## 계산 비용

Global supported는 입력당 평균 `370.8`개의 point를 찾기 위해 평균 `740.6`번의
exact scalarized solve를 수행했다.

- 평균 build time: `44.61 ms/input`
- 중앙값: `34.36 ms/input`
- 최대: `76.71 ms/input`

이는 현재 Python/Numba 구현의 offline oracle timing이다. Quant의 기존 dense
grid build는 Experiment 13에서 입력당 약 `756 ms`였으므로 adaptive enumeration이
더 적은 multiplier를 계산한다. 다만 두 timing은 실행 환경과 병렬화 방식이
다르므로 엄밀한 kernel benchmark로 해석해서는 안 된다.

## 제한

이 실험은 Experiment 13의 54개 입력을 사용한다. Experiment 15/14의 900개
입력 결과와 evaluator는 같지만 표본 크기와 budget grid가 다르다. 따라서
Paper 대비 효과 크기의 정밀 추정에는 Experiment 14의
`6.64% [6.16%, 7.12%]`가 더 적합하다.

또한 Global supported는 이름 그대로 supported frontier에 대해서만 완전하다.
정확한 coverage optimum이 unsupported Pareto point이면 Global과 Quant 모두
놓칠 수 있다.

## 결론

Experiment 16의 주 latency 기준을 Experiment 15와 동일한 released lookup으로
통일했다. 이 기준에서 Global supported는 Paper보다 평균 `7.3276%`, Quant는
`7.3224%` 빠르다.

가장 중요한 결과는 Global이 Quant를 크게 개선했다는 것이 아니라,
`q=131,072` Quant가 exact supported frontier와 `99.47%` 동일한 mask를 선택했다는
것이다. 남은 최적성 차이는 더 촘촘한 \(\lambda\)-grid가 아니라 unsupported
coverage point를 직접 표현하는 방법에서 찾아야 한다.
