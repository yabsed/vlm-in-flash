# Experiment 13 보고서: saturation-aware two-line + high-q Lagrangian

## 질문

Experiment 11은 단일 affine DP-Cover가 지나치게 긴 chunk를 만들고, 그 결과
lookup table의 측정 범위 밖 외삽에 과도하게 노출된다는 사실을 보였다.

Experiment 12의 `note2.md`가 제안한 대로 비용 모델을 saturation-aware
two-line 형태로 교체하고 매우 조밀한 multiplier grid를 사용하면 다음을 달성할
수 있는지 확인한다.

1. affine DP의 과도한 merge를 줄이는가?
2. Paper보다 낮은 latency를 안정적으로 얻는가?
3. Experiment 11의 released lookup 및 tail-linear 평가에서도 개선이 유지되는가?

## 용어: 여기서 q는 importance bucket 수가 아니다

이 실험의 `q=131,072`는 이전 Quantized-Pareto의 importance bucket 수가 아니라
탐색하는 Lagrange multiplier `λ`의 수다. 각 `λ`에서 푸는 문제는

\[
\max_M\{\lambda I(M)-L_{pw}(M)\}
\]

이며, two-line 구조를 이용해 정확히 `O(N)`에 푼다. 그러나 `q`개의 해 중
coverage target을 만족하는 최소비용 해를 고르는 전체 방법은 supported
Pareto point만 볼 수 있으므로 coverage-constrained problem의 exact solver가
아니다. `q`를 아무리 키워도 unsupported point는 복구되지 않는다.

## Two-line 비용 모델

breakpoint는 논문이 제시한 AGX saturation point `236 KiB`로 고정했다. 한
행이 `1.75 KiB`이므로

\[
r_{sat}=236/1.75=134.857\text{ rows}.
\]

연속성 조건을 둔 constrained fit은 다음과 같다.

\[
T_{pw}(r)=
\begin{cases}
a+c_1r,&r\le r_{sat},\\
c_2r,&r>r_{sat},
\end{cases}
\]

\[
a=0.0110806,\quad
c_1=0.000243154,\quad
c_2=0.000325320.
\]

fit 품질은 `R²=0.98763`, `MAPE=4.11%`로, 단일 affine의 `R²=0.98672`와
비슷하다. 핵심 차이는 측정 범위 이후의 긴 chunk에 더 높은 saturated
throughput slope를 적용한다는 점이다.

mask 전체의 비용은

\[
L_{pw}(M)=aK+c_1R+(c_2-c_1)E_{sat}(M)
\]

이다.

## O(N) penalized solver

마지막 chunk가 `[i,j]`일 때 후보를 두 집합으로 나눈다.

- 길이 `≤ floor(r_sat)=134`: sliding-window maximum
- 길이 `≥135`: prefix maximum

두 maximum을 deque와 누적 maximum으로 관리하므로 각 `λ`의 내부 문제는
`O(N)` 시간과 `O(N)` backtracking memory에 정확히 풀린다. 작은 `N`에서는
모든 `2^N` mask를 열거하여 short/saturated branch 모두를 검증했다.

`λ=0`과 양의 log-spaced multiplier `131,071`개를 사용했다. 각 입력의 최소
및 최대 `λ`는 empty/full 영역을 포함하도록 importance 범위에 맞춰 정했다.

## 설정

- `N=4,864`, row size `1.75 KiB`
- 여섯 VLM CV: `1.07, 1.25, 1.44, 2.48, 3.30, 4.55`
- random, local, persistent hot-cold ordering
- CV당 세 trial
- 54개 입력
- Paper가 달성한 importance target 378개
- 공통 coverage target 378개
- `q=131,072`

Paper 및 affine DP-Cover mask는 Experiment 10의 RLE를 재사용했다. 새로운
two-line mask만 선택된 `λ`에서 backtracking했다. `O(N^3)` DP는 실행하지
않았다.

## 결과 1: Paper 대비 성능

아래 ratio는 `Paper latency / matched method latency`다. 1보다 크면 분모의
방법이 Paper보다 빠르다.

| 평가 비용 | Paper / affine DP-Cover | Paper / two-line λ-grid | λ-grid가 Paper보다 빠른 비율 |
| --- | ---: | ---: | ---: |
| Single affine | 1.300 | 1.132 | 92.1% |
| Two-line | 1.087 | **1.119** | **95.0%** |
| Released lookup | 1.053 | **1.087** | **87.8%** |
| Anchored tail-linear | 1.143 | **1.090** | **86.5%** |
| 255-KiB block split | 1.050 | 1.029 | 56.1% |

새 solver가 실제로 최적화한 two-line 비용에서는 Paper 비용이 λ-grid보다
평균 `11.94%` 컸다. λ-grid 관점의 Paper 대비 latency 절감률은 평균
`10.03%`이며, 378개 중 `95.0%`에서 이겼다. Released lookup으로 다시
평가하면 Paper/λ-grid ratio는 `1.0866`, λ-grid의 Paper 대비 절감률은
`7.32%`, 승률은 `87.8%`다. Tail-linear에서도 절감률 `7.54%`, 승률
`86.5%`로 유지된다.

따라서 Experiment 11에서 보인 `5.35%`는 Paper가 본질적으로 거의
최적이어서 생긴 결과가 아니다. 비용 모델을 saturation-aware하게 바꾸고 그
모델을 직접 최적화하면 Paper보다 의미 있게 낮은 latency를 찾을 수 있다.

## 결과 2: 기존 affine DP-Cover와의 관계

Paper importance target에서 two-line λ-grid mask를 affine DP-Cover mask와
비교하면:

| 평가 비용 | λ-grid / affine DP-Cover 평균 | λ-grid가 더 낮은 비율 |
| --- | ---: | ---: |
| Two-line | **0.972** | 77.2% |
| Released lookup | **0.971** | 76.2% |
| Tail-linear | 1.051 | 22.8% |
| Block split | 1.023 | 41.3% |

즉 native two-line과 공개 lookup에서는 새 mask가 평균 약 `2.8--2.9%`
낮다. 하지만 tail-linear는 affine DP-Cover의 매우 긴 chunk를 상대적으로 덜
벌주므로 결과가 반대다. 이는 Experiment 11의 외삽 민감도가 여전히
사라지지 않았다는 뜻이다.

공통 coverage grid에서는 λ-grid/affine-DP two-line ratio의 평균이 `1.018`,
중앙값이 `0.984`였다. λ-grid가 낮은 경우는 `62.4%`다. 낮은 coverage에서
supported point가 target을 크게 overshoot하는 사례가 있어 평균은 중앙값보다
나쁘다.

## 결과 3: 해의 구조

| 방법 | 평균 K | case별 run 중앙값의 중앙값 | 255 KiB 밖 chunk에 포함된 행 |
| --- | ---: | ---: | ---: |
| Paper | 25.24 | 84행 | 20.8% |
| Affine DP-Cover | 4.72 | 655행 | 90.3% |
| Two-line λ-grid | **16.24** | **144행** | **52.5%** |

새 방법은 Paper와 affine DP-Cover 사이의 구조를 만든다. chunk 수를 Paper보다
줄이지만, affine DP처럼 수백--수천 행을 한 chunk로 과도하게 합치지 않는다.
전형적인 run 길이 `144행`은 fitted saturation point `134.9행`과 공개 table
끝 `145.7행` 근처다. 이는 two-line penalty가 의도한 구조적 효과다.

## 결과 4: q 수렴성과 unsupported point

`q=131,072` grid의 결과를 기준으로 nested subset을 비교했다.

| q | 평균 gap | p95 | 최대 gap |
| ---: | ---: | ---: | ---: |
| 1,024 | 1.613% | 9.306% | 47.714% |
| 4,096 | 0.419% | 1.271% | 20.129% |
| 16,384 | 0.052% | 약 0% | 7.047% |
| 65,536 | 0.020% | 약 0% | 7.047% |
| 131,072 | 기준 | 기준 | 기준 |

평균적으로는 `q=16,384`부터 거의 수렴하지만, 매우 좁은 multiplier 구간에서만
나오는 mask 때문에 worst case에는 `q=131,072`가 실제 차이를 만들었다.

그러나 최대 `q`에서도 two-line 기준으로 Paper가 더 빠른 사례가 `5.0%`
남는다. 이 사례는 주로 낮은 row budget에서 발생했고, λ-grid mask가 target을
상당히 overshoot했다. 이는 grid 해상도가 아니라 unsupported Pareto point
문제다. 실제 평균 importance overshoot도 `0.00733`으로 affine DP-Cover의
`0.000116`보다 크다.

## 계산 비용

- `q=131,072` grid build: 입력당 평균 `751 ms`, 중앙값 `741 ms`
- Experiment 07 Paper reference: query당 평균 `7.66 ms`
- 입력당 실제 distinct supported mask: 평균 `368개`

grid build는 16-thread Numba CPU 실행이고 Paper 시간은 저장된 single-query
CPU 측정이므로 완전히 동일한 timing scope는 아니다. 그래도 현재 구현은
Paper보다 온라인에서 훨씬 무겁다. 또한 131,072개의 multiplier 중 실제 서로
다른 supported mask는 평균 368개뿐이어서 grid 중복이 매우 크다. 향후에는
breakpoint를 직접 추적하는 parametric search가 더 적합하다.

## Released lookup 기준의 3-method 그래프

결과 그림은 비용 기준을 혼동하지 않도록 두 디렉터리로 분리했다.

- [`results/two_line/`](results/two_line/): surrogate 검증용
  `q_convergence`와 `run_distributions`만 유지
- [`results/lookup/`](results/lookup/): released lookup으로 재평가한
  `Quant (high-q) / Paper greedy / Top-R` 비교

lookup 디렉터리에는 다음 다섯 그래프가 있다.

- [`importance_latency_frontiers.pdf`](results/lookup/importance_latency_frontiers.pdf):
  lookup latency--importance frontier
- [`r_importance.pdf`](results/lookup/r_importance.pdf): 선택 행 수--importance
- [`r_latency.pdf`](results/lookup/r_latency.pdf): 선택 행 수--lookup latency
- [`coverage_opt_ratio.pdf`](results/lookup/coverage_opt_ratio.pdf): 동일한 Paper
  importance에서 high-q Quant를 1로 둔 latency ratio
- [`chunk_count.pdf`](results/lookup/chunk_count.pdf): 동일한 Paper importance에서
  세 방법의 chunk 수 분포

I--L, R--I, R--L의 Top-R은 fixed-R 결과다. 반면 공정한 target-matched 비교가
필요한 ratio와 chunk histogram에서는 Paper importance를 넘는 최소 Top-R
prefix를 사용했다.

released lookup 기준으로 Paper/Quant latency ratio는 평균 `1.0866`, 중앙값
`1.0713`이다. 즉 앞에서 보고한 평균 `7.32%` 절감과 같은 결과를 새 I--L
곡선에서도 직접 확인할 수 있다. 동일 importance의 Top-R/Quant ratio는 평균
`6.99`로 매우 크다. Paper-matched chunk 수의 평균은 Quant `16.24`, Paper
`25.24`, Top-R `475.92`다.

단, 여기서 `coverage_opt_ratio`의 Quant 분모는 exact lookup-aware optimum이
아니다. two-line 비용으로 선택한 `q=131,072` supported-point 해를 released
lookup으로 재평가한 reference다. 따라서 파일명은 기존 실험 계열과 맞췄지만,
그래프 자체도 이를 “high-q Quant reference”라고 명시한다.

## 결론

two-line 비용 모델은 affine DP가 보인 과도한 장거리 merge를 실제로
교정했다. 결과 mask의 전형적인 chunk 길이는 saturation point 근처로 이동했고,
Paper보다 native two-line 비용을 평균 `10.03%`, released lookup 비용을 평균
`7.32%` 줄였다.

하지만 high-q Lagrangian은 최종 해답은 아니다.

1. coverage optimum의 unsupported point를 놓친다.
2. `q=131,072`는 Paper보다 약 두 자릿수 이상 느리다.
3. 255 KiB 이후 실제 측정이 없으므로 lookup 평가의 외삽 민감도가 남아 있다.

따라서 다음 알고리즘 목표는 명확하다. Two-line 구조를 유지하면서 coverage를
직접 보존하여 unsupported point를 표현하고, dense λ-grid의 중복 계산을 피하는
solver가 필요하다.
