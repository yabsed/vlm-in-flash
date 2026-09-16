# Experiment 02: paper-calibrated CV sweep

## 질문

논문 Appendix C의 neuron-importance CV를 재현했을 때 다음 네 방법은 어떻게
달라지는가?

1. Paper greedy
2. Exact fixed-`R` DP
3. Exact Coverage DP
4. Quantized Pareto `O(qN)` (`q=256`)

## 설정

- `N=256`, CV당 독립 multiset 100개
- 목표 CV: `1.07, 1.25, 1.44, 2.48, 3.30, 4.55, 9.19`
- 앞의 여섯 값은 논문 VLM 범위이고 `9.19`는 ReLU OPT 참고값이다.
- 각 multiset은 sample CV가 목표값과 일치하도록 lognormal-shaped sample을
  보정했다. 관측된 최대 CV 오차는 `1.07e-14`였다.
- 동일한 값 multiset을 `random`, `locally clustered`, `persistent hot-cold`
  세 순서로 재배치했다. 따라서 ordering 비교에서 marginal distribution과
  CV는 변하지 않는다.
- fixed-`R` 비교는 `R/N = 12.5%, 25%, ..., 87.5%`의 7개 budget을 사용한다.
- Paper greedy가 실제 선택한 행 수를 exact fixed-`R` DP에도 동일하게 주고,
  `I/L`을 비교한다.
- Exact Coverage DP는 각 방법이 달성한 importance를 하한으로 받아 가능한
  최소 latency를 계산한다. 모든 latency overhead의 0% 기준선이다.
- Quantized Pareto도 동일한 importance 하한을 받으므로 approximation error를
  Exact Coverage와 직접 분리할 수 있다.
- affine latency는 Orin AGX 측정표의 적합식
  `L = 0.0110009 K + 0.000139813 R ms` (`R^2=0.9867`)을 사용한다.

총 2,100개 ordered input, 14,700개 fixed-`R` 비교, 14,700개 exact/quantized
coverage frontier 점을 계산했다.

공간 구조가 실제로 달라졌는지도 확인했다. 평균 lag-1 correlation은 random
`-0.008`, local `0.608`, hot-cold `0.205`였다. 앞 절반에 놓인 importance
비율은 각각 `0.497`, `0.499`, `0.876`으로, hot-cold만 persistent spatial
bias를 갖는다.

## 결과 1: Exact fixed-R이 Paper greedy보다 얼마나 좋은가

아래 값은 7개 `R`에서의 **exact fixed-`R` I/L 개선율 평균**이다.

| CV | Random | Local | Hot-cold |
|---:|---:|---:|---:|
| 1.07 | 8.40% | 14.08% | 7.29% |
| 1.25 | 8.81% | 16.26% | 7.32% |
| 1.44 | 9.15% | 17.01% | 9.22% |
| 2.48 | 13.83% | 23.08% | 11.02% |
| 3.30 | 17.93% | 26.67% | 12.00% |
| 4.55 | 26.67% | 36.97% | 15.67% |
| 9.19 (OPT) | 77.00% | 60.37% | 34.77% |

VLM 범위만 평균하면 개선율은 **15.63%**다. Ordering별로는 random
`14.13%`, local `22.34%`, hot-cold `10.42%`다. 즉 CV가 커질수록 exact
optimization의 이점이 대체로 커지며, local cluster에서 greedy의 손실이
가장 컸다. Hot-cold ordering은 이미 큰 값들을 가깝게 두므로 greedy와
optimal의 차이를 줄였다.

LLaVA-OneVision-7B의 세 CV만 ordering 전체에서 평균하면 middle `CV=1.25`
에서 `10.79%`, first `CV=1.44`에서 `11.79%`, last `CV=3.30`에서
`18.87%` 개선됐다.

![Fixed-R gain](results/cv_fixed_r_gain.png)

## 결과 2: 같은 importance에서의 latency

각 방법의 output importance를 Exact Coverage DP의 하한으로 다시 주고
`L_method/L_exact-1`을 계산했다. Paper와 fixed-`R`은 각자 달성한
importance를 사용한다. Quantized Pareto 값은 같은 두 target에서 얻은
overhead를 평균한 것이다.

| CV | Paper greedy | Exact fixed-R | Quantized Pareto |
|---:|---:|---:|---:|
| 1.07 | 9.59% | 0.11% | 0.41% |
| 1.25 | 10.19% | 0.13% | 0.45% |
| 1.44 | 11.06% | 0.11% | 0.55% |
| 2.48 | 14.16% | 0.22% | 1.42% |
| 3.30 | 16.48% | 0.54% | 2.14% |
| 4.55 | 21.76% | 1.06% | 4.49% |
| 9.19 (OPT) | 40.14% | 2.35% | 17.76% |

VLM 범위 평균은 Paper greedy **13.87%**, Exact fixed-`R` **0.36%**,
Quantized Pareto **1.58%**다. Exact fixed-`R`은 row count를 고정하는 다른
문제를 풀면서도 자신이 달성한 importance에서 coverage optimum에 매우
가까웠다. Quantized Pareto는 greedy보다 훨씬 가깝지만 고CV에서 bucket
근사 손실이 커졌다.

Ordering별 VLM 평균에서 Paper greedy overhead는 random `13.80%`, local
`18.22%`, hot-cold `9.60%`였다. Quantized Pareto overhead는 각각 `0.41%`,
`3.00%`, `1.31%`였다.

![Matched latency](results/cv_matched_latency.png)

## 결과 3: `q=256` Pareto의 한계

고정 coverage grid `0.10, ..., 0.99`에서는 Quantized Pareto의 Exact Coverage
대비 latency overhead가 VLM 범위 평균 **0.60%**, 중앙값 `0%`, 95th
percentile `2.51%`였다. 그러나 fixed-`R` 방법들이 실제 달성한 불규칙한
importance target에서는 위 표처럼 평균 `1.58%`였다. 즉 평균 frontier는
가깝지만 target과 입력에 따라 pruning 손실이 커질 수 있다.

고정 grid의 CV별 평균 gap은 `1.07: 0.28%`, `1.25: 0.31%`, `1.44: 0.38%`,
`2.48: 0.72%`, `3.30: 0.80%`, `4.55: 1.09%`, `9.19: 1.88%`였다.
따라서 Experiment 01의 low-CV half-normal에서 관찰한 `q=256`의 거의 exact한
결과는 평균적으로는 상당 부분 유지되지만, high-CV 개별 입력에 대한 강한
보장은 아니다. 다음 단계는 `q` sensitivity 또는 nonuniform/adaptive bucket
실험이다.

네 방법의 평균 frontier는 다음 그림에 함께 표시했다. Exact Coverage와
Quantized Pareto가 대부분 겹치므로 세부 approximation gap은 위 수치로 읽는
편이 정확하다. Coverage target과 fixed-`R` budget point는 서로 다른 격자를
사용하므로 선 자체보다 latency-importance envelope를 비교해야 한다.

![Importance-latency frontiers](results/importance_latency_frontiers.png)

## 결론

1. 논문 CV 범위에서도 Paper greedy의 suboptimality는 사라지지 않았고,
   exact fixed-`R` DP는 평균 `15.63%`의 I/L 이득을 보였다.
2. 동일 importance의 Exact Coverage optimum을 기준으로 Paper greedy는
   `13.87%`, Exact fixed-`R`은 `0.36%`, Quantized Pareto는 `1.58%`의 평균
   latency overhead를 보였다.
3. 공간 구조가 중요하다. 같은 CV와 같은 multiset이어도 local ordering은
   greedy의 gap을 키우고, hot-cold ordering은 이를 줄였다.
4. `q=256`은 고정 coverage grid에서 Exact Coverage 대비 평균 `0.60%`로
   가깝지만, last-layer와 OPT 수준의 개별 heavy-tail에서는 오차가 커질 수
   있다.

이는 실제 activation vector를 사용한 결과가 아니라 논문 CV를 맞춘 synthetic
experiment다. CV와 ordering을 분리한 stress test로 해석해야 하며, 최종
주장은 TempCompass activation 원자료로 재검증해야 한다.
