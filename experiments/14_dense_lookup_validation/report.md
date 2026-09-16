# Experiment 14 보고서: dense released-lookup validation

## 질문

Experiment 13의 “Quant가 Paper보다 released lookup latency를 평균 7.32% 줄인다”는
결과는 CV당 trial이 세 개뿐이었다. 독립 입력과 R 지점을 크게 늘렸을 때 효과의
크기와 방향이 유지되는지, 그리고 신뢰구간이 얼마나 좁아지는지 확인한다.

## 설정

- `N=4,864`
- CV: `1.07, 1.25, 1.44, 2.48, 3.30, 4.55`
- 50 trials per CV
- three spatial orderings
- 19 budgets, `R/N=0.05,0.10,...,0.95`
- high-q Quant: `q=131,072` lambda multiplier
- Paper가 얻은 importance 이상을 만족하는 Quant supported point 선택
- released lookup latency: 255 KiB 이후에는 saturation 뒤 throughput이
  일정하다는 가정으로 마지막 table endpoint에 비례해 선형 확장
- bootstrap replicate: 20,000
- 독립 resampling 단위: `(trial, CV)` cluster

총 300개의 독립 `(trial, CV)` cluster, 900개 spatial input, 17,100개 paired
budget case를 측정했다. 동일 base importance multiset에서 나온 세 ordering과
19개 budget은 서로 독립인 것처럼 세지 않고 같은 cluster 안에 유지했다.

여기서 `q`는 importance bucket 수가 아니라 Lagrange multiplier 수다. Quant는
lookup table 자체가 아니라 two-line surrogate를 최적화하고, 선택된 mask를
released lookup으로 다시 평가한다.

## 주 결과

Paper와 동일 이상의 importance를 달성하는 조건에서:

| 지표 | 결과 |
| --- | ---: |
| Quant의 평균 lookup latency 절감 | **6.641%** |
| cluster-bootstrap 95% CI | **[6.160%, 7.124%]** |
| 중앙값 절감 | 6.574% |
| Quant 승률 | **85.164%** |
| 승률 95% CI | [84.123%, 86.170%] |
| Paper의 Quant 대비 평균 추가 latency | 7.950% |

Experiment 13의 `7.32% [5.27%, 9.58%]`와 비교하면 중심값은 다소 낮아졌지만
방향은 유지됐다. 특히 평균 절감의 95% CI 폭은 `4.31%p`에서 `0.96%p`로
약 4.5배 좁아졌다. 현재 합성 benchmark와 tail 가정 아래에서는 “약 6.6%,
95% CI 6.2--7.1%”가 더 믿을 만한 요약이다.

전체 평균은 여섯 CV, 세 ordering, `0.05--0.95`의 19개 R budget을 동일하게
가중한 macro average다. 실제 workload의 CV 또는 R 분포가 다르면 그 분포로
다시 가중해야 한다.

## R budget에 따른 차이

효과는 R에 매우 강하게 의존한다.

| R/N | 평균 절감 | 95% CI | Quant 승률 |
| ---: | ---: | ---: | ---: |
| 0.05 | **-6.601%** | [-7.561%, -5.640%] | 37.3% |
| 0.10 | **-0.960%** | [-1.560%, -0.360%] | 50.8% |
| 0.15 | 1.342% | [0.806%, 1.869%] | 60.1% |
| 0.25 | 4.111% | [3.639%, 4.579%] | 76.1% |
| 0.50 | 7.574% | [7.114%, 8.035%] | 91.9% |
| 0.75 | 10.342% | [9.803%, 10.873%] | 99.4% |
| 0.95 | 12.881% | [12.242%, 13.537%] | 100.0% |

따라서 Quant가 모든 영역에서 Paper보다 좋은 것은 아니다. `R/N=0.05`와
`0.10`에서는 평균적으로 Paper가 더 빠르다. 낮은 coverage에서는 supported
point 사이가 넓어 Quant가 target importance를 크게 overshoot하는 사례가 있기
때문이다. 최악 사례에서는 Paper importance `0.0779`를 맞추려다가 Quant가
`0.1363`까지 선택해 latency가 Paper보다 `69.8%` 높아졌다. 이는 q 해상도보다
unsupported Pareto point 문제다.

반면 `R/N>=0.15`에서는 평균 개선의 95% CI 전체가 0보다 크고, `R/N>=0.80`
에서는 거의 모든 사례에서 Quant가 이겼다.

주 그래프는
[`latency_saving_ci.pdf`](results/latency_saving_ci.pdf)다.

## CV와 spatial ordering

| CV | 평균 절감 | 95% CI |
| ---: | ---: | ---: |
| 1.07 | 2.002% | [1.840%, 2.163%] |
| 1.25 | 2.943% | [2.772%, 3.120%] |
| 1.44 | 3.693% | [3.451%, 3.934%] |
| 2.48 | 7.799% | [7.546%, 8.055%] |
| 3.30 | 10.200% | [9.781%, 10.619%] |
| 4.55 | 13.210% | [12.739%, 13.677%] |

importance dispersion이 클수록 Quant의 이점이 일관되게 증가한다.

| Ordering | 평균 절감 | 95% CI | 승률 |
| --- | ---: | ---: | ---: |
| Random | 3.060% | [2.617%, 3.504%] | 75.9% |
| Local clustering | 10.994% | [10.478%, 11.515%] | 93.2% |
| Persistent hot-cold | 5.870% | [5.314%, 6.426%] | 86.4% |

local clustering에서 gap merge의 이점이 가장 크고, random ordering에서는
상대적으로 작다.

## Chunk 구조와 계산 비용

Paper-matched target에서 평균 chunk 수는:

| 방법 | 평균 K | 중앙값 K |
| --- | ---: | ---: |
| Quant | 16.23 | 18 |
| Paper | 24.78 | 24 |
| Top-R | 458.15 | 360 |

Quant는 Paper보다 chunk 수를 줄이되 Top-R처럼 파편화되지 않는다.

현재 구현의 시간은:

- Quant q-grid build: spatial input당 평균 `756.5 ms`
- Paper query: 평균 `10.06 ms`
- 입력당 서로 다른 supported solution: 평균 `376.5개`

Quant build 하나로 19개 coverage query를 처리했지만, 단일 online query로 보면
현재 dense-grid 구현은 Paper보다 훨씬 비싸다. 이 실험은 더 좋은 mask가
존재함을 정밀하게 측정한 것이며, 아직 online replacement의 실행시간을
달성한 것은 아니다.

## 산출물

- [`latency_saving_ci.pdf`](results/latency_saving_ci.pdf)
- [`importance_latency_frontiers.pdf`](results/importance_latency_frontiers.pdf)
- [`r_importance.pdf`](results/r_importance.pdf)
- [`r_latency.pdf`](results/r_latency.pdf)
- [`coverage_opt_ratio.pdf`](results/coverage_opt_ratio.pdf)
- [`chunk_count.pdf`](results/chunk_count.pdf)
- [`summary.json`](results/summary.json)

## 결론

constant-throughput tail 가정과 현재 synthetic CV/order benchmark 아래에서
Quant는 Paper와 같은 importance를 유지하면서 lookup latency를 평균
**6.64% [95% CI 6.16%, 7.12%]** 줄인다. 높은 CV와 중간--큰 R에서는 개선이
크고 안정적이다.

단, 매우 작은 R에서는 supported-point overshoot 때문에 Paper보다 나쁘다.
따라서 최종 알고리즘은 low-budget 영역에서 Paper fallback을 사용하거나,
coverage를 직접 보존하여 unsupported point를 표현하는 보완이 필요하다.
