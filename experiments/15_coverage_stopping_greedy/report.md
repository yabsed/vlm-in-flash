# Experiment 15 보고서: Paper greedy with coverage stopping

## 질문

Paper greedy의 candidate와 utility score는 유지하되 fixed-R 종료조건을
importance coverage 종료조건으로 바꾸면 high-q Quant의 이점을 얼마나
회수할 수 있는가?

## 변경한 알고리즘

Paper와 동일하게 multi-scale window를 만들고

\[
\operatorname{score}(C)=\frac{I(C)}{T(|C|)}
\]

순서로 정렬한다. 순서대로 기존 mask와 겹치지 않는 window를 선택하는 것도
동일하다. 차이는 종료조건뿐이다.

- Original Paper: 선택 행 수가 fixed-R budget을 채울 때까지 진행
- Coverage-Paper: 누적 importance가 target \(B\)를 처음 넘을 때 종료

각 사례의 \(B\)는 Experiment 14에서 original Paper가 얻은 importance다.
따라서 Original Paper, Coverage-Paper, Quant는 모두 같은 lower-bound target을
만족한다. 후보 순서는 target과 무관하므로 입력당 greedy non-overlap prefix를
한 번 구축한 뒤 19개 target에 재사용했다.

## 설정

Experiment 14의 300개 독립 cluster, 900개 spatial input, 17,100개 budget
case를 그대로 사용했다. Original Paper와 high-q Quant 결과는 저장된 CSV에서
재사용했고, Coverage-Paper만 새로 계산했다. latency는 saturation 이후
constant-throughput을 가정한 released lookup 기준이다.

## 주 결과

| 방법 | Original Paper 대비 평균 latency 절감 | 95% CI | 승률 |
| --- | ---: | ---: | ---: |
| Paper + coverage stop | **-2.088%** | **[-2.150%, -2.027%]** | 10.73% |
| Quant (high-q) | **6.641%** | **[6.159%, 7.129%]** | 85.16% |

종료조건만 coverage로 바꾼 Paper는 개선되지 않았다. 오히려 Original Paper보다
평균 `2.09%` 느렸다. Coverage-Paper가 Quant보다 낮은 latency를 얻은 비율도
`8.47%`뿐이며, Coverage-Paper latency는 Quant보다 평균 `10.09%` 높았다
(`95% CI 9.54--10.65%`).

따라서 Quant의 이점은 “fixed-R 대신 coverage에서 멈춘다”는 차이로 설명되지
않는다.

## 왜 느려지는가

Coverage-Paper도 target 직전에는 다음 candidate window 전체를 선택해야 한다.
마지막 window가 필요한 importance보다 크면 importance와 행 수가 함께
overshoot한다. 하지만 candidate의 구조는 여전히 Original Paper와 거의 같아
chunk 수는 줄어들지 않는다.

| 방법 | 평균 선택 행 수 | 평균 chunk 수 | 평균 importance |
| --- | ---: | ---: | ---: |
| Original Paper | 2,429.68 | 24.78 | 0.71446 |
| Paper + coverage stop | 2,479.21 | 24.73 | 0.72119 |
| Quant | **2,370.17** | **16.23** | 0.72211 |

Coverage-Paper는 Original Paper보다 평균 약 50행을 더 선택하면서 chunk 수는
사실상 그대로다. 반면 Quant는 importance를 더 많이 얻으면서도 평균 약 60행을
덜 선택하고 chunk 수를 약 8.5개 줄인다. 즉 Quant의 핵심은 coverage 종료가
아니라 latency-aware한 전역 chunk 조합과 merge 구조다.

## R에 따른 결과

| R/N | Coverage-Paper 절감 | Quant 절감 |
| ---: | ---: | ---: |
| 0.05 | -8.36% | -6.60% |
| 0.10 | -5.26% | -0.96% |
| 0.15 | -3.68% | 1.34% |
| 0.25 | -2.30% | 4.11% |
| 0.50 | -1.51% | 7.57% |
| 0.75 | -1.02% | 10.34% |
| 0.95 | -0.20% | 12.88% |

Coverage-Paper는 R이 커지면 Original Paper에 가까워지지만 전 구간 평균에서
개선되지 않는다. 작은 R에서는 한 window의 상대적 크기가 커서 overshoot
penalty가 가장 크다.

## 계산 비용

- Coverage-Paper prefix build: 입력당 평균 `13.08 ms`
- Coverage-Paper target query: 평균 `0.016 ms`
- Original Paper query: 평균 `10.06 ms`
- Quant q-grid build: 입력당 평균 `756.55 ms`

Coverage-Paper는 하나의 입력에서 여러 target을 질의한다면 매우 싸다. 하지만
빠른 계산이 더 낮은 I/O latency로 이어지지는 않았다. 단일 target에서는
build를 포함해 약 `13.10 ms`이므로 Original Paper와 같은 수준이고, Quant보다는
훨씬 싸다.

## 산출물

- [`latency_saving_ci.pdf`](results/latency_saving_ci.pdf)
- [`importance_latency_frontiers.pdf`](results/importance_latency_frontiers.pdf)
- [`latency_ratio.pdf`](results/latency_ratio.pdf)
- [`importance_overshoot.pdf`](results/importance_overshoot.pdf)
- [`summary.json`](results/summary.json)

## 결론

Paper greedy의 종료조건만 importance coverage로 바꾸는 것은 답이 아니다.
Coverage-Paper는 매우 빠르지만 Original Paper보다 평균 `2.09%` 느리고,
Quant보다 평균 `10.09%` 느리다.

Quant의 성능 차이는 종료조건이 아니라 평균 chunk 수를 `24.8 -> 16.2`로 줄이고
더 적은 행으로 같은 target을 덮는 전역적 mask 구조에서 발생한다. 다음 개선은
Paper의 종료조건 변경보다 candidate 선택 이후의 merge/trim 또는 unsupported
coverage point를 직접 표현하는 방법에 집중해야 한다.
