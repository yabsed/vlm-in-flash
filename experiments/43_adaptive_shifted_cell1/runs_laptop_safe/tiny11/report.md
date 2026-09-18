# Experiment 43: adaptive shifted Cell-1

고정된 원점의 s-cell만 쓰는 Cell-1을 8개 offset으로 확장하고, 가장 
불확실한 selected/unselected cell 각각 4개만 s/2 subcell로 다시 
최적화했다. 모든 방법은 실제 score + selector + O_DIRECT/upload + 
activation gather + compact GEMM 시간을 포함한다.

## 동일 projection error

양수 gain은 기존 Cell-1 X²보다 빠르다는 뜻이다.

| error | Paper | Cell-1 X² | Shifted-8 | gain | Adaptive-8 | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 2.322 ms | 1.959 ms | 2.054 ms | -4.85% | 2.085 ms | -6.43% |
| 0.20 | 2.123 ms | 1.822 ms | 1.917 ms | -5.23% | 1.937 ms | -6.34% |
| 0.25 | 1.936 ms | 1.639 ms | 1.754 ms | -7.02% | 1.757 ms | -7.17% |
| 0.30 | 1.810 ms | 1.532 ms | 1.632 ms | -6.51% | 1.652 ms | -7.84% |
| 0.35 | 1.614 ms | 1.424 ms | 1.541 ms | -8.24% | 1.535 ms | -7.83% |
| 0.40 | 1.440 ms | 1.258 ms | 1.358 ms | -7.97% | 1.344 ms | -6.84% |
| 0.42 | 1.387 ms | 1.191 ms | 1.316 ms | -10.51% | 1.302 ms | -9.32% |
| 0.45 | 1.304 ms | 1.115 ms | 1.238 ms | -11.06% | 1.223 ms | -9.68% |
| 0.50 | 1.177 ms | 1.012 ms | 1.106 ms | -9.31% | 1.082 ms | -6.99% |

## 동일-error 구성요소 평균

| method | score | selector | SSD/upload | gather | GEMM | total |
|---|---:|---:|---:|---:|---:|---:|
| Paper |x| | 0.0318 | 0.4268 | 1.0865 | 0.0194 | 0.1148 | 1.6793 |
| Cell-1 X² | 0.0411 | 0.1235 | 1.1268 | 0.0192 | 0.1282 | 1.4389 |
| Shifted-8 X² | 0.0411 | 0.2277 | 1.1288 | 0.0196 | 0.1288 | 1.5462 |
| Adaptive-8 X² | 0.0411 | 0.2415 | 1.1170 | 0.0193 | 0.1272 | 1.5463 |

## R을 없앤 전역 allocation

각 workload의 sampled projection 전체에서 하나의 quality 
budget을 공유한다. `importance_bound`는 공통 mean-|x| retention 
하한만 사용하며, `error_oracle`은 실측 projection error를 직접 
사용하는 비배포형 상한이다.

| objective | paper target R | method | resulting R | error | latency |
|---|---:|---|---:|---:|---:|
| error_oracle | 30% | Adaptive-8 X² | 34.1% | 0.5405 | 0.6060 ms |
| error_oracle | 30% | Cell-1 X² | 35.6% | 0.5332 | 0.5127 ms |
| error_oracle | 50% | Adaptive-8 X² | 50.5% | 0.4007 | 0.7544 ms |
| error_oracle | 50% | Cell-1 X² | 53.4% | 0.3968 | 0.6730 ms |
| error_oracle | 70% | Adaptive-8 X² | 69.4% | 0.2596 | 1.2266 ms |
| error_oracle | 70% | Cell-1 X² | 70.2% | 0.2629 | 1.1203 ms |
| error_oracle | 90% | Adaptive-8 X² | 87.4% | 0.1322 | 1.8689 ms |
| error_oracle | 90% | Cell-1 X² | 89.5% | 0.1272 | 1.7879 ms |
| importance_bound | 30% | Adaptive-8 X² | 35.3% | 0.5407 | 0.6070 ms |
| importance_bound | 30% | Cell-1 X² | 35.5% | 0.5620 | 0.4997 ms |
| importance_bound | 50% | Adaptive-8 X² | 56.3% | 0.3826 | 0.8165 ms |
| importance_bound | 50% | Cell-1 X² | 57.0% | 0.3903 | 0.7085 ms |
| importance_bound | 70% | Adaptive-8 X² | 75.0% | 0.2530 | 1.3428 ms |
| importance_bound | 70% | Cell-1 X² | 77.5% | 0.2356 | 1.3000 ms |
| importance_bound | 90% | Adaptive-8 X² | 92.0% | 0.1276 | 1.9551 ms |
| importance_bound | 90% | Cell-1 X² | 91.7% | 0.1326 | 1.8113 ms |

## 판정

동일-error에서 Shifted-8의 Cell-1 대비 평균 gain은 `-7.86%`, Adaptive-8은 `-7.60%`다.
결론은 selector 자체의 mask 개선이 추가 selector 시간보다 큰지, 그리고 
고정 R을 없앤 전역 quality allocation이 그보다 더 큰지로 나누어 해석해야 
한다. error oracle은 달성 가능한 상한이지 배포 가능한 알고리즘이 아니다.

## 측정 범위

- projection 후보 `9072`개; fallback `0`건.
- offset `8`개, boundary cell 방향당 `4`개, refine divisor `2`.
- R=10%..95%, 5% 간격. 그래프의 allocation 점 위 %는 최적화 결과로 
나온 평균 selected fraction이다.

![Frontier](tiny11/adaptive_quality_latency_frontier.png)

![Components](tiny11/adaptive_component_breakdown.png)

![Allocation](tiny11/global_quality_allocation.png)
