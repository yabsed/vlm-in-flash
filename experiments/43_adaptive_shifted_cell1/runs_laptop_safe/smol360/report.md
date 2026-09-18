# Experiment 43: adaptive shifted Cell-1

고정된 원점의 s-cell만 쓰는 Cell-1을 8개 offset으로 확장하고, 가장 
불확실한 selected/unselected cell 각각 4개만 s/2 subcell로 다시 
최적화했다. 모든 방법은 실제 score + selector + O_DIRECT/upload + 
activation gather + compact GEMM 시간을 포함한다.

## 동일 projection error

양수 gain은 기존 Cell-1 X²보다 빠르다는 뜻이다.

| error | Paper | Cell-1 X² | Shifted-8 | gain | Adaptive-8 | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.900 ms | 0.561 ms | 0.608 ms | -8.33% | 0.620 ms | -10.58% |
| 0.20 | 0.840 ms | 0.549 ms | 0.591 ms | -7.48% | 0.596 ms | -8.50% |
| 0.25 | 0.789 ms | 0.506 ms | 0.554 ms | -9.56% | 0.567 ms | -12.11% |
| 0.30 | 0.761 ms | 0.491 ms | 0.545 ms | -10.98% | 0.549 ms | -11.72% |
| 0.35 | 0.734 ms | 0.467 ms | 0.516 ms | -10.47% | 0.520 ms | -11.37% |
| 0.40 | 0.686 ms | 0.439 ms | 0.503 ms | -14.43% | 0.500 ms | -13.89% |
| 0.42 | 0.674 ms | 0.429 ms | 0.480 ms | -11.82% | 0.479 ms | -11.68% |
| 0.45 | 0.652 ms | 0.414 ms | 0.453 ms | -9.51% | 0.454 ms | -9.69% |
| 0.50 | 0.631 ms | 0.404 ms | 0.416 ms | -2.99% | 0.422 ms | -4.47% |

## 동일-error 구성요소 평균

| method | score | selector | SSD/upload | gather | GEMM | total |
|---|---:|---:|---:|---:|---:|---:|
| Paper |x| | 0.0324 | 0.3411 | 0.3095 | 0.0168 | 0.0407 | 0.7407 |
| Cell-1 X² | 0.0309 | 0.0946 | 0.2889 | 0.0167 | 0.0424 | 0.4734 |
| Shifted-8 X² | 0.0309 | 0.1238 | 0.3048 | 0.0167 | 0.0422 | 0.5183 |
| Adaptive-8 X² | 0.0309 | 0.1328 | 0.3008 | 0.0168 | 0.0419 | 0.5231 |

## R을 없앤 전역 allocation

각 workload의 sampled projection 전체에서 하나의 quality 
budget을 공유한다. `importance_bound`는 공통 mean-|x| retention 
하한만 사용하며, `error_oracle`은 실측 projection error를 직접 
사용하는 비배포형 상한이다.

| objective | paper target R | method | resulting R | error | latency |
|---|---:|---|---:|---:|---:|
| error_oracle | 30% | Adaptive-8 X² | 33.2% | 0.5229 | 0.3335 ms |
| error_oracle | 30% | Cell-1 X² | 34.8% | 0.5345 | 0.3081 ms |
| error_oracle | 50% | Adaptive-8 X² | 47.0% | 0.4000 | 0.3664 ms |
| error_oracle | 50% | Cell-1 X² | 51.2% | 0.4076 | 0.3428 ms |
| error_oracle | 70% | Adaptive-8 X² | 65.2% | 0.2769 | 0.4300 ms |
| error_oracle | 70% | Cell-1 X² | 68.8% | 0.2737 | 0.4097 ms |
| error_oracle | 90% | Adaptive-8 X² | 87.3% | 0.1390 | 0.5688 ms |
| error_oracle | 90% | Cell-1 X² | 91.6% | 0.1384 | 0.5405 ms |
| importance_bound | 30% | Adaptive-8 X² | 35.8% | 0.5413 | 0.3334 ms |
| importance_bound | 30% | Cell-1 X² | 35.2% | 0.5591 | 0.3057 ms |
| importance_bound | 50% | Adaptive-8 X² | 55.1% | 0.3832 | 0.3795 ms |
| importance_bound | 50% | Cell-1 X² | 54.7% | 0.4058 | 0.3465 ms |
| importance_bound | 70% | Adaptive-8 X² | 75.6% | 0.2432 | 0.4752 ms |
| importance_bound | 70% | Cell-1 X² | 74.6% | 0.2665 | 0.4293 ms |
| importance_bound | 90% | Adaptive-8 X² | 91.7% | 0.1479 | 0.5804 ms |
| importance_bound | 90% | Cell-1 X² | 91.7% | 0.1578 | 0.5269 ms |

## 판정

동일-error에서 Shifted-8의 Cell-1 대비 평균 gain은 `-9.51%`, Adaptive-8은 `-10.45%`다.
결론은 selector 자체의 mask 개선이 추가 selector 시간보다 큰지, 그리고 
고정 R을 없앤 전역 quality allocation이 그보다 더 큰지로 나누어 해석해야 
한다. error oracle은 달성 가능한 상한이지 배포 가능한 알고리즘이 아니다.

## 측정 범위

- projection 후보 `9072`개; fallback `0`건.
- offset `8`개, boundary cell 방향당 `4`개, refine divisor `2`.
- R=10%..95%, 5% 간격. 그래프의 allocation 점 위 %는 최적화 결과로 
나온 평균 selected fraction이다.

![Frontier](smol360/adaptive_quality_latency_frontier.png)

![Components](smol360/adaptive_component_breakdown.png)

![Allocation](smol360/global_quality_allocation.png)
