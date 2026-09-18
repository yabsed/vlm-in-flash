# Experiment 39 보고서: weight-aware Cell-1

기존 Cell-1의 `mean(|X_i|)` 점수를 실제 projection squared-error의 대각 근사 `q_i = mean(X_i^2) * ||W[:,i]||_2^2`로 교체했다. `Cell-1 X²`를 추가해 activation-energy 변경과 weight norm 효과를 분리했다. 모든 latency에는 온라인 score 생성 시간이 포함되고, weight norm은 모델 로딩 때 한 번 사전 계산하므로 포함하지 않는다.

## 동일 projection error 보간

5% R grid 사이의 선형 보간 진단이다. 양수 gain은 기존 Cell-1보다 빠르다는 뜻이다.

| error | Paper | Cell-1 | Cell-1 X² | gain | WCell-1 | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.594 ms | 0.497 ms | 0.503 ms | -1.121% | 0.501 ms | -0.737% |
| 0.45 | 0.638 ms | 0.538 ms | 0.533 ms | 0.948% | 0.548 ms | -1.904% |
| 0.42 | 0.668 ms | 0.571 ms | 0.570 ms | 0.171% | 0.573 ms | -0.461% |
| 0.40 | 0.695 ms | 0.591 ms | 0.582 ms | 1.403% | 0.590 ms | 0.148% |
| 0.35 | 0.745 ms | 0.656 ms | 0.644 ms | 1.834% | 0.648 ms | 1.124% |
| 0.30 | 0.808 ms | 0.672 ms | 0.663 ms | 1.450% | 0.672 ms | 0.009% |
| 0.25 | 0.855 ms | 0.711 ms | 0.725 ms | -1.875% | 0.723 ms | -1.630% |
| 0.20 | 0.890 ms | 0.731 ms | 0.735 ms | -0.591% | 0.744 ms | -1.751% |
| 0.15 | 0.932 ms | 0.765 ms | 0.771 ms | -0.854% | 0.768 ms | -0.354% |

## 동일-error 구성요소 평균

| method | score | selector | SSD/upload | gather | GEMM | total |
|---|---:|---:|---:|---:|---:|---:|
| Paper |x| | 0.0298 ms | 0.2487 ms | 0.4127 ms | 0.0163 ms | 0.0508 ms | 0.7583 ms |
| Cell-1 |x| | 0.0298 ms | 0.0968 ms | 0.4384 ms | 0.0162 ms | 0.0557 ms | 0.6369 ms |
| Cell-1 X² | 0.0314 ms | 0.0976 ms | 0.4353 ms | 0.0162 ms | 0.0556 ms | 0.6362 ms |
| WCell-1 X²·W² | 0.0351 ms | 0.0948 ms | 0.4395 ms | 0.0160 ms | 0.0555 ms | 0.6408 ms |

## 모델별 실측-grid 최적점

| model | error | Cell-1 | Cell-1 X² | WCell-1 |
|---|---:|---:|---:|---:|
| qwen05 | 0.15 | 0.780 ms (R 95%) | 0.792 ms (R 95%) | 0.779 ms (R 95%) |
| qwen05 | 0.25 | 0.718 ms (R 85%) | 0.727 ms (R 85%) | 0.735 ms (R 85%) |
| qwen05 | 0.42 | 0.585 ms (R 60%) | 0.579 ms (R 60%) | 0.591 ms (R 60%) |

## End-to-end logit KL

| KL ceiling | Paper | Cell-1 | Cell-1 X² | WCell-1 |
|---:|---:|---:|---:|---:|
| 12 | 0.494 ms (KL 9.567) | 0.516 ms (KL 11.879) | 0.343 ms (KL 10.377) | 0.591 ms (KL 11.421) |
| 10 | 0.494 ms (KL 9.567) | 0.671 ms (KL 9.816) | 0.658 ms (KL 9.463) | 0.672 ms (KL 7.769) |
| 8 | 0.824 ms (KL 6.748) | 0.706 ms (KL 7.376) | 0.727 ms (KL 7.020) | 0.672 ms (KL 7.769) |
| 6 | 0.875 ms (KL 5.294) | 0.740 ms (KL 5.745) | 0.741 ms (KL 5.653) | 0.749 ms (KL 5.153) |
| 4 | 0.966 ms (KL 2.573) | 0.780 ms (KL 0.800) | 0.792 ms (KL 0.798) | 0.779 ms (KL 0.833) |
| 2 | — | 0.780 ms (KL 0.800) | 0.792 ms (KL 0.798) | 0.779 ms (KL 0.833) |
| 1 | — | 0.780 ms (KL 0.800) | 0.792 ms (KL 0.798) | 0.779 ms (KL 0.833) |

## 판정

WCell-1의 기존 Cell-1 대비 동일-error gain 범위는 `-1.90%..1.12%`, 평균 `-0.62%`다. X²-only gain 평균은 `0.15%`다. ceiling별 전체 winner 횟수는 `{'cell1_abs': 4, 'cell1_x2': 5}`다. projection proxy 개선과 실제 logit KL 결과가 다르면 최종 판정은 logit KL을 우선한다.

## 측정 범위

- projection 후보 `9072`개.
- non-empty O_DIRECT fallback `0`건.
- 모델 3개, prompt 6개, 모델당 표본 layer 3개, projection 7종.
- R=10%..95%, 5% 간격; actual total은 score + selector + O_DIRECT/upload wall + gather + compact GEMM이다.
- weight norm metadata는 GPU에 상주하며 사전 계산 시간은 online latency에서 제외했다.

![Frontiers](qwen05/weight_aware_frontiers.png)

![Components](qwen05/weight_aware_components.png)

![Gain](qwen05/weight_aware_gain.png)
