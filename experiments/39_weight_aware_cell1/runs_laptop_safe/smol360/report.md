# Experiment 39 보고서: weight-aware Cell-1

기존 Cell-1의 `mean(|X_i|)` 점수를 실제 projection squared-error의 대각 근사 `q_i = mean(X_i^2) * ||W[:,i]||_2^2`로 교체했다. `Cell-1 X²`를 추가해 activation-energy 변경과 weight norm 효과를 분리했다. 모든 latency에는 온라인 score 생성 시간이 포함되고, weight norm은 모델 로딩 때 한 번 사전 계산하므로 포함하지 않는다.

## 동일 projection error 보간

5% R grid 사이의 선형 보간 진단이다. 양수 gain은 기존 Cell-1보다 빠르다는 뜻이다.

| error | Paper | Cell-1 | Cell-1 X² | gain | WCell-1 | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.578 ms | 0.386 ms | 0.377 ms | 2.318% | 0.380 ms | 1.601% |
| 0.45 | 0.610 ms | 0.404 ms | 0.396 ms | 2.015% | 0.397 ms | 1.813% |
| 0.42 | 0.627 ms | 0.415 ms | 0.410 ms | 1.129% | 0.411 ms | 0.906% |
| 0.40 | 0.640 ms | 0.422 ms | 0.418 ms | 0.826% | 0.419 ms | 0.629% |
| 0.35 | 0.660 ms | 0.447 ms | 0.438 ms | 2.118% | 0.442 ms | 1.249% |
| 0.30 | 0.695 ms | 0.470 ms | 0.471 ms | -0.036% | 0.468 ms | 0.481% |
| 0.25 | 0.728 ms | 0.487 ms | 0.482 ms | 1.179% | 0.484 ms | 0.740% |
| 0.20 | 0.757 ms | 0.525 ms | 0.525 ms | 0.044% | 0.530 ms | -0.811% |
| 0.15 | 0.806 ms | 0.538 ms | 0.533 ms | 0.953% | 0.535 ms | 0.616% |

## 동일-error 구성요소 평균

| method | score | selector | SSD/upload | gather | GEMM | total |
|---|---:|---:|---:|---:|---:|---:|
| Paper |x| | 0.0303 ms | 0.3043 ms | 0.2878 ms | 0.0161 ms | 0.0394 ms | 0.6779 ms |
| Cell-1 |x| | 0.0303 ms | 0.0900 ms | 0.2781 ms | 0.0156 ms | 0.0411 ms | 0.4550 ms |
| Cell-1 X² | 0.0288 ms | 0.0891 ms | 0.2757 ms | 0.0156 ms | 0.0408 ms | 0.4500 ms |
| WCell-1 X²·W² | 0.0321 ms | 0.0877 ms | 0.2756 ms | 0.0156 ms | 0.0407 ms | 0.4517 ms |

## 모델별 실측-grid 최적점

| model | error | Cell-1 | Cell-1 X² | WCell-1 |
|---|---:|---:|---:|---:|
| smol360 | 0.15 | 0.540 ms (R 95%) | 0.533 ms (R 95%) | 0.533 ms (R 95%) |
| smol360 | 0.25 | 0.516 ms (R 85%) | 0.516 ms (R 85%) | 0.520 ms (R 85%) |
| smol360 | 0.42 | 0.417 ms (R 55%) | 0.416 ms (R 55%) | 0.418 ms (R 55%) |

## End-to-end logit KL

| KL ceiling | Paper | Cell-1 | Cell-1 X² | WCell-1 |
|---:|---:|---:|---:|---:|
| 12 | 0.577 ms (KL 11.939) | 0.351 ms (KL 11.778) | 0.372 ms (KL 11.834) | 0.428 ms (KL 9.061) |
| 10 | 0.632 ms (KL 9.874) | 0.432 ms (KL 8.801) | 0.430 ms (KL 8.217) | 0.428 ms (KL 9.061) |
| 8 | 0.661 ms (KL 7.290) | 0.450 ms (KL 6.633) | 0.466 ms (KL 6.286) | 0.454 ms (KL 7.841) |
| 6 | 0.705 ms (KL 5.855) | 0.476 ms (KL 5.942) | 0.479 ms (KL 4.596) | 0.481 ms (KL 4.170) |
| 4 | 0.743 ms (KL 3.487) | 0.516 ms (KL 3.206) | 0.516 ms (KL 2.905) | 0.520 ms (KL 2.704) |
| 2 | 0.815 ms (KL 1.207) | 0.535 ms (KL 1.539) | 0.533 ms (KL 0.706) | 0.533 ms (KL 0.637) |
| 1 | 0.851 ms (KL 0.450) | 0.540 ms (KL 0.700) | 0.533 ms (KL 0.706) | 0.533 ms (KL 0.637) |

## 판정

WCell-1의 기존 Cell-1 대비 동일-error gain 범위는 `-0.81%..1.81%`, 평균 `0.80%`다. X²-only gain 평균은 `1.17%`다. ceiling별 전체 winner 횟수는 `{'cell1_x2': 8, 'wcell1': 1}`다. projection proxy 개선과 실제 logit KL 결과가 다르면 최종 판정은 logit KL을 우선한다.

## 측정 범위

- projection 후보 `9072`개.
- non-empty O_DIRECT fallback `0`건.
- 모델 3개, prompt 6개, 모델당 표본 layer 3개, projection 7종.
- R=10%..95%, 5% 간격; actual total은 score + selector + O_DIRECT/upload wall + gather + compact GEMM이다.
- weight norm metadata는 GPU에 상주하며 사전 계산 시간은 online latency에서 제외했다.

![Frontiers](smol360/weight_aware_frontiers.png)

![Components](smol360/weight_aware_components.png)

![Gain](smol360/weight_aware_gain.png)
