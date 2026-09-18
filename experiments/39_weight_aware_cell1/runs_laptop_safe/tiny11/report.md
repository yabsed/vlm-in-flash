# Experiment 39 보고서: weight-aware Cell-1

기존 Cell-1의 `mean(|X_i|)` 점수를 실제 projection squared-error의 대각 근사 `q_i = mean(X_i^2) * ||W[:,i]||_2^2`로 교체했다. `Cell-1 X²`를 추가해 activation-energy 변경과 weight norm 효과를 분리했다. 모든 latency에는 온라인 score 생성 시간이 포함되고, weight norm은 모델 로딩 때 한 번 사전 계산하므로 포함하지 않는다.

## 동일 projection error 보간

5% R grid 사이의 선형 보간 진단이다. 양수 gain은 기존 Cell-1보다 빠르다는 뜻이다.

| error | Paper | Cell-1 | Cell-1 X² | gain | WCell-1 | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.082 ms | 0.951 ms | 0.910 ms | 4.380% | 0.926 ms | 2.661% |
| 0.45 | 1.196 ms | 1.019 ms | 0.993 ms | 2.614% | 1.003 ms | 1.598% |
| 0.42 | 1.270 ms | 1.086 ms | 1.052 ms | 3.149% | 1.063 ms | 2.054% |
| 0.40 | 1.318 ms | 1.147 ms | 1.113 ms | 2.946% | 1.125 ms | 1.865% |
| 0.35 | 1.485 ms | 1.288 ms | 1.269 ms | 1.464% | 1.270 ms | 1.419% |
| 0.30 | 1.666 ms | 1.390 ms | 1.371 ms | 1.390% | 1.377 ms | 0.930% |
| 0.25 | 1.812 ms | 1.501 ms | 1.492 ms | 0.621% | 1.486 ms | 1.048% |
| 0.20 | 2.013 ms | 1.668 ms | 1.671 ms | -0.173% | 1.708 ms | -2.359% |
| 0.15 | 2.183 ms | 1.808 ms | 1.778 ms | 1.655% | 1.784 ms | 1.321% |

## 동일-error 구성요소 평균

| method | score | selector | SSD/upload | gather | GEMM | total |
|---|---:|---:|---:|---:|---:|---:|
| Paper |x| | 0.0319 ms | 0.4255 ms | 0.9707 ms | 0.0168 ms | 0.1134 ms | 1.5583 ms |
| Cell-1 |x| | 0.0319 ms | 0.1192 ms | 1.0208 ms | 0.0170 ms | 0.1288 ms | 1.3177 ms |
| Cell-1 X² | 0.0401 ms | 0.1191 ms | 0.9909 ms | 0.0171 ms | 0.1270 ms | 1.2942 ms |
| WCell-1 X²·W² | 0.0427 ms | 0.1194 ms | 0.9989 ms | 0.0171 ms | 0.1266 ms | 1.3047 ms |

## 모델별 실측-grid 최적점

| model | error | Cell-1 | Cell-1 X² | WCell-1 |
|---|---:|---:|---:|---:|
| tiny11 | 0.15 | 1.813 ms (R 95%) | 1.794 ms (R 95%) | 1.779 ms (R 95%) |
| tiny11 | 0.25 | 1.554 ms (R 80%) | 1.571 ms (R 80%) | 1.576 ms (R 80%) |
| tiny11 | 0.42 | 1.109 ms (R 55%) | 1.120 ms (R 55%) | 1.132 ms (R 55%) |

## End-to-end logit KL

| KL ceiling | Paper | Cell-1 | Cell-1 X² | WCell-1 |
|---:|---:|---:|---:|---:|
| 12 | 0.749 ms (KL 10.536) | 0.450 ms (KL 10.482) | 0.450 ms (KL 10.456) | 0.449 ms (KL 10.200) |
| 10 | 0.886 ms (KL 9.374) | 0.712 ms (KL 9.320) | 0.517 ms (KL 9.847) | 0.634 ms (KL 9.771) |
| 8 | 1.429 ms (KL 7.298) | 1.026 ms (KL 7.941) | 1.120 ms (KL 7.635) | 1.132 ms (KL 7.628) |
| 6 | 1.648 ms (KL 5.683) | 1.314 ms (KL 5.965) | 1.455 ms (KL 3.517) | 1.321 ms (KL 5.084) |
| 4 | 1.838 ms (KL 3.249) | 1.554 ms (KL 3.378) | 1.455 ms (KL 3.517) | 1.450 ms (KL 3.074) |
| 2 | 2.089 ms (KL 1.314) | 1.695 ms (KL 1.492) | 1.571 ms (KL 1.943) | 1.760 ms (KL 1.311) |
| 1 | 2.215 ms (KL 0.697) | 1.808 ms (KL 0.730) | 1.778 ms (KL 0.612) | 1.779 ms (KL 0.295) |

## 판정

WCell-1의 기존 Cell-1 대비 동일-error gain 범위는 `-2.36%..2.66%`, 평균 `1.17%`다. X²-only gain 평균은 `2.01%`다. ceiling별 전체 winner 횟수는 `{'cell1_x2': 7, 'cell1_abs': 1, 'wcell1': 1}`다. projection proxy 개선과 실제 logit KL 결과가 다르면 최종 판정은 logit KL을 우선한다.

## 측정 범위

- projection 후보 `9072`개.
- non-empty O_DIRECT fallback `0`건.
- 모델 3개, prompt 6개, 모델당 표본 layer 3개, projection 7종.
- R=10%..95%, 5% 간격; actual total은 score + selector + O_DIRECT/upload wall + gather + compact GEMM이다.
- weight norm metadata는 GPU에 상주하며 사전 계산 시간은 online latency에서 제외했다.

![Frontiers](tiny11/weight_aware_frontiers.png)

![Components](tiny11/weight_aware_components.png)

![Gain](tiny11/weight_aware_gain.png)
