# Experiment 45: causal local importance-bound Cell-1

Global R allocation과 미래 projection 정보는 사용하지 않는다. 각 projection 
호출에서 현재 X² importance만 계산하고 `I(S) >= τ`를 만족하는 fixed-origin 
s-cell mask 중 lookup read latency가 작은 mask를 선택한 뒤 run endpoint를 
importance 하한이 깨지지 않는 범위에서 trim한다. R은 입력이 아니라 결과다.

## 동일 projection error

| error | Paper | Cell-1 fixed R | Cell-1-τ | gain vs Paper | gain vs fixed |
|---:|---:|---:|---:|---:|---:|
| 0.15 | 2.494 ms | 2.054 ms | 2.091 ms | 16.17% | -1.77% |
| 0.20 | 2.380 ms | 1.981 ms | 2.015 ms | 15.33% | -1.70% |
| 0.25 | 2.256 ms | 1.950 ms | 1.896 ms | 16.00% | 2.82% |
| 0.30 | 2.068 ms | 1.838 ms | 1.758 ms | 14.97% | 4.35% |
| 0.35 | 1.847 ms | 1.668 ms | 1.627 ms | 11.91% | 2.43% |
| 0.40 | 1.658 ms | 1.529 ms | 1.470 ms | 11.30% | 3.84% |
| 0.42 | 1.583 ms | 1.486 ms | 1.399 ms | 11.66% | 5.88% |
| 0.45 | 1.495 ms | 1.381 ms | 1.316 ms | 11.98% | 4.75% |
| 0.50 | 1.408 ms | 1.249 ms | 1.180 ms | 16.18% | 5.54% |

## 동일-error 구성요소 평균

| method | score | selector | SSD/upload | gather | GEMM | total |
|---|---:|---:|---:|---:|---:|---:|
| Paper fixed R |x| | 0.0301 | 0.2894 | 1.5121 | 0.0177 | 0.0607 | 1.9099 |
| Cell-1 X² fixed R | 0.0329 | 0.1040 | 1.4603 | 0.0176 | 0.0672 | 1.6819 |
| Cell-1-τ X² local I-bound | 0.0329 | 0.0930 | 1.4287 | 0.0173 | 0.0672 | 1.6390 |

## 모델별 동일-error 평균 gain

| model | vs Paper | vs fixed-R Cell-1 |
|---|---:|---:|
| qwen05 | 13.94% | 2.90% |

## 판정

공통 error 구간에서 Cell-1-τ의 평균 gain은 Paper 대비 `13.94%`, fixed-R Cell-1 대비 `2.90%`다.

이 결과는 global allocation 상한과 다르다. 각 호출은 현재 activation만 
사용하므로 causal하고, 하나의 τ만 quality knob로 사용한다. actual error는 
평가에만 사용하고 selector에는 입력하지 않았다.

## 측정 범위

- 모델 `1`개, 후보 `7938`개.
- fallback `0`건; importance 하한 위반 `0`건.
- control 10%..95% 5% 간격 + 97/98/99%; baseline은 fixed R, Cell-1-τ는 τ.
- actual total은 score + selector + O_DIRECT/upload + gather + compact GEMM.

![Frontier](qwen05/local_importance_frontier.png)

![Components](qwen05/local_importance_components.png)

![Tau to R](qwen05/tau_to_resulting_r.png)
