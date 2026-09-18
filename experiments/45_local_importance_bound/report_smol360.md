# Experiment 45: causal local importance-bound Cell-1

Global R allocation과 미래 projection 정보는 사용하지 않는다. 각 projection 
호출에서 현재 X² importance만 계산하고 `I(S) >= τ`를 만족하는 fixed-origin 
s-cell mask 중 lookup read latency가 작은 mask를 선택한 뒤 run endpoint를 
importance 하한이 깨지지 않는 범위에서 trim한다. R은 입력이 아니라 결과다.

## 동일 projection error

| error | Paper | Cell-1 fixed R | Cell-1-τ | gain vs Paper | gain vs fixed |
|---:|---:|---:|---:|---:|---:|
| 0.15 | 1.860 ms | 1.309 ms | 1.468 ms | 21.08% | -12.18% |
| 0.20 | 1.775 ms | 1.283 ms | 1.342 ms | 24.41% | -4.62% |
| 0.25 | 1.614 ms | 1.204 ms | 1.231 ms | 23.76% | -2.18% |
| 0.30 | 1.560 ms | 1.153 ms | 1.136 ms | 27.17% | 1.50% |
| 0.35 | 1.661 ms | 1.072 ms | 1.070 ms | 35.56% | 0.21% |
| 0.40 | 1.402 ms | 1.020 ms | 0.978 ms | 30.23% | 4.09% |
| 0.42 | 1.356 ms | 0.983 ms | 0.948 ms | 30.07% | 3.57% |
| 0.45 | 1.274 ms | 0.922 ms | 0.915 ms | 28.14% | 0.69% |
| 0.50 | 1.146 ms | 0.863 ms | 0.851 ms | 25.71% | 1.38% |

## 동일-error 구성요소 평균

| method | score | selector | SSD/upload | gather | GEMM | total |
|---|---:|---:|---:|---:|---:|---:|
| Paper fixed R |x| | 0.0306 | 0.3220 | 1.1069 | 0.0165 | 0.0405 | 1.5165 |
| Cell-1 X² fixed R | 0.0286 | 0.0941 | 0.9089 | 0.0163 | 0.0420 | 1.0900 |
| Cell-1-τ X² local I-bound | 0.0286 | 0.0866 | 0.9244 | 0.0158 | 0.0490 | 1.1045 |

## 모델별 동일-error 평균 gain

| model | vs Paper | vs fixed-R Cell-1 |
|---|---:|---:|
| smol360 | 27.35% | -0.84% |

## 판정

공통 error 구간에서 Cell-1-τ의 평균 gain은 Paper 대비 `27.35%`, fixed-R Cell-1 대비 `-0.84%`다.

이 결과는 global allocation 상한과 다르다. 각 호출은 현재 activation만 
사용하므로 causal하고, 하나의 τ만 quality knob로 사용한다. actual error는 
평가에만 사용하고 selector에는 입력하지 않았다.

## 측정 범위

- 모델 `1`개, 후보 `7938`개.
- fallback `0`건; importance 하한 위반 `0`건.
- control 10%..95% 5% 간격 + 97/98/99%; baseline은 fixed R, Cell-1-τ는 τ.
- actual total은 score + selector + O_DIRECT/upload + gather + compact GEMM.

![Frontier](smol360/local_importance_frontier.png)

![Components](smol360/local_importance_components.png)

![Tau to R](smol360/tau_to_resulting_r.png)
