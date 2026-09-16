# 랜덤 입력에서 paper greedy와 exact bounded-`R` 해의 비교

## 결론

200개의 랜덤 중요도 벡터에서, fixed-budget affine 목적에 대한 exact
DP는 paper greedy보다 `I/L`을 전체 조건 평균 **7.15%** 높였다. 다만
전체 관측치의 중앙값은 **1.23%**이므로 이 평균을 “일반적으로 항상
7% 정도 개선된다”고 읽으면 안 된다. 개선은 `R/N=75%`처럼 greedy가
여러 run을 만드는 구간에 집중됐다.

가장 큰 차이는 `R/N=75%`에서 나타났다. 이때 exact DP의 평균 개선은
**25.28%**, 중앙값은 **27.28%**였다. 200개 입력 중 paper greedy는
187개에서 두 개의 maximal run을 만들었지만 exact 해는 200개 모두 한
run을 선택했다. Affine 모델의 chunk당 고정비 `a`가 두 번째 run을 크게
벌점화하기 때문이다.

## 실험 설정

- 입력: `v_i = |N(0,1)|`, 각 벡터의 합을 1로 정규화
- 입력 수와 seed: 200개, `20260916`
- 크기: `N=256`, 1 KiB/row
- `R`: 32, 64, 96, 128, 160, 192, 224 (`R/N=12.5%`부터 `87.5%`)
- latency: bundled Jetson Orin AGX lookup table
- fitted affine model:
  `L_aff(M) = 0.0110009 K(M) + 0.000139813 R(M)` ms,
  table fit `R^2=0.9867`
- paper heuristic: 저장소의 실제 `select_chunks(..., impl="torch")`,
  `start=8 KiB`, `jump_cap=8 KiB`, multiscale end=256 KiB
- Top-`R`: 저장소의 실제 latency-blind `select_topk`; fixed-`R`에서
  importance 합을 최대화하는 기준선
- 비교는 paired comparison이다. 모든 방법이 같은 `v`와 `R`을 받는다.

기본 `R`들이 8의 배수라 paper greedy는 모든 시행에서 정확히 예산을
채웠다. 따라서 fixed-budget 결과에는 underfill에 따른 행 수 차이가 없다.
스크립트는 다른 설정에서 underfill이 생기면 greedy가 실제로 선택한 행
수와 같은 cardinality로 exact DP를 실행한다.

## Fixed-budget 결과: `R(M)=R`

아래의 주 지표는 두 mask를 같은 affine 목적
`I(M)/(aK(M)+cR)`로 평가한 값이다. 괄호는 입력별 개선률의 5th--95th
percentile이며 신뢰구간이 아니다.

| `R/N` | 평균 개선 | 중앙값 | 5th--95th percentile | lookup 재평가 평균 |
|---:|---:|---:|---:|---:|
| 12.5% | 2.78% | 2.38% | 0.00--6.74% | 2.78% |
| 25.0% | 1.49% | 1.17% | 0.00--4.14% | 1.49% |
| 37.5% | 1.10% | 0.81% | 0.00--2.53% | 1.13% |
| 50.0% | 2.59% | 0.67% | 0.00--31.02% | 2.85% |
| 62.5% | 1.55% | 0.51% | 0.00--2.00% | 1.74% |
| 75.0% | 25.28% | 27.28% | 0.75--29.16% | 23.58% |
| 87.5% | 15.25% | 22.73% | 0.00--26.50% | 12.77% |

“lookup 재평가”는 affine 목적에서 고른 exact mask를 원래 lookup table로
다시 평가한 민감도 검사다. 이는 fixed-`R` lookup 목적의 exact optimum을
뜻하지 않는다. 이번 표본에서는 lookup 재평가도 모든 시행에서 음수가
아니었고, 전체 평균 개선은 **6.62%**였다.

Exact 방법은 1,400개 fixed-`R` 문제 중 1,189개에서 strict improvement를
보였고 나머지는 greedy와 동률이었다. Dinkelbach outer iteration은 최대
5회, 평균 3.90회였다. 이번 synthetic 설정에서는 1,400개의 exact
fixed-`R` 해가 모두 한 run이었다. 이는 이 입력·latency 설정의 관측
결과이지 fixed-`R` 문제에 대한 일반적인 single-chunk 정리는 아니다.

### Top-`R`과 importance--latency trade-off

Top-`R`은 정의상 같은 `R`에서 가장 큰 importance를 보존했다. 그러나
선택 위치가 산재하면서 평균 28--65개의 run을 만들었고, paper greedy와
exact DP의 1--2개 run보다 latency가 훨씬 컸다.

| `R/N` | Top-`R` importance | Greedy importance | Exact importance | Top-`R` affine ms | Greedy affine ms | Exact affine ms |
|---:|---:|---:|---:|---:|---:|---:|
| 12.5% | 0.3078 | 0.1570 | 0.1613 | 0.3140 | 0.0155 | 0.0155 |
| 25.0% | 0.5151 | 0.2859 | 0.2902 | 0.5366 | 0.0199 | 0.0199 |
| 37.5% | 0.6740 | 0.4104 | 0.4140 | 0.6784 | 0.0245 | 0.0244 |
| 50.0% | 0.7962 | 0.5339 | 0.5359 | 0.7301 | 0.0296 | 0.0289 |
| 62.5% | 0.8872 | 0.6531 | 0.6550 | 0.6906 | 0.0338 | 0.0334 |
| 75.0% | 0.9503 | 0.7851 | 0.7735 | 0.5632 | 0.0481 | 0.0378 |
| 87.5% | 0.9875 | 0.8976 | 0.8917 | 0.3480 | 0.0491 | 0.0423 |

그 결과 Top-`R`의 affine `I/L`은 paper greedy보다 평균적으로 각
`R`에서 **84.4--94.1% 낮았다**. 이는 Top-`R`의 importance가 낮아서가
아니라 높은 fragmentation latency 때문이다. 반대로 exact DP는
importance를 최대화하지 않고 `I/L`을 최대화하므로, Top-`R`보다 낮은
importance를 받아들이면서 latency를 크게 줄인다.

[`results/latency_importance.png`](results/latency_importance.png)는 이
관계를 직접 표시한다. 각 점은 하나의 fixed `R/N`에서 200개 입력의
평균이며, 왼쪽은 fitted affine latency, 오른쪽은 lookup-table latency다.
가로축은 큰 latency 범위를 읽기 위해 log scale을 사용했다.

### 입력 분포 민감도

같은 200개 입력 수, seed, `R`, latency 설정으로 lognormal heavy-tail과
7-point moving average로 만든 locally correlated 중요도도 검사했다.

| 랜덤 중요도 분포 | 모든 `R`을 합친 평균 개선 | 모든 `R`을 합친 중앙값 | `R/N=75%` 평균 개선 |
|---|---:|---:|---:|
| Half-normal (주 실험) | 7.15% | 1.23% | 25.28% |
| Lognormal | 9.00% | 2.01% | 23.15% |
| Locally correlated half-normal | 6.26% | 0.52% | 25.16% |

따라서 정확한 수치는 분포에 의존하지만, “대부분의 구간은 작은 개선이고
greedy가 run을 나누는 큰 `R`에서 큰 개선”이라는 패턴은 세 분포에서
공통적이었다. 집계 원자료는
[`results/sensitivity_summary.json`](results/sensitivity_summary.json)에
기록했다.

## Upper-bound 결과: `1 <= R(M) <= R`

논문에 표시된 상한 문제에서는 best single interval이 lookup-table
`I/L`의 exact optimum이다. Greedy 대비 개선은 전체 평균 **7.48%**였다.
하지만 exact 해의 전체 평균 예산 사용률은 **96.62%**였고, 큰 예산에서는
다음처럼 더 적은 행을 선택했다.

| `R/N` | lookup `I/L` 평균 개선 | exact 해의 평균 `rows/R` |
|---:|---:|---:|
| 12.5% | 2.78% | 100.0% |
| 25.0% | 1.49% | 100.0% |
| 37.5% | 1.16% | 99.7% |
| 50.0% | 2.98% | 98.5% |
| 62.5% | 1.84% | 98.7% |
| 75.0% | 26.93% | 94.1% |
| 87.5% | 15.14% | 85.4% |

즉 upper-bound 결과의 일부는 중요도를 더 보존해서가 아니라 `I/L`에
불리한 추가 행을 선택하지 않는 데서 온다. 이 해는 수학적으로는 정확하지만
fixed sparsity나 동일 accuracy를 요구하는 실제 정책과는 다른 문제다.

## 해석과 한계

1. 새 exact 방법의 이점은 랜덤 입력에서도 관찰되지만, 효과 크기는
   `R`과 greedy가 만든 run 수에 강하게 의존한다.
2. `R/N=75%`의 큰 점프는 보편적인 상수 이득이 아니라 candidate-window
   greedy의 조합 효과와 chunk 고정비가 만난 결과다.
3. 이 실험은 synthetic importance에 대한 optimization-quality 실험이다.
   실제 VLM accuracy, 실제 activation 분포, selection runtime, SSD 실측
   latency를 검증하지 않는다.
4. Fixed-budget exact 보장은 fitted affine latency에 대한 것이다. 원래
   lookup-table fixed-`R` 목적에 대한 exact 보장은 아니다.

재현 코드는 [`run_experiment.py`](run_experiment.py), 원자료는
[`results/trials.csv`](results/trials.csv), 집계값은
[`results/summary.json`](results/summary.json), 그림은
[`results/comparison.png`](results/comparison.png)에 있다. 실행 시 먼저
작은 `N`의 모든 binary mask와 두 exact solver를 대조하는 self-check를
수행한다.
