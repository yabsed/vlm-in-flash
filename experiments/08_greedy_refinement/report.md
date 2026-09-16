# Experiment 08: Paper greedy의 coverage-preserving refinement

## 목적

Experiment 07의 `N=4,864` 비교에 다음 알고리즘 하나를 추가했다.

`Paper + merge + trim`은 Paper greedy가 달성한 importance를 하한으로 두고,
짧은 gap을 합친 뒤 chunk 경계의 낮은-importance row를 제거한다. 목표는 Paper와
비슷한 선택 비용으로 다중 chunk coverage 문제의 exact solution에 가까워지는
것이다.

Experiment 07의 DP-Cover, DP-R, Top-R 결과는 CSV에서 그대로 읽었다. Exact
DP-Cover table은 다시 만들지 않았다. Experiment 07이 mask를 저장하지 않았기
때문에 동일한 seed로 입력과 Paper mask만 재생했으며, 모든 Paper importance,
R, K, latency가 저장된 값과 일치하는지 검사했다.

## 알고리즘

Paper mask의 importance를 coverage target `B`로 설정한다.

1. 두 chunk 사이의 gap 길이가 `g`일 때 `c*g < a`이면 gap을 모두 선택한다.
   latency 변화는 `c*g-a < 0`이고 importance는 nonnegative이므로 이 병합은
   항상 해를 개선한다. Experiment 07의 latency model에서는 `g <= 44`인
   gap이 해당한다.
2. 병합 후 생긴 importance surplus 안에서 chunk의 양끝 row를 제거한다.
   후보는 `잃는 importance / 절약하는 latency`가 작은 순서로 처리한다.
   singleton을 제거할 때는 row 비용 `c`와 chunk-open 비용 `a`를 함께
   절약한다.

병합은 `O(N)`, heap 기반 trimming은 `O(N log N)`이다. 추가 메모리는 `O(N)`이다.
현재 Paper 구현도 후보 점수 정렬을 사용하므로 전체 점근 복잡도는 Paper와 같은
`O(N log N)`이다.

이 방법은 fixed-R을 보존하지 않는다. Gap을 읽으면서 R이 증가할 수 있으며,
Paper가 달성한 importance 이상을 유지하면서 latency를 최소화하는 coverage
문제를 푼다.

## 설정

- `N=4,864`, shape `4,864 x 896`, FP16 row `1.75 KiB`
- Experiment 07과 동일한 Orin AGX affine latency `L=aK+cR`
- VLM CV `1.07, 1.25, 1.44, 2.48, 3.30, 4.55`
- random, locally clustered, persistent hot-cold ordering
- CV별 3개 multiset, 7개 Paper row budget
- VLM 결과 378개; CV `9.19` stress case는 CSV에 별도로 유지

## 결과

Exact gap은 각 방법의 latency를 Paper importance target에 대한 DP-Cover 최소
latency로 나눈 값에서 1을 뺀 것이다.

| 방법 | Paper 대비 latency 절감 | Exact 대비 초과 latency |
|---|---:|---:|
| Paper greedy | 0% | 30.01% |
| gap merge까지만 적용 | 11.00% | 14.97% |
| Paper + merge + trim | **16.44%** | **7.59%** |
| DP-Cover exact | 22.48% | 0% |

전체 VLM 사례의 `93.39%`에서 refined method가 Paper보다 엄격하게 빨랐고,
나머지에서도 같았다. 나빠진 사례는 없었다. 절감률 중앙값은 `16.76%`,
95-percentile은 `31.28%`, 최대는 `48.39%`였다.

Paper와 exact 사이에 존재했던 초과 latency 가운데 사례별 평균 `66.79%`,
중앙값 `72.98%`를 회수했다. Refined solution이 exact target latency와 완전히
같아진 비율은 `5.56%`였다. 따라서 이 후처리는 gap의 대부분을 싸게 줄이지만
exact algorithm은 아니다.

Ordering별 결과는 다음과 같다.

| Ordering | Paper 대비 절감 | Exact gap | Strict win | 평균 K |
|---|---:|---:|---:|---:|
| Random | 10.64% | 11.38% | 85.71% | 13.26 |
| Locally clustered | 15.42% | 7.63% | 94.44% | 13.95 |
| Persistent hot-cold | **23.27%** | **3.76%** | 100% | 2.25 |

Hot-cold ordering에는 가까운 유용 구간 사이의 작은 gap이 많기 때문에 병합과
trimming이 특히 잘 작동했다. Random ordering에서는 여전히 exact와 평균
11.38% 차이가 남았다.

## Chunk 구조

| 방법 | 평균 K | 중앙값 K |
|---|---:|---:|
| DP-Cover at Paper target | 4.72 | 3 |
| Paper greedy | 25.24 | 24 |
| Paper + merge + trim | **9.82** | **10** |
| DP-R exact | 2.98 | 3 |
| Top-R | 557.75 | 535 |

새 알고리즘은 Paper의 chunk 수를 평균 `55.01%` 줄였다. Exact의 평균 4.72개에는
아직 미치지 못하지만, Paper의 과도한 fragmentation이 성능 차이의 큰 원인이었음이
확인된다.

## 선택 알고리즘 실행시간

같은 Experiment 08 replay에서 측정한 평균 CPU wall time은 다음과 같다.

| 단계 | 평균 |
|---|---:|
| Paper greedy | 7.17 ms |
| merge + trim 후처리 | 1.39 ms |
| Paper + 전체 후처리 | **8.56 ms** |

후처리는 Paper 실행시간의 약 19%를 추가했다. Experiment 07의 DP-R 평균
`321 ms`, DP-Cover table build 평균 `14.13 s`와 비교하면 online selection에
사용할 수 있는 범위다. 이 시간은 단일 프로세스 CPU reference이며 구현 간
절대적인 하드웨어 성능 비교로 해석하면 안 된다.

## 해석

Experiment 07에서 Paper는 exact보다 평균 30.01% 비쌌다. 그 차이의 상당 부분은
더 복잡한 전역 DP가 필요한 효과가 아니라, 짧은 gap을 남기고 필요 이상으로 많은
chunk를 여는 데서 발생했다. Latency model이 제공하는 지배 규칙과 coverage
surplus만 활용해 exact gap을 7.59%까지 줄였다.

남은 차이는 boundary row를 개별 비율로 제거하는 greedy trimming의 한계다.
여러 row 또는 chunk 하나를 묶어서 제거해야 이득인 경우와, 긴 gap을 합친 뒤 다른
영역을 크게 줄여야 하는 교환은 현재 알고리즘이 찾지 못한다.

## 결과 파일

- `importance_latency_frontiers.pdf`
- `r_importance.pdf`
- `r_latency.pdf`
- `coverage_optimality_ratio.pdf`
- `chunk_count_histograms.pdf`
- `fixed_r_trials.csv`, `coverage_trials.csv`, `input_trials.csv`
- `summary.json`
