# Experiment 09: Paper greedy versus merge-and-trim at realistic scales

## 연구 질문

Experiment 08에서 제안한 `Paper + merge + trim`은 Paper greedy가 달성한
importance를 그대로 하한으로 사용한다. 따라서 두 방법은 동일한 입력과 동일한
coverage target에서 직접 짝지어 비교할 수 있다.

이 실험은 다음 질문을 검증한다.

1. Experiment 08의 개선이 더 많은 random input에서도 재현되는가?
2. 개선이 실제 VLM down-projection에 해당하는 여러 channel count에서 유지되는가?
3. Latency 감소가 chunk 수, row 수, 계산시간과 어떤 관계를 갖는가?
4. Affine latency에서 얻은 개선이 논문이 공개한 lookup table에서도 유지되는가?

## 실험 설계

| Model family | N | Shape | FP16 row | Profitable affine gap |
|---|---:|---:|---:|---:|
| LLaVA-OneVision-0.5B | 4,864 | 4,864 x 896 | 1.75 KiB | 44 rows |
| NVILA-Lite-2B | 8,960 | 8,960 x 1,536 | 3 KiB | 26 rows |
| VILA1.5-8B | 14,336 | 14,336 x 4,096 | 8 KiB | 9 rows |
| LLaVA-OneVision-7B / LongVA-7B | 18,944 | 18,944 x 3,584 | 7 KiB | 11 rows |

- VLM CV: `1.07, 1.25, 1.44, 2.48, 3.30, 4.55`
- 별도 stress CV: `9.19`
- ordering: random, locally clustered, persistent hot-cold
- Paper row budget: `0.125N`부터 `0.875N`까지 7개
- 각 `(N, CV)`에서 독립 multiset 20개
- 전체 paired cases: `11,760`
- 주 분석 VLM cases: `10,080`
- 주 분석의 독립 multiset 수: `4 x 6 x 20 = 480`

한 multiset에서 세 ordering과 일곱 budget을 파생하므로 이 21개 관측값은 서로
독립이 아니다. 95% confidence interval은 `(N, trial, CV)` multiset을 하나의
cluster로 묶은 20,000회 percentile bootstrap으로 계산했다. Ordering과 budget은
항상 같은 cluster 안에서 함께 resampling했다.

모든 비교는 paired comparison이다. Paper mask를 먼저 계산하고

$$
B=I(M_{\mathrm{Paper}})
$$

를 refinement의 importance lower bound로 사용했다. 모든 refined mask는

$$
I(M_{\mathrm{refined}})\ge B
$$

를 만족한다.

## 평가한 두 latency model

주 평가는 Experiment 07과 08에서 사용한 affine model이다.

$$
L_{\mathrm{aff}}(M)=aK(M)+cR(M).
$$

Refinement의 merge threshold와 trimming score가 이 model을 직접 사용하므로,
affine latency는 Paper보다 커지지 않는다는 알고리즘적 보장이 있다.

민감도 분석으로 같은 최종 mask를 공개된 Orin AGX lookup table에도 넣었다.

$$
L_{\mathrm{lookup}}(M)=
\sum_{C\in\mathcal C(M)}T[|C|].
$$

Lookup 결과는 알고리즘이 직접 최적화한 값이 아니며 non-worsening 보장이 없다.

각 model의 primary metric은

$$
\boxed{
Q(M)=\frac{L(M_{\mathrm{refined}})}{L(M_{\mathrm{Paper}})},
\qquad
\operatorname{Saving}(M)=100(1-Q(M)).
}
$$

이다. `Q=0.9`는 refined latency가 Paper의 90%, 즉 10% 감소했다는 뜻이다.

## Affine latency 결과

| N | Refined / Paper | Cluster-bootstrap 95% CI | 평균 절감 | Strict win | Non-worse |
|---:|---:|---:|---:|---:|---:|
| 4,864 | **83.97%** | 83.41--84.50% | **16.03%** | 92.46% | 100% |
| 8,960 | **89.43%** | 89.28--89.58% | **10.57%** | 97.74% | 100% |
| 14,336 | **91.04%** | 90.94--91.14% | **8.96%** | 99.96% | 100% |
| 18,944 | **91.91%** | 91.75--92.06% | **8.09%** | 99.88% | 100% |
| 전체 | **89.09%** | 88.77--89.40% | **10.91%** | 97.51% | 100% |

전체 10,080 cases에서 refined affine latency는 평균적으로 Paper의 `89.09%`였다.
절감률 중앙값은 `10.14%`, 5--95 percentile은 `2.29--24.65%`였다. 최대 절감은
`52.66%`였다.

모든 사례에서 coverage를 지키면서 affine latency가 Paper 이하였으므로 구현의
non-worsening invariant도 대규모 입력에서 확인됐다. Strict win이 아닌 사례는
merge 또는 trim할 수 있는 유효 후보가 없었던 경우다.

N이 커질수록 평균 절감률은 감소했다. Row 크기가 커지면 $c$가 증가하고
profitable gap threshold $a/c$가 44 rows에서 9--11 rows로 줄어들기 때문이다.
합칠 수 있는 gap의 수와 길이가 함께 감소한다.

## Ordering, CV, budget 효과

| Ordering | Affine 절감 | Lookup 절감 | Chunk 감소 | Final R / Paper R |
|---|---:|---:|---:|---:|
| Random | 10.44% | 1.09% | 39.73% | 102.04% |
| Locally clustered | 7.37% | 1.79% | 30.06% | 100.49% |
| Persistent hot-cold | **14.92%** | **2.25%** | **72.13%** | 99.44% |

Hot-cold ordering에서 nearby hot regions 사이의 gap을 합치는 효과가 가장 컸다.
Random ordering은 더 많은 row를 읽는 대신 chunk를 줄였고, locally clustered
ordering은 큰 N에서 이미 긴 local run을 만들기 때문에 추가 개선폭이 작았다.

CV와 budget별 결과는 `saving_heatmaps.pdf`에 기록했다. `N=4,864`에서는 높은
CV와 큰 budget에서 개선이 대체로 커졌다. 더 큰 N의 local/hot-cold ordering은
중간 budget에서 peak를 보이고, 높은 budget에서는 합칠 수 있는 짧은 gap이 줄어
개선폭이 다시 감소했다.

## 개선 메커니즘

| N | Paper 평균 K | Refined 평균 K | Chunk 감소 | Final R / Paper R |
|---:|---:|---:|---:|---:|
| 4,864 | 24.99 | 9.66 | 54.95% | 100.61% |
| 8,960 | 57.21 | 27.82 | 50.59% | 101.40% |
| 14,336 | 207.30 | 110.67 | 44.20% | 100.48% |
| 18,944 | 241.05 | 140.73 | 39.49% | 100.12% |
| 전체 | 132.64 | 72.22 | 47.31% | 100.65% |

전체적으로 row 수는 평균 `0.65%` 늘었지만 chunk 수는 `47.31%` 감소했다. Affine
latency 개선은 row를 적게 읽어서 생긴 것이 아니라, 거의 같은 row 수를 더 적은
chunk로 배치해 startup cost를 줄인 결과다.

Merge-only와 full refinement의 차이도 분명하다.

| N | Merge-only affine 절감 | Merge + trim affine 절감 |
|---:|---:|---:|
| 4,864 | 10.69% | **16.03%** |
| 8,960 | 5.64% | **10.57%** |
| 14,336 | 3.82% | **8.96%** |
| 18,944 | 3.95% | **8.09%** |

Gap merge가 chunk-open cost를 없애고, boundary trim이 merge로 추가된 row와 낮은
importance 경계를 다시 제거한다. 두 단계 모두 성능에 실질적으로 기여한다.

Refinement가 Paper target을 초과한 importance는 전체 importance가 1로 정규화된
상태에서 평균 `0.000075`, 즉 `0.0075 percentage points`였다. 따라서 성능 차이가
큰 coverage overshoot에서 나온 것은 아니다.

## 선택 알고리즘 실행시간

| N | Paper median | Postprocess median | 전체 median | Per-case overhead median |
|---:|---:|---:|---:|---:|
| 4,864 | 7.16 ms | 1.40 ms | 8.51 ms | 18.99% |
| 8,960 | 14.38 ms | 2.55 ms | 17.01 ms | 16.89% |
| 14,336 | 13.31 ms | 4.35 ms | 17.49 ms | 30.14% |
| 18,944 | 20.07 ms | 5.44 ms | 25.28 ms | 25.54% |

후처리는 가장 큰 N에서도 median `5.44 ms`였다. Paper와 후처리를 합친 전체
선택시간은 `8.51--25.28 ms` 범위였다. 이 값은 단일 프로세스 CPU reference이며
device inference latency가 아니다.

## Lookup-table 민감도

| N | Refined / Paper | Cluster-bootstrap 95% CI | 평균 절감 | Strict win | Non-worse |
|---:|---:|---:|---:|---:|---:|
| 4,864 | 96.05% | 95.62--96.47% | 3.95% | 80.32% | 87.86% |
| 8,960 | 99.27% | 99.16--99.37% | 0.73% | 67.70% | 69.96% |
| 14,336 | 99.15% | 99.06--99.23% | 0.85% | 85.00% | 85.04% |
| 18,944 | 98.69% | 98.58--98.80% | 1.31% | 91.83% | 91.94% |
| 전체 | **98.29%** | 98.12--98.45% | **1.71%** | 81.21% | 83.70% |

Lookup table에서도 평균적으로는 네 N 모두 개선됐지만, affine model보다 효과가
훨씬 작았다. 전체 사례의 `16.30%`에서는 lookup latency가 증가했으며, 최악의
증가는 `5.50%`였다.

Merge-only mask는 lookup 기준으로 평균 `2.11--4.35%` 악화됐다. Boundary trim이
추가 row를 제거하면서 최종 평균을 다시 Paper 아래로 내렸다. 따라서 lookup
model에서는 trim이 단순 보조 단계가 아니라 merge의 비용을 상쇄하는 핵심 단계다.

이 차이는 latency model의 외삽 방식과 관련된다. 공개 AGX table은 제한된 chunk
크기까지만 측정되어 있으며, 범위 밖에서는 마지막 측정점을 크기 비율로 scale한다.
이 외삽은 긴 chunk에 대해 affine startup cost $a$를 명시적으로 유지하지 않는다.
따라서 affine rule로 여러 chunk를 합쳤을 때 얻는 $a$ 절감이 lookup 평가에서는
사라지고, 추가로 읽는 gap row 비용만 커질 수 있다.

결과적으로 Experiment 08의 큰 개선은 **affine objective에 대해서는 강하게
재현되지만 실제 flash latency 개선으로 그대로 해석할 수 없다.** 물리적 성능을
주장하려면 다음 중 하나가 필요하다.

1. 실제 target device에서 refinement mask의 mixed-size I/O를 측정한다.
2. Merge 조건을
   $T[\ell_1+g+\ell_2]<T[\ell_1]+T[\ell_2]$로 바꾸고 trimming도 lookup의 실제
   marginal latency로 평가한다.

## CV 9.19 stress test

| N | Affine 절감 | Lookup 절감 | Lookup non-worse |
|---:|---:|---:|---:|
| 4,864 | 24.99% | 13.29% | 99.29% |
| 8,960 | 13.13% | 3.11% | 84.76% |
| 14,336 | 10.38% | 2.28% | 85.48% |
| 18,944 | 10.33% | 2.79% | 94.76% |

높은 CV에서는 중요도가 강하게 집중되어 trimming할 수 있는 저-importance 경계가
많아 두 latency model 모두 평균 개선폭이 커졌다.

## 결론

Paper greedy에 merge-and-trim을 붙이는 방법은 계산비가 작은 안정적인 affine
refinement다. 네 realistic N, 10,080 VLM paired cases에서 coverage를 유지하고
affine latency를 평균 `10.91%` 줄였으며 악화 사례가 없었다. 계산비 증가는
median 기준 약 `17--30%`였다.

동시에 이 실험은 현재 결과의 가장 중요한 한계를 드러냈다. 공개 lookup model에서
평균 개선은 `1.71%`로 줄고 사례의 `16.30%`가 악화된다. 따라서 알고리즘의 다음
버전은 affine gap threshold를 고정해서 쓰기보다 실제 $T[\ell]$의 merge delta를
직접 계산해야 한다.

## 결과 파일

- `comparison_overview.pdf`: N별 affine/lookup latency, K, runtime
- `latency_ratio_ecdf.pdf`: ordering별 affine latency ratio 분포
- `lookup_ratio_ecdf.pdf`: ordering별 lookup latency ratio 분포
- `saving_heatmaps.pdf`: N, ordering, CV, budget별 affine 절감률
- `paired_trials.csv`: 11,760개 paired raw observations
- `grouped_summary.csv`: N/order/CV/budget별 집계
- `summary.json`: cluster-bootstrap CI와 전체 요약
- `n_<N>/paired_trials.csv`, `n_<N>/summary.json`: N별 결과
