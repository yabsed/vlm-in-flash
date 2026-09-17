# Experiment 16 보고서: Exact coverage와 supported optimizer

## 질문

두 가지 남은 최적성 차이를 분리해 측정한다.

1. Two-line 모델에서 adaptive \(\lambda\) 탐색이 놓치는 unsupported coverage
   optimum은 실제로 얼마나 큰가?
2. Two-line surrogate 대신 released lookup을 직접 최적화하면 released latency가
   얼마나 개선되는가?

Paper가 달성한 importance를 target \(Q\)로 두고 다음 다섯 방법을 비교했다.

1. **Exact coverage**: two-line 제약문제의 전역 최적해
2. **Lookup supported**: released lookup scalarization의 supported frontier
3. **Two-line supported**: two-line scalarization의 supported frontier
4. **Quant**: two-line 목적함수의 `q=131,072` 고정 multiplier grid
5. **Paper greedy**: Neuron Chunking의 fixed-\(R\) greedy

## Exact constrained DP

Two-line 모델은

\[
T(\ell)=c_2\ell+\delta(s-\ell)_+,
\qquad \delta=c_2-c_1>0
\]

이므로 전체 mask 비용은

\[
\boxed{
L_{\mathrm{2line}}(M)=c_2R(M)+\delta H(M)
},
\qquad
H(M)=\sum_{C\in\operatorname{runs}(M)}(s-|C|)_+.
\]

실험의 비정수 breakpoint는

\[
s=134.857142\ldots,
\qquad m=\lceil s\rceil=135
\]

이다. 현재 active run 상태를

\[
k\in\{0,1,\ldots,m\}
\]

로 저장하며, \(k=m\)은 길이가 135 이상임을 뜻한다. 각 상태에는
\((L,I)\) Pareto label을 모두 보존한다. 이는 \((R,H)\)를 저장하는 DP와
동등하지만, 최종 목적함수 값 \(L=c_2R+\delta H\)가 이미 정해졌으므로
지배성 검사에는 scalar cost \(L\)만 있으면 충분하다.

같은 \((i,k)\)에서 label \(A\)가

\[
L_A\le L_B,
\qquad I_A\ge I_B
\]

를 만족하면 \(B\)만 제거한다. 두 prefix는 미래 전이가 완전히 같으므로 이
제거는 최적해를 잃지 않는다. Importance binning, label cap, beam search,
epsilon-dominance는 사용하지 않았다.

추가로 다음 exact-safe pruning만 사용했다.

- 남은 importance를 모두 더해도 \(Q\)에 못 미치는 label 제거
- 알려진 feasible upper bound보다 이미 비싼 label 제거
- suffix fixed-\(\lambda\) DP가 주는 Lagrangian lower bound로도 upper bound를
  이길 수 없는 label 제거
- 동일 상태의 cost--importance 지배 label 제거

Run 비용은 행을 추가할 때 exact marginal cost로 모두 청구하므로 마지막 active
run을 별도로 닫을 필요가 없다. 이는 끝에 unselected sentinel을 붙이는 점화식과
동일하다. 64행마다 lineage checkpoint를 저장하여 전역 최적 mask도 복원했다.

작은 \(N=9\)에서 모든 \(2^N\) mask를 열거하여 여러 target의 최적 비용과 복원
mask를 비교했고 모두 일치했다. Full experiment에서도 모든 저장 mask에 대해
coverage, two-line 비용, row--shortfall 항등식을 다시 계산했다.

## Released lookup supported DP

Released evaluator는 255 KiB까지 실제 table lookup과 선형보간을 사용하고,
그 이후에는 마지막 endpoint를 비례 확장한다.

\[
T_{\mathrm{released}}(z)=
\begin{cases}
T_{\mathrm{table}}(z),&z\le255\text{ KiB},\\
T_{\mathrm{table}}(255)z/255,&z>255\text{ KiB}.
\end{cases}
\]

한 row가 1.75 KiB이므로 lookup run state는

\[
m_{\mathrm{lookup}}=\left\lceil255/1.75\right\rceil=146
\]

에서 cap한다. 고정 \(\lambda\)의

\[
\max_M\{\lambda I(M)-L_{\mathrm{released}}(M)\}
\]

은 실제 lookup 증분비용을 사용하는 \(O(Nm_{\mathrm{lookup}})\) DP로 정확히
푼다. Adaptive intersection search는 이 목적함수의 strongly-supported
frontier 전체를 찾지만, Lookup supported 역시 exact constrained-coverage
oracle은 아니다.

## 설정

- `N=4,864`, matrix `4864x896`
- 6 CV, CV당 3 trials
- random, locally clustered, persistent hot-cold ordering
- 54 spatial inputs, 378 Paper-matched targets
- Paper budgets: `R/N=0.125, 0.25, ..., 0.875`
- Two-line breakpoint: `s=134.857 rows`, capped state `m=135`
- Lookup proportional-tail state: `m_lookup=146`
- Quant: `q=131,072`

## 주 결과: Released evaluator

모든 headline latency와 I--L/L--R 그래프는 Experiment 15와 같은 released
lookup evaluator를 사용한다.

| 방법 | Paper 대비 평균 released 절감 | 중앙값 | 승률 |
| --- | ---: | ---: | ---: |
| **Exact coverage** | **9.475%** | **8.469%** | **100.00%** |
| Lookup supported | 8.563% | 7.639% | 92.59% |
| Two-line supported | 7.328% | 6.687% | 87.83% |
| Quant (`q=131,072`) | 7.322% | 6.658% | 87.83% |

Exact coverage는 released evaluator를 직접 최적화하지 않는데도 평균 결과가 가장
좋았다. 주된 이유는 target을 거의 정확히 맞추기 때문이다. 평균 importance
overshoot는 다음과 같다.

| 방법 | 평균 importance overshoot |
| --- | ---: |
| Exact coverage | **0.000031** |
| Lookup supported | 0.006031 |
| Two-line supported | 0.007308 |
| Quant | 0.007328 |

Exact coverage의 Paper 대비 released 절감 최솟값도 `0.417%`여서 378개 모든
target에서 Paper보다 빨랐다.

## Unsupported optimality gap

핵심 비교는 같은 two-line 목적함수 아래의

\[
L_{\mathrm{exact}}(Q)
\le L_{\mathrm{supported}}(Q)
\le L_{\mathrm{Quant}}(Q)
\]

이다. 다음 비율을 target별로 계산했다.

\[
\operatorname{Gap}_{\mathrm{unsupported}}
=\frac{L_{\mathrm{supported}}-L_{\mathrm{exact}}}
{L_{\mathrm{supported}}}.
\]

| 통계 | Unsupported gap |
| --- | ---: |
| 평균 | **2.143%** |
| 중앙값 | 1.181% |
| 5 percentile | 0.013% |
| 95 percentile | 7.365% |
| 최대 | **18.306%** |
| 양수인 target | **369/378 (97.62%)** |

따라서 `q=131,072` Quant가 supported frontier를 거의 완전히 재현한다는 이전
결과는 맞지만, 실제 coverage optimum에 사실상 도달한다는 결론은 틀렸다.
남은 차이는 multiplier grid 해상도가 아니라 unsupported point였다.

Budget별 gap은 낮은 coverage에서 가장 크다.

| Paper R/N | 평균 unsupported gap | Exact의 Paper 대비 released 절감 |
| ---: | ---: | ---: |
| 0.125 | **5.303%** | 6.449% |
| 0.250 | 3.034% | 8.042% |
| 0.375 | 2.063% | 8.246% |
| 0.500 | 2.063% | 9.234% |
| 0.625 | 1.007% | 10.137% |
| 0.750 | 0.922% | 11.568% |
| 0.875 | 0.606% | 12.648% |

Ordering별 평균 unsupported gap은 Random `2.613%`, Persistent hot-cold
`2.354%`, Locally clustered `1.461%`였다.

## Exact coverage와 Lookup supported

두 방법은 서로 다른 목적함수를 푼다.

- Exact coverage: two-line 모델의 constrained global optimum
- Lookup supported: released lookup의 supported scalarized optimum

Released evaluator에서 Exact coverage는 Lookup supported보다 평균 `0.899%`
빨랐고 `217/378` target에서 이겼다. 반대로 `161/378`에서는 Lookup supported가
더 빨랐다. 이는 모순이 아니다. Exact coverage는 released lookup의 전역해가
아니고, Lookup supported는 coverage-constrained 전역해가 아니다.

다음 단계의 완전한 lookup oracle은 현재 exact Pareto-label DP의 run increments를
released lookup increments로 교체하면 만들 수 있다.

## I--L, I--R, L--R 곡선

- [`importance_latency.pdf`](results/importance_latency.pdf)
- [`importance_rows.pdf`](results/importance_rows.pdf)
- [`latency_rows.pdf`](results/latency_rows.pdf)

파란색 별표가 Exact coverage다. Exact curve는 supported 방법보다 importance
target에 훨씬 가깝게 위치하며, 그 결과 평균적으로 더 적은 행을 선택한다.

| 구조 지표 | Exact | Lookup supported | Two-line supported | Quant | Paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| 평균 rows | **2,337.21** | 2,347.62 | 2,374.47 | 2,374.65 | 2,430.00 |
| 평균 chunks | **15.90** | 18.30 | 16.24 | 16.24 | 25.24 |
| 평균 shortfall rows | 193.19 | 347.85 | 192.73 | 192.72 | 1,270.04 |
| pooled chunk 중앙값 | 144 | 135 | 143 | 143 | 84 |
| two-line saturation 이상 chunk | 67.10% | 58.05% | 68.16% | 68.16% | 18.46% |

Exact는 Two-line supported보다 평균 37.26행을 덜 선택하면서도 chunk 수와
shortfall 구조는 비슷하다. 주된 개선은 더 낮은 target overshoot를 허용하는
unsupported 조합을 선택하는 데서 나온다.

## 청크 길이 히스토그램

[`chunk_length_histogram.pdf`](results/chunk_length_histogram.pdf)는 다섯 방법의
진짜 maximal run 분포다. 다음 두 기준선을 함께 표시했다.

- two-line breakpoint: `s=134.857 rows`
- released proportional-tail cap: `m=146 rows`

Exact coverage는 파란색 분포다. Two-line supported와 비슷하게 saturation 부근에
집중되지만 동일한 분포는 아니다. Lookup supported는 실제 table의 불규칙한
증분비용 때문에 135행과 144행 부근의 다른 구조를 만든다.

## Tail 민감도

각 값은 해당 mask를 각 evaluator로 다시 계산한 Paper 대비 평균 절감이다.

| mask selector | Two-line | Released | Anchored tail | Block split |
| --- | ---: | ---: | ---: | ---: |
| Exact coverage | **12.099%** | **9.475%** | **9.765%** | 4.109% |
| Lookup supported | 9.640% | 8.563% | 8.245% | **6.323%** |
| Two-line supported | 10.034% | 7.328% | 7.549% | 1.912% |
| Quant | 10.029% | 7.322% | 7.545% | 1.907% |

Two-line evaluator에서는 정의상 Exact coverage가 가장 좋다. Released와 anchored
tail에서도 평균적으로 가장 좋았지만, block-splitting evaluator에서는 Lookup
supported가 더 견고했다. Selector와 evaluator가 다르면 순위 보장은 없다.

## 계산 비용

Exact Pareto-label pass의 target별 runtime은 다음과 같다.

| 통계 | Runtime | Peak labels |
| --- | ---: | ---: |
| 평균 | 9.666 s | 143,082 |
| 중앙값 | 2.370 s | 44,493 |
| 95 percentile | 38.802 s | 587,048 |
| 최대 | 199.485 s | 1,340,660 |

입력 하나의 7개 target에 대한 solver CPU time 합은 평균 `67.66 s`였다. 실험은
4개 worker로 병렬 실행했다. 이 수치는 supported optimizer의 입력당 약
`44 ms`와 큰 차이가 난다. Exact DP는 offline oracle이고, Quant 및 supported
방법은 훨씬 가벼운 approximation이다.

중요하게도 runtime을 줄이기 위한 label cap이나 근사 dominance는 사용하지
않았다. Frontier 크기 차이가 target별 runtime 분산을 직접 만든다.

## 제한

- Exact coverage의 전역 최적성은 **two-line latency 모델 내부**의 주장이다.
  실제 하드웨어 latency나 released lookup에 대한 전역 최적성은 아니다.
- Released lookup과 block-splitting tail에서의 수치는 exact mask를 재평가한
  sensitivity 결과다.
- 실험은 Experiment 13의 54개 spatial input과 378개 target을 사용한다.
- Activation importance는 downstream task accuracy의 surrogate다.
- 계산은 float64와 보수적인 numerical margin을 사용한다. 작은 문제에서는
  exhaustive enumeration과 일치했다.

## 결론

\((R,H)\) 구조를 보존하는 direct constrained DP로 two-line coverage 문제의
진짜 전역 최적해를 구했다. 결과는 명확하다.

> Quant와 adaptive \(\lambda\) 탐색의 차이는 거의 없지만, supported frontier와
> 실제 coverage optimum의 차이는 평균 2.143%이며 97.62%의 target에서 존재한다.

따라서 Experiment 16의 최종 관계는

\[
\boxed{
L_{\mathrm{exact\ coverage}}
< L_{\mathrm{Two\mbox{-}line\ supported}}
\approx L_{\mathrm{Quant}}
}
\]

이다. Exact oracle은 Paper와 Quant를 실제 전역해에 대해 평가할 수 있게 했고,
unsupported point가 단순한 이론적 가능성이 아니라 측정 가능한 개선 원천임을
확인했다.
