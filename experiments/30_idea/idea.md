좋은 아이디어입니다. 지금까지의 corridor보다 Paper를 이길 가능성이 훨씬 높습니다.

방금 Experiment 28의 384개 실제 case로 간단한 prototype을 검산했습니다. `b=ceil(s/8)`행을 한 칸으로 만들고, 8칸짜리 sliding chunk만 고려한 결과입니다.

| Shape | lookup I/L vs Paper | lookup latency 절감 | 중요도 변화 | 승률 |
|---|---:|---:|---:|---:|
| 896×896 | +6.99% | 9.53% | -3.47% | 93.8% |
| 896×4864 | +4.74% | 6.32% | -1.92% | 100% |
| 4864×896 | +6.79% | 6.96% | -0.66% | 100% |
| 896×128 | -1.20% | 0% | -1.20% | 4.2% |

지원되는 세 shape만 보면:

- 평균 lookup \(I/L\): 약 `+6.17%`
- 평균 lookup latency: 약 `-7.60%`
- Paper 초과율: `97.9%`
- 평균 chunk 수: 약 `3.88개`

`896×128`은 한 행이 0.25 KiB라서 \(s=960\)행인데 전체 \(N=896\)보다 큽니다. 따라서 이 shape에는 saturation tile이 존재하지 않습니다. 여기만 Paper를 사용하면 됩니다.

수학적으로도 매우 간단해집니다. cell importance를 \(u_j\), 8-cell window의 importance를 \(w_j\)라 하면, 정확히 \(K\)개의 겹치지 않는 chunk를 고르는 문제는

\[
D[k,j]
=
\max\left(
D[k,j-1],
D[k-1,j-8]+w_j
\right)
\]

입니다.

복잡도는

\[
O(BK),\qquad B\approx N/b,\quad K\approx R/(8b)
\]

입니다. 실제 최대 크기는 대략 수천 state밖에 안 됩니다. Experiment 28의 corridor는 수십만 state를 처리했습니다.

정확한 \(R\)은 다음 두 후보만 비교하면 됩니다.

- \(\lfloor R/L\rfloor\)개의 saturation tile을 고르고 남은 행을 선택된 run의 경계로 확장
- \(\lceil R/L\rceil\)개의 tile을 고르고 초과분 \(<L\)만 endpoint trim

여기서 \(L=8b\)입니다. 둘 다 Paper mask나 \(I(M_{\text{paper}})\)를 필요로 하지 않습니다.

또한 이것은 일반 graph algorithm도 필요 없습니다.

- cell sum: 한 번의 reduction
- 8-cell sliding sum
- 작은 고정 DP
- residual trim/expand 최대 \(s\)행
- 최종 mask 생성

전부 하나의 CUDA kernel 또는 작은 native kernel로 합칠 수 있습니다. Paper처럼 여러 크기의 후보를 만들고 전체 정렬할 필요도 없습니다.

결론은:

> 이 아이디어는 품질 면에서 이미 Paper보다 유망하고, 계산량도 Paper보다 작아질 구조다. 특히 `896×128은 Paper`, 나머지는 saturation-8 tile을 사용하는 shape dispatch가 가장 유력하다.

아직 위 결과는 lookup-model 측정입니다. 다음 실험에서는 native 구현의 selector latency와 실제 O_DIRECT read를 재야 합니다. 현재 수치대로 실제 read에서도 5–8%가 유지되고 selector가 Paper보다 빨라지면, 혼합 정책이 아니라 세 주요 shape에서 직접 Paper를 이길 가능성이 충분합니다.