# Experiment 41: 노트북 Cell-1 lookup vs 2-line

실험 39의 실제 노트북 측정 경로를 그대로 사용하고, Cell-1 selector가 청크 비용을 평가할 때만 SSD lookup table과 아래 2-line 근사를 교체했다.

```text
T(r) = a + c1*r   (r <= s)
     = c2*r       (r > s)
```

노트북 SSD profile fit은 `a=0.012833 ms`, `c1=0.000120629 ms/KiB`, `c2=0.000174100 ms/KiB`, `s=240 KiB`이다. 적합도는 `R²=0.98541`, `MAPE=3.29%`다.

저장된 laptop lookup은 profiler가 throughput saturation에서 trim했기 때문에 1–240 KiB만 포함한다. 따라서 pre-s branch는 240개 실측점에 적합했지만, post-s의 `c2*r`은 continuity가 정하는 선형 외삽이다. 기존 lookup도 240 KiB 밖에서는 마지막 점을 크기에 비례해 외삽한다.

## 동일 projection error

양수 gain은 2-line이 lookup보다 빠르다는 뜻이다. R=5% grid 사이를 선형 보간했으므로 작은 차이는 보간 오차로 해석해야 한다.

| error | score | lookup | 2-line | gain | selector Δ | SSD Δ | chunk Δ |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.50 | ABS | 0.374 ms | 0.372 ms | 0.434% | -0.0012 ms | -0.0004 ms | 0.01 |
| 0.50 | X2 | 0.364 ms | 0.364 ms | -0.107% | 0.0012 ms | -0.0009 ms | 0.01 |
| 0.45 | ABS | 0.390 ms | 0.390 ms | 0.220% | -0.0009 ms | 0.0001 ms | -0.00 |
| 0.45 | X2 | 0.382 ms | 0.381 ms | 0.358% | 0.0008 ms | -0.0020 ms | 0.02 |
| 0.42 | ABS | 0.403 ms | 0.404 ms | -0.267% | 0.0002 ms | 0.0007 ms | -0.03 |
| 0.42 | X2 | 0.396 ms | 0.397 ms | -0.214% | 0.0011 ms | -0.0002 ms | -0.01 |
| 0.40 | ABS | 0.411 ms | 0.412 ms | -0.196% | 0.0003 ms | 0.0004 ms | -0.02 |
| 0.40 | X2 | 0.404 ms | 0.407 ms | -0.602% | 0.0023 ms | 0.0002 ms | -0.01 |
| 0.35 | ABS | 0.436 ms | 0.436 ms | 0.010% | -0.0005 ms | 0.0004 ms | 0.00 |
| 0.35 | X2 | 0.428 ms | 0.430 ms | -0.448% | 0.0016 ms | 0.0003 ms | 0.00 |
| 0.30 | ABS | 0.458 ms | 0.459 ms | -0.162% | 0.0002 ms | 0.0004 ms | 0.01 |
| 0.30 | X2 | 0.458 ms | 0.458 ms | -0.006% | 0.0000 ms | 0.0001 ms | 0.01 |
| 0.25 | ABS | 0.474 ms | 0.474 ms | 0.062% | -0.0003 ms | 0.0000 ms | 0.00 |
| 0.25 | X2 | 0.469 ms | 0.468 ms | 0.125% | -0.0006 ms | 0.0000 ms | 0.00 |
| 0.20 | ABS | 0.508 ms | 0.508 ms | -0.087% | 0.0004 ms | 0.0000 ms | 0.00 |
| 0.20 | X2 | 0.505 ms | 0.505 ms | 0.028% | -0.0001 ms | 0.0000 ms | 0.00 |
| 0.15 | ABS | 0.525 ms | 0.524 ms | 0.099% | -0.0005 ms | 0.0000 ms | 0.00 |
| 0.15 | X2 | 0.519 ms | 0.519 ms | 0.045% | -0.0002 ms | 0.0000 ms | 0.00 |

## 같은 R에서 마스크가 얼마나 바뀌었나

| score | cases | same mask | mean error Δ | mean |error Δ| | total Δ | selector Δ | SSD Δ | chunk Δ |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ABS | 1134 | 93.65% | -0.000870 | 0.001499 | -0.0001 ms | -0.0001 ms | +0.0000 ms | +0.009 |
| X2 | 1134 | 91.98% | -0.001179 | 0.001299 | +0.0004 ms | +0.0005 ms | -0.0000 ms | +0.011 |

## 판정

2-line의 동일-error 평균 gain은 ABS `+0.013%`, X² `-0.091%`이며, ceiling 승수는 `{'abs:2-line': 5, 'x2:lookup': 5, 'x2:2-line': 4, 'abs:lookup': 4}`다. 같은 R의 마스크 일치율은 두 score 평균 `92.81%`다.

2-line은 실제 측정치를 대체하는 latency 예측기가 아니라 **Cell-1의 repair 결정을 위한 목적함수**로만 사용했다. 최종 x축은 두 방법 모두 동일하게 실제 O_DIRECT SSD read/upload, activation gather, compact GEMM을 재측정한 `actual_total_ms`다. 따라서 결과 차이는 비용모델이 선택한 마스크 차이이며, 2-line 예측값을 성능값으로 그린 것이 아니다.

두 경로는 같은 로컬 cost-array cache와 같은 Cell-1 kernel을 사용한다. 따라서 lookup table을 매 호출마다 hash하는 구현 비용은 비교에서 제거했다.

Cell-1의 본체는 동일 길이 s-cell을 고르므로 두 비용모델의 차이는 주로 exact-R을 맞추는 floor-expand 대 ceil-trim에서 발생한다. 따라서 높은 마스크 일치율과 작은 frontier 차이가 나와도 버그가 아니라 구조적으로 예상되는 결과다.

## 측정 범위

- projection 후보 `11340`개, holdout aggregate point `90`개.
- Cell-1 selector fallback `0`건, Cell-1 row mismatch `0`건. Paper는 원 논문의 8-row chunk granularity 때문에 exact-R 대상이 아니다.
- score 생성 + selector + 실제 SSD/upload + gather + compact GEMM을 total에 포함했다.
- 모델별 별도 프로세스, CUDA allocator 55%, CPU/O_DIRECT thread 2개, projection 사이 25 ms 양보를 사용했다.

- end-to-end aggregate point `90`개도 저장했다.
![SSD fit](smol360/twoline_fit.png)

![Frontier](smol360/lookup_vs_twoline_frontier.png)

![Gain](smol360/twoline_gain.png)

![Components](smol360/twoline_components.png)
