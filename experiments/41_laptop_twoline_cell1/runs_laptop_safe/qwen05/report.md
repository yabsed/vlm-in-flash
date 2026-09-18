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
| 0.50 | ABS | 0.482 ms | 0.483 ms | -0.208% | 0.0016 ms | -0.0001 ms | 0.05 |
| 0.50 | X2 | 0.483 ms | 0.484 ms | -0.191% | -0.0001 ms | 0.0009 ms | 0.00 |
| 0.45 | ABS | 0.526 ms | 0.525 ms | 0.237% | -0.0016 ms | 0.0008 ms | -0.00 |
| 0.45 | X2 | 0.519 ms | 0.524 ms | -1.072% | 0.0009 ms | 0.0037 ms | -0.02 |
| 0.42 | ABS | 0.558 ms | 0.556 ms | 0.375% | -0.0005 ms | -0.0016 ms | -0.01 |
| 0.42 | X2 | 0.559 ms | 0.561 ms | -0.409% | 0.0018 ms | -0.0000 ms | 0.00 |
| 0.40 | ABS | 0.574 ms | 0.572 ms | 0.337% | 0.0008 ms | -0.0028 ms | -0.02 |
| 0.40 | X2 | 0.573 ms | 0.579 ms | -1.079% | 0.0017 ms | 0.0044 ms | 0.01 |
| 0.35 | ABS | 0.618 ms | 0.621 ms | -0.513% | 0.0016 ms | 0.0016 ms | 0.00 |
| 0.35 | X2 | 0.613 ms | 0.615 ms | -0.327% | 0.0003 ms | 0.0018 ms | -0.04 |
| 0.30 | ABS | 0.642 ms | 0.647 ms | -0.803% | 0.0002 ms | 0.0047 ms | -0.01 |
| 0.30 | X2 | 0.652 ms | 0.652 ms | -0.041% | 0.0011 ms | -0.0008 ms | 0.00 |
| 0.25 | ABS | 0.697 ms | 0.694 ms | 0.420% | -0.0002 ms | -0.0027 ms | 0.00 |
| 0.25 | X2 | 0.690 ms | 0.690 ms | -0.061% | 0.0001 ms | 0.0004 ms | -0.02 |
| 0.20 | ABS | 0.712 ms | 0.712 ms | -0.080% | 0.0005 ms | 0.0002 ms | 0.01 |
| 0.20 | X2 | 0.708 ms | 0.709 ms | -0.125% | 0.0004 ms | 0.0005 ms | -0.01 |
| 0.15 | ABS | 0.734 ms | 0.736 ms | -0.227% | 0.0005 ms | 0.0012 ms | 0.01 |
| 0.15 | X2 | 0.731 ms | 0.732 ms | -0.153% | 0.0009 ms | 0.0002 ms | -0.01 |

## 같은 R에서 마스크가 얼마나 바뀌었나

| score | cases | same mask | mean error Δ | mean |error Δ| | total Δ | selector Δ | SSD Δ | chunk Δ |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ABS | 1134 | 69.31% | -0.000638 | 0.000967 | +0.0000 ms | -0.0001 ms | +0.0002 ms | +0.048 |
| X2 | 1134 | 63.49% | -0.000556 | 0.001079 | +0.0014 ms | +0.0006 ms | +0.0007 ms | +0.006 |

## 판정

2-line의 동일-error 평균 gain은 ABS `-0.051%`, X² `-0.384%`이며, ceiling 승수는 `{'abs:lookup': 5, 'x2:lookup': 9, 'abs:2-line': 4}`다. 같은 R의 마스크 일치율은 두 score 평균 `66.40%`다.

2-line은 실제 측정치를 대체하는 latency 예측기가 아니라 **Cell-1의 repair 결정을 위한 목적함수**로만 사용했다. 최종 x축은 두 방법 모두 동일하게 실제 O_DIRECT SSD read/upload, activation gather, compact GEMM을 재측정한 `actual_total_ms`다. 따라서 결과 차이는 비용모델이 선택한 마스크 차이이며, 2-line 예측값을 성능값으로 그린 것이 아니다.

두 경로는 같은 로컬 cost-array cache와 같은 Cell-1 kernel을 사용한다. 따라서 lookup table을 매 호출마다 hash하는 구현 비용은 비교에서 제거했다.

Cell-1의 본체는 동일 길이 s-cell을 고르므로 두 비용모델의 차이는 주로 exact-R을 맞추는 floor-expand 대 ceil-trim에서 발생한다. 따라서 높은 마스크 일치율과 작은 frontier 차이가 나와도 버그가 아니라 구조적으로 예상되는 결과다.

## 측정 범위

- projection 후보 `11340`개, holdout aggregate point `90`개.
- Cell-1 selector fallback `0`건, Cell-1 row mismatch `0`건. Paper는 원 논문의 8-row chunk granularity 때문에 exact-R 대상이 아니다.
- score 생성 + selector + 실제 SSD/upload + gather + compact GEMM을 total에 포함했다.
- 모델별 별도 프로세스, CUDA allocator 55%, CPU/O_DIRECT thread 2개, projection 사이 25 ms 양보를 사용했다.

- end-to-end aggregate point `90`개도 저장했다.
![SSD fit](qwen05/twoline_fit.png)

![Frontier](qwen05/lookup_vs_twoline_frontier.png)

![Gain](qwen05/twoline_gain.png)

![Components](qwen05/twoline_components.png)
