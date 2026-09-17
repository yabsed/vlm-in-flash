# Experiment 22 보고서: Paper vs Predicted-lambda TD-2L(8) + Endpoint Trim

Paper는 논문의 고정 row-budget 알고리즘을 그대로 실행했다. 각 paired input에서 Paper가 달성한 importance를 `Q = I(M_paper)`로 두고 proposed method가 최소한 같은 importance를 유지하도록 했다. 따라서 아래 lookup 비교는 importance-matched 비교다.

논문 Table 2에는 총 16개 matrix shape가 있으며 모두 포함했다. importance는 실제 activation trace가 아니라 CV `3.3`인 synthetic lognormal이고, lookup latency는 노트북 NVMe 실측값이 아니라 `orin-agx` 공개 profile의 예측값이다.

## 전체 결과

### Host input -> CPU mask

| 방법 | 중앙 runtime | case-p95 | worst case-p95 | 유효+2ms | Paper 대비 평균/가중 lookup | Paper보다 비악화 | trim 순증분 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper | 0.291 ms | 0.597 ms | 0.940 ms | 100.0% | 0.00% / 0.00% | 100.0% | n/a |
| Predicted-lambda TD-2L(8) | 0.663 ms | 3.952 ms | 6.049 ms | 85.9% | 5.23% / 6.31% | 91.4% | n/a |
| Predicted-lambda TD-2L(8) + trim 64 | 0.869 ms | 4.794 ms | 6.657 ms | 82.4% | 6.83% / 6.89% | 94.2% | 1.54% |
| Predicted-lambda TD-2L(8) + trim 256 | 0.876 ms | 4.697 ms | 6.507 ms | 82.2% | 7.74% / 6.95% | 95.1% | 2.15% |

### CUDA input -> CUDA mask

| 방법 | 중앙 runtime | case-p95 | worst case-p95 | 유효+2ms | Paper 대비 평균/가중 lookup | Paper보다 비악화 | trim 순증분 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper | 0.323 ms | 0.604 ms | 1.124 ms | 100.0% | 0.00% / 0.00% | 100.0% | n/a |
| Predicted-lambda TD-2L(8) | 0.749 ms | 4.069 ms | 5.987 ms | 85.4% | 5.23% / 6.31% | 91.4% | n/a |
| Predicted-lambda TD-2L(8) + trim 64 | 0.961 ms | 4.891 ms | 7.044 ms | 81.5% | 6.83% / 6.89% | 94.2% | 1.54% |
| Predicted-lambda TD-2L(8) + trim 256 | 0.966 ms | 4.970 ms | 6.986 ms | 81.5% | 7.74% / 6.95% | 95.1% | 2.15% |

## 판정

`cuda` track에서는 모든 case에서 2 ms를 통과한 trim 설정이 없다. 품질 최고 설정은 `Predicted-lambda TD-2L(8) + trim 256`이고 Paper 대비 평균 lookup 절감은 `7.74%`다.

`Predicted-lambda TD-2L(8) + trim 256`은 432개 paired case 중 `21`개에서 Paper보다 느린 lookup을 보였고, 비악화율은 `95.1%`다. 절대 latency로 가중한 전체 절감은 `6.95%`다. Endpoint trim 자체는 평균 `2.15%`를 추가로 줄였지만, noisy lookup table 기준으로는 `2.5%` case에서 소폭 악화됐다. Two-line 목적의 악화는 0건이다.

Shape 단위로는 `12/16`개가 모든 case에서 2 ms를 통과했다. 평균 lookup이 Paper보다 나쁜 shape는 `896x128`다.

## Shape별 trim 256 (Host)

| Shape | rows | row KiB | runtime case-p95 | worst p95 | Paper 대비 lookup | 유효+2ms |
|---|---:|---:|---:|---:|---:|---:|
| 896x128 | 896 | 0.25 | 0.943 ms | 0.974 ms | -0.20% | 100.0% |
| 896x896 | 896 | 1.75 | 0.268 ms | 0.333 ms | 8.35% | 100.0% |
| 896x4864 | 896 | 9.50 | 0.236 ms | 0.245 ms | 8.18% | 100.0% |
| 1536x256 | 1,536 | 0.50 | 0.714 ms | 0.834 ms | 7.39% | 100.0% |
| 1536x1536 | 1,536 | 3.00 | 0.505 ms | 0.577 ms | 14.03% | 100.0% |
| 1536x8960 | 1,536 | 17.50 | 0.611 ms | 0.631 ms | 8.58% | 100.0% |
| 3584x512 | 3,584 | 1.00 | 1.208 ms | 1.724 ms | 9.74% | 100.0% |
| 3584x3584 | 3,584 | 7.00 | 1.097 ms | 1.155 ms | 7.36% | 100.0% |
| 3584x18944 | 3,584 | 37.00 | 1.772 ms | 1.871 ms | 5.49% | 100.0% |
| 4096x1024 | 4,096 | 2.00 | 1.105 ms | 1.183 ms | 5.99% | 100.0% |
| 4096x4096 | 4,096 | 8.00 | 1.295 ms | 1.300 ms | 7.88% | 100.0% |
| 4096x14336 | 4,096 | 28.00 | 2.019 ms | 2.049 ms | 6.59% | 92.6% |
| 4864x896 | 4,864 | 1.75 | 1.343 ms | 1.472 ms | 12.51% | 100.0% |
| 8960x1536 | 8,960 | 3.00 | 2.479 ms | 2.646 ms | 7.61% | 22.2% |
| 14336x4096 | 14,336 | 8.00 | 4.794 ms | 4.996 ms | 6.85% | 0.0% |
| 18944x3584 | 18,944 | 7.00 | 6.422 ms | 6.507 ms | 7.47% | 0.0% |

## Shape별 trim 256 (CUDA round trip)

| Shape | rows | row KiB | runtime case-p95 | worst p95 | Paper 대비 lookup | 유효+2ms |
|---|---:|---:|---:|---:|---:|---:|
| 896x128 | 896 | 0.25 | 0.777 ms | 1.070 ms | -0.20% | 100.0% |
| 896x896 | 896 | 1.75 | 0.337 ms | 0.410 ms | 8.35% | 100.0% |
| 896x4864 | 896 | 9.50 | 0.336 ms | 0.349 ms | 8.18% | 100.0% |
| 1536x256 | 1,536 | 0.50 | 0.616 ms | 0.815 ms | 7.39% | 100.0% |
| 1536x1536 | 1,536 | 3.00 | 0.541 ms | 0.569 ms | 14.03% | 100.0% |
| 1536x8960 | 1,536 | 17.50 | 0.722 ms | 0.876 ms | 8.58% | 100.0% |
| 3584x512 | 3,584 | 1.00 | 1.256 ms | 1.452 ms | 9.74% | 100.0% |
| 3584x3584 | 3,584 | 7.00 | 1.216 ms | 1.291 ms | 7.36% | 100.0% |
| 3584x18944 | 3,584 | 37.00 | 1.865 ms | 1.958 ms | 5.49% | 100.0% |
| 4096x1024 | 4,096 | 2.00 | 1.190 ms | 1.369 ms | 5.99% | 100.0% |
| 4096x4096 | 4,096 | 8.00 | 1.379 ms | 1.405 ms | 7.88% | 100.0% |
| 4096x14336 | 4,096 | 28.00 | 2.361 ms | 3.439 ms | 6.59% | 88.9% |
| 4864x896 | 4,864 | 1.75 | 1.621 ms | 1.671 ms | 12.51% | 100.0% |
| 8960x1536 | 8,960 | 3.00 | 2.516 ms | 2.606 ms | 7.61% | 14.8% |
| 14336x4096 | 14,336 | 8.00 | 5.193 ms | 5.302 ms | 6.85% | 0.0% |
| 18944x3584 | 18,944 | 7.00 | 6.733 ms | 6.986 ms | 7.47% | 0.0% |

## 측정 범위와 한계

- `host`: host float32 importance에서 CPU bool mask까지 측정했다. CUDA가 있으면 Paper native 구현의 GPU sort도 포함된다.
- `cuda`: 미리 만들어 둔 CUDA float32 importance에서 시작해 CPU proposal의 D2H, float64 변환, 8회 이하 DP, mask 복원, trim, H2D 및 synchronization을 모두 포함했다.
- 최초 JIT/native compilation, importance 생성, 실제 NVMe I/O, model compute는 제외했다.
- lookup latency는 Orin AGX profile 기반이므로 RTX 3050 노트북 selector 시간과 서로 다른 축이다. Jetson의 2 ms 충족 여부는 Jetson에서 다시 측정해야 한다.
- endpoint trim은 two-line 목적을 단조 감소시키지만 측정 잡음이 있는 lookup table의 각 개별 점까지 단조 감소한다고 보장하지는 않는다.

![Overall runtime-quality](results/runtime_quality.png)

![Shape comparison](results/shape_comparison.png)

