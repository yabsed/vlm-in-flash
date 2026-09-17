# Experiment 24 보고서: 동일 row 수의 fixed-R 비율 최적화

각 paired case에서 Paper를 원래 nominal budget으로 실행한 뒤 실제 선택 행 수 `R_eff = |M_paper|`를 측정했다. Top-R와 모든 proposed mask는 정확히 `R_eff`개를 선택한다. 따라서 Experiment 22와 달리 실제 row 수가 완전히 같다.

제안법은 two-line 목적 `I/L`을 최적화한다. 공개 lookup table의 `I/L`, importance, lookup latency는 별도 지표이며 서로 혼동하지 않는다.

## 전체 결과

### Host input -> CPU mask

| 방법 | 중앙 runtime | case-p95 | 유효+2ms | lookup I/L | two-line I/L | importance | lookup 절감 | 혼합 손익분기 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper | 0.292 ms | 0.549 ms | 100.0% | 0.00% | 0.00% | 0.00% | 0.00% | 0.0% |
| Top-R | 0.031 ms | 0.149 ms | 100.0% | -62.22% | -60.55% | 14.14% | -149.44% | 1.9% |
| Fixed-R 2L (rho1, mu4) | 0.428 ms | 2.282 ms | 92.8% | 3.71% | 5.80% | 0.68% | 5.48% | 47.7% |
| Fixed-R 2L (rho1, mu8) | 0.606 ms | 3.084 ms | 87.7% | 2.77% | 4.94% | 1.67% | 5.42% | 31.2% |
| Fixed-R 2L (rho2, mu4) | 0.807 ms | 4.099 ms | 83.8% | 7.40% | 9.50% | -3.02% | 7.13% | 24.1% |

### CUDA input -> CUDA mask

| 방법 | 중앙 runtime | case-p95 | 유효+2ms | lookup I/L | two-line I/L | importance | lookup 절감 | 혼합 손익분기 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper | 0.309 ms | 0.576 ms | 100.0% | 0.00% | 0.00% | 0.00% | 0.00% | 0.0% |
| Top-R | 0.063 ms | 0.212 ms | 100.0% | -62.22% | -60.55% | 14.14% | -149.44% | 1.2% |
| Fixed-R 2L (rho1, mu4) | 0.498 ms | 2.383 ms | 91.4% | 3.71% | 5.80% | 0.68% | 5.48% | 35.6% |
| Fixed-R 2L (rho1, mu8) | 0.688 ms | 3.193 ms | 86.6% | 2.77% | 4.94% | 1.67% | 5.42% | 26.2% |
| Fixed-R 2L (rho2, mu4) | 0.892 ms | 4.297 ms | 83.8% | 7.40% | 9.50% | -3.02% | 7.13% | 20.4% |

## Small-N exact oracle

`N=18` exhaustive fixed-R optimum과 비교했다.

| 방법 | 평균 optimum 회수 | 최악 회수 | p95 gap | exact hit |
|---|---:|---:|---:|---:|
| Top-R | 72.918% | 47.440% | 47.652% | 7.8% |
| Fixed-R 2L (rho1, mu4) | 99.264% | 82.529% | 3.830% | 68.9% |
| Fixed-R 2L (rho1, mu8) | 99.357% | 82.529% | 3.182% | 72.2% |
| Fixed-R 2L (rho2, mu4) | 99.264% | 82.529% | 3.830% | 68.9% |

## Shape-adaptive hybrid

각 timing track에서 모든 case가 2 ms를 통과한 shape에만 fixed-R selector를 사용하고 나머지는 Paper로 되돌리는 정적 정책이다.

| Track | 후보 | fixed-R shape | deadline | lookup I/L | importance | 가중 lookup 절감 | case-p95 | worst p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| host | Fixed-R 2L (rho1, mu4) | 14/16 | 100.0% | 3.38% | 0.64% | 4.54% | 1.029 ms | 1.902 ms |
| host | Fixed-R 2L (rho2, mu4) | 13/16 | 100.0% | 6.81% | -2.70% | 5.39% | 1.297 ms | 1.884 ms |
| cuda | Fixed-R 2L (rho1, mu4) | 13/16 | 100.0% | 3.49% | 0.51% | 4.65% | 0.832 ms | 1.442 ms |
| cuda | Fixed-R 2L (rho2, mu4) | 13/16 | 100.0% | 6.81% | -2.70% | 5.39% | 1.376 ms | 1.978 ms |

## 판정

lookup-table `I/L` 평균이 가장 높은 fixed-R 설정은 `Fixed-R 2L (rho2, mu4)`다. Paper 대비 lookup `I/L`은 평균 `7.40%`, two-line `I/L`은 `9.50%` 변했다.

모든 method case의 정확한 row-match와 deterministic rate는 각각 `100.0%`, `100.0%`다. `cuda` track에서 모든 case가 2 ms를 통과한 shape는 `13/16`개다.

importance가 Paper보다 낮지 않은 비율은 `19.2%`, lookup도 동시에 나쁘지 않은 Pareto 비악화율은 `18.1%`다. 따라서 비율 개선만으로 동일 accuracy를 주장할 수 없다.

노트북 selector overhead와 Orin lookup 예측을 섞은 진단적 손익분기 통과율은 `20.4%`다. 서로 다른 장치의 시간을 더한 값이므로 end-to-end 결과가 아니라 Jetson에서 검증할 조건이다.

2 ms 우선의 현실적인 선택은 `Fixed-R 2L (rho1, mu4)` shape hybrid다. 모든 case의 deadline을 지키면서 lookup `I/L` 평균 `3.49%`, importance 평균 `0.51%`, 가중 lookup 절감 `4.65%`를 보였다. 다만 case별 importance 비악화율은 `44.4%`라 동일 accuracy는 여전히 보장되지 않는다.

`mu=8`은 small-N oracle 평균을 소폭 높였지만 production-N에서는 `mu=4`보다 느리고 lookup `I/L`도 낮았다. `mu=4`도 한 ratio iteration당 평균 `660.2`개 endpoint를 repair하므로, 다음 최적화 지점은 더 촘촘한 row-price 탐색보다 repair 대상 cardinality를 줄이는 예측이다.

## Shape별 Fixed-R 2L (rho2, mu4) (cuda)

| Shape | rows | lookup I/L | importance | lookup 절감 | case-p95 | 유효+2ms |
|---|---:|---:|---:|---:|---:|---:|
| 896x128 | 896 | 10.20% | -6.97% | 14.52% | 0.391 ms | 100.0% |
| 896x896 | 896 | 9.61% | -4.85% | 11.67% | 0.405 ms | 100.0% |
| 896x4864 | 896 | 13.28% | -3.25% | 14.50% | 0.400 ms | 100.0% |
| 1536x256 | 1,536 | 4.33% | -6.11% | 9.42% | 0.693 ms | 100.0% |
| 1536x1536 | 1,536 | 15.33% | -1.64% | 14.21% | 0.659 ms | 100.0% |
| 1536x8960 | 1,536 | 12.11% | -3.07% | 12.62% | 0.635 ms | 100.0% |
| 3584x512 | 3,584 | 6.04% | -3.11% | 8.56% | 1.364 ms | 100.0% |
| 3584x3584 | 3,584 | 5.22% | -2.28% | 6.57% | 1.389 ms | 100.0% |
| 3584x18944 | 3,584 | 7.29% | -2.29% | 7.87% | 1.334 ms | 100.0% |
| 4096x1024 | 4,096 | 2.92% | -2.75% | 5.00% | 1.569 ms | 100.0% |
| 4096x4096 | 4,096 | 5.85% | -2.55% | 7.16% | 1.551 ms | 100.0% |
| 4096x14336 | 4,096 | 10.80% | -2.60% | 10.80% | 1.489 ms | 100.0% |
| 4864x896 | 4,864 | 5.96% | -1.78% | 7.64% | 1.856 ms | 100.0% |
| 8960x1536 | 8,960 | 2.45% | -2.06% | 4.11% | 3.284 ms | 40.7% |
| 14336x4096 | 14,336 | 3.37% | -1.53% | 4.07% | 5.162 ms | 0.0% |
| 18944x3584 | 18,944 | 3.68% | -1.51% | 4.32% | 6.631 ms | 0.0% |

## 측정 한계

- importance는 CV 3.30 synthetic lognormal이며 실제 activation trace가 아니다.
- lookup은 released Orin AGX profile의 예측값이고 실제 NVMe I/O를 실행하지 않았다.
- `cuda` track은 D2H, CPU solve/repair, H2D와 synchronization을 포함하지만 model compute는 제외한다.
- row-price scalarization과 endpoint repair는 unsupported exact-R 해를 건너뛸 수 있으므로 전역 최적 알고리즘이 아니다.
- shape hybrid의 허용 목록은 같은 benchmark에서 사후 선택했으므로 별도 trace에서 재검증해야 한다.

![Runtime-quality](results/runtime_quality.png)

![Shape comparison](results/shape_comparison.png)

![Small-N oracle](results/oracle_optimality.png)

