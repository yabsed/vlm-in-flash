# Experiment 25 보고서: GPU Saturation Tiles

Experiment 18의 `Tiles(s/2)`, `Tiles(s)`, `Tiles(2s)`를 CUDA-resident PyTorch 연산으로 옮겼다. 입력 importance부터 최종 boolean mask까지 GPU에 머물며, data-dependent CPU readback은 없다.

importance는 `vlm-flash`를 실제 `Qwen/Qwen2.5-0.5B-Instruct` projection에 attach한 dense forward에서 수집했으며 Experiment 22/24와 같은 trace archive를 사용했다.

## 한 문장 판정

\[\boxed{\text{Tiles (s/2)는 2 ms를 지켰지만, lookup 개선은 0.08\%이고 CPU 대비 10.05배 느렸다.}}\]

## 측정 설정

- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- model: Qwen/Qwen2.5-0.5B-Instruct
- 실제 activation trace / Table-2 shape: 128 / 4
- trace × scenarios: 128 × 3
- case당 warm-up/repetitions: 5/30
- CUDA event: device 실행시간
- synchronized wall: Python dispatch + GPU 실행 + synchronization
- 2ms 판정: synchronized wall p95 <= 2 ms

## 전체 결과

| 방법 | GPU wall 중앙 | GPU wall case-p95 | CUDA-event case-p95 | CPU case-p95 | GPU/CPU | coverage | coverage+R | Paper 대비 lookup 평균/가중 | CPU mask 일치 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiles (s/2) | 0.712 ms | 0.843 ms | 0.822 ms | 0.145 ms | 10.05× | 100.0% | 1.0% | 1.97% / 0.08% | 98.4% |
| Tiles (s) | 0.708 ms | 0.868 ms | 0.855 ms | 0.098 ms | 13.57× | 100.0% | 0.3% | -0.65% / -0.58% | 92.7% |
| Tiles (2s) | 0.708 ms | 0.883 ms | 0.863 ms | 0.078 ms | 16.78× | 100.0% | 0.3% | -3.85% / -3.28% | 97.1% |

## 판정

가중 lookup 품질이 가장 높은 설정은 `Tiles (s/2)`이다. Paper coverage selector 대비 전체 가중 lookup은 소폭 개선됐지만 크기는 `0.08%`에 그쳤다.

wall case-p95가 가장 낮은 설정은 `Tiles (s/2)`의 `0.843 ms`다. 모든 설정의 2ms case 통과율은 Tiles (s/2) 100.0%, Tiles (s) 100.0%, Tiles (2s) 100.0%.

동일 입력의 Experiment 18 CPU 구현 대비 GPU wall/CPU 중앙시간 비율은 Tiles (s/2) 10.05×, Tiles (s) 13.57×, Tiles (2s) 16.78×다. 1보다 크면 현재 eager GPU 구현이 더 느리다는 뜻이다. GPU가 실제로 더 빠른 case 비율은 Tiles (s/2) 0.0%, Tiles (s) 0.0%, Tiles (2s) 0.0%다.

CPU와 GPU 모두 2ms를 통과한 비율은 각각 Tiles (s/2) CPU 100.0%/GPU 100.0%, Tiles (s) CPU 100.0%/GPU 100.0%, Tiles (2s) CPU 100.0%/GPU 100.0%다. `s/2` GPU는 일부 tile 수가 많은 큰 shape의 CPU tail을 줄였지만, 대부분의 case 중앙시간은 CPU가 더 짧았다.

Paper 대비 가중 lookup이 양수인 shape 수는 Tiles (s/2) 3/4, Tiles (s) 1/4, Tiles (2s) 1/4다. Case별 lookup 비악화율은 Tiles (s/2) 50.8%, Tiles (s) 37.0%, Tiles (2s) 25.0%다.

Experiment 18 CPU mask와는 `1107/1152`개가 정확히 일치했다. 나머지 `45`개는 float32 GPU와 float64 CPU의 정렬 경계 차이이며 coverage 위반은 없었고 최대 two-line 차이는 `0.006104 ms`였다.

row-budget은 알고리즘이 직접 강제하지 않는 audit이다. Paper fixed-R의 coverage+R 성공률은 `1.8%`이고, Tiles는 Tiles (s/2) 1.0%, Tiles (s) 0.3%, Tiles (2s) 0.3%다.

## Target별 결과

| Q/total | 방법 | wall case-p95 | coverage+R | 가중 lookup 절감 |
|---:|---|---:|---:|---:|
| 0.50 | Tiles (2s) | 0.900 ms | 0.0% | -5.44% |
| 0.50 | Tiles (s/2) | 0.841 ms | 0.0% | -1.77% |
| 0.50 | Tiles (s) | 0.910 ms | 0.0% | -1.79% |
| 0.70 | Tiles (2s) | 0.856 ms | 0.8% | -4.09% |
| 0.70 | Tiles (s/2) | 0.833 ms | 2.3% | 0.24% |
| 0.70 | Tiles (s) | 0.867 ms | 0.8% | -0.34% |
| 0.90 | Tiles (2s) | 0.851 ms | 0.0% | -1.53% |
| 0.90 | Tiles (s/2) | 0.852 ms | 0.8% | 0.92% |
| 0.90 | Tiles (s) | 0.853 ms | 0.0% | -0.13% |

## Shape별 wall case-p95

| Shape | rows | row KiB | s/2 | s | 2s |
|---|---:|---:|---:|---:|---:|
| 896x128 | 896 | 0.25 | 0.815 ms | 0.881 ms | 0.820 ms |
| 896x896 | 896 | 1.75 | 0.847 ms | 0.863 ms | 0.919 ms |
| 896x4864 | 896 | 9.50 | 0.834 ms | 0.859 ms | 0.864 ms |
| 4864x896 | 4,864 | 1.75 | 0.856 ms | 0.856 ms | 0.893 ms |

## 측정 한계

- 실제 activation은 `Qwen/Qwen2.5-0.5B-Instruct`의 짧은 text prompt에서 얻었으며 다른 model, 긴 context, multimodal frame-append workload까지 대표하지 않는다.
- lookup latency는 released Orin AGX profile 예측값이고 실제 NVMe I/O를 실행하지 않았다.
- NVIDIA GeForce RTX 3050 6GB Laptop GPU timing이므로 Jetson의 절대시간을 증명하지 않는다.
- static tile layout은 shape만으로 결정되어 online timing에서 제외했다.
- eager PyTorch 구현이며 CUDA graph나 custom fused kernel은 사용하지 않았다.
- row-budget 결과는 제약을 직접 푼 것이 아니라 coverage mask에 대한 사후 audit이다.

![Overall](results/runtime_quality.png)

![Shape scaling](results/shape_scaling.png)

