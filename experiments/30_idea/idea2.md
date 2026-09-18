결론부터 말하면, 더 줄일 수 있습니다. 그런데 다음 단계는 `s/8`을 더 미세하게 조정하는 게 아닙니다.

Experiment 29가 드러낸 핵심은 이것입니다.

> 현재 최적화하는 \(\sum_j T(\ell_j)\)와 이 노트북 reader의 실제 실행시간 목적함수가 다릅니다.

가장 유망한 다음 알고리즘은 **6-thread wave-balanced exact-\(R\) interval DP**입니다.

## 왜 tile의 실제 read가 느려졌는가

현재 native reader는 다음 순서로 동작합니다.

1. mask를 CPU로 복사
2. 연속 run으로 변환
3. run을 최대 768 KiB task로 분할
4. 최대 6개 thread에 `task t, t+6, t+12...` 방식으로 배정
5. 모든 thread가 끝난 후 GPU upload

실제 구현은 [native.cc](/home/yabsed/Documents/fall26/work/vlm-in-flash/preliminary_research/vlm-flash/src/vlmflash/csrc/native.cc:305)에 그대로 보입니다.

따라서 실제 read 비용은

\[
L_{\mathrm{sum}}(M)=\sum_jT(\ell_j)
\]

보다는 다음에 가깝습니다.

\[
\boxed{
L_{\mathrm{wall}}(M)
\approx
\tau_0+
\max_{0\le t<6}
\sum_{j\equiv t\pmod 6}g(q_j)
+U(R)
}
\]

- \(q_j\): 768 KiB 이하로 분할된 실제 read task
- \(g(q)\): task 하나의 실제 service time
- \(U(R)\): GPU upload 시간. fixed-\(R\)에서는 거의 상수
- 핵심은 합이 아니라 **가장 일이 많은 thread의 완료시간**, 즉 makespan입니다.

실험 29 mask를 실제 reader task로 다시 분해해 보니:

| 평균 | Paper | ceil tile |
|---|---:|---:|
| run 수 | 10.53 | 7.27 |
| reader task 수 | 10.55 | 7.41 |
| 가장 바쁜 lane의 데이터 | 552.7 KiB | 632.2 KiB |
| 실제 I/O | 0.6357 ms | 0.6610 ms |

tile은 chunk와 wave 수를 줄였지만, 작업을 6개 lane에 더 불균형하게 배치했습니다. 그 결과 가장 바쁜 lane이 약 79.5 KiB 더 읽었고 실제 I/O가 0.0253 ms 느려졌습니다. Pooled case에서 실제 I/O와 최대 lane load의 상관계수는 0.958이었습니다.

또한 현재 profile은 여러 chunk를 읽어 얻은 포화 throughput을 단일 \(T(\ell)\)로 축약합니다. [profile_flash.py](/home/yabsed/Documents/fall26/work/vlm-in-flash/preliminary_research/vlm-flash/scripts/profile_flash.py:110)에서 chunk 개수 차원이 사라집니다. 이건 latency model 자체가 틀렸다기보다, **유한 batch와 6-way scheduling 정보가 소실된 것**입니다.

## 제안: Wave-balanced exact-\(R\) DP

연속 run의 개수 \(K\)를 포화점만으로 정하지 말고, reader thread 수 \(p=6\)의 배수로 만듭니다.

\[
K\in\{6,12,18,24,\ldots\}
\]

총 읽기 크기를 \(B=R\cdot\text{rowBytes}\)라 하면, 이상적인 동일 크기 task에서는

\[
C_K(B)\approx
\left\lceil\frac Kp\right\rceil
g\left(\frac BK\right).
\]

\(K\)가 6의 배수면 모든 lane이 정확히 같은 수의 task를 받습니다. 각 task가 포화 크기 \(s\) 이상인 범위에서는 \(g(x)\)가 거의 선형이므로, \(K\)를 늘려 importance 자유도를 얻어도 read makespan은 거의 증가하지 않습니다.

간단한 초기값은

\[
K_0
=
p\max\left(
1,
\left\lfloor\frac{B}{ps}\right\rfloor
\right)
\]

이고, 실제로는 \(K_0\) 주변의 6 배수만 직접 profile하면 됩니다.

이 노트북의 \(p=6,\ s=240\) KiB에서는 큰 두 shape의 세 budget에 대해 대략

\[
K=(6,12,24)
\]

가 나옵니다.

### Exact-\(R\)과 alignment를 동시에 보장

O_DIRECT alignment가 \(A\), row 크기가 \(b_r\)이면 cell 크기를

\[
a=\frac{A}{\gcd(A,b_r)}
\]

행으로 둡니다. 이 노트북의 512-byte alignment를 적용하면:

- 0.25 KiB row: \(a=2\)
- 1.75 KiB row: \(a=2\)
- 9.5 KiB row: \(a=1\)

따라서 run 시작과 길이를 cell 단위로 제한하면 bounce buffer와 추가 memcpy를 대부분 없앨 수 있습니다. 실제 구현에서는 파일별 O_DIRECT 제약을 `statx(STATX_DIOALIGN)`로 확인하는 것이 맞습니다. [Linux `statx(2)` 문서](https://linuxman7.org/linux/man-pages/man2/statx.2.html)

\(R'=R/a\)에 대해

\[
\ell=\left\lfloor\frac{R'}K\right\rfloor,\qquad
q=R'-K\ell
\]

로 두면, \(q\)개 run은 \(\ell+1\) cell, 나머지는 \(\ell\) cell이면 정확히 \(R\)행이 됩니다.

DP 상태를

\[
D[i,k,q]
\]

즉, 처음 \(i\)개 cell에서 \(k\)개 run과 \(q\)개의 long run을 선택한 최대 importance로 둡니다. 전이는 skip, short run, long run 세 가지입니다.

\[
D[i,k,q]=\max
\begin{cases}
D[i-1,k,q],\\
D[i-\ell-1,k-1,q]+W(i-\ell,i),\\
D[i-\ell-2,k-1,q-1]+W(i-\ell-1,i).
\end{cases}
\]

한 cell의 gap을 강제하므로 실제 run 수가 정확히 \(K\)로 유지됩니다.

이 방식의 장점은 다음과 같습니다.

- 정확히 \(R\)행
- 정확히 \(K\)개 reader task/run
- O_DIRECT alignment 보장
- floor-expand/ceil-trim 제거
- 해당 hardware cost class 안에서 importance 전역 최적
- Paper mask와 \(I(M_{\text{paper}})\) 불필요

일반적인 \(K\)-disjoint interval 선택 자체도 거의 선형시간 알고리즘이 알려진 문제입니다. 다만 여기서는 exact row 수와 alignment가 있어 위의 작은 DP가 더 직접적입니다. [Bae–Takaoka의 \(K\)-disjoint maximum-subarray 알고리즘](https://ir.canterbury.ac.nz/bitstreams/cb7deff9-bea1-4dd6-a6d7-b67fafbca038/download)

## 방금 실제 trace로 한 수학적 probe

아직 O_DIRECT를 다시 돌린 결과는 아니고, 128개 실제 Qwen trace에 aligned equal-run DP만 적용한 결과입니다.

단순히 큰 shape에 \(K=(6,12,24)\), 작은 shape에 \(K=6\)을 적용하면 Paper 대비 importance 변화는:

| Shape | 25% budget | 50% budget | 75% budget |
|---|---:|---:|---:|
| 4864×896 | -2.23% | -0.90% | +0.45% |
| 896×4864 | -4.02% | -1.55% | +0.27% |
| 896×896 | +7.04% | +2.19% | +0.22% |
| 896×128 | +20.87% | +10.37% | +4.97% |

전체 평균은 Paper보다 약 **+3.14% importance**입니다. Experiment 29의 `-1.64%`보다 훨씬 낫습니다.

더 흥미로운 것은 큰 shape의 \(K\)를 조금 늘리는 경우입니다.

- 낮은 budget에서 \(K=12\): Paper importance 대비 `+0.97%`, `+1.26%`
- 중간 budget에서 \(K=18\): `+0.15%`, `-0.18%`
- 높은 budget에서 \(K=24\): `+0.45%`, `+0.27%`

즉, 6-lane 균형을 유지하면서 Paper 수준 이상의 importance까지 회복할 가능성이 큽니다.

## \(K\)는 Paper 없이 어떻게 고르는가

각 \(K\)에 대해 DP가 그 cost class의 최대 importance \(I_K^*(v)\)를 줍니다. 로컬에서 미리 측정한 batch cost를 \(\widehat C_K\)라 하면 온라인 선택은

\[
\boxed{
K^*(v)=
\arg\max_{K\in\mathcal K}
\frac{I_K^*(v)}
{C_{\mathrm{selector}}(K)+\widehat C_K}
}
\]

로 끝납니다.

Paper importance는 전혀 필요하지 않습니다. 정확도 하한을 원하면 다음도 가능합니다.

\[
\min_K\widehat C_K
\quad\text{s.t.}\quad
I_K^*(v)\ge(1-\epsilon)I_{\mathrm{TopR}}(v).
\]

\(I_{\mathrm{TopR}}\)은 fixed-\(R\) mask가 가질 수 있는 importance의 절대 상한이므로 Paper보다 수학적으로 더 자연스러운 기준입니다.

## selector 자체도 더 줄일 수 있다

현재 CUDA 경로는 다음 낭비가 있습니다.

1. importance를 GPU→CPU로 복사
2. CPU에서 DP
3. bool mask를 CPU→GPU로 복사
4. reader 진입 후 같은 mask를 다시 GPU→CPU로 복사

특히 4번은 [native.cc](/home/yabsed/Documents/fall26/work/vlm-in-flash/preliminary_research/vlm-flash/src/vlmflash/csrc/native.cc:307)에 있습니다.

새 DP는 dense mask 대신 `(start, length)` interval \(K\)개를 반환해야 합니다.

- CPU reader는 interval을 직접 받음
- GPU는 동일한 작은 interval descriptor만 받음
- GPU에서 interval gather kernel로 `x_sel` 생성
- dense bool mask 두 번의 왕복과 O(N) run scan 제거

추가로 reader의 fd와 6개 worker thread를 projection마다 새로 만들지 말고 유지해야 합니다. 현재는 매 호출마다 `open`하고 thread를 다시 생성합니다. 이것도 [native.cc](/home/yabsed/Documents/fall26/work/vlm-in-flash/preliminary_research/vlm-flash/src/vlmflash/csrc/native.cc:367)에서 확인됩니다.

## 현실적인 다음 목표

현재 수치에서:

- Experiment 29 고정 ceil: `1.0018 ms`
- 현재 후보들을 case마다 사후 최적으로 선택한 총시간: `0.9756 ms`
- Paper read + tile selector를 조합한 가상값: `0.9757 ms`
- 후보별 최선 read와 최선 tile selector를 독립 조합한 낙관적 값: `0.9496 ms`

따라서 제 판단은:

- mask 알고리즘만 개선: **0.97 ms 근처**
- wave-balanced DP + alignment + interval API: **0.95–0.98 ms가 현실적인 1차 목표**
- persistent reader와 동적 task scheduling까지 적용: **0.95 ms 아래 가능성 있음**
- `0.90 ms` 아래는 selector 개선만으로는 어렵고, 평균 0.211 ms인 GPU upload를 read와 겹치거나 streaming matmul까지 해야 합니다.

논문의 알고리즘이 최적인 것이 아닙니다. 논문은 importance/추정 latency utility를 쓰는 greedy chunking이며, 그 목적 자체도 실제 장치에 맞춰 정의할 수 있습니다. [VLM in a Flash 논문](https://arxiv.org/abs/2511.18692)

가장 중요한 다음 실험은 `s/8` 변형 추가가 아니라 다음 네 후보입니다.

1. `Wave-K6`
2. `Wave-K12/18/24 shape-budget dispatch`
3. `Wave adaptive-I/C` — 여러 \(K\) 중 자기 자신의 \(I/C\)로 선택
4. `Wave interval-native` — mask 왕복과 thread 생성까지 제거

이게 현재 가장 수학적으로 정당하고, Paper보다 더 빠르면서 importance 손실도 회복할 가능성이 높은 방향입니다.