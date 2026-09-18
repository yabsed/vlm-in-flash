맞습니다. 다만 핵심 통찰은 단순히 **“\(R\)을 가변화한다”**가 아니라 다음과 같이 정리해야 합니다.

\[
\boxed{\text{품질의 하한을 만족하면서 실제 latency를 최소화한다}}
\]

현재까지의 문제는 사실 이렇게 풀고 있었습니다.

\[
\max_{M:\,|M|=R} I(M)
\qquad\text{또는}\qquad
\min_{M:\,|M|=R} L(M)
\]

여기서 \(M\)은 읽을 row들의 집합입니다. 하지만 고정된 \(R\)은 실험을 공정하게 비교하기 위한 통제변수일 뿐, 실제 배포에서 원하는 목적은 아닙니다.

진짜 문제는 다음입니다.

\[
\boxed{
\min_M T(M)
\quad\text{s.t.}\quad
Q(M)\ge \tau
}
\]

- \(T(M)\): 노트북에서 실제로 걸리는 총 시간
- \(Q(M)\): 선택한 row들이 보존하는 품질
- \(\tau\): 허용할 수 있는 최소 품질

그러면 \(R=|M|\)은 입력·레이어·projection마다 최적화 결과로 자동 결정됩니다.

---

## 1. 왜 이것이 고정 \(R\)보다 수학적으로 우월한가

고정된 \(R\)에서 얻은 어떤 mask를 \(M_R\)라 하고, 그 품질을

\[
\tau=Q(M_R)
\]

로 놓겠습니다. 새로운 문제에서는 \(M_R\) 자체가 feasible solution입니다. 따라서 최적해 \(M^\star\)는 반드시

\[
T(M^\star)\le T(M_R)
\]

를 만족합니다.

즉, 같은 품질을 요구한다면 품질 제약식 방식은 고정-\(R\) 방식보다 절대로 느릴 수 없습니다. 더 적은 row나 더 효율적인 chunk 구조가 같은 품질을 달성하면 부등식은 strict해집니다.

\[
T(M^\star)<T(M_R)
\]

이것은 휴리스틱한 주장이 아니라 feasible set inclusion에서 직접 나오는 결과입니다.

---

## 2. \(R\)은 왜 입력마다 달라야 하는가

activation importance가 몇 개의 row에 몰린 입력을 생각하면 작은 \(R\)만으로도 충분합니다.

\[
I_{(1)} \gg I_{(2)} \gg \cdots
\]

반대로 importance가 넓게 퍼진 입력에서는 같은 품질을 얻기 위해 큰 \(R\)이 필요합니다.

고정 \(R\)은 따라서:

- 쉬운 입력에서는 필요 이상으로 읽고
- 어려운 입력에서는 품질을 충분히 보존하지 못합니다.

가변 \(R\)에서는

\[
R^\star(x,l,p,\tau)=|M^\star|
\]

가 되어 입력 \(x\), layer \(l\), projection \(p\), 목표 품질 \(\tau\)에 따라 달라집니다.

이것이 fixed sparsity가 아니라 **adaptive rate allocation** 문제가 되는 지점입니다.

---

## 3. 그러나 \(I(M)\)만 하한으로 두는 것은 부족하다

현재 importance를 대략

\[
I(M)=\sum_{i\in M}\operatorname{mean}|X_i|
\]

로 정의한다면, 이는 activation만 봅니다. 하지만 projection은

\[
Y=XW^\top+b
\]

이므로 row \(i\)를 버렸을 때의 영향은 activation뿐 아니라 대응되는 weight column에도 의존합니다.

선택한 집합을 \(M\)이라고 하면 정확한 projection 오차는

\[
Y-Y_M=X_{\bar M}W_{\bar M}^{\top}
\]

이고,

\[
\|Y-Y_M\|_F^2
=
\sum_{i,j\notin M}
\langle X_i,X_j\rangle
\langle W_{:,i},W_{:,j}\rangle
\]

입니다.

즉 raw importance가 놓치는 것은:

1. weight column의 크기
2. 서로 다른 channel 사이의 상관관계
3. downstream layer의 민감도

입니다.

Experiment 32에서 importance와 local projection error의 순위 상관이 매우 높았지만 완벽하지 않았고, local error가 작아져도 최종 logit/KL이 반드시 좋아지지는 않았습니다. 따라서 최종 논지는

\[
I(M)\ge\tau
\]

에서 끝나면 안 됩니다.

---

## 4. 현실적으로 측정 가능한 더 좋은 품질 함수

가장 간단한 weight-aware score는

\[
s_i=\|X_i\|_2\|W_{:,i}\|_2
\]

입니다.

삼각부등식으로

\[
\|Y-Y_M\|_F
\le
\sum_{i\notin M}
\|X_i\|_2\|W_{:,i}\|_2
=
\sum_{i\notin M}s_i
\]

이므로 다음 제약은 실제 projection error의 상한을 제공합니다.

\[
\sum_{i\notin M}s_i\le\epsilon
\]

전체 score \(S_{\mathrm{all}}=\sum_i s_i\)를 미리 계산하면 이는 retained-score 하한으로도 쓸 수 있습니다.

\[
\sum_{i\in M}s_i
\ge
S_{\mathrm{all}}-\epsilon
\]

weight norm은 모델별로 한 번 offline에서 계산하면 됩니다. 온라인에서는 activation norm만 계산하면 되므로 \(I(M_{\text{paper}})\)를 알아야 할 필요도 없습니다.

상관관계 항을 무시한 조금 더 공격적인 근사는

\[
s_i^{(2)}
=
\|X_i\|_2^2\|W_{:,i}\|_2^2
\]

입니다. 이는 conservative guarantee는 약하지만 실제 오차 예측에는 더 정확할 수 있습니다. 실제로는 두 score를 모두 측정해서 calibration set에서 어떤 것이 end-to-end error를 더 잘 예측하는지 확인해야 합니다.

---

## 5. latency \(L\)도 진짜 목적함수로 다시 정의해야 한다

\(R\)이 달라지면 단순 read latency만 비교해서는 안 됩니다. 실제 목적함수는 최소한

\[
T(M)=
T_{\mathrm{select}}(M)
+
T_{\mathrm{read}}(M)
+
T_{\mathrm{upload}}(M)
+
T_{\mathrm{GEMM}}(|M|)
\]

이어야 합니다.

이 노트북의 reader가 6개 lane이고 chunk가 병렬로 실행된다면 read time은 단순 합이 아닙니다.

\[
T_{\mathrm{read}}(M)
=
\tau_{\mathrm{open}}
+
\max_{1\le k\le6}
\sum_{j\in\mathcal J_k(M)}
g(\operatorname{size}_j)
\]

따라서 같은 \(R\)에서도:

- 연속 run 수
- 768 KiB splitting
- 6개 lane의 imbalance
- alignment
- 작은 I/O task의 개수

에 따라 시간이 달라집니다.

반대로 \(R\)을 줄이면 read/upload/GEMM은 줄지만 selector가 복잡해질 수 있습니다. 그러므로 “최소 row 수”가 아니라 “최소 actual total”을 찾아야 합니다.

---

## 6. 최적화 문제는 rate–distortion 문제와 같다

이 문제는 본질적으로 다음 형태입니다.

\[
\min_M T(M)
\quad\text{s.t.}\quad
E(M)\le\epsilon
\]

- \(T\): rate 또는 hardware cost
- \(E\): distortion 또는 실제 모델 오차

Lagrangian으로 바꾸면

\[
M_\lambda
=
\arg\min_M
\left[
T(M)+\lambda E(M)
\right]
\]

입니다.

\(\lambda\)가 크면 품질을 중요하게 보아 더 많은 row를 읽고, 작으면 latency를 중요하게 보아 더 적은 row를 읽습니다. \(\lambda\)를 조절하며 원하는 \(\epsilon\)을 만족시키는 지점을 찾을 수 있습니다.

중요한 점은 cardinality constraint가 사라진다는 것입니다.

\[
|M|=R
\]

을 강제하지 않으므로 최적화기가 자연스럽게 \(R^\star\)도 함께 결정합니다.

---

## 7. cell8 구조와 결합하면 온라인 최적화가 가능하다

길이 8 tile을 atomic unit으로 삼으면 tile \(q\)마다:

- 품질 기여량 \(s_q\)
- 실제 I/O cost
- 인접 tile과 합쳐질 때의 run cost
- lane scheduling cost

를 정의할 수 있습니다.

그 후 여러 tile count에 대해 다음 frontier를 한 번에 계산합니다.

\[
\mathcal F
=
\operatorname{Pareto}
\left\{
\left(S(M),T(M)\right)
\right\}
\]

온라인에서는 frontier에서

\[
M^\star
=
\arg\min_{M\in\mathcal F}
T(M)
\quad\text{s.t.}\quad
S(M)\ge\tau
\]

만 찾으면 됩니다.

이 방식은 “cell8인데 \(R\)은 고정”인 Experiment 29/31보다 일반적입니다. 결과적으로 easy case에서는 25%보다 아래로 내려갈 수 있고, hard case에서는 50%나 75% 이상을 사용할 수도 있습니다.

---

## 8. 전체 모델에서는 layer별로 예산을 다르게 배분해야 한다

모든 layer에 같은 품질 하한을 줄 이유도 없습니다. 최종 logit에 민감한 layer에는 더 많은 row를 주고, 둔감한 layer에서는 더 공격적으로 줄여야 합니다.

대략적인 end-to-end 제약을

\[
\sum_l \kappa_l E_l(M_l)
\le
\epsilon_{\mathrm{global}}
\]

로 두면 최적화는

\[
\min_{\{M_l\}}
\sum_l T_l(M_l)
\quad\text{s.t.}\quad
\sum_l \kappa_l E_l(M_l)
\le\epsilon_{\mathrm{global}}
\]

이 됩니다.

\(\kappa_l\)는 해당 layer projection error가 최종 logit/KL/NLL에 얼마나 증폭되는지를 calibration prompt에서 측정한 민감도입니다.

Lagrangian은 layer별로 분리됩니다.

\[
M_l^\star(\lambda)
=
\arg\min_{M_l}
\left[
T_l(M_l)+
\lambda\kappa_l E_l(M_l)
\right]
\]

하나의 \(\lambda\)를 조절하면 전체 오차 예산이 layer들 사이에 자동으로 배분됩니다. 통신 이론의 water-filling과 유사한 구조입니다.

이 단계까지 가야 “모든 projection에 같은 \(R\)”이라는 논문의 근본적인 제약을 벗어날 수 있습니다.

---

## 9. 기존 데이터가 주는 제한적인 증거

Experiment 32 결과에 사후적으로 Paper의 실제 projection error를 품질 기준으로 사용해 oracle을 계산하면:

- Paper 평균 actual total: 약 \(1.562\) ms
- 같은 \(R\)에서 최적 geometry 선택: 약 \(1.439\) ms
- \(R\)까지 자유롭게 한 oracle: 약 \(1.428\) ms
- Paper 대비 약 \(8.62\%\) 절약

다만 가변 \(R\)이 같은-\(R\) oracle에 추가로 준 이득은 약 \(0.81\%\)였고, 실제로 다른 budget을 택해 이긴 경우도 약 \(16.4\%\)였습니다.

이 결과는 통찰이 틀렸다는 뜻이 아닙니다. 기존 실험에는 \(R=25,50,75\%\) 세 점밖에 없어서 가변 \(R\)의 frontier를 거의 관측하지 못했습니다. 동시에 현재 범위에서는 \(R\) 변경보다 같은 \(R\)에서 geometry와 scheduling을 잘 선택하는 효과가 더 컸다는 뜻입니다.

그리고 이 oracle은 Paper가 얻은 실제 error를 사후적으로 알고 사용했으므로 배포 가능한 알고리즘은 아닙니다.

---

## 결론

우리의 중심 명제는 이렇게 써야 합니다.

\[
\boxed{
\text{Fixed sparsity ratio is not the optimization objective;}
\quad
\text{it should emerge from quality-constrained latency minimization.}
}
\]

더 구체적으로는:

\[
\boxed{
\min_{\{M_l\}}
T_{\mathrm{actual}}(\{M_l\})
\quad
\text{s.t.}\quad
E_{\mathrm{end-to-end}}(\{M_l\})
\le\epsilon
}
\]

입니다.

다만 논문 기여가 단순히 “\(R\)을 가변화했다”여서는 약합니다. rate–distortion formulation 자체는 고전적이기 때문입니다. 진짜 기여 후보는 다음의 결합입니다.

- 이 노트북의 실제 병렬 I/O schedule을 반영한 latency 함수
- weight-aware하고 측정 가능한 error surrogate
- tile/run 구조를 보존하는 빠른 Pareto-frontier solver
- layer별 sensitivity를 이용한 global error-budget allocation
- 실제 logit KL/NLL을 보장하도록 calibration된 adaptive \(R\)

따라서 다음 실험은 dense한 \(R\) sweep과 GEMM까지 포함한 total latency, 그리고 calibration prompt에서 정한 품질 하한을 holdout prompt에서 검증해야 합니다. 그 실험에서만 이것이 정말 Paper ideal을 이기는 핵심 통찰인지 판정할 수 있습니다.