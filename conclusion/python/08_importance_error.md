# 08 · importance는 무엇을 보장하는가: 새 activation과 실제 projection 오차

## 입력과 관찰 범위

로컬 원본 SmolLM2 checkpoint를 읽고 고정된 두 텍스트 prompt를 **새로 forward**한다. 12번 MLP down projection의 입력 $X\in\mathbb R^{t\times n}$과 가중치 $W\in\mathbb R^{n\times d}$를 사용한다. 이전 activation·importance·error 결과는 읽지 않는다. model snapshot, weight SHA256, prompt, layer, dtype를 기록한다. `VLM_CHECKPOINT`로 호환되는 로컬 원본 checkpoint를 지정할 수 있다. 없으면 명확히 실패하며 synthetic 데이터로 대체하지 않는다.

이는 LLM projection 실험이다. VLM의 visual tokens나 downstream task accuracy를 측정했다는 뜻은 아니다. 모든 mask는 고정 dense forward에서 얻은 같은 $X,W$로 평가하므로 여러 층을 동시에 sparsify한 누적 오차도 아니다.

$$Y=XW,\qquad Y_M=X\operatorname{diag}(M)W,\qquad
E(M)=\frac{\|Y-Y_M\|_F}{\|Y\|_F}.$$

## 두 importance

$$v_i^{abs}=\frac1t\sum_u|X_{ui}|,\qquad
v_i^{energy}=\|X_{:i}\|_2^2\|W_{i:}\|_2^2.$$

후자는 rank-one contribution $A_i=X_{:i}W_{i:}$의 Frobenius norm 제곱이다. $I_v(M)=\sum_iv_iM_i$를 사용한다. $v^{abs}$를 많이 보존하는 것과 projection error 최소화는 일반적으로 동치가 아니다.

## 정리 6 · energy importance로 얻는 deterministic bound

누락 집합 $U=\{i:M_i=0\}$에 대해

$$Y-Y_M=\sum_{i\in U}A_i,\qquad
\|Y-Y_M\|_F\le\sum_{i\in U}\|A_i\|_F
\le\sqrt{|U|\sum_{i\in U}\|A_i\|_F^2}.$$

따라서 $D=\|Y\|_F^2>0$일 때

$$\boxed{E(M)^2\le\frac{(n-|M|)\,[I_{energy}(\mathbf1)-I_{energy}(M)]}{D}.}$$

$(n-|M|)$ 인자를 지운 식은 일반적으로 보장이 아니다. 서로 정렬된 contribution이 있으면 cross term이 누적된다. 두 contribution이 서로 상쇄하면 반대로 sum of energies가 실제 error를 크게 과대평가할 수도 있다.

$$\|\sum_{i\in U}A_i\|_F^2=\sum_{i\in U}\|A_i\|_F^2+2\sum_{i<j\in U}\langle A_i,A_j\rangle_F.$$

pairwise orthogonality가 성립하면 첫 항만 남지만 trained projection에서 이를 무검증으로 가정하지 않는다. bound가 실제 error보다 매우 큰 경우는 정리가 틀렸다는 뜻이 아니라 실용적 certificate가 약하다는 뜻이다.

## 비교 설계

같은 실제 $X,W$에서 abs/energy score 각각으로 Top-$R$, paper greedy, saturation tile mask를 새로 만든다. normalized importance, $E$, 위 bound, 선택 수를 모두 계산한다. 이 노트북의 $T$는 명시적인 **two-line 수학 모델**이며 SSD 실측값으로 표시하지 않는다. $s=32$도 가정한 행 길이다. 실제 I/O 연구와 error proxy 연구의 인과를 섞지 않기 위함이다.

모든 점에서 bound 위반 여부를 assert한다. $I$–$E$ scatter, error–model latency, score별 분포와 대표 mask를 그린다. synthetic 두-column 예제로 같은 importance에서도 상쇄 때문에 error가 달라지는 현상을 별도로 재현한다.

## 해석 기준

importance가 같으면 품질이 같다는 주장을 입증하려는 실험이 아니다. 대리 목적의 유효 범위와 실패 양상을 문서화한다. downstream 품질 결론에는 별도의 task 데이터와 전체 모델 forward가 필요하며, 이 노트북은 그 결과를 대신하지 않는다.
