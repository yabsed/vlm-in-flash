# Predicted-\(\lambda\) TD-2L(8) + Endpoint Trim

각 행 \(i\)의 중요도를 \(v_i\ge0\), 선택 마스크를 \(M\)이라 하겠습니다.

$$
I(M)=\sum_{i\in M}v_i
$$

목표는 필요한 importance \(Q\)를 유지하면서 lookup latency를 최소화하는 것입니다.

$$
\boxed{
\min_M L(M)
\qquad\text{s.t.}\qquad
I(M)\ge Q
}
$$

전체 알고리즘은 세 단계로 구성됩니다.

$$
\boxed{
\lambda\text{ 예측}
\;\longrightarrow\;
O(N)\text{ exact DP를 최대 8회}
\;\longrightarrow\;
\text{endpoint trim}
}
$$

---

## 1. 고정된 \(\lambda\)에서 scalarized 문제를 푼다

coverage 제약 대신 importance에 가격 \(\lambda\)를 붙입니다.

$$
\boxed{
M_\lambda
=
\arg\max_M
\left\{
\lambda I(M)-L(M)
\right\}
}
$$

* 작은 \(\lambda\): latency를 더 중시
* 큰 \(\lambda\): importance를 더 중시

각 \(\lambda\)에서 이 문제는 two-line latency 구조를 이용해 \(O(N)\)에 정확히 풀 수 있습니다.

---

# Part I. \(M_\lambda\)를 구하는 \(O(N)\) exact DP

## 2. 마스크를 연속된 run으로 분해한다

예를 들어

$$
M=(0,1,1,1,0,1,1,0)
$$

이면 선택된 run은

$$
[2,4],\qquad[6,7]
$$

입니다.

마스크의 latency는 각 run의 비용 합입니다.

$$
L(M)=\sum_{C\in\operatorname{runs}(M)}T(|C|)
$$

Two-line run 비용은

$$
T(\ell)=
\begin{cases}
a+c_1\ell,&\ell<s,\\[2mm]
c_2\ell,&\ell\ge s.
\end{cases}
$$

따라서 run \([i,j]\)을 선택했을 때의 scalarized 이득은

$$
\lambda\sum_{t=i}^{j}v_t-T(j-i+1)
$$

입니다.

---

## 3. 마지막 run을 기준으로 DP를 세운다

importance prefix sum을

$$
P_j=\sum_{t=1}^{j}v_t,\qquad P_0=0
$$

라고 정의합니다.

그리고

$$
D_j=
\text{처음 }j\text{개 행에서 얻을 수 있는 최대 scalarized score}
$$

라고 하겠습니다.

최적 마스크는 두 경우 중 하나입니다.

### \(j\)번째 행을 선택하지 않는 경우

$$
D_{j-1}
$$

### 마지막 run이 \([i,j]\)인 경우

행 \(i-1\)은 선택되지 않아야 하므로 앞부분의 최적값은 \(D_{i-2}\)입니다.

$$
D_{i-2}
+
\lambda(P_j-P_{i-1})
-
T(j-i+1)
$$

따라서 정확한 recurrence는

$$
\boxed{
D_j=
\max\left\{
D_{j-1},
\max_{1\le i\le j}
\left[
D_{i-2}
+\lambda(P_j-P_{i-1})
-T(j-i+1)
\right]
\right\}.
}
$$

모든 가능한 마지막 run \([i,j]\)을 고려하므로 정확하지만, 그대로 계산하면 \(O(N^2)\)입니다.

---

## 4. Two-line 구조로 시작점과 끝점을 분리한다

run 길이를

$$
\ell=j-i+1
$$

이라고 하겠습니다.

### 짧은 run: \(\ell<s\)

$$
T(\ell)=a+c_1\ell
$$

을 대입하면

$$
\begin{aligned}
&D_{i-2}
+\lambda(P_j-P_{i-1})
-a-c_1(j-i+1)\\
&=
\lambda P_j-a-c_1j
+
\left[
D_{i-2}-\lambda P_{i-1}+c_1(i-1)
\right].
\end{aligned}
$$

다음을 정의합니다.

$$
\boxed{
A_i^{(1)}
=
D_{i-2}-\lambda P_{i-1}+c_1(i-1)
}
$$

그러면 짧은 run의 최댓값은

$$
\lambda P_j-a-c_1j
+
\max_{\max(1,j-s+2)\le i\le j}A_i^{(1)}
$$

입니다.

이것은 최근 \(s-1\)개 시작점에 대한 sliding-window maximum입니다. Monotone deque를 사용하면 각 시작점이 한 번 들어가고 한 번 빠지므로 전체 계산량은 \(O(N)\)입니다.

---

### 긴 run: \(\ell\ge s\)

$$
T(\ell)=c_2\ell
$$

을 대입하면

$$
\begin{aligned}
&D_{i-2}
+\lambda(P_j-P_{i-1})
-c_2(j-i+1)\\
&=
\lambda P_j-c_2j
+
\left[
D_{i-2}-\lambda P_{i-1}+c_2(i-1)
\right].
\end{aligned}
$$

다음을 정의합니다.

$$
\boxed{
A_i^{(2)}
=
D_{i-2}-\lambda P_{i-1}+c_2(i-1)
}
$$

그러면 긴 run의 최댓값은

$$
\lambda P_j-c_2j
+
\max_{1\le i\le j-s+1}A_i^{(2)}
$$

입니다.

여기서 시작점 범위는 계속 커지기만 하므로 running maximum 하나만 유지하면 됩니다.

---

## 5. 최종 \(O(N)\) recurrence

$$
\boxed{
D_j=
\max
\begin{cases}
D_{j-1},\\[1mm]
\lambda P_j-a-c_1j+
\displaystyle\max_{\max(1,j-s+2)\le i\le j}A_i^{(1)},\\[3mm]
\lambda P_j-c_2j+
\displaystyle\max_{1\le i\le j-s+1}A_i^{(2)}.
\end{cases}
}
$$

각 \(j\)에서 필요한 작업은 다음뿐입니다.

* skip 후보 계산: \(O(1)\)
* 짧은 run의 deque 갱신: amortized \(O(1)\)
* 긴 run의 running maximum 갱신: \(O(1)\)

따라서

$$
\boxed{
M_\lambda\text{ 계산 시간}=O(N)
}
$$

입니다.

각 \(D_j\)가 어느 선택에서 왔는지 parent pointer를 저장하면 실제 마스크도 \(O(N)\)에 복원할 수 있습니다.

이 DP가 정확한 이유는 모든 마스크가 유일한 마지막 run \([i,j]\)을 가지며, DP가 모든 가능한 마지막 run을 고려하기 때문입니다. Deque는 후보를 근사적으로 줄이는 것이 아니라 동일한 최댓값을 빠르게 계산합니다.

---

# Part II. 목표 \(Q\) 근처의 \(\lambda\)를 예측하고 보정한다

## 6. 선형 완화로 첫 \(\lambda\)를 예측한다

청크 시작 비용과 위치 관계를 잠시 무시하고

$$
L(M)\approx c_2|M|
$$

라고 근사합니다.

그러면 scalarized objective는

$$
\lambda I(M)-L(M)
\approx
\sum_i(\lambda v_i-c_2)x_i
$$

가 됩니다.

행 \(i\)는

$$
\lambda v_i-c_2>0
$$

즉,

$$
v_i>\frac{c_2}{\lambda}
$$

일 때 선택됩니다.

importance를 내림차순으로 정렬하여

$$
v_{(1)}\ge v_{(2)}\ge\cdots\ge v_{(N)}
$$

라고 하고,

$$
k=
\min\left\{
r:
\sum_{j=1}^{r}v_{(j)}\ge Q
\right\}
$$

를 구합니다.

목표 \(Q\)에 도달하기 위해 필요한 마지막 행의 importance를

$$
\tau_Q=v_{(k)}
$$

라고 하면, 선택 경계에서

$$
\lambda\tau_Q\approx c_2
$$

이므로 첫 multiplier를

$$
\boxed{
\lambda_0=\frac{c_2}{\tau_Q}
}
$$

로 예측합니다.

---

## 7. 예측값에서 exact DP를 실행한다

$$
M_0
=
\arg\max_M
\left\{
\lambda_0I(M)-L_{\mathrm{2L}}(M)
\right\}
$$

을 앞의 \(O(N)\) DP로 정확히 계산합니다.

* \(I(M_0)<Q\): \(\lambda_0\)가 작음
* \(I(M_0)\ge Q\): \(\lambda_0\)가 큼

목표를 만족하지 못하는 해를

$$
A=(I_A,L_A),\qquad I_A<Q
$$

목표를 만족하는 해를

$$
B=(I_B,L_B),\qquad I_B\ge Q
$$

로 유지합니다.

초기 빈 마스크와 전체 마스크는 계산 없이 각각 infeasible·feasible endpoint로 사용할 수 있습니다.

---

## 8. 두 endpoint의 교차점으로 \(\lambda\)를 보정한다

두 해의 scalarized score가 같아지는 지점은

$$
\lambda I_A-L_A
=
\lambda I_B-L_B
$$

이므로

$$
\boxed{
\lambda_{\mathrm{next}}
=
\frac{L_B-L_A}{I_B-I_A}
}
$$

입니다.

이 값에서 다시 exact DP를 실행합니다.

$$
M_{\mathrm{next}}
=
\arg\max_M
\left\{
\lambda_{\mathrm{next}}I(M)-L_{\mathrm{2L}}(M)
\right\}
$$

그리고

$$
I(M_{\mathrm{next}})<Q
\quad\Longrightarrow\quad
A\leftarrow M_{\mathrm{next}}
$$

$$
I(M_{\mathrm{next}})\ge Q
\quad\Longrightarrow\quad
B\leftarrow M_{\mathrm{next}}
$$

로 bracket을 좁힙니다.

이 과정을 exact DP 호출이 최대 8회가 될 때까지 반복합니다.

$$
\boxed{
\tau_Q
\longrightarrow
\lambda_0
\longrightarrow
M_{\lambda_0}
\longrightarrow
\lambda_{\mathrm{next}}
\longrightarrow
\cdots
\longrightarrow
B
}
$$

마지막 feasible endpoint \(B\)가 TD-2L의 출력입니다.

---

# Part III. Endpoint trim으로 남은 낭비를 제거한다

## 9. Supported solution의 overshoot

TD-2L이 반환한 마스크를 \(M^{(0)}\)이라 하겠습니다.

$$
I(M^{(0)})\ge Q
$$

일반적으로 importance가 정확히 \(Q\)와 같지는 않고 약간 초과합니다. 이 surplus는

$$
\boxed{
\sigma=I(M^{(0)})-Q\ge0
}
$$

입니다.

이 surplus 안에서 중요도가 낮은 행을 제거하면 coverage를 유지하면서 latency를 더 줄일 수 있습니다.

---

## 10. run의 양 끝 행만 제거한다

길이 \(\ell\)인 run의 endpoint \(e\)를 제거한다고 하겠습니다.

importance 손실은

$$
\Delta I_e=v_e
$$

이고, latency 절감은

$$
\boxed{
\Delta L_e=T(\ell)-T(\ell-1)
}
$$

입니다. 단일 행 run에는 \(T(0)=0\)을 사용합니다.

제거가 가능한 조건은

$$
\Delta I_e\le\sigma
$$

이며, 제거 효율은

$$
\boxed{
\rho_e=
\frac{\Delta I_e}{\Delta L_e}
}
$$

로 평가합니다.

\(\rho_e\)가 작을수록 적은 importance를 버리고 많은 latency를 줄일 수 있습니다.

따라서 가능한 endpoint 중

$$
e^*=\arg\min_e\rho_e
$$

를 제거합니다.

그 후

$$
\sigma\leftarrow\sigma-v_{e^*}
$$

로 surplus를 갱신하고, 짧아진 run에서 새롭게 노출된 endpoint를 후보에 추가합니다.

---

## 11. 왜 내부 행은 제거하지 않는가

run 내부의 행을 제거하면 하나의 run이 두 개로 갈라질 수 있습니다.

$$
[\,i,\ldots,e-1,e+1,\ldots,j\,]
$$

그러면 새로운 청크 시작 비용이 생겨 latency가 오히려 증가할 수 있습니다.

반면 endpoint를 제거하면 run 하나가 단순히 짧아집니다.

$$
[i,j]\longrightarrow[i+1,j]
$$

또는

$$
[i,j]\longrightarrow[i,j-1].
$$

따라서 latency 변화가 정확히

$$
T(\ell)-T(\ell-1)
$$

로 계산됩니다.

---

## 12. Trim의 불변조건

모든 삭제는 다음 두 조건을 만족할 때만 허용합니다.

$$
v_e\le\sigma
$$

$$
\Delta L_e>0
$$

그러므로 삭제 후에도

$$
I(M')=I(M)-v_e\ge Q
$$

이고

$$
L(M')=L(M)-\Delta L_e<L(M)
$$

입니다.

즉 trim은 항상

$$
\boxed{
\text{coverage 유지}
\quad+\quad
\text{latency 감소}
}
$$

를 보장합니다.

이를 최대 \(E\)회 반복합니다.

* `trim 64`: 최대 64개 endpoint 삭제
* `trim 256`: 최대 256개 endpoint 삭제

---

# 전체 알고리즘

$$
\boxed{
\begin{aligned}
&\textbf{1. Predict:}
&&\lambda_0=\frac{c_2}{\tau_Q}\\[1mm]
&\textbf{2. Correct:}
&&M_\lambda\text{를 }O(N)\text{ exact DP로 최대 8회 계산}\\[1mm]
&\textbf{3. Select:}
&&I(B)\ge Q\text{인 마지막 feasible endpoint 선택}\\[1mm]
&\textbf{4. Trim:}
&&\rho_e=\frac{v_e}{T(\ell)-T(\ell-1)}
\text{가 작은 endpoint부터 제거}
\end{aligned}
}
$$

최종 출력은

$$
\boxed{
M_{\mathrm{out}}
=
\operatorname{Trim}
\left(
\operatorname{TD\text{-}2L}_8(\lambda_0)
\right)
}
$$

이며

$$
I(M_{\mathrm{out}})\ge Q
$$

가 유지됩니다.

---

## 정확성과 복잡도

| 부분                | 성질                                         |
| ----------------- | ------------------------------------------ |
| 고정 \(\lambda\) DP | two-line scalarized objective의 전역 최적해      |
| 8회 \(\lambda\) 탐색 | 목표 \(Q\) 주변의 supported solution 탐색         |
| Endpoint trim     | coverage를 유지하며 latency를 단조 감소시키는 heuristic |
| 전체 방법             | 원래 constrained problem의 전역 최적성은 보장하지 않음    |

현재 구현은 \(\tau_Q\) 계산에 정렬을 사용하므로

$$
O(N\log N)+8O(N)+O(N+E\log N)
$$

이고 전체적으로

$$
\boxed{O(N\log N)}
$$

입니다.

threshold 예측을 선형시간 selection 또는 histogram으로 바꾸고, endpoint 후보를 heapify로 구성하면 고정된 \(E\)에 대해

$$
\boxed{O(N)}
$$

으로 구현할 수 있습니다.

핵심은 다음 한 줄로 압축됩니다.

$$
\boxed{
\text{선형 완화로 목표 근처의 }\lambda\text{를 예측하고,}
\quad
\text{exact }O(N)\text{ DP로 보정한 뒤,}
\quad
\text{남은 coverage surplus를 endpoint trim으로 회수한다.}
}
$$

현재 실험의 `Lambda predict + correct (8)` 결과에는 trim이 붙어 있지 않습니다. 따라서 `Predicted-\lambda TD-2L(8)+trim`은 이 방법들을 결합한 다음 실험 후보입니다.
