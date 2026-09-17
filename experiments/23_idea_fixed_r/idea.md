가능합니다. 오히려 **\(R\)을 고정하면 two-line 구조가 아주 깔끔하게 드러납니다.** 다만 현재의 \(O(N)\) 알고리즘을 그대로 쓰면서 전역 최적성을 유지할 수 있는 것은 아닙니다.

## 1. 새 목적함수

행 중요도를 \(v_i\), 선택 마스크를 \(M\)이라 하면

$$
I(M)=\sum_{i\in M}v_i,\qquad R(M)=|M|.
$$

새 문제는

$$
\boxed{
\max_{M:\,|M|=R}\frac{I(M)}{L(M)}
}
$$

입니다. 이는 “동일한 row 계산량에서 lookup latency 1ms당 최대한 많은 importance를 가져오라”는 문제입니다.

## 2. 꺾이는 점 구조를 적용하면

$$
\delta=c_2-c_1,\qquad a=\delta s
$$

이고 오른쪽 직선이 원점을 지나므로

$$
L(M)=c_2R(M)+\delta H(M),
$$

$$
H(M)=\sum_{C\in\operatorname{runs}(M)}(s-|C|)_+.
$$

\(R\)을 고정하면

$$
\boxed{
\frac{I(M)}{L(M)}
=
\frac{I(M)}{c_2R+\delta H(M)}
}
$$

입니다.

즉 알고리즘은 정확히 다음 두 가지를 동시에 추구합니다.

* importance \(I(M)\)를 크게 한다.
* 짧고 파편화된 run의 총 shortfall \(H(M)\)를 작게 한다.

길이 \(s\) 이상인 run은 별도의 구조적 벌점을 받지 않습니다. 고정된 \(R\)에서는 \(c_2R\)이 모든 마스크에 공통인 baseline이기 때문입니다.

## 3. 비율 문제를 scalarized DP로 바꾸기

최적 비율을 \(\rho^*\)라 하겠습니다.

$$
\rho^*=\max_{|M|=R}\frac{I(M)}{L(M)}.
$$

다음 함수를 정의합니다.

$$
F(\rho)=\max_{|M|=R}\{I(M)-\rho L(M)\}.
$$

그러면

$$
\begin{cases}
F(\rho)>0,&\rho<\rho^*,\\
F(\rho)=0,&\rho=\rho^*,\\
F(\rho)<0,&\rho>\rho^*.
\end{cases}
$$

따라서 Dinkelbach 반복을 사용할 수 있습니다.

$$
M_t
=
\arg\max_{|M|=R}
\{I(M)-\rho_tL(M)\},
$$

$$
\boxed{
\rho_{t+1}=\frac{I(M_t)}{L(M_t)}
}
$$

이고

$$
I(M_t)-\rho_tL(M_t)=0
$$

이 되면 \(M_t\)가 원래 비율 문제의 전역 최적해입니다.

고정된 \(R\)에서는

$$
I(M)-\rho L(M)
=
I(M)-\rho\delta H(M)-\rho c_2R.
$$

마지막 항은 상수이므로 내부 문제는

$$
\boxed{
\max_{|M|=R}
\{I(M)-\rho\delta H(M)\}
}
$$

로 단순화됩니다.

## 4. 기존 fixed-\(\lambda\) DP와의 관계

기존 알고리즘은

$$
\max_M\{\lambda I(M)-L(M)\}
$$

을 풉니다. 그런데

$$
I(M)-\rho L(M)
=
\rho\left(\frac{1}{\rho}I(M)-L(M)\right)
$$

이므로

$$
\boxed{\lambda=\frac1\rho}
$$

라고 보면 scalarized 목적함수 자체는 동일합니다.

따라서 기존의 핵심 아이디어인

* 모든 마지막 run 고려
* two-line 비용 분리
* sliding maximum
* running maximum

은 그대로 재사용할 수 있습니다.

차이는 단 하나입니다.

$$
\boxed{|M|=R}
$$

을 정확히 강제해야 한다는 것입니다.

## 5. 정확한 fixed-\(R\) DP

중요도 prefix sum을

$$
P_j=\sum_{i=1}^jv_i
$$

라 하고,

$$
D_{j,r}
=
\text{처음 }j\text{개 행에서 정확히 }r\text{개를 선택한 최대 점수}
$$

라고 합시다.

마지막 run의 길이를 \(\ell\)이라 하면 정확한 recurrence는

$$
\boxed{
D_{j,r}
=
\max\left\{
D_{j-1,r},
\max_{1\le\ell\le\min(j,r)}
\left[
D_{j-\ell-1,r-\ell}
+
(P_j-P_{j-\ell})
-\rho T(\ell)
\right]
\right\}.
}
$$

* 첫 항: \(j\)번째 행을 선택하지 않음
* 둘째 항: 길이 \(\ell\)인 마지막 run을 선택함

모든 마지막 run을 검사하므로 정확합니다. Two-line 선형성을 이용해 각 \((j,r)\)의 내부 최댓값을 sliding/running maximum으로 관리하면 한 번의 fixed-\(\rho\) solve를

$$
\boxed{O(NR)}
$$

까지 줄일 수 있습니다.

Dinkelbach 반복 횟수를 \(K\)라 하면 전체는

$$
\boxed{O(KNR)}
$$

입니다.

즉 전역 최적해는 구할 수 있지만, \(R\)이 수천이면 현재의 \(O(N)\) online DP보다 훨씬 무겁습니다. 이것은 2ms용이라기보다는 **offline oracle**에 가깝습니다.

## 6. 2ms용 근사 버전

정확한 cardinality 차원을 저장하는 대신 \(R\)에 가격 \(\mu\)를 붙입니다.

$$
\boxed{
M_{\rho,\mu}
=
\arg\max_M
\left\{
I(M)-\rho L(M)-\mu|M|
\right\}.
}
$$

* \(\mu\)가 크면 적은 행을 선택
* \(\mu\)가 작으면 많은 행을 선택

각 행의 점수를 단순히

$$
v_i\longmapsto v_i-\mu
$$

로 바꾸면 되므로, 기존 two-line DP로 각 \((\rho,\mu)\) 문제를 여전히

$$
O(N)
$$

에 풀 수 있습니다.

실용적인 online 알고리즘은 다음과 같습니다.

1. \(\rho\)와 \(\mu\)를 예측한다.
2. \(O(N)\) DP를 실행한다.
3. 선택 행 수가 \(R\)보다 많으면 \(\mu\)를 증가시킨다.
4. 선택 행 수가 \(R\)보다 적으면 \(\mu\)를 감소시킨다.
5. 4–8회 보정한다.
6. 마지막에 endpoint trim/fill로 정확히 \(R\)개를 맞춘다.
7. \(\rho\leftarrow I/L\)로 한두 번 갱신한다.

이 방식은 대략

$$
O(KN)
$$

이고 현재 `Predicted-λ TD-2L`과 유사한 속도를 기대할 수 있습니다. 다만 \(\mu\)-scalarization이 특정 정확한 \(R\)의 unsupported 해를 건너뛸 수 있으므로 전역 최적 보장은 없습니다.

## 이 목적함수의 주의점

\(I/L\) 최대화는 단위 문제를 완전히 없애는 것은 아닙니다. 단위는

$$
\frac{\text{importance}}{\text{ms}}
$$

가 됩니다. 다만 importance 전체에 상수를 곱해도 최적 마스크가 바뀌지 않으므로 정규화에는 강합니다.

더 중요한 문제는 다음과 같은 선택입니다.

$$
(I,L)=(0.9,1.0)
\quad\Rightarrow\quad I/L=0.9,
$$

$$
(I,L)=(0.7,0.6)
\quad\Rightarrow\quad I/L\approx1.17.
$$

비율 목적은 두 번째를 선택하지만, 실제 모델에는 \(I=0.9\)가 필요할 수 있습니다. 따라서 이것은 coverage 목적을 대체하기보다는 보조 평가로 쓰는 것이 좋습니다.

제가 권하는 구조는 다음입니다.

$$
\boxed{
\begin{aligned}
\text{주 목적:}\quad&
\min L(M)
&&\text{s.t. }I(M)\ge Q,\ |M|\le R,\\[1mm]
\text{보조 목적:}\quad&
\max \frac{I(M)}{L(M)}
&&\text{s.t. }|M|=R.
\end{aligned}
}
$$

결론적으로, **가능하며 수학적으로도 아주 자연스럽습니다.** 특히

$$
L(M)=c_2R+\delta H(M)
$$

덕분에 fixed-\(R\) 비율 최적화의 본질이

$$
\boxed{\text{importance 최대화}-
\text{짧은 run의 fragmentation 벌점}}
$$

으로 정확히 드러납니다. 전역해는 \(O(KNR)\) offline oracle로, 2ms 후보는 row-price \(\mu\)를 추가한 \(O(KN)\) target-directed 근사로 만드는 것이 가장 현실적입니다.
