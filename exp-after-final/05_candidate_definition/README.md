# 1.2절 보완: 후보 구간 C의 정의

## 교체 문단

기존의 “실제 알고리즘은 연속 후보 구간 $C$마다…”로 시작하는 문단을 다음과 같이 보완한다.

---

실제 알고리즘은 중요도–지연시간 비율이 높은 mask $M$을 구성하기 위해 연속 후보 구간들의 집합 $\mathcal G$를 생성한다. 각 후보 $C\in\mathcal G$에 대해

$$
U(C)=\frac{\sum_{i\in C}v_i}{T[|C|]}
$$

를 계산하고, 점수가 높은 후보부터 이미 선택한 행과의 겹침 및 잔여 예산을 검사한다. 조건을 만족하면 해당 구간의 행들을 선택하도록 mask를 갱신하는 greedy 방식을 이용한다. [원 논문 §3.2.2, p. 6](https://arxiv.org/pdf/2511.18692v1#page=6), [Algorithm 1, p. 17](https://arxiv.org/pdf/2511.18692v1#page=17)

여기서 후보 구간 $C$는 여러 길이의 window를 행 인덱스 축을 따라 이동시켜 생성한다. 시작 행이 $j$이고 길이가 $\ell$인 후보를

$$
C_{j,\ell}=\{j,j+1,\ldots,j+\ell-1\},
\qquad 0\le j\le N-\ell
$$

로 정의한다. 행 단위로 환산한 최소·최대 후보 길이를 $\ell_{\min},\ell_{\max}$, 길이 증가 간격을 $\Delta\ell$, 시작 위치의 이동 간격 상한을 $J$라 하면, 원 논문이 고려하는 $(j,\ell)$ 쌍은 다음과 같다.

$$
\begin{aligned}
\ell&=\ell_{\min}+k\Delta\ell\le\min(\ell_{\max},N),\\
j&=m\min(\ell,J)\le N-\ell,
\end{aligned}
\qquad k,m=0,1,2,\ldots
$$

후보 길이는 $\Delta\ell$ 간격으로, 시작 위치는 $\min(\ell,J)$ 간격으로 정한다. 이 규칙은 $\ell>J$일 때 같은 길이의 후보 간 중첩을 허용한다. 최대 후보 크기는 장치의 처리량 포화 지점을 기준으로 정하고, $\ell_{\min}$, $\Delta\ell$, $J$는 탐색의 세밀함과 계산량을 조절하는 매개변수로 사용한다. [§3.2.2, p. 6](https://arxiv.org/pdf/2511.18692v1#page=6), [Algorithm 1, p. 17](https://arxiv.org/pdf/2511.18692v1#page=17)

---

## 표기상의 구분

$\mathcal G$는 **선택 전에 생성한 후보 구간의 집합**이고, $\mathcal C(M)$는 **선택 결과인 mask $M$의 최대 연속 구간 집합**이다. 따라서 utility를 계산하는 후보에는 $C\in\mathcal G$라고 쓴다.

후보가 선택되지 않을 수도 있고, 선택된 인접 후보들이 하나의 최대 연속 구간으로 합쳐질 수도 있으므로, 후보를 처음부터 $C\in\mathcal C(M)$라고 정의하면 안 된다.
