### 2.4 타일 기반 행 선택

Tiles는 일정 길이의 타일을 단위로 행을 선택하고, 남은 예산을 하나의 연속 구간으로 채운다.

타일 길이를 양의 정수 $B$라 하자. 시작 행을 $0$과 $d=\lfloor B/2\rfloor$로 두면 두 배치의 타일은

$$
\begin{aligned}
\mathcal T_1&=\{C_{0,B},C_{B,B},C_{2B,B},\ldots\},\\
\mathcal T_2&=\{C_{d,B},C_{d+B,B},C_{d+2B,B},\ldots\}
\end{aligned}
$$

이다. 끝 행이 $N-1$을 넘는 타일은 제외하고, 각 배치에서 mask를 독립적으로 구성한다.

행 예산을

$$
R=KB+r,
\qquad
K=\left\lfloor\frac RB\right\rfloor,
\qquad
0\le r<B
$$

로 나누고, 각 배치에서 중요도 $\sum_{i\in C}v_i$가 큰 $K$개 타일을 선택한다.

$r>0$이면 선택한 타일과 겹치지 않는 길이 $r$의 연속 구간 중 중요도가 가장 큰 구간을 추가한다. $R<B$이면 중요도가 가장 큰 길이 $R$의 연속 구간을 선택한다.

예산을 채운 mask 중 $I(M)/L(M)$이 큰 것을 $M_{\mathrm{Tiles}}$로 선택한다.

$B\ge\ell_0$이고 $R\ge B$이면, two-line 모델의 추가 비용은 완전 타일과 합쳐지지 않은 잔여 구간에서만 발생할 수 있다. 따라서

$$
\alpha R
\le L(M_{\mathrm{Tiles}})
\le\alpha(R-r)+T(r).
$$

이다. $r=0$이거나 잔여 구간이 완전 타일과 합쳐지면 $L(M_{\mathrm{Tiles}})=\alpha R$이다. 한편 $B=1$이면 Tiles는 Top-$R$과 같다.

중요도 누적합과 상위 $K$개 타일의 부분 선택을 이용하기 때문에, 고정된 $B$에서 시간 복잡도는 평균 $O(N)$, 메모리 복잡도는 $O(N)$이다.