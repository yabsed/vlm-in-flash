### 2.5 최소 구간 길이 제약과 최적값의 보장

Two-line 모델에서는 같은 수의 행을 선택하고 모든 구간이 충분히 길면 읽기 비용이 동일해진다. 이 성질을 이용하여 길이 제약 아래에서의 최적화 문제를 정리하고, 중요도 보존율로 원래 최적값에 얼마나 가까운지 평가해 보자.

mask $M$의 중요도–지연시간 비율을 $q(M)=I(M)/L(M)$이라 하자.

#### 2.5.1 길이 제약 아래에서의 최적해

정확히 $R$개 행을 선택하는 문제에, 모든 최대 연속 구간 $C\in\mathcal C(M)$의 길이가 $\ell_0$ 이상이어야 한다는 조건을 추가하자. 이 조건이 가능하도록 $\lceil\ell_0\rceil\le R\le N$을 가정한다. 이때 길이 $R$의 단일 구간은 항상 조건을 만족한다.

각 구간은 $T(|C|)=\alpha|C|$를 만족하고, 구간 길이의 합은 $R$이므로 전체 읽기 비용은

$$
\begin{aligned}
L(M)
&=\sum_{C\in\mathcal C(M)}T(|C|)\\
&=\alpha\sum_{C\in\mathcal C(M)}|C|
=\alpha R.
\end{aligned}
$$

따라서 선택한 행의 위치나 구간의 수가 달라도 읽기 비용은 같다. 중요도–지연시간 비율의 분모가 고정되므로, 이 조건 아래에서는 중요도를 최대화하는 mask가 비율도 최대화한다.

그 mask를 $M_{\mathrm{sat}}^\star$, 보존 중요도를 $I_{\mathrm{sat}}=I(M_{\mathrm{sat}}^\star)$이라 하자. 길이 제약 아래에서의 최적 비율은

$$
q_{\mathrm{sat}}
=q(M_{\mathrm{sat}}^\star)
=\frac{I_{\mathrm{sat}}}{\alpha R}
$$

이다.

#### 2.5.2 일반적인 mask에 대한 보장

이제 같은 행 예산에서 길이 조건을 두지 않은 문제의 최적 비율 $q^\star$와 비교하자.

중요도가 가장 높은 $R$개 행을 선택한 Top-$R$ mask를 $M_T$, 그 중요도를 $I_T=I(M_T)$라 하자. 정확히 $R$개 행을 선택한 모든 mask는 $I(M)\le I_T$를 만족한다. 또한 two-line 모델에서 $L(M)\ge\alpha R$이므로

$$
q^\star\le\frac{I_T}{\alpha R}.
$$

이 상한은 중요도 상한 $I_T$와 비용 하한 $\alpha R$을 결합한 것이다. 두 값을 동시에 달성하는 mask가 존재하지 않더라도 성립한다.

$I_T>0$이면 $q^\star>0$이다. 정확히 $R$개 행을 선택하는 임의의 mask $M$에 대해 $q(M)\ge0$이므로

$$
\boxed{
\begin{aligned}
\frac{q(M)}{q^\star}
&\ge\frac{q(M)}{I_T/(\alpha R)}\\
&=\frac{I(M)}{I_T}\,
\frac{\alpha R}{L(M)}.
\end{aligned}
}
$$

오른쪽 마지막 식의 두 항은 각각 Top-$R$ 대비 중요도 보존율과 읽기 효율의 상한 대비 달성률이다. 후자는 행 단위 처리량 $R/L(M)$을 그 상한 $1/\alpha$로 나눈 값이다. 두 항은 모두 $0$과 $1$ 사이에 있으며, 그 곱으로 최적 비율 대비 달성률의 하한을 얻는다.

#### 2.5.3 중요도 보존율에 따른 보장

모든 최대 연속 구간의 길이가 $\ell_0$ 이상인 mask는 비용 하한 $L(M)=\alpha R$을 달성한다. 따라서 앞 절의 보장에서 읽기 효율에 해당하는 항이 $1$이 되어

$$
\frac{q(M)}{q^\star}
\ge
\frac{I(M)}{I_T}
\underbrace{\frac{\alpha R}{L(M)}}_{=1}
=\frac{I(M)}{I_T}.
$$

이 조건 아래에서는 읽기 비용을 더 줄일 수 없으므로, 중요도 보존율만으로 최적 비율 대비 달성률을 보장할 수 있다. 중요도 상한 $I_T$에 도달하지 못하더라도, 비용 쪽의 하한은 이미 달성한 상태이기 때문이다.

특히 길이 제약 아래에서의 최적 mask $M_{\mathrm{sat}}^\star$에 적용하면

$$
\boxed{
\frac{q_{\mathrm{sat}}}{q^\star}
\ge
\frac{I_{\mathrm{sat}}}{I_T}.
}
$$

$M_{\mathrm{sat}}^\star$도 원래 문제에서 선택할 수 있는 mask이므로 $q_{\mathrm{sat}}\le q^\star$이다. 따라서 길이 제약 아래에서 Top-$R$ 중요도의 $95\%$ 이상을 보존하면, 중요도–지연시간 비율도 원래 최적값의 $95\%$ 이상이다.

이 보장은 원래 최적 mask를 직접 구하지 않고도 $I_{\mathrm{sat}}$과 $I_T$만으로 확인할 수 있다. 여기서 비용 하한의 달성과 비율에 대한 보장은 two-line 모델의 추정 비용을 기준으로 한다.
