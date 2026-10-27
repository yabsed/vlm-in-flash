### 2.1 논문에서 제안한 문제의 실제 최적해

원 논문의 가산 비용 모델에서 $1\le R\le N$이고 모든 길이 $1\le\ell\le R$에 대해 $T[\ell]>0$이라 하자.

행 예산 $1\le\|M\|_1\le R$ 아래에서 임의의 최적 mask $M^\star$를 생각하자. $\mathcal C(M^\star)$의 구간들은 선택된 행들을 분할하므로

$$
\begin{aligned}
\frac{I(M^\star)}{\widehat L_{\mathrm{table}}(M^\star)}
&=
\sum_{C\in\mathcal C(M^\star)}
\underbrace{
\frac{T[|C|]}{\widehat L_{\mathrm{table}}(M^\star)}
}_{w_C}
\underbrace{
\frac{\sum_{i\in C}v_i}{T[|C|]}
}_{U(C)}\\
&=
\sum_{C\in\mathcal C(M^\star)}w_CU(C)
\le
\max_{C\in\mathcal C(M^\star)}U(C).
\end{aligned}
$$

여기서 $w_C>0$이고 $\sum_Cw_C=1$이므로, 전체 비율은 각 구간의 비율을 latency로 가중한 평균이다. 따라서 가장 큰 구간 비율을 넘을 수 없다.

$U(C)$가 가장 큰 구간 $C^\star\in\mathcal C(M^\star)$만 남긴 mask를 $\widetilde M$이라 하자. $1\le\|\widetilde M\|_1=|C^\star|\le\|M^\star\|_1\le R$이므로 $\widetilde M$도 행 예산을 만족한다. 위 부등식에 의해 $\widetilde M$의 비율은 $M^\star$보다 작지 않고, $M^\star$의 최적성에 의해 더 클 수도 없다. 따라서 $\widetilde M$도 최적해다.

결국 단일 연속 구간으로 이루어진 최적해가 존재하며, 원래 문제는 길이가 $R$ 이하인 모든 연속 구간 중 $U(C)$가 가장 큰 구간을 찾는 문제로 환원된다. 앞서 정의한 $C_{j,\ell}=\{j,\ldots,j+\ell-1\}$을 사용하면

$$
\max_{\substack{M\in\{0,1\}^N\\1\le\|M\|_1\le R}}
\frac{I(M)}{\widehat L_{\mathrm{table}}(M)}
=
\max_{\substack{1\le\ell\le R\\0\le j\le N-\ell}}
U(C_{j,\ell}).
$$

오른쪽의 비교 대상은 후보 집합 $\mathcal G$에 한정되지 않는 모든 연속 구간이다. 또한 이 결론은 행 수의 상한 제약에 대한 것이며, 정확히 $R$개 행을 선택하는 문제에 그대로 적용되지는 않는다.

위 증명은 [원 논문 §3.1–3.2.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5)의 모델과 목적식을 바탕으로 한 본 연구의 분석이다.
