### 2.2 포화 제약 구간별 선형 지연 모델의 성질

앞 절에서 정한 $\|M\|_1=R$ 아래에서, 선택된 행들의 배치가 읽기 비용에 미치는 영향을 살펴보자. 선행연구의 구간별 비용을 합산하는 구조를 유지하고, 측정표 $T[\ell]$를 다음의 two-line 모델로 근사한다.

길이가 $\ell$행인 구간의 비용을 $T_{\mathrm{2L}}(\ell)$이라 하자. 행 단위로 환산한 전환점을 $\ell_0>0$, 비용 계수를 $\alpha\ge\lambda\ge0$, $\alpha>0$이라 두면

$$
\begin{aligned}
T_{\mathrm{2L}}(\ell)
&=\alpha\ell+\lambda(\ell_0-\ell)_+\\
&=
\begin{cases}
(\alpha-\lambda)\ell+\lambda\ell_0,
&0<\ell<\ell_0,\\
\alpha\ell,
&\ell\ge\ell_0.
\end{cases}
\end{aligned}
$$

여기서 $(x)_+=\max(x,0)$이며, 읽을 행이 없는 경우는 별도로 $T_{\mathrm{2L}}(0)=0$이라 둔다. 이는 행 단위 [노트북의 비용식](../../final/results/02/02_02_global_oracle.ipynb) $b\ell+\delta\max(s,\ell)$에서 $\alpha=b+\delta$, $\lambda=\delta$, $\ell_0=s$로 표기한 것이다. $\ell_0$는 정수로 반올림하지 않는다.

두 직선은 $\ell=\ell_0$에서 연속이며, 행 단위 처리량은

$$
\frac{\ell}{T_{\mathrm{2L}}(\ell)}
=
\frac{1}{\alpha+\lambda(\ell_0/\ell-1)_+}
\le\frac1\alpha
$$

이다. $\lambda>0$이면 전환점 이전에는 길이가 늘수록 처리량이 증가하고, 전환점 이후에는 $1/\alpha$로 일정해진다. 이러한 의미에서 이 모델을 **포화 제약 구간별 선형 지연 모델**이라 부른다. 전환점은 측정값에 대한 적합으로 정하며, 그 이후의 처리량이 일정하다는 것은 모델에 부과한 가정이다.

이제 정확히 $R$개 행을 선택한 mask $M$을 생각하자. $\sum_{C\in\mathcal C(M)}|C|=R$이므로

$$
\begin{aligned}
\widehat L_{\mathrm{2L}}(M)
&=\sum_{C\in\mathcal C(M)}T_{\mathrm{2L}}(|C|)\\
&=
\underbrace{\alpha R}_{\text{고정된 기본 전송 비용}}
+
\underbrace{
\lambda\sum_{C\in\mathcal C(M)}(\ell_0-|C|)_+
}_{\text{짧은 구간의 추가 비용}}.
\end{aligned}
$$

따라서 같은 $R$에서 모델 비용의 차이는 전환점보다 짧은 구간들의 길이 부족분으로 결정된다. 각 구간은 길이가 $\ell_0$에 도달할 때까지 추가 비용을 줄일 수 있으며, 그 이후에는 추가 비용이 없다.

또한 모든 구간의 길이가 $R$ 이하이므로 $\sum_C(\ell_0-|C|)_+\ge(\ell_0-R)_+$이고,

$$
\widehat L_{\mathrm{2L}}(M)
\ge
\alpha R+\lambda(\ell_0-R)_+
=T_{\mathrm{2L}}(R).
$$

길이 $R$의 단일 구간은 이 하한을 달성한다. $\lambda>0$일 때 $R<\ell_0$이면 단일 구간만 하한을 달성하지만, $R\ge\ell_0$이면 모든 구간의 길이가 $\ell_0$ 이상인 mask가 하한을 달성한다. 즉, 여러 구간으로 나뉘어 있더라도 각 구간이 충분히 길면 모델 비용은 동일한 $\alpha R$이다.

한편 행의 배치를 바꾸면 보존 중요도 $I(M)$도 달라진다. 따라서 다음 절에서는 중요도와 짧은 구간의 추가 비용을 함께 고려하는 다음 문제의 최적해를 구한다.

$$
\max_{\substack{M\in\{0,1\}^N\\\|M\|_1=R}}
\frac{I(M)}
{\displaystyle
\alpha R+
\lambda\sum_{C\in\mathcal C(M)}(\ell_0-|C|)_+
}.
$$

위 모델과 비용 분해는 [실험 01의 모델 분석](../../final/results/01/01_02_chunk_latency.ipynb)에, 비용 하한은 [실험 02의 최적화 분석](../../final/results/02/02_02_global_oracle.ipynb)에 근거한다. 이 절의 성질은 해당 근사 모델 아래에서의 결론이다.
