### 1.2 Yang 등이 진행한 I/O-Efficient Sparsification 연구(2025)

LLM이나 VLM과 같은 모델을 휴대용 하드웨어에서 실행할 때의 난점은, 장치의 DRAM 용량이 모델 전체를 담기에 충분하지 않은 경우가 많다는 것이다. 이에 모델 가중치를 flash에 저장하고, 추론 시 필요한 가중치만 DRAM으로 가져오는 weight offloading이 연구되어 왔다. Activation sparsification은 이 과정에서 중요도가 높은 채널에 대응하는 가중치 행들을 선별하여 읽는 양을 줄이는 방법이다. [Yang 등, 2025, p. 1](https://arxiv.org/pdf/2511.18692v1#page=1)

이영기(Youngki Lee)가 교신저자로 참여한 Yang 등(2025)의 연구팀은, 읽는 데이터의 양만으로 I/O 지연시간을 설명하는 가정에 문제를 제기하였다. 같은 양의 가중치를 읽더라도 선택된 행들이 작은 구간으로 분산되어 있으면 처리량이 낮아질 수 있기 때문이다. 이에 연구팀은 행들의 연속성을 반영하는 chunk-based latency model과 중요도–지연시간 기반 선택 방법인 Neuron Chunking을 제안하였다. 아래 수식은 원 논문의 정의와 목적식을 일관된 기호로 정리한 것이다. [Introduction 및 §2.3, pp. 2–4](https://arxiv.org/pdf/2511.18692v1#page=2)

먼저 가중치 행의 수를 $N$, 선택 mask를 $M\in\{0,1\}^N$이라 하자. $M_i=1$이면 $i$번째 행을 읽는다. 토큰 수가 $H$인 입력에서 $i$번째 채널의 중요도와 mask가 보존하는 중요도를

$$
v_i=\frac1H\sum_{t=1}^{H}|a_{t,i}|,
\qquad
I(M)=\sum_{i=0}^{N-1}v_iM_i
$$

로 나타낸다. 여기서 $a_{t,i}$는 토큰 $t$의 채널 $i$ activation이다. 첫 식은 원 논문이 설명한 토큰별 절댓값 평균을 수식화한 것이며, $I(M)$은 실제 정확도를 대신하여 사용하는 중요도 지표다. [Appendix B.2, pp. 14–15](https://arxiv.org/pdf/2511.18692v1#page=14), [§3.2.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5)

행 하나의 저장 크기를 $w$라 하면 읽는 데이터의 양은 $w\|M\|_1$이다. 처리량을 일정한 값 $\beta_0$로 보는 가정은

$$
L_{\mathrm{linear}}(M)=\frac{w}{\beta_0}\|M\|_1
$$

로 표현할 수 있다. 이 식은 원 논문이 비판한 data size 비례 가정을 재표현한 것이다. 이 모델에서는 선택 행 수가 같으면 비용도 같으므로 중요도가 가장 큰 행들만 고르면 된다. 그러나 원 논문의 실험에서는 연속 읽기 크기에 따라 처리량이 변하므로, 같은 선택 행 수라도 접근 구간의 구성이 다른 mask를 구분할 필요가 있다. [p. 2 및 §2.3, pp. 3–4](https://arxiv.org/pdf/2511.18692v1#page=2)

이를 위해 선택된 행들을 최대 연속 구간으로 분해하고 그 집합을 $\mathcal C(M)$라 하자. 길이가 $\ell$인 구간의 개수를

$$
h_\ell(M)=
\left|\{C\in\mathcal C(M):|C|=\ell\}\right|
$$

로 정의하면, $h_\ell(M)$은 원 논문의 contiguity distribution을 나타낸다. 이 표현은 각 구간의 정확한 위치와 구간 사이의 간격을 생략하고, 연속 구간의 길이별 개수를 보존한다. [§3, p. 4](https://arxiv.org/pdf/2511.18692v1#page=4)

길이 $\ell$인 행 구간의 profiled 비용을 $T[\ell]$이라 쓰면, 원 논문의 chunk-based latency model은 다음과 같이 표현된다.

$$
\boxed{
\widehat L_{\mathrm{table}}(M)
=\sum_{C\in\mathcal C(M)}T[|C|]
=\sum_{\ell=1}^{N}h_\ell(M)T[\ell].
}
$$

$T[\ell]$은 행 크기 $w$를 반영한 크기별 lookup 비용이다. 연구팀은 충분한 수의 같은 크기 chunk를 읽어 처리량이 안정된 조건에서 이를 profile하였다. 따라서 $\widehat L_{\mathrm{table}}$은 chunk별 상각 비용의 합으로 전체 읽기 시간을 근사하는 모델이다. 실제 latency와 항상 일치하는 등식이나, 개별 요청의 응답 시간을 단순 합한 값으로 해석하지 않는다. [§3.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5), [Appendix D, pp. 15–16](https://arxiv.org/pdf/2511.18692v1#page=15)

연구팀은 이 비용 모델을 이용하여, 행 예산 $R$ 아래에서 단위 추정 지연시간당 보존 중요도를 최대화하는 문제를 제시하였다.

$$
\boxed{
\max_{\substack{M\in\{0,1\}^N\\1\le\|M\|_1\le R}}
\frac{I(M)}{\widehat L_{\mathrm{table}}(M)}.
}
$$

이는 원 논문 §3.2.1의 목적식이며, 여기서는 비율이 정의되지 않는 빈 mask를 제외하기 위한 하한만 명시하였다. 실제 알고리즘은 연속 후보 구간 $C$마다

$$
U(C)=\frac{\sum_{i\in C}v_i}{T[|C|]}
$$

를 계산하고, 점수가 높은 후보부터 겹침과 잔여 예산을 검사하며 선택한다. 이는 실행 시간을 고려한 greedy 방법이며 위 목적식의 전역 최적해를 보장하는 방법으로 제시되지는 않았다. [§3.2.1–3.2.2, pp. 5–6](https://arxiv.org/pdf/2511.18692v1#page=5), [Algorithm 1, p. 17](https://arxiv.org/pdf/2511.18692v1#page=17)

본 연구는 이러한 **chunk별 가산 비용 모델과 중요도–지연시간 비율을 선행연구에서 계승한다**. 이후 도입하는 두 직선 비용 근사와 짧은 연속 구간의 추가 비용 분해는 이 모델을 바탕으로 한 후속 분석이다. 또한 우리 연구의 Global DP는 정확히 $R$개 행을 선택하는 조건 $\|M\|_1=R$을 사용하므로, 원 논문의 상한 조건 $\|M\|_1\le R$과 구분한다. 이에 대한 수식 전개는 문제 정의와 제안 방법 절에서 다룬다.
