# Mathematical background for Section 1.2: *VLM in a flash*

선행연구의 수식, 전제, 인용 범위에 대한 검토

작성 기준: 2026-10-27. [preliminary_research/idea.md](../../preliminary_research/idea.md)의 주장별 절 구성, 정의–전개–해석 순서, 독립된 수식과 박스 표기를 따른다.

---

## Chunk 기반 비용 모델과 중요도–지연시간 목적함수는 선행연구에 귀속한다.

### 검토 대상과 인용 방식

대상은 Kichang Yang, Seonjun Kim, Minjae Kim, Nairan Zhang, Chi Zhang, Youngki Lee의 *VLM in a flash: I/O-Efficient Sparsification of Vision-Language Model via Neuron Chunking*, arXiv:2511.18692v1 (2025)이다. 첫 두 저자는 공동 제1저자이고 이영기(Youngki Lee)는 교신저자다. 본문에서는 **Yang 등(2025)**으로 인용하고, 필요하면 “이영기가 교신저자로 참여한 연구팀”이라고 설명하는 것이 저자 목록과 일치한다. [원 논문 p. 1](https://arxiv.org/pdf/2511.18692v1#page=1)

이 보고서는 [저장소의 v1 PDF](../../final/references/2511.18692v1.pdf)를 기준으로 한다. 아래 페이지 번호는 논문 하단의 쪽수이며 이 PDF의 물리적 페이지 번호와 같다. 원 논문 §3.1과 §3.2.1의 핵심 수식에는 번호가 없으므로, 존재하지 않는 식 번호를 붙여 인용하지 않는다.

### 세 종류의 수식

다음 세 종류를 구분해야 한다.

| 구분 | 이 보고서에서의 사례 | 보고서에 적을 표현 |
|---|---|---|
| 원 논문에 직접 제시된 수식 | chunk 비용의 합, 중요도/지연시간 목적식 | “Yang 등(2025, §3.1, p. 5)의 모델을 따르면…” |
| 원 논문의 설명을 수식으로 재표현 | 토큰별 절댓값 평균, chunk 길이 히스토그램, 후보 utility | “원 논문의 정의를 다음과 같이 표기한다.” |
| 우리 연구에서 추가한 모델·분석 | 두 직선 비용 함수, 짧은 구간 비용 분해, 고정 예산 DP | “선행연구의 가산 모델을 바탕으로, 본 연구에서는…” |

수학적 표기를 새로 정리했다는 이유만으로 모델이나 목적함수의 출처가 달라지지는 않는다. 반대로, 원 논문의 실측 lookup을 우리가 택한 두 직선 함수와 동일시해서도 안 된다.

### 권장 서술 순서

1. activation과 선택 mask를 정의한다.
2. 행 수만으로 I/O 비용을 설명하는 가정을 수식화한다.
3. 선택된 행들을 maximal contiguous chunk로 분해한다.
4. 원 논문의 chunk별 비용 가산 모델을 제시한다.
5. 중요도/지연시간 목적함수와 greedy utility를 설명한다.
6. 우리 연구가 추가하는 비용 함수와 예산 조건을 구분한다.

[section_1_2.md](section_1_2.md)는 이 순서를 압축한 삽입용 초안이다. 이 작업의 변경 대상은 이 디렉토리의 보고서와 초안이며, [final/readme.md](../../final/readme.md)에 직접 반영하지 않았다.

---

## Activation 중요도의 합은 보존 품질을 나타내는 대리 지표다.

### 가중치 행과 선택 mask

입력 채널 수를 $N$, 출력 차원을 $d$라 하자. 설명을 위해 입력 채널이 행에 대응하는 방향으로 가중치 행렬을

$$
W\in\mathbb R^{N\times d}
$$

로 두고, activation을 $a\in\mathbb R^N$라 쓰면 출력은

$$
y=a^\top W=\sum_{i=0}^{N-1}a_iW_i
$$

이다. 선택 mask와 선택 행 수는

$$
M\in\{0,1\}^N,\qquad R(M)=\|M\|_1
$$

로 정의한다. 선택된 가중치만 사용한 출력은

$$
\widetilde y(M)=\sum_{i=0}^{N-1}M_i a_iW_i
$$

이다. 이 표기는 원 논문 Appendix B.2의 행렬 방향을 따른다. 프레임워크의 실제 weight tensor가 이 방향으로 저장된다는 뜻은 아니다. [Appendix B.2, pp. 14–15](https://arxiv.org/pdf/2511.18692v1#page=14)

### Activation magnitude와 보존 중요도

원 논문은 단일 토큰에서 $v_i=|a_i|$를 사용하고, 여러 토큰을 처리하는 VLM 입력에서는 토큰별 절댓값의 평균을 사용한다고 설명한다. 토큰 수를 $H$라고 두어 이를 수식화하면

$$
\boxed{
v_i=\frac1H\sum_{t=1}^{H}|a_{t,i}|,
\qquad
I(M)=\sum_{i=0}^{N-1}v_iM_i.
}
$$

첫 식은 Appendix B.2의 설명을 수식으로 옮긴 것이고, 두 번째 식은 §3.2.1의 목적함수 분자다. activation magnitude를 중요도의 대리 지표로 쓰는 관행 자체는 이 논문이 소개하는 기존 연구에도 등장한다. Yang 등의 기여로 귀속할 핵심은 이를 flash 접근 비용과 결합한 선택 방식이다. [Appendix B.2, p. 15](https://arxiv.org/pdf/2511.18692v1#page=15), [§3.2.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5)

우리 trace 수집에서는 배치 축까지 평균한다.

$$
v_i=\frac1{BH}\sum_{b=1}^{B}\sum_{t=1}^{H}|x_{b,t,i}|.
$$

이는 동일한 중요도 정의의 적용이다. [02_01_capture_traces.ipynb](../../final/results/02/02_01_capture_traces.ipynb)

### 지표의 해석 범위

$I(M)$은 실제 출력 오차나 downstream 정확도와 같은 양이 아니다. 예를 들어 단일 토큰의 출력 오차는

$$
y-\widetilde y(M)
=\sum_{i=0}^{N-1}(1-M_i)a_iW_i
$$

이므로 가중치 행의 크기와 방향에도 영향을 받는다. 이 등식은 위 출력 정의에서 바로 얻는 설명이며, 논문이 중요도 보존을 정확도 보장으로 증명했다는 뜻이 아니다.

따라서 “정확도를 최대화한다”보다 “activation 중요도를 보존한다” 또는 “중요도를 품질의 대리 지표로 사용한다”라고 쓰는 것이 정확하다.

---

## 같은 행 수라도 선택된 행의 연속성이 다르면 추정 I/O 비용이 달라진다.

### 전송량에만 비례하는 모델

행 하나의 저장 크기를 $w$라 하자. 크기와 처리량의 단위는 일관되게 사용한다. 총 전송량은

$$
D(M)=wR(M)
$$

이다. 일정한 처리량 $\beta_0$를 가정하면

$$
L_{\mathrm{linear}}(M)
=\frac{D(M)}{\beta_0}
=\frac{w}{\beta_0}R(M).
$$

이 식은 논문 p. 2의 “I/O latency가 data size에 선형 비례한다는 가정”을 이 보고서의 기호로 재표현한 것이다. 원 논문에 이 형태의 식이 직접 인쇄되어 있지는 않다. 또한 모든 기존 sparsification 연구가 명시적으로 이 비용 함수를 사용했다고 확대해서 서술하지 않는다. [Introduction, p. 2](https://arxiv.org/pdf/2511.18692v1#page=2)

이 모델에서는 $R(M)=R$일 때 비용이 고정되므로, 중요도/비용 최대화는 중요도 최대화로 환원된다.

$$
\arg\max_{R(M)=R}
\frac{I(M)}{L_{\mathrm{linear}}(M)}
=\arg\max_{R(M)=R}I(M).
$$

따라서 top-$R$ 선택이 자연스럽다. 이 동치 관계는 선형 비용 가정의 의미를 설명하기 위한 대수적 전개다.

### Maximal contiguous chunk

원 논문은 선택된 행들의 **최대 연속 구간** 집합을 사용한다. 이를 $\mathcal C(M)$로 쓰고 길이별 chunk 수를

$$
h_\ell(M)=
\left|\left\{C\in\mathcal C(M):|C|=\ell\right\}\right|
$$

로 정의하면

$$
R(M)=\sum_{\ell=1}^{N}\ell h_\ell(M),
\qquad
K(M)=|\mathcal C(M)|=\sum_{\ell=1}^{N}h_\ell(M).
$$

$h_\ell$은 확률로 정규화한 값이 아니라 빈도다. 원 논문의 *contiguity distribution*을 수식으로 옮긴 것이다. [§3, p. 4](https://arxiv.org/pdf/2511.18692v1#page=4)

예를 들어 선택 위치가 $\{0,1,3,5,6\}$이면

$$
\mathcal C(M)=
\big\{\{0,1\},\{3\},\{5,6\}\big\},
\qquad h_1(M)=1,\quad h_2(M)=2.
$$

후보 생성 단계에서 사용한 window와 최종 mask의 maximal chunk는 구분해야 한다. 서로 인접한 두 후보가 모두 선택되면 최종 mask에서는 하나의 chunk가 된다.

---

## 원 논문은 실측 chunk 비용을 합하여 전체 읽기 지연시간을 근사한다.

### Chunk 비용의 의미

크기 $z$에 대한 profiled chunk 비용을 $T_{\mathrm{size}}(z)$라 하고, 행 단위 함수를

$$
T[\ell]=T_{\mathrm{size}}(w\ell)
$$

로 정의한다. 이후 식은 $T[\ell]$을 사용한다. 이는 원 논문의 chunk 크기 표기와 우리 노트북의 행 단위 표기를 연결하기 위한 정의다.

원 논문은 같은 크기의 chunk를 충분히 많이 읽어 처리량이 안정된 조건에서 비용을 profile한다. $k$개 chunk의 배치 읽기 시간이 $t_k(z)$이고 처리량을 $\beta(z)=kz/t_k(z)$로 정의하면, chunk당 상각 비용은

$$
T_{\mathrm{size}}(z)
\approx\frac{t_k(z)}{k}
=\frac{z}{\beta(z)}
$$

로 설명할 수 있다. 이 환산식은 profiling 절차를 수식화한 것으로, 고립된 요청 하나의 응답 시간을 의미하지 않는다. 특히 병렬 읽기에서는 배치 시간/요청 수와 개별 요청의 응답 시간을 구분한다. [§3.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5), [Appendix D, pp. 15–16](https://arxiv.org/pdf/2511.18692v1#page=15)

### 가산 모델

원 논문에 직접 제시된 핵심 모델은

$$
L_{\mathrm{total}}=\sum_{j=1}^{n}T[s_j]
$$

이다. 이를 mask와 행 단위 표기로 다시 쓰면

$$
\boxed{
\widehat L_{\mathrm{table}}(M)
=\sum_{C\in\mathcal C(M)}T[|C|]
=\sum_{\ell=1}^{N}h_\ell(M)T[\ell].
}
$$

마지막 등호는 같은 길이의 항을 묶은 결과다. 새로운 latency model을 제안한 식으로 제시하면 안 된다. [§3.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5)

이 모델의 두 전제는 다음과 같다.

1. 각 chunk의 비용을 길이별 lookup으로 나타낼 수 있다. 정확한 위치, chunk 사이 간격 등은 명시적으로 사용하지 않는다.
2. 서로 다른 크기의 chunk가 섞인 mask의 비용도 개별 profile 비용의 합으로 근사할 수 있다.

따라서

$$
h_\ell(M)=h_\ell(M')\ \text{for all }\ell
\quad\Longrightarrow\quad
\widehat L_{\mathrm{table}}(M)
=\widehat L_{\mathrm{table}}(M')
$$

가 **모델 안에서** 성립한다. 실제 읽기 시간이 반드시 같다는 명제는 아니다. 서로 다른 크기와 간격이 섞일 때 생기는 controller·queue 효과는 원 논문도 오차 원인으로 설명한다. [§3.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5)

### 비례 오차와 순위 보존

원 논문은 추정치와 실측치 사이에 대체로 비례 관계가 관찰되며, 공통 배율 오차라면 utility 순위가 보존된다고 설명한다. 그 논리를 정확한 등식으로 표현할 수 있는 충분조건은 모든 비교 대상에 대해 같은 $c>0$가 적용되는 경우다.

$$
L_{\mathrm{actual}}(M)=c\widehat L_{\mathrm{table}}(M)
\quad\Longrightarrow\quad
\frac{I(M)}{L_{\mathrm{actual}}(M)}
=\frac1c\frac{I(M)}{\widehat L_{\mathrm{table}}(M)}.
$$

실측 관계는 근사적이다. 높은 상관관계만으로 모든 mask나 후보의 순위가 반드시 보존된다고 결론 내릴 수는 없다. [§3.1–3.2.2, pp. 5–6](https://arxiv.org/pdf/2511.18692v1#page=5)

---

## 중요도–지연시간 비율과 chunk utility는 원 논문의 선택 기준이다.

### 원 논문의 목적함수

$v_i\ge0$, $T[\ell]>0$, $1\le R\le N$라 하자. 원 논문의 §3.2.1은 다음 목적식을 제시한다.

$$
\boxed{
\max_{\substack{M\in\{0,1\}^{N}\\1\le R(M)\le R}}
\frac{I(M)}{\widehat L_{\mathrm{table}}(M)}.
}
$$

원문에는 상한 $R(M)\le R$만 적혀 있다. 여기서는 $0/0$인 빈 mask를 제외하기 위해 $1\le R(M)$을 명시했다. 목적함수 자체는 우리 연구가 새로 제안한 것이 아니다. [§3.2.1, p. 5](https://arxiv.org/pdf/2511.18692v1#page=5)

### Greedy utility

연속 후보 구간 $C$의 점수는

$$
\boxed{
U(C)=\frac{I(C)}{T[|C|]},
\qquad
I(C)=\sum_{i\in C}v_i.
}
$$

원 논문은 여러 크기의 sliding window로 후보를 만들고 $U(C)$가 큰 순서로 검토한다. 이미 선택한 후보 집합을 $\mathcal S$라고 하면 새 후보 $C$를 받아들이는 조건은

$$
C\cap\bigcup_{D\in\mathcal S}D=\varnothing,
\qquad
\sum_{D\in\mathcal S}|D|+|C|\le R.
$$

예산을 채우거나 후보를 모두 검토할 때까지 진행한다. 후보 크기와 겹침 때문에 반환 행 수가 $R$보다 작을 수 있다. [§3.2.2, p. 6](https://arxiv.org/pdf/2511.18692v1#page=6), [Appendix E 및 Algorithm 1, pp. 16–17](https://arxiv.org/pdf/2511.18692v1#page=16)

### 후보 점수와 최종 mask 비용

인접한 후보 $C,D$를 함께 선택하면 최종 chunk는 $C\cup D$가 된다. 이때 그 구간의 모델 비용은

$$
T[|C|+|D|]
$$

이고, 후보를 각각 평가할 때 사용한 비용 합

$$
T[|C|]+T[|D|]
$$

과 같을 필요가 없다. 원 논문도 인접한 낮은 점수의 chunk가 결합할 때 얻는 이점을 greedy가 충분히 고려하지 못한다고 설명한다. 따라서 $U(C)$를 현재 mask에 추가할 때의 정확한 한계 효용으로 해석하거나, greedy의 전역 최적성을 주장해서는 안 된다. [§3.2.2, p. 6](https://arxiv.org/pdf/2511.18692v1#page=6)

---

## 상한 예산과 고정 예산은 서로 다른 최적화 문제다.

### 상한 예산에서의 가중평균 구조

이 절은 원 논문에 제시된 정리가 아니라, [기존 idea.md의 첫 절](../../preliminary_research/idea.md)에서 이미 전개한 해석을 요약한다.

각 maximal chunk의 utility를 사용하면

$$
\frac{I(M)}{\widehat L_{\mathrm{table}}(M)}
=\sum_{C\in\mathcal C(M)}
\underbrace{\frac{T[|C|]}
{\sum_{D\in\mathcal C(M)}T[|D|]}}_{\omega_C}
U(C),
\qquad
\sum_C\omega_C=1,\quad\omega_C>0.
$$

따라서 전체 비율은 각 chunk utility의 가중평균이고,

$$
\frac{I(M)}{\widehat L_{\mathrm{table}}(M)}
\le\max_{C\in\mathcal C(M)}U(C).
$$

상한 조건만 있으면 이 최고 utility chunk 하나만 남긴 mask도 여전히 feasible하다. 그러므로

$$
\boxed{
\max_{1\le R(M)\le R}
\frac{I(M)}{\widehat L_{\mathrm{table}}(M)}
=
\max_{\substack{C\text{ contiguous}\\1\le|C|\le R}}U(C).
}
$$

즉, 이 명시적 목적식과 가산 모델 아래에서는 단일 구간으로 표현되는 최적해가 적어도 하나 존재한다. 모든 최적해가 단일 구간이라는 뜻은 아니다. 이 등식의 오른쪽은 모든 허용 길이의 연속 구간을 대상으로 하며, greedy의 제한된 후보 집합만을 뜻하지 않는다. 여기서는 $T[\ell]$이 모든 비교 길이 $1\le\ell\le R$에 대해 정의되어 있다고 가정한다. 실측 profile이 그 범위를 덮지 않는다면, 그 밖의 길이에 적용할 비용은 별도의 모델링 대상이다.

### 우리 연구의 고정 예산

현재 [02_02_global_oracle.ipynb](../../final/results/02/02_02_global_oracle.ipynb)는 정확히 $R$개 행을 선택한다.

$$
\boxed{
\eta_R^\star
=\max_{R(M)=R}
\frac{I(M)}{L_{\mathrm{2L}}(M)}.
}
$$

최고 utility chunk만 남기면 $R(M)=R$을 위반할 수 있으므로 앞의 단일 구간 논리는 일반적으로 적용되지 않는다.

따라서 “원 논문의 목적함수를 그대로 정확히 풀었다”보다 “원 논문의 중요도–지연시간 기준을 계승하고, 정확히 $R$개 행을 선택하는 비교 문제를 정의했다”라고 써야 한다.

또한 Paper가 실제로 $r<R$개 행을 반환한 사례는 고정-$R$ DP와 feasible set이 다르다. 동일한 요청 예산에서의 실험 비교는 가능하지만, 그 수치를 그대로 같은 제약 아래의 최적성 gap이라고 부르면 안 된다. 같은 비용 함수에서 반환 행 수 $r$에 대한 oracle을 사용하거나, 요청 예산 기준의 비교임을 명시해야 한다.

---

## 두 직선 비용 함수와 짧은 구간 비용 분해는 우리 연구의 추가 모델링이다.

### 실측 lookup과 해석용 surrogate

원 논문은 실측 lookup $T[\ell]$을 사용한다. 우리 연구의 [01_02_chunk_latency.ipynb](../../final/results/01/01_02_chunk_latency.ipynb)는 측정한 곡선을 두 직선 함수로 근사하고, [02_02_global_oracle.ipynb](../../final/results/02/02_02_global_oracle.ipynb)는 이를 행 단위로 환산한다.

$$
T_{\mathrm{2L}}(0)=0,\qquad
\boxed{
T_{\mathrm{2L}}(\ell)
=b\ell+\delta\max(s,\ell)
\quad(\ell\ge1).
}
$$

여기서 $s\ge0$는 행 단위 경계, $b,\delta\ge0$, $b+\delta>0$이다. $s$는 정수일 필요가 없다. $b,\delta$는 ms/row 단위이며, KiB 단위 계수를 행 단위 계수와 섞어 쓰지 않는다.

### 정확한 대수적 분해

$(x)_+=\max(x,0)$라 두면

$$
T_{\mathrm{2L}}(\ell)
=(b+\delta)\ell+\delta(s-\ell)_+.
$$

원 논문의 chunk 가산 구조에 이 함수를 대입하여

$$
\boxed{
L_{\mathrm{2L}}(M)
=(b+\delta)R(M)
+\delta\sum_{C\in\mathcal C(M)}(s-|C|)_+.
}
$$

를 얻는다. 앞 항은 선택 행 수에 비례하는 비용이고, 뒤 항은 경계보다 짧은 chunk의 부족 길이에 따른 추가 비용이다.

이 식의 **합산 구조는 Yang 등의 모델**에 근거하고, **두 직선 함수의 선택 및 그 함수에서의 비용 분해는 우리 연구의 분석**이다. 이 구분을 한 문장에 함께 적는 것이 적절하다.

### 고정 예산에서의 해석

$R(M)=R$이면

$$
L_{\mathrm{2L}}(M)
=(b+\delta)R+\delta D_s(M),
\qquad
D_s(M)=\sum_{C\in\mathcal C(M)}(s-|C|)_+.
$$

따라서 동일 행 수에서 모델 비용 차이는 $D_s(M)$에 의해 결정된다. 그러나 비율의 분자 $I(M)$도 달라지므로, 짧은 chunk를 없애는 것만으로 전체 중요도–지연시간 비율의 최적성을 보장할 수는 없다.

또한 이 모델에서는

$$
\frac{T_{\mathrm{2L}}(\ell)}{\ell}=b+\delta
\qquad(\ell\ge1,\ \ell\ge s)
$$

가 정확히 성립한다. 이는 포화 이후의 유효 처리량을 상수로 두는 surrogate의 구조적 가정이다. 원 논문의 경험적 처리량 포화 관측이 모든 길이에서 이 정확한 등식을 증명하는 것은 아니다. 이 차이는 [기존 idea.md의 마지막 절](../../preliminary_research/idea.md)에서도 논의한다.

### 고정 예산 DP의 위치

현재 oracle은 비율 문제를

$$
F_R(\lambda)
=\max_{R(M)=R}
\{I(M)-\lambda L_{\mathrm{2L}}(M)\}
$$

로 바꾸어 푼다. $P_j=\sum_{i<j}v_i$라 하고, $D_{z,r}$를 처음 $z+r$개 행에서 $z$개를 제외하고 $r$개를 선택한 최대 $I-\lambda L$로 정의하면

$$
\boxed{
D_{z,r}
=\max_{0\le t\le r}
\left[
D_{z-1,t}
+P_{z+r}-P_{z+t}
-\lambda T_{\mathrm{2L}}(r-t)
\right],
\quad z\ge1.
}
$$

마지막 0 앞의 prefix에 $t$개 선택 행이 있고, 그 뒤 마지막 chunk에 $r-t$개 행이 있다는 분해다. 초기조건과 끝값은

$$
D_{0,r}=P_r-\lambda T_{\mathrm{2L}}(r),
\qquad
F_R(\lambda)=D_{N-R,R}
$$

이다. 이 구체적인 상태와 점화식은 현재 노트북의 후속 전개에 해당하며, 원 논문에 제시된 알고리즘은 아니다. 분수계획법과 동적계획법이라는 일반 기법 자체를 우리 연구의 발명으로 주장하는 것도 피한다. 자세한 구현과 계산량은 [02_02_global_oracle.ipynb](../../final/results/02/02_02_global_oracle.ipynb)를 따른다.

---

## 1.2절에서는 선행연구의 수식과 우리 연구로 넘어가는 경계를 제시한다.

### 본문에 남길 수식

1. 중요도와 mask: $I(M)=\sum_i v_iM_i$.
2. contiguity distribution: $h_\ell(M)=|\{C\in\mathcal C(M):|C|=\ell\}|$.
3. 비용 모델: $\widehat L_{\mathrm{table}}(M)=\sum_C T[|C|]=\sum_\ell h_\ell(M)T[\ell]$.
4. 원 논문의 목적식: $\max_{1\le R(M)\le R}I(M)/\widehat L_{\mathrm{table}}(M)$.
5. greedy 후보 점수: $U(C)=I(C)/T[|C|]$.

이 순서면 모델의 입력, 공간적 추상화, 비용, 최적화 기준이 연결된다. 두 직선 함수의 비용 분해와 DP 점화식은 후속 문제 정의·제안 방법 또는 실험 절에서 다루는 편이 자연스럽다.

### 원 논문 페이지별 근거

| 서술할 내용 | v1의 위치 | 주의할 점 |
|---|---|---|
| 저자, DRAM 부족과 weight offloading 배경 | p. 1 | 표준 저자 인용은 Yang 등 |
| data size 비례 가정에 대한 문제 제기 | p. 2, Introduction | $L_{\mathrm{linear}}$ 식은 이 보고서의 재표현 |
| 접근 연속성과 처리량 포화 관측 | pp. 3–4, §2.3, Figures 3–4 | 실험적 관측과 정확한 비용 법칙을 구분 |
| maximal chunk와 contiguity distribution | p. 4, §3 | $h_\ell$ 표기는 이 보고서에서 도입 |
| chunk 비용의 합과 offline profiling | p. 5, §3.1 | 전체 latency의 근사 모델 |
| 중요도/latency 목적함수, 행 예산 상한 | p. 5, §3.2.1 | 원문 조건은 $\le R$ |
| utility 정렬과 greedy 선택, 인접 chunk의 결합 한계 | p. 6, §3.2.2 | 전역 최적성 보장은 제시하지 않음 |
| activation 기반 행 선택과 토큰별 평균 | pp. 14–15, Appendix B.2 | 중요도는 품질의 대리 지표 |
| profiling 조건과 99% peak 처리량 기준 | pp. 15–16, Appendix D | 개별 요청 응답 시간과 상각 비용을 구분 |
| 후보 생성, 겹침·잔여 예산 검사 | pp. 16–17, Appendix E, Algorithm 1 | 예산 미충족 가능 |

Figure 4 캡션과 Appendix D의 장치별 포화 크기 수치는 서로 일치하지 않는다. 이 보고서와 삽입용 초안에서는 특정 수치를 선택해 전제하지 않았다. 이 문제는 [외부 saturation 자료 보고서](../02_external_saturation_data/README.md)에도 기록되어 있다.

### 권장 인용 문장

> 본 연구는 Yang 등(2025, §3.1–3.2.1, p. 5)의 chunk 기반 가산 비용 모델과 중요도–지연시간 비율을 분석의 출발점으로 삼는다. 이후 전개하는 두 직선 비용 근사, 짧은 연속 구간의 추가 비용 분해, 고정 행 수 제약하의 최적화는 해당 모델을 바탕으로 본 연구에서 구체화한 분석이다.

이 문장은 모델과 목적함수의 출처를 함께 밝히면서, 후속 수식 전개까지 모두 원 논문의 결과인 것처럼 읽히는 것을 피한다.

### 확인 범위

원 논문의 해당 페이지, 기존 idea.md, 현재 final/results 노트북의 수식과 제약을 대조했다. 추가로 $N=1,\ldots,8$의 작은 입력에서 1,506개의 비어 있지 않은 mask를 열거하여 가중평균 관계와 두 직선 비용 분해를 확인했다. 108개의 예산 문제에서는 단일 구간으로의 환원 및 DP 점화식을 전수 열거 결과와 대조했다. 이 검산은 유리수의 정확 연산을 사용한 수식 확인이며, 현재 selector 구현 전체나 실제 SSD 모델의 타당성을 검증한 것은 아니다.

추가 SSD 벤치마크나 모델 정확도 실험은 수행하지 않았다. 삽입용 초안은 [section_1_2.md](section_1_2.md)에 있다.
