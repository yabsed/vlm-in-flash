# 09 · 고정 sparsity 대신 전역 importance 하한으로 예산 배분

## 문제

3개 실제 projection layer의 activation을 새 forward로 수집한다. 층 $j$의 importance는 $\sum_i v_{ji}=1$로 정규화한다. 동일 가중치 $w_j=1/3$로 평균 보존율을 정의한다. 이 가중치는 명시적 실험 선택이며 모델 loss로부터 유도된 상수는 아니다.

$$\min_{M_1,M_2,M_3}\sum_jL_j(M_j)
\quad\text{s.t.}\quad\sum_jw_jI_j(M_j)\ge q.$$

모든 층에서 $|M_j|/n_j=r$을 고정하는 전략은 위 문제의 작은 부분집합만 탐색한다. 같은 importance 하한에서 층마다 선택률을 다르게 배분할 여지가 있는지 확인한다.

## finite candidate 문제의 정확해

방법 $A$가 층 $j$에서 생성한 후보 집합을 $\mathcal C_{j,A}$라 두자. $r\in\{0,.15,.3,.45,.6,.75,.9,1\}$마다 새 mask를 만든다. 이산 문제는

$$L_A^{grid}(q)=\min_{(M_j)\in\prod_j\mathcal C_{j,A}}
\left\{\sum_jL_j(M_j):\sum_jw_jI_j(M_j)\ge q\right\}.$$

3층 × 8후보라 $8^3=512$개 조합을 전부 계산한다. 이 범위에서는 전역 정확해다. **모든 channel mask를 포함하는 원래 문제의 global optimum은 아니다.** 이 큰 채널 실험의 paper baseline은 minimum/step 1 row, stride cap 4 rows로 명시적으로 설정한다. 작은 oracle 실험의 stride cap 1과 구분한다. 같은 grid에서 fixed-$r$ 조합이 포함되므로

$$L_A^{grid}(q)\le L_A^{fixed-grid}(q).$$

이 관계는 실험으로 기대하는 경험 법칙이 아니라 feasible set 포함관계의 정리다. 구현에서 매 target에 대해 검증한다. 다른 방법끼리의 우열에는 이런 포함관계가 없다.

## 다항시간 구조와 남는 제약

전역 scalarized 문제는 층별로 분해된다.

$$\max_{M_1,\ldots,M_J}\sum_j[w_jI_j(M_j)-\lambda L_j(M_j)]
=\sum_j\max_{M_j}[w_jI_j(M_j)-\lambda L_j(M_j)].$$

실험 04의 DP로 각 층의 scalarized 정확해를 얻을 수 있다. 그러나 shared $\lambda$만 바꾸어 discrete coverage의 모든 optimum을 찾는다는 보장은 없다. unsupported point 문제가 층을 합친 공간에도 남는다.

후보 coverage를 $\delta$ 단위로 정수화해 $q_{jc}=\lfloor w_jI_j(M_{jc})/\delta\rfloor$로 두면

$$D_j[z]=\min_{c\in\mathcal C_j}\{D_{j-1}[z-q_{jc}]+L_j(M_{jc})\}$$

를 사용할 수 있다. $z\ge\lceil q/\delta\rceil$을 요구하면 실제 quality 하한을 보수적으로 만족한다. 복잡도 $O(JKZ)$는 quantization 정밀도에 의존하며 원래 실수 제약 문제의 강다항 해법이라고 할 수 없다. 노트북은 quantization 없이 작은 후보 곱집합을 직접 열거한다.

## 새 hardware 모델과 단위

이 노트북 안에서 SSD curve를 새로 측정한다. 원본 모델의 각 저장 행을 4 KiB page 배수로 padding한 가상 layout을 명시한다. 실제 저장된 모델 weight를 읽는 end-to-end benchmark는 아니다.

512 MiB 새 파일에서 측정한 최대 chunk $B=2048$ pages보다 긴 run은 외삽 대신 **명시적인 request 분할** 모델을 쓴다.

$$T_{split}(\ell)=\lfloor\ell/B\rfloor\hat T(B)+\hat T(\ell\bmod B),\qquad\hat T(0)=0.$$

이 역시 병렬 wall time이 아닌 가산 예측값이다. $s$를 row로 환산할 때 padding된 bytes/row를 사용한다. 나눠 읽는 정책과 가산 근사가 결과에 들어간다는 점을 숨기지 않는다.

## 출력과 결론의 범위

fixed/global candidate frontier, 같은 $q$의 비용 비교, layer별 실제 선택률 heatmap, 각 방법의 per-layer importance를 표시한다. 각 점이 만족한 평균 중요도와 **최소 층별 보존율**도 기록한다. 평균이 높아도 특정 층이 희생될 수 있다. 층별 quality 제약을 별도로 요구하면 새로운 문제다.

이 실험은 국소 가산 대리 목적의 전역 배분을 닫는다. 08에서 확인한 cross term과 실제 여러 층의 오차 전파를 없애지 않으며, 전역 task accuracy 최적화라는 결론으로 확대하지 않는다.
