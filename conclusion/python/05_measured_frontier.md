# 05 · 새로 측정한 latency 위의 importance–latency 비교

## 목적

실험 02의 결과를 불러오지 않는다. 이 노트북이 직접 생성한 새 파일에서 $T$를 다시 측정하고, 새 importance 입력에 대해 여러 selector가 만든 mask를 같은 좌표계에 놓는다.

$$x(M)=\widehat L(M)=\sum_{C\in\mathcal R(M)}\widehat T(|C|),\qquad
y(M)=I(M)/I(\mathbf1).$$

$x$는 새 실측 table을 합산한 **예측 I/O**이며 mask 전체를 직접 잰 wall time은 아니다. 실제 mask time 비교는 07이다. $y$는 retained importance이며 task accuracy는 아니다. synthetic importance는 synthetic이라고 명시한다.

## 설계

$n=1024$개의 4 KiB 행, flat·smooth·spiky 세 입력을 사용한다. 512 MiB 새 파일에서 4 KiB–8 MiB의 chunk를 측정한다. 원본 채널 가중치 크기를 흉내 낸 숨은 단위 변환은 없다. 측정 grid 안에서는 선형 보간하고 grid 밖에는 외삽하지 않는다. $s=s_{.05}$는 새 curve에서 선택한 near-best 길이다. finite grid의 near-best 기준을 수학적 포화점으로 확정하지 않는다.

Top-$R$, dense Algorithm 1, shifted saturation tiles는 같은 $R$ sweep을 사용한다. 비율 정확해는 $|M|\le R$로 별도 표시한다. 일반 table DP의 $\lambda$ sweep은 다른 objective에서 얻은 supported point들을 추가한다. 공집합과 전체 mask를 포함하여 양 끝의 feasibility를 드러낸다.

## 같은 quality에서의 비교

방법 $A$의 실제 생성 mask 집합을 $\mathcal C_A$라 할 때

$$\widehat L_A(q)=\min_{M\in\mathcal C_A:\ I(M)\ge qI(\mathbf1)}\widehat L(M).$$

feasible mask가 없으면 NaN이다. 두 점을 연결해 그린 선은 시각화용일 뿐, 이 계산에는 보간을 사용하지 않는다. $q\in\{.25,.5,.75,.9\}$를 사전에 고정한다. 원래의 global constrained optimum이 아니라 **평가한 candidate 집합 안에서의 최선**이다.

$$\Delta_q(A,P)=100\left(1-\frac{\widehat L_A(q)}{\widehat L_P(q)}\right).$$

양수는 같은 importance 하한에서 $A$의 예측 latency가 작다는 뜻이다. 이 값과 함께 실제 달성 importance와 선택 행 수를 저장하여 서로 다른 overshoot를 확인할 수 있게 한다.

## 곡선 오차가 선택에 미치는 영향

각 run에 대해 $|T(\ell)-\hat T(\ell)|\le\epsilon$이면

$$|L_T(M)-L_{\hat T}(M)|\le K(M)\epsilon,\qquad K(M)=|\mathcal R(M)|.$$

따라서 fragmented mask는 같은 per-run 오차를 더 많이 누적할 수 있다. 상대오차 $|T/\hat T-1|\le\eta$가 모든 사용 길이에 성립하면 $(1-\eta)\hat L\le L\le(1+\eta)\hat L$. 하지만 실측 일부 길이에서의 fit 오차만으로 이 uniform bound가 성립했다고 간주할 수 없다.

## 보고서의 출력

각 입력의 importance 곡선, latency–importance cloud와 방법별 attainable envelope, 같은 quality의 latency table, 대표 mask heatmap을 생성한다. raw I/O·모든 candidate point·mask를 저장한다. 새 machine에서 성능 순위가 바뀌어도 실험의 실패로 처리하지 않는다. 이는 확인할 사실이다.
