# 02 · 실제 chunk length–latency와 포화점의 식별

## 검증할 명제

$\ell$은 4 KiB로 정렬된 저장 행의 수다. 원점에서 $(\ell,T(\ell))$로 가는 직선의 기울기와 throughput의 관계는

$$\rho(\ell)=T(\ell)/\ell,\qquad B(\ell)=4096\ell/T(\ell),\qquad
\arg\min\rho=\arg\max B.$$

이는 정의로부터의 항등식이다. **유한한 최적 길이의 존재**, **최초 포화점의 유일성**, **전후 두 직선 적합성**은 항등식이 아니라 측정할 가설이다. 예를 들어 $T(\ell)=a+b\ell$이면 $\rho=b+a/\ell$은 계속 감소해 유한 최소점이 없다.

## fresh 측정과 단위

각 실행에서 대상 filesystem에 512 MiB 난수 파일을 새로 쓰고 `fsync`한 뒤 Linux `O_DIRECT + preadv`로 읽는다. 4 KiB–8 MiB chunk를 측정하며 가장 큰 chunk도 batch 64개를 서로 다른 위치에서 읽도록 파일 크기를 정한다. 정렬된 버퍼를 사용하고 direct I/O 실패 시 중단한다. `/tmp` 같은 tmpfs는 SSD로 측정하지 않는다. 파일은 실행 후 삭제한다. 이는 page cache를 우회하지만 SSD controller cache까지 제거하는 실험은 아니다.

workers $p=4$, batch size $q\in\{1,8,32,64\}$, 서로 다른 offset들을 사용한다. 같은 반복 안에서 $(\ell,q)$ 순서를 무작위화한다. 7개 독립 반복 블록의 원시 시간 $t_{\ell,q,b}$를 모두 저장한다.

$$\widehat T_q(\ell)=\operatorname{median}_b(t_{\ell,q,b})/q.$$

이는 thread dispatch·Python 호출·completion을 포함한 **batch당 상각 비용**이다. 단일 read의 service time도, 논문의 C++/Jetson 성능 재현도 아니다. $q=64$를 기본 모델로 쓰되 $q=32$와의 차이를 반드시 표시한다. batch plateau가 확인되지 않으면 “포화 latency table”이라고 단정하지 않는다.

## 최소점과 plateau의 구별

측정한 길이 집합 $\mathcal G$에 대해서만

$$\hat s_{\min}=\arg\min_{\ell\in\mathcal G}\widehat T_{64}(\ell)/\ell,\qquad
\hat s_\varepsilon=\min\{\ell\in\mathcal G:\hat\rho(\ell)\le(1+\varepsilon)\min\hat\rho\},\quad\varepsilon=.05.$$

최대 throughput의 99% 기준은 $\rho\le\rho_{\min}/.99$와 같고 $\varepsilon=.05$와는 다른 정의다. 위 $s_\varepsilon$는 near-best 집합의 첫 원소이지 이후 모든 길이가 plateau라는 보장은 아니다. 최소점이 측정 상한에 붙거나 bootstrap 분포가 넓으면 범위를 넓혀야 한다. 상한을 포화점이라고 자동 대입하지 않는다.

반복 block을 길이 전체에 걸쳐 함께 bootstrap한다. 매 복제마다 median, 최소점, near-best 길이를 다시 계산한다. argmin의 선택 편향과 다중 후보 경쟁 때문에 한 점의 작은 오차막대를 포화점의 확신으로 해석하지 않는다.

## 두 직선 가설의 경쟁 모델

$$
\begin{aligned}
T_A(\ell)&=a+b\ell,\\
T_H(\ell)&=a+b\ell+d(\ell-s)_+,\\
T_S(\ell)&=b_1\ell+d\max(s,\ell),\quad b_1,d\ge0.
\end{aligned}
$$

$T_H$는 연속 hinge이지만 기울기·절편을 강제하지 않는다. $T_S$는 다음의 강한 plateau 가설이다.

$$T_S(\ell)=\begin{cases}ds+b_1\ell&\ell\le s,\\(b_1+d)\ell&\ell>s.\end{cases}
\quad \frac{T_S(\ell)}\ell=b_1+d\max(s/\ell,1).$$

따라서 $\ell\ge s$의 모든 점이 최소 기울기를 갖는다. “포화점만이 유일한 최적 길이”라는 결론은 나오지 않는다. $d=0$이면 모든 길이의 효율이 같다. $T_S(0)=0$은 별도로 정의한다.

처음 4개 반복의 median으로 계수와 $s$를 선택한다. 남은 3개 반복에는 **계수와 breakpoint를 다시 맞추지 않는다**. 검증 RMSE·상대오차와 residual plot으로 가설을 비교한다. $s$는 양쪽에 측정점이 남는 interior grid만 탐색한다. 모델의 in-sample $R^2$만으로 두 직선을 채택하지 않는다.

## 출력과 해석

길이–latency 오차막대, 원점 접선, $T/\ell$, batch 수별 throughput, holdout residual, bootstrap 포화점 분포를 같은 노트북에 둔다. 원시 CSV와 mount·seed·workers·소스 hash를 실행별 디렉토리에 남긴다. 다른 노트북은 이 결과를 읽지 않고 필요한 측정을 새로 수행한다.

측정 범위를 벗어난 값은 외삽하지 않는다. 행 크기 변경은 $s_{rows}=s_{bytes}/bytes_{row}$로 환산해야 하며, 4 KiB 실험의 행 수를 원본 모델의 채널 수와 자동으로 동일시할 수 없다.
