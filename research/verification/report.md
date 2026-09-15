# `idea.md` 검증 보고서

## 결론

**핵심 수학 정리와 세 종류의 DP 점화식은 문서의 가정 아래 맞다.** 다만 실제 구현의 예산 반올림에 관한 서술은 다르며, 빈 greedy 결과와 coverage 초과 달성을 평가할 때 보완이 필요하다.

본문의 간결한 스타일을 유지하기 위해 `idea.md`는 수정하지 않았다. 아래에 증명, 재현 가능한 계산, 구현상의 차이를 따로 기록한다.

## 검증 범위와 방법

- 대상: 현재 `idea.md` 전체. 정확한 파일 SHA-256은 [results.json](results.json)의 `idea_sha256`에 기록한다.
- 서브모듈: `9023679ebfd1900fa2912c4098f51bfb9efa7bb8`.
- 일반적인 수학 주장은 아래의 직접 증명과 귀납법으로 검토했다.
- 알고리즘은 별도로 모든 이진 마스크를 열거하는 기준 계산과 비교했다. DP·비율·종료 조건에는 Python `Fraction`의 정확한 유리수 연산을 사용했다.
- 프로파일 피팅에는 부동소수점 최소제곱 계산을 사용했다. 실제 하드웨어의 읽기 실험을 새로 수행한 것은 아니다.
- Lean 형식 증명은 수행하지 않았다. 아래 증명은 사람이 검토할 수 있는 수학 증명이며, 유한 전수조사는 일반 증명을 대체하지 않는다.

재현:

```bash
python3 verification/verify_idea.py --implementation --output verification/results.json
```

기본 수학·프로파일 검증에는 Python 표준 라이브러리만 필요하다. `--implementation`은 설치된 PyTorch와 NumPy를 사용하며, 서브모듈의 순수 PyTorch 경로와 실제 `_select` 메서드를 호출한다. CUDA/native 커널 실행 검증은 포함하지 않는다.

### 실행 결과

모든 비교가 통과했다. 테스트 입력은 $N=1,\ldots,6$의 모든 $\{0,1,2\}^N$ 벡터 1,092개와 $N=1,\ldots,10$의 유리수 벡터 200개다. 비용 $a,c$도 양의 정수·유리수로 바꾸었고, 단일 청크 검증에는 단조성을 가정하지 않는 양의 lookup 비용을 사용했다. 전부 0인 벡터는 문서의 가정 밖인 경계 테스트로 포함했다.

| 검증 항목 | 비교 수 |
|---|---:|
| 중요도 벡터 | 1,292 |
| 최종 마스크 열거 | 96,906 |
| 모든 prefix의 도달 가능한 $F$ 상태 | 73,965 |
| Fixed-$R$ 비율 최적화 및 Dinkelbach | 7,115 |
| Coverage threshold 최적화 | 27,941 |
| Lagrangian 내부 문제 | 6,460 |
| 실제 PyTorch greedy와 독립 선택 계산 | 72 |

$G$의 모든 도달 가능한 최종 상태도 네 multiplier 값에서 비교했다. Dinkelbach의 관측 최대 평가 횟수는 4회이며, 이는 테스트 결과이지 일반적인 4회 상한이 아니다. 검증 스크립트가 복원한 마스크의 비율도 직접 계산했다.

## 1. 전체 주장 판정

| 주장 | 판정 | 근거 또는 조건 |
|---|---|---|
| 비율은 청크 utility의 latency 가중평균이다 | 맞음 | 양의 청크 비용, 중요도·비용의 가산성 |
| 상한 예산 문제에 단일 청크 최적해가 존재한다 | 맞음 | 모든 길이 $1\le r\le R$의 연속 구간을 허용 |
| 최적 다중 청크는 각각 전역 최대 utility를 갖는다 | 맞음 | 모든 가중치가 엄밀히 양수 |
| prefix sum과 $O(NR)$ 탐색 | 맞음 | 각 $T[r]$를 $O(1)$에 참조하는 계산 모델 |
| fixed-$R$/일반 cardinality band에는 같은 단일 청크 논증이 적용되지 않는다 | 맞음 | 아래의 직접 반례 |
| top-$R$가 고정 행 수에서 중요도를 최대화한다 | 맞음 | 교환 논증 |
| greedy의 utility 정렬·중복 배제·예산 검사 | 맞음 | `policy.py`의 실제 선택 경로와 대조 |
| 구현 예산은 $\lfloor(1-\rho)N\rfloor$이다 | **구현과 다름** | `_select`는 $N-\lfloor\rho N\rfloor$ 사용 |
| fixed-$R$는 재구성한 비교 문제이다 | 맞음 | greedy의 등호 제약 또는 근사비 보장은 없음 |
| 프로파일 피팅 수치·단위 변환 | 맞음 | 두 JSON의 모든 항목으로 재계산 |
| affine throughput 증가·점근값 | 맞음 | $a,c_{\mathrm{KiB}}>0$에서 미분·극한 계산 |
| $L_{\mathrm{aff}}=aK+cR$ 및 고정 $R$에서의 $K$ 순서 | 맞음 | affine 모델 안의 정확한 등식 |
| $K=M_1+\sum_{i=2}^N(1-M_{i-1})M_i$ | 맞음 | 각 선택 run의 시작을 정확히 한 번 계산 |
| $K_{\max}=\min(R,N-R+1)$ | 맞음 | 선택 행 수와 사이의 0 개수로 상한, 구성으로 달성 |
| $F_i^s(r,k)$ DP와 고정 $R$ 비율 최적화 | 맞음 | prefix 귀납법, 같은 $(r,k)$의 동일 비용 |
| exact-$k$ 중요도는 비단조일 수 있다 | 맞음 | 아래의 직접 반례 |
| nondominated 상태가 fixed-$R$ frontier를 이룬다 | 맞음 | 같은 $(r,k)$에서 최대 중요도만 보존하면 충분 |
| $\Psi(\eta)$의 부호와 Dinkelbach 갱신 | 맞음 | 양의 분모, 정확한 내부 최대화 |
| $G_i^s(r;\eta)$ DP와 $O(qNR)$ | 맞음 | run 시작 때만 $\eta a$, 마지막에 $\eta cR$ 차감 |
| coverage의 모든 $(r,k)$를 이용한 exact DP | 맞음 | 같은 비용 상태의 최대 중요도가 feasible 여부 결정 |
| coverage 비용 동률에서 최대 중요도를 택하면 Pareto 최적 | 맞음 | 지배점이 있다고 가정하면 최소 비용 또는 동률 규칙에 모순 |
| $D_i^s(\lambda)$ DP와 $O(qN)$ | 맞음 | 선택 행마다 $c-\lambda v_i$, run 시작마다 $a$ |
| Lagrangian 해는 coverage 최적해를 놓칠 수 있다 | 맞음 | 아래의 unsupported point 반례 |
| full-selection fallback의 feasibility | 맞음 | $\alpha\le1$, $I_{\mathrm{tot}}>0$ |
| threshold가 커질수록 최적 latency가 감소하지 않는다 | 맞음 | feasible set의 포함 관계 |
| 여섯 threshold는 frontier의 표본이다 | 맞음 | 중복 점과 표본 사이의 미포착 점이 가능 |
| 달성 coverage 기준 affine gap은 음수가 아니다 | 맞음 | 해당 마스크가 비교 문제의 feasible solution |
| 그 gap이 요청 coverage에 대한 최적화 손해를 모두 나타낸다 | **그렇지 않음** | 초과 달성으로 생긴 손해는 별도 평가 필요 |
| affine 최적성이 실측 latency·VLM 정확도 최적성을 뜻한다 | 문서도 주장하지 않음 | 실제 장치·태스크 실험이 필요 |
| $O(qN)$이면 약 2 ms 이내이다 | 미검증 실험 가설 | 점근 복잡도만으로 실제 실행 시간을 결정할 수 없음 |

## 2. 단일 청크 정리와 prefix sum

### 가중평균과 전역 최적성

마스크의 최대 연속 run을 $C_1,\ldots,C_k$라 하고 $t_j=T[|C_j|]>0$, $u_j=I(C_j)/t_j$라 하자. 그러면

$$
\frac{I(M)}{L_{\mathrm{table}}(M)}
=\frac{\sum_j t_j u_j}{\sum_j t_j}
\le\max_j u_j.
$$

$U^\star$를 길이 $1,\ldots,R$의 모든 구간 중 최대 utility라고 하자. 모든 feasible 마스크의 각 run도 길이가 $R$ 이하이므로 마스크 비율은 $U^\star$ 이하이다. $U^\star$를 달성하는 구간 하나를 선택하면 feasible하면서 같은 값을 달성한다. 따라서 단일 청크 최적해가 존재한다.

또한

$$
U^\star-\frac{I(M)}{L_{\mathrm{table}}(M)}
=\sum_j\frac{t_j}{\sum_h t_h}(U^\star-u_j).
$$

모든 항이 음이 아니고 가중치가 양수이므로 등호는 모든 $u_j=U^\star$일 때에만 성립한다. $T$의 단조성이나 affine 성질은 필요 없다. 후보 구간을 특정 window 목록으로 제한하는 다른 문제와는 구분해야 한다.

### 탐색 비용

0 기반 인덱스에서 $P_j=\sum_{i<j}v_i$이므로 구간 $[i,i+\ell)$의 중요도는 $P_{i+\ell}-P_i$이다. 길이 $\ell$인 구간은 $N-\ell+1$개이며,

$$
\sum_{\ell=1}^R(N-\ell+1)
=R(N+1)-\frac{R(R+1)}2
=\frac{R(2N-R+1)}2.
$$

$R=N$이면 $\binom{N+1}{2}$이다. Prefix 계산 $O(N)$, 각 점수 계산 $O(1)$이므로 총 $O(NR)$, prefix 저장 $O(N)$이다. 실제 lookup 함수가 이진 탐색을 수행하면 그 비용을 포함하거나 행 길이별 비용을 사전 계산해야 한다.

### fixed-$R$에 같은 결론이 없는 반례

$v=(3,0,3)$, $a=c=1$, $R=2$:

- `101`: $I=6$, $K=2$, $L=4$, 비율 $3/2$.
- `110`, `011`: $I=3$, $K=1$, $L=3$, 비율 $1$.

최적해는 두 청크다. 개별 청크만 남기면 $R(M)=R$ 제약을 위반하므로 상한 문제의 증명을 옮길 수 없다. 일반 band에도 등호 제약이 특수한 경우로 포함된다.

### top-$R$

선택한 값보다 큰 값을 선택하지 않았다면 두 위치를 교환해 목적값을 증가시킬 수 있다. 그런 교환이 더 이상 불가능할 때 선택 집합은 상위 $R$개 값으로 이루어진다. 동률이면 여러 최적 마스크가 가능하다.

## 3. affine 구조와 fixed-$R$ DP

### 행 단위 변환과 chunk count

행 크기가 일정한 $b$ KiB이면 $z=br$, $c=bc_{\mathrm{KiB}}$이다. 따라서

$$
\sum_C(a+c|C|)=aK+cR.
$$

같은 $R$에서는 $L_1-L_2=a(K_1-K_2)$이므로 $a>0$에서 순서 동치가 성립한다. Lookup 모델에 이 동치를 적용할 수는 없다.

각 run의 첫 행은 앞이 0이거나 배열의 시작이다. 따라서 $K$의 이진 전이 공식이 맞다. $K\le R$이고, $K$개 run 사이에는 최소 $K-1$개의 0이 필요하므로 $K\le N-R+1$이다. 역으로 $k\le\min(R,N-R+1)$이면 $k$개의 양의 run 길이로 $R$을 나누고 사이에 0을 배치할 수 있으므로 모든 $1\le k\le K_{\max}$가 가능하다.

### $F$ 점화식의 귀납 증명

$i=0$에서는 가상 이전 상태를 0으로 두고 $(r,k,s)=(0,0,0)$만 값 0을 갖는다. 나머지는 도달 불가능하다.

$i-1$ 단계가 정확하다고 가정한다.

1. $M_i=0$이면 행 수·청크 수·중요도는 바뀌지 않는다. 이전 마지막 상태 0 또는 1 중 최댓값이 $F_i^0(r,k)$다.
2. $M_i=1$, $M_{i-1}=1$이면 행 하나와 중요도 $v_i$만 추가된다. 이전 상태는 $(r-1,k,1)$이다.
3. $M_i=1$, $M_{i-1}=0$이면 새 run이 생긴다. 이전 상태는 $(r-1,k-1,0)$이다.

모든 feasible prefix가 정확히 이 경우 중 하나이며, 각 전이는 feasible prefix를 만든다. 따라서 문서의 두 점화식이 정확하다. 음수 인덱스 등 불가능한 상태는 $-\infty$로 처리한다.

같은 $(R,k)$의 모든 마스크는 양의 분모 $ak+cR$을 공유한다. 그러므로 중요도가 최대인 마스크만 남긴 후 $k$별 비율을 비교하면 원래 affine 비율의 최적값을 얻는다.

### Frontier와 비단조성

$v=(0,10,0)$, $R=2$이면 $I^\star(2,1)=10$, $I^\star(2,2)=0$이다. Exact-$k$ 중요도가 감소하는 직접 반례다. 반면 at-most-$k$ 값은 후보 집합이 커지므로 비감소한다.

각 $(R,k)$에서 최대 중요도를 보존하면 모든 다른 마스크는 같은 비용의 보존된 점에 지배되거나 동일한 점이다. 따라서 보존된 점들 중 지배되지 않는 점을 추리면 정확한 fixed-$R$ frontier다.

## 4. Dinkelbach 변환과 $G$ DP

$A_M=I(M)$, $B_M=aK(M)+cR>0$, $\eta^\star=\max_M A_M/B_M$라 두면

$$
A_M-\eta B_M=B_M(A_M/B_M-\eta).
$$

- $\eta<\eta^\star$: 비율 최적 마스크를 대입하면 양수다.
- $\eta=\eta^\star$: 모든 값이 0 이하이고 최적 마스크가 0을 달성한다.
- $\eta>\eta^\star$: 유한한 모든 후보의 값이 음수이므로 최댓값도 음수다.

따라서 문서의 부호 판정이 맞다.

$G$는 중요도에서 새 run의 $\eta a$만 차감한다. $F$와 같은 두 이전 상태로 분기하되 run 수를 저장하는 대신 그 비용을 즉시 반영한다. 고정된 $\eta cR$을 마지막에 차감하면 $\Psi(\eta)$가 된다. 같은 prefix 귀납 논증으로 내부 최대화가 정확하다.

$\eta_t\le\eta^\star$이고 잔차가 양수이면

$$
\eta_{t+1}-\eta_t
=\frac{\Psi(\eta_t)}{B_{M_t}}>0.
$$

갱신값은 feasible 마스크의 비율이므로 최적값을 넘지 않는다. 가능한 비율이 유한하므로 정확 산술에서 종료한다. $\eta_0=0$으로 시작하면 이 논증이 바로 적용된다. 초기값이 최적값보다 크더라도 한 번 갱신하면 feasible 비율로 돌아온다.

추가로, fixed-$R$에서 내부 최적 마스크는 자기 $k$에 대해 최대 중요도를 갖는다. 따라서 갱신되는 비율은 $K_{\max}$개의 $I^\star(R,k)/(ak+cR)$ 중 하나다. $\eta_0=0$일 때 평가 횟수는 $K_{\max}+1$ 이하라는 상한도 얻지만, 문서의 $O(qNR)$은 실제 반복 수를 사용하는 올바른 표현이다.

작은 수치 잔차는 정확한 0과 다르다. 정확한 내부 해를 가정하고 $0\le\Psi(\eta)\le\varepsilon$, $\eta\le\eta^\star$이면

$$
0\le\eta^\star-\eta\le\frac{\varepsilon}{a+cR}.
$$

이는 최적 마스크의 분모가 $a+cR$ 이상임을 이용한 절대 비율 오차 한계다. 부동소수점 내부 DP의 오차까지 자동으로 보장하는 식은 아니다.

## 5. coverage DP, Lagrangian, Pareto 성질

### 정확한 coverage DP

같은 $(r,k)$의 비용은 $ak+cr$로 같다. 그 상태에서 feasible 마스크가 존재할 필요충분조건은 $I^\star(r,k)\ge B_\alpha$다. 필요성은 최대의 정의에서, 충분성은 그 최대를 달성하는 마스크의 존재에서 나온다. 따라서 해당 상태들의 비용 최솟값이 정확한 coverage 최적값이다.

최소 비용끼리 최대 중요도를 선택했는데 이를 지배하는 마스크가 있다고 가정하자. 그 마스크도 coverage를 만족한다. 더 싼 비용이면 최소성에, 같은 비용에서 더 큰 중요도이면 동률 규칙에 모순이다. 따라서 출력은 Pareto 최적이다.

$\alpha_1\le\alpha_2$이면 두 번째 feasible set이 첫 번째의 부분집합이므로 최소 비용은 비감소한다. 임의의 $I(M)>0$인 마스크는 자기 달성 coverage 문제의 feasible solution이며 최적 비용은 양수다. 따라서 문서의 achieved-coverage gap은 정의 가능하고 음수가 아니다.

### Lagrangian 내부 문제

완화 목적은

$$
\lambda B_\alpha+\sum_i(c-\lambda v_i)M_i+aK(M).
$$

첫 항은 마스크와 무관하다. 마지막 행을 버리면 이전 두 상태 중 최솟값, 선택하면 $c-\lambda v_i$를 더하고 이전 상태가 0일 때만 $a$를 더한다. 이것이 문서의 $D$ 점화식이며 $F$와 같은 prefix 귀납법으로 정확하다. 빈 마스크는 완화 문제에서는 허용해야 한다.

$\lambda_1<\lambda_2$에서 정확한 최소해의 중요도를 $I_1,I_2$라 하자. 두 최적성 부등식을 더하면

$$
(\lambda_2-\lambda_1)(I_2-I_1)\ge0.
$$

따라서 중요도는 multiplier에 따라 비감소한다. 다만 계단형으로 변해 threshold를 건너뛸 수 있다. Full-selection은 $I=I_{\mathrm{tot}}\ge B_\alpha$이므로 fallback의 feasibility는 보장된다.

### 놓치는 최적해: 두 채널 반례

$v=(2,1)$, $a=2$, $c=1$, $B=2$를 사용한다.

| 마스크 | 중요도 | 비용 | 완화 목적 |
|---|---:|---:|---|
| `00` | 0 | 0 | $0$ |
| `10` | 2 | 3 | $3-2\lambda$ |
| `01` | 1 | 3 | $3-\lambda$ |
| `11` | 3 | 4 | $4-3\lambda$ |

Coverage 최적해는 `10`, 비용은 3이다. 그러나 `10`이 `00`보다 좋거나 같으려면 $\lambda\ge3/2$, `11`보다 좋거나 같으려면 $\lambda\le1$이어야 한다. 동시에 만족할 수 없다. 즉, multiplier를 무한히 많이 탐색해도 이 최적해를 얻지 못한다. 이는 문서의 unsupported point 경고를 증명한다.

### 현재 gap이 감추는 부분

위 예에서 relaxed solver가 `11`을 반환하면, 목표 coverage $\alpha=2/3$에 대한 비용 손해는 $(4-3)/3=1/3$이다. 하지만 달성 coverage는 1이고 그 최적 비용도 4라서 문서의 $\operatorname{Gap}_{\mathrm{aff}}$는 0이다.

현재 수식은 틀리지 않았다. 다만 빠른 solver가 **요청한 coverage 문제**를 얼마나 잘 풀었는지도 평가하려면, feasible 출력에 대해 다음을 함께 기록해야 한다.

$$
\operatorname{Gap}_{\mathrm{target}}(M;\alpha)
=\frac{L_{\mathrm{aff}}(M)-L_{\mathrm{aff}}^\star(\alpha)}
{L_{\mathrm{aff}}^\star(\alpha)}.
$$

## 6. 시간·메모리 복잡도

| 방법 | 연산 수 | 값 계산용 추가 메모리 | 모든 부모 포인터를 저장할 때 |
|---|---:|---:|---:|
| prefix 구간 탐색 | $O(NR)$ | $O(N)$ | 최적 구간 양 끝만 저장하면 충분 |
| fixed-$R$의 $F$ DP | $O(NRK_{\max})$ | $O(RK_{\max})$ | $O(NRK_{\max})$ |
| Dinkelbach 내부 $G$ DP | $O(NR)$ / 회 | $O(R)$ | $O(NR)$ / 회 |
| 모든 $(r,k)$의 coverage DP | $O(N^3)$ | $O(N^2)$ | $O(N^3)$ |
| Lagrangian 내부 $D$ DP | $O(N)$ / 회 | $O(1)$ | $O(N)$ |

각 상태가 상수 개의 이전 상태만 보므로 상태 수를 세면 이 복잡도를 얻는다. Fixed-$R$ DP에서 $r>R$ 또는 $k>K_{\max}$인 prefix는 최종 목표에 도달할 수 없어 생략 가능하다. $R\approx N/2$에서는 $R,K_{\max}=\Theta(N)$이므로 cubic worst case가 실제로 가능하다.

모든 복잡도는 기본 산술·비교를 $O(1)$로 보는 통상적인 연산 수다. 정확한 유리수 구현의 비트 연산 비용이나 Python 객체 메모리 바이트 수를 뜻하지 않는다. $O(1)$ rolling memory는 이미 주어진 입력 벡터를 제외한 추가 메모리다. Coverage 최종 테이블은 threshold마다 $O(N^2)$ 스캔으로 재사용할 수 있다.

## 7. 구현과 문서 사이의 차이

### 7.1 행 예산의 반올림 — 수정할 사실

문서의 두 $R=\lfloor(1-\rho)N\rfloor$ 표현 중 특히 “implementation treats”라는 서술은 현재 서브모듈과 다르다. [linear.py](../vlm-flash/src/vlmflash/linear.py)의 `_select`는 다음 계산을 한다.

```python
num_skip = int(self.in_features * self.nc_sparsity)
budget = self.in_features - num_skip
```

양의 정확 산술로 해석하면 $R_{\mathrm{impl}}=N-\lfloor\rho N\rfloor=\lceil(1-\rho)N\rceil$이다. 실제 코드는 부동소수점 곱셈에 `int`를 적용한다. $N=5$, $\rho=0.25$에서는 문서의 값이 3, 실제 `_select`가 정책에 넘긴 값이 4임을 직접 실행해 확인했다.

이는 $R$을 입력받는 정리·DP 자체를 무효화하지 않는다. 구현과 동일한 예산으로 실험하려면 실제 전달된 $R$을 사용해야 한다.

### 7.2 후보 청크 비용과 최대 run 비용 — 실험 시 구분

[policy.py](../vlm-flash/src/vlmflash/policy.py)의 `Selection.est_cost_ms`는 수락한 후보별 비용의 합이다. [latency.py](../vlm-flash/src/vlmflash/latency.py)의 `mask_elat_ms`는 최종 마스크의 인접 행을 합친 최대 run별 비용의 합이다.

AGX 프로파일에서 1 KiB짜리 행 네 개를 각각 후보로 수락하면:

$$
\texttt{est\_cost\_ms}=4T[1]\approx0.03996317\text{ ms},
\qquad
L_{\mathrm{table}}(1111)=T[4]\approx0.01011320\text{ ms}.
$$

문서의 정의는 후자와 일치한다. 따라서 해당 열에 `est_cost_ms`를 그대로 넣으면 안 된다. 후보 정렬 규칙 자체는 문서와 일치한다.

### 7.3 빈 greedy 마스크 — 빠진 경계조건

실제 구현은 양의 예산에서도 빈 마스크를 반환할 수 있다. 기본 설정, $N=4$, $R=2$, 행 크기 1 KiB의 예를 실행해 확인했다.

따라서 “실제 선택 행 수의 fixed-$R$ frontier와 비교한다”는 문장은 선택 행 수가 양수일 때 적용된다. 0이면 문서의 $1\le R\le N$ 범위 밖이며 중요도/비용 비율도 $0/0$이다. 빈 결과는 별도로 기록해야 한다. 선택 결과의 중요도가 0인 경우에도 문서의 positive-coverage gap을 적용할 수 없다.

### 7.4 범위 밖 lookup

`chunk_ms`는 표 내부의 비정수 크기에 선형 보간을 적용하며, 상한 밖에서는 마지막 값을 크기 비율로 확대한다. `read_ms`는 최소 크기보다 작은 읽기를 최소 표 항목으로 올린다. 따라서 $T_{\mathrm{KiB}}(br)$를 일반적인 $br$에 적용하려면 이 규칙을 포함한 함수를 뜻해야 한다. 범위 밖 값은 측정값이 아니다.

## 8. 프로파일 수치와 외부 근거

재계산에는 [AGX JSON](../vlm-flash/src/vlmflash/profiles/orin-agx.json), [Nano JSON](../vlm-flash/src/vlmflash/profiles/orin-nano.json)의 전체 항목을 사용했다. OLS는 intercept를 포함한다. $R^2=1-\mathrm{SSE}/\mathrm{SST}$, 상대오차 분모는 관측 latency다. 정확한 수치는 [results.json](results.json)에 저장한다.

| 지표 | AGX | Nano |
|---|---:|---:|
| 표 범위 / 항목 수 | 1–255 KiB / 255 | 1–350 KiB / 350 |
| $a$ (ms) | 0.011000885814 | 0.015728789772 |
| $c_{\mathrm{KiB}}$ (ms/KiB) | 0.000139812728 | 0.000284206278 |
| Latency $R^2$ | 0.986719 | 0.994038 |
| MAPE | 4.218751% | 3.207409% |
| 최대 상대오차 | 28.629979% | 23.411260% |
| Throughput $R^2$ | 0.987535 | 0.986635 |

문서에 적힌 반올림 수치와 일치한다. Throughput은 $z$ KiB를 $T(z)$ ms에 읽으므로

$$
\beta(z)=\frac{z/1024}{T(z)/1000}
=\frac{1000z}{1024T(z)}\;\mathrm{MiB/s}.
$$

Affine 예측에서는

$$
\widehat\beta'(z)=\frac{1000a}{1024(a+c_{\mathrm{KiB}}z)^2}>0,
\qquad
\lim_{z\to\infty}\widehat\beta(z)=\frac{1000}{1024c_{\mathrm{KiB}}}.
$$

문서의 saturation 지점 throughput, 점근 throughput, MB/s에서 MiB/s로의 변환, 광고값과의 차이, 점근값 대비 75.0%·86.3%도 재계산 대상이다. 모두 문서의 반올림과 일치한다. 평균 fit 정확도가 모든 크기의 정확도나 새로운 장치에서의 성능을 보장하지 않는다는 문서의 한계 설명도 맞다.

원 논문의 §3.1–3.2에서 가산 모델·상한 예산·greedy 구조를, §4.1에서 SSD 사양을, Appendix D/H에서 236/348 KiB와 2 ms 기준을 확인했다. 논문 Figure 4 caption의 크기 표기는 Appendix D/H와 일치하지 않지만, `idea.md`는 D/H의 수치를 명시적으로 사용하고 있다. [원 논문](https://arxiv.org/html/2511.18692v1)

## 9. 이 검증으로 확정되지 않는 것

- Affine 또는 lookup 비용이 새 마스크의 실제 장치 latency를 얼마나 정확히 예측하는지.
- 빠른 solver의 실측 실행 시간이 2 ms 이하인지.
- Importance coverage가 VLM 태스크 정확도와 어떤 관계를 갖는지.
- Native/GPU 경로의 전체 수치 동작, 동시 I/O, cache, alignment 효과.

이들은 문서의 수학적 최적성에서 따라오는 결론이 아니며 별도의 실험 대상이다.
