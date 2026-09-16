난이도는 한 단계 올라갔지만, 다시 \(2^N\) 지수시간 문제가 된 것은 아닙니다. 다만 이제는 다음 세 가지를 구분해야 합니다.

## 1. 주어진 \(\lambda\)에 대한 문제는 여전히 \(O(N)\)

piecewise latency를

$$
T(r)=a+c_1r+d(r-s)_+,
\qquad d=c_2-c_1>0
$$

라고 합시다. 여기서 \(s\)는 saturation 길이입니다.

주어진 \(\lambda\)에 대해

$$
\boxed{
\max_M\left\{\lambda I(M)-\widehat L(M)\right\}
}
$$

를 푸는 문제는 여전히 정확히 \(O(N)\)에 풀 수 있습니다.

마지막 chunk가 \([i,j]\)라고 하면

$$
D_j
=
\max\left\{
D_{j-1},
\max_{i\le j}
\left[
D_{i-2}
+\lambda(P_j-P_{i-1})
-T(j-i+1)
\right]
\right\}.
$$

후보를

$$
j-i+1\le s
$$

인 짧은 chunk와

$$
j-i+1>s
$$

인 긴 chunk로 나눕니다.

* 짧은 chunk: sliding-window maximum
* 긴 chunk: prefix maximum

으로 관리할 수 있기 때문에 각 \(j\)를 amortized \(O(1)\)에 처리할 수 있습니다.

따라서 \(q\)개의 \(\lambda\)를 탐색하는 Lagrangian 알고리즘은 여전히

$$
\boxed{O(qN)}
$$

입니다.

즉, **두 직선이 되었다고 해서 우리의 lightweight heuristic이 망가진 것은 아닙니다.**

## 2. 하지만 \(O(qN)\)은 coverage 문제의 exact solver는 아님

우리가 진짜 풀고 싶은 문제는

$$
\boxed{
\min_M\widehat L(M)
\quad\text{s.t.}\quad
I(M)\ge B
}
$$

입니다.

여러 \(\lambda\)에 대해

$$
\max_M\{\lambda I(M)-\widehat L(M)\}
$$

를 풀면 importance–latency frontier의 supported point는 찾을 수 있습니다. 그러나 이산적인 마스크 집합에는 unsupported Pareto point가 존재할 수 있습니다.

따라서

$$
\boxed{
O(qN)\text{은 매우 빠른 휴리스틱이지만 coverage 최적해를 항상 보장하지는 않는다.}
}
$$

각 \(\lambda\)의 내부 문제는 exact이지만, 원래 제약 문제에 대해서는 heuristic이라는 뜻입니다.

## 3. coverage 문제의 정확해도 polynomial DP로 구할 수 있음

piecewise 모델에서는

$$
\widehat L(M)
=
aK(M)+c_1R(M)+dE_s(M),
$$

$$
E_s(M)
=
\sum_{C\in\mathcal C(M)}(|C|-s)_+
$$

입니다.

따라서 다음 상태를 정의할 수 있습니다.

$$
F_i(r,k,e,\ell)
=
\text{처음 \(i\)개 채널에서 얻는 최대 importance}.
$$

각 변수는 다음을 뜻합니다.

* \(r\): 선택한 행 수
* \(k\): 형성된 chunk 수
* \(e\): saturation 길이를 초과한 행 수
* \(\ell\): 현재 chunk 길이. 단, \(s\) 이상은 하나의 상태로 묶음

즉,

$$
\ell\in\{0,1,\ldots,s\},
$$

에서 \(\ell=s\)는 “현재 chunk 길이가 \(s\) 이상”이라는 뜻입니다.

전이는 다음과 같습니다.

* 현재 행을 선택하지 않음: \(\ell\leftarrow0\)
* 새로운 chunk 시작: \(r\leftarrow r+1,\ k\leftarrow k+1,\ \ell\leftarrow1\)
* 짧은 chunk 연장: \(r\leftarrow r+1,\ \ell\leftarrow\ell+1\)
* 포화된 chunk 연장: \(r\leftarrow r+1,\ e\leftarrow e+1\)

마지막에는

$$
\boxed{
\min_{\substack{r,k,e\\
\max_\ell F_N(r,k,e,\ell)\ge B}}
\left(
ak+c_1r+de
\right)
}
$$

를 선택합니다. importance를 양자화할 필요 없이 각 구조적 상태에서 최대 importance만 보존하므로 정확합니다.

단순 구현의 상한은 대략

$$
\boxed{O(N^4s)}
$$

시간과

$$
O(N^3s)
$$

rolling memory입니다. \(s=\Theta(N)\)까지 허용하면 최악에는 \(O(N^5)\)입니다. 이는 타이트한 하한이 아니라 가장 직접적인 exact DP의 상한입니다.

따라서 이 방법은 이론적 exact oracle로는 가능하지만, 실제 inference 중 매번 실행하기에는 지나치게 큽니다.

## 실제 lookup table을 그대로 쓰면

정확한 비용이

$$
T(r)=
\begin{cases}
T_{\mathrm{table}}(r),&r\le s,\\
T(s)+c_2(r-s),&r>s
\end{cases}
$$

라면 짧은 구간은 임의의 lookup 값이므로 단순한 두 직선 최적화는 쓸 수 없습니다.

주어진 \(\lambda\)의 문제는

$$
O(Ns)
$$

에 풀 수 있고, \(q\)개 \(\lambda\)를 사용하면

$$
\boxed{O(qNs)}
$$

입니다. AGX에서 \(s\approx135\text{--}146\)이므로, 단일 affine/piecewise-line 모델보다 약 100배 이상 무거울 수 있지만 여전히 지수시간은 아닙니다.

## 정리

| 문제                                     | 성격                       |               복잡도 |
| -------------------------------------- | ------------------------ | ----------------: |
| Two-line, 주어진 \(\lambda\)              | exact penalized solution |          \(O(N)\) |
| Two-line, \(q\)개 \(\lambda\)           | coverage heuristic       |         \(O(qN)\) |
| Two-line coverage exact                | 구조 상태 전체 DP              | 단순 상한 \(O(N^4s)\) |
| Lookup+linear tail, 주어진 \(\lambda\)    | exact penalized solution |         \(O(Ns)\) |
| Lookup+linear tail, \(q\)개 \(\lambda\) | coverage heuristic       |        \(O(qNs)\) |

핵심은 이것입니다.

$$
\boxed{
\text{정확한 coverage 최적해는 어려워졌지만,
빠른 }O(qN)\text{ 휴리스틱은 여전히 유지된다.}
}
$$

따라서 졸업논문의 좋은 구도는 다음과 같습니다.

1. saturation-aware two-line 모델을 제시한다.
2. 무거운 exact DP를 offline oracle로 사용한다.
3. \(O(qN)\) Lagrangian 알고리즘을 online heuristic으로 제안한다.
4. Paper Greedy와 정확도·latency·실행시간을 비교한다.

단일 affine 모델의 약점이 드러났지만, 알고리즘 연구의 중심은 오히려 더 선명해진 셈입니다.
