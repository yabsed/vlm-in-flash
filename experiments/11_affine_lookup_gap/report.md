# Experiment 11 보고서: affine 격차가 lookup에서 줄어드는 이유

## 질문

Experiment 07에서 Paper Greedy는 동일 importance의 affine DP-Cover보다
평균 `30.0%` 느렸다. 그런데 동일한 mask를 공개 Orin AGX lookup table로
재평가한 Experiment 10에서는 격차가 `5.35%`로 줄었다.

이 실험은 다음 세 가지 가설을 검증한다.

1. Paper와 DP-Cover의 chunk-length 분포가 근본적으로 다르다.
2. 그 차이 때문에 affine에서 lookup으로 옮길 때 두 방법의 latency가 서로
   다른 비율로 재조정된다.
3. DP-Cover의 긴 chunk 대부분은 공개 table 범위를 벗어나므로 결과가
   외삽 규칙에 민감하다.

## 방법

Experiment 10에 저장한 모든 mask의 `start:length` RLE를 재사용했다. Paper,
DP-R, Top-R 또는 exact DP를 다시 실행하지 않았다. 기본 분석은 여섯 VLM
CV에 해당하는 378개 fixed-R case를 사용한다.

한 행은 `1.75 KiB`이고 공개 table은 `255 KiB`까지 측정되어 있으므로,
직접 측정 범위는 chunk 길이 약

\[
\ell_{\max}=255/1.75=145.7\text{ rows}
\]

까지다. 공개 구현은 이 범위를 넘으면 마지막 table 값을 비례 확대한다.

\[
T_{\mathrm{released}}(s)=T(255)\frac{s}{255},\qquad s>255\text{ KiB}.
\]

이 외삽의 기울기는 `0.0003323 ms/row`다. 전체 table의 affine fit은

\[
L_{\mathrm{aff}}(M)=0.0110009K+0.000244672R
\]

이므로, 긴 chunk에서는 공개 외삽의 행당 비용이 affine 기울기보다 약
`35.8%` 높다. 두 모델의 단일-chunk 비용은 약 `125.5행`에서 교차한다.

외삽 민감도는 다음 네 latency 정책으로 확인했다.

- `Affine fit`: Experiment 07의 원래 목적함수
- `Released extrapolation`: 공개 코드의 endpoint proportional scaling
- `Anchored tail-linear`: 128--255 KiB 측정값에 선형 기울기를 fit한 후
  `T(255)`에서 연속적으로 연장
- `255-KiB block split`: 긴 요청을 255 KiB block과 나머지로 나눠 합산

Anchored tail fit의 기울기는 `0.00029185 ms/row`, `R²=0.9948`이다. 이는
민감도 분석용 대안이지 실제 장치의 정답으로 간주하지 않는다.

## 1. Chunk-length 분포

| 방법 | 평균 chunk 수 | 전체 run 중앙값 | case별 평균 run 중앙값 | 범위 밖 chunk에 속한 행 |
| --- | ---: | ---: | ---: | ---: |
| Paper Greedy | 25.24 | 84행 | 94.0행 | 20.8% |
| DP-Cover@Paper | 4.72 | 206.5행 | 678.7행 | 90.3% |
| DP-R | 2.98 | 423행 | 912행 | 95.3% |
| DP-Cover@DP-R | 4.74 | 179행 | 608행 | 87.2% |
| Top-R | 557.75 | 2행 | 4.59행 | 6.0% |
| DP-Cover@Top-R | 3.89 | 333행 | 1,479.8행 | 94.8% |

Paper의 전형적인 chunk는 측정 범위 안에 있지만, matched DP-Cover가 선택한
행의 `90.3%`는 측정 범위보다 긴 chunk에 포함된다. 따라서 두 방법을
lookup으로 비교하면 분모인 DP-Cover가 외삽 정책에 훨씬 강하게 노출된다.

## 2. Paper 격차의 정확한 분해

각 case에서 다음 항등식이 성립한다.

\[
\frac{L_P^{lookup}}{L_C^{lookup}}
=
\frac{L_P^{aff}}{L_C^{aff}}
\cdot
\frac{L_P^{lookup}/L_P^{aff}}
     {L_C^{lookup}/L_C^{aff}}.
\]

측정된 평균은 다음과 같다.

| 항목 | 평균 |
| --- | ---: |
| Affine Paper/DP-Cover ratio | 1.300 |
| Paper lookup/affine rescaling | 1.009 |
| DP-Cover lookup/affine rescaling | 1.247 |
| 상대 보정계수 | 0.813 |
| 최종 lookup Paper/DP-Cover ratio | 1.053 |

즉 Paper mask의 비용은 거의 변하지 않지만, matched DP-Cover mask는 평균
`24.7%` 비싸게 재평가된다. case별 항등식의 최대 수치 오차는
`5.33e-15`였다. 따라서 `30.0% → 5.35%` 변화는 집계상의 우연이 아니라
chunk-length-dependent rescaling으로 완전히 설명된다.

Ordering별로 보면 random 입력에서 이 효과가 가장 강하다. 공개 lookup
평가에서 Paper/DP-Cover 평균 ratio는 random `0.984`, local `1.098`,
hot-cold `1.078`이다. Random에서는 Paper가 lookup-aware exact optimum을
이긴 것이 아니라, affine-optimal mask보다 이 lookup evaluator에서 싸게
평가된 것이다.

## 3. 외삽 민감도

| Latency 정책 | Paper 평균 ratio | Paper 중앙값 | Paper가 DP-Cover보다 싼 비율 |
| --- | ---: | ---: | ---: |
| Affine fit | 1.300 | 1.285 | 0.0% |
| Released extrapolation | 1.053 | 1.038 | 25.4% |
| Anchored tail-linear | 1.143 | 1.127 | 0.53% |
| 255-KiB block split | 1.050 | 1.043 | 31.5% |

공개 외삽과 block split에서는 Paper 격차가 약 5%지만, 측정 tail의 기울기를
연장하면 `14.3%`로 커진다. 어느 경우에도 affine의 `30.0%`보다는 작지만,
정확한 격차와 “Paper가 DP-Cover보다 싼 case”의 빈도는 외삽 선택에 크게
의존한다.

DP-R은 상대적으로 안정적이다. 평균 matched ratio는 affine `1.016`, 공개
외삽 `1.048`, anchored tail `1.034`, block split `1.046`이다. DP-R과 matched
DP-Cover가 모두 긴 few-chunk mask이므로 extrapolation exposure가 비슷하기
때문이다.

Top-R은 모든 정책에서 나쁘다. 평균 ratio는 `6.53--8.60x`이며, 결론이
외삽 선택에 의해 뒤집히지 않는다.

## 산출물

- `latency_model_and_runs.pdf`: chunk 길이에 따른 모델 차이, run ECDF,
  측정 범위 밖 행의 비율
- `paper_gap_decomposition.pdf`: ordering별 Paper 격차, unequal rescaling,
  외삽 노출
- `extrapolation_sensitivity.pdf`: 네 latency 정책의 결과 비교
- `run_length_summary.csv`: 방법 및 ordering별 chunk 통계
- `ratio_decomposition.csv`: 모든 case의 항등식 분해
- `extrapolation_sensitivity.csv`: 모든 mask와 latency 정책의 재평가 결과

## 결론

Experiment 10에서 격차가 줄어든 직접적인 이유는 Paper가 lookup 범위 안의
중간 길이 chunk를 주로 선택하는 반면, affine DP-Cover는 chunk 수를 줄이기
위해 측정 범위를 훨씬 넘는 긴 chunk를 선택하기 때문이다. 공개 evaluator는
후자를 affine 예상보다 비싸게 외삽한다.

따라서 현재 근거로 말할 수 있는 것은 다음과 같다.

> Affine DP-Cover는 lookup objective의 신뢰할 수 있는 oracle이 아니며,
> Experiment 07의 30% 격차를 실제 lookup 격차로 해석해서는 안 된다.

반면 Paper가 진짜 lookup optimum에 가깝다는 결론도 아직 낼 수 없다.
DP-Cover mask의 대부분을 포함하는 255 KiB 이상의 직접 측정과
`\sum_C T(|C|)`를 목적함수로 사용하는 lookup-aware exact solver가 모두
필요하다.

