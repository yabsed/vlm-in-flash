# 07 · mask의 예측 latency와 실제 읽기 시간

## 질문

per-chunk 비용의 합이 실제 혼합 mask 읽기 시간을 설명하는가? 좋은 I/O mask를 만드는 데 걸리는 selector 시간이 그 이득을 지우는가? batch로 얻은 $T$를 개별 read의 물리 법칙처럼 취급하지 않고 검증한다.

$$\hat L(M)=\sum_{C\in\mathcal R(M)}\hat T(|C|),\qquad
W_p(M)=\operatorname{walltime}(\text{all requests of }M;\ p\text{ workers}).$$

가산 모델과 병렬 실행의 차이를 이해하기 위한 이상적인 scheduling bound는

$$\max\{\max_j\tau_j,\;p^{-1}\sum_j\tau_j\}\le\operatorname{makespan}\le\sum_j\tau_j.$$

이는 독립 job service time $\tau_j$를 가정한 bound다. SSD에서 service time 자체가 concurrency에 따라 변할 수 있으므로 측정 $T$를 바로 $\tau$에 넣어 hardware bound라고 선언하지 않는다.

## 독립 calibration과 validation

이번 실행에서 새 512 MiB 파일로 길이별 profile을 측정한다. 새로운 importance로 여러 $R$과 방법의 mask를 만든 뒤, **별도의 새 128 MiB 파일**에서 각 최대 run을 한 개의 aligned request로 바꾸어 실행한다. 1 worker와 4 workers를 각각 측정하며 모든 method×budget 조합의 순서를 반복마다 무작위화한다. buffer warmup 후 9회 반복의 원시 시간을 보존한다.

mask는 $n=256$, row size 4096 byte로 모두 정렬된다. 실험 범위에서 run은 최대 1 MiB이므로 request splitting은 필요하지 않다. 파일 내 base offset도 반복마다 바꾸어 하나의 주소에만 종속되는 결과를 줄인다. page cache는 O_DIRECT로 우회하지만 장치 cache·다른 프로세스·온도·전력 상태는 통제했다고 주장하지 않는다.

## fit과 진단

$$W(M)\approx\alpha\hat L(M),\qquad
W(M)\approx\beta+\alpha\hat L(M).$$

예산의 앞 절반 mask로 $\alpha,\beta$를 fit하고 뒤 절반의 다른 mask로 검증한다. 같은 mask의 다른 반복만 holdout으로 쓰는 것보다 강한 분할이다. holdout RMSE와 $(\hat L,W)$ scatter, residual 대 run 수를 출력한다. proportional bias가 모든 mask에 일정하면 ratio 순위는 보존되지만, mask별 오차나 절편이 있으면 그런 결론은 자동으로 나오지 않는다.

$$\frac{I}{\alpha L}=\alpha^{-1}\frac IL,\qquad
\frac{I}{\alpha L+\beta}\text{ 는 }\frac IL\text{ 와 같은 순위를 보장하지 않는다.}$$

## 실제로 측정하는 총 비용

$$H(M)=S(M)+W_p(M).$$

$S$는 importance와 cost table이 이미 메모리에 있을 때 **선택 함수 전체**의 CPU 시간이다. prefix sum·후보 생성·정렬·mask 구성을 포함한다. profiling은 offline 비용으로 제외한다. $H$는 selector+I/O subtotal이며 GPU upload, gather, GEMM, 전체 모델 추론 시간은 포함하지 않는다.

runtime은 이 노트북의 Python 구현에 한정한다. 논문의 C++/GPU 구현보다 빠르거나 느리다는 주장은 하지 않는다. 다른 구현 비용을 공정하게 비교하려면 같은 backend가 필요하다.

## 출력

예측–실측 관계, workers에 따른 변화, held-out 오차, $S$와 $W$의 누적 막대, retained importance 대 $H$를 표시한다. 동일 $R$의 I/O 비교와 importance가 다른 방법의 utility 비교를 함께 둔다. 값이 기대와 반대여도 수동으로 결론을 고치지 않고 실행 결과에 그대로 남긴다.
