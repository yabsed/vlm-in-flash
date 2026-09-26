# 02 · 읽기 크기와 SSD latency

이 실험은 연속해서 읽는 데이터 크기를 바꾸며 latency 곡선을 직접 측정한다. 크기는 KiB로 표시하고 CSV에는 byte와 bit 수도 저장한다. `1 KiB = 1024 bytes = 8192 bits`다.

## 측정 방법

`preliminary_research/vlm-flash/scripts/profile_flash.py`의 `prepare_blob`, `measure`, `saturation_throughput`을 그대로 호출한다. 해당 서브모듈의 native C++ `O_DIRECT` reader를 사용한다. 기존 실험의 결과 파일은 입력하지 않는다.

- 크기: 1–256 KiB는 1 KiB 간격, 260–768 KiB는 4 KiB 간격, 총 384개.
- 각 크기에서 chunk 수: `1, 2, 4, 8, 16, 28, 32, 48, 64, 96, 128, 192, 256, 384, 512`. 128 MiB 파일에 들어가는 조합만 측정한다.
- chunk 사이에는 32 KiB 간격을 둔다. CPU reader thread 6개, 각 조합마다 warmup 3회와 측정 10회.
- 같은 크기들을 오름차순·내림차순으로 각각 측정한다. 두 곡선을 모두 보존해 순서에 따른 차이를 확인한다.
- native reader의 요청 분할 한도가 768 KiB이므로 이번 sweep도 그 크기까지다. 작은 요청의 물리 I/O에는 alignment padding이 포함될 수 있다. 크기와 처리량의 분자는 논리적으로 요청한 byte다.

매 실행에서 새 난수 파일을 만들고 preallocation과 fsync를 수행한다. 공용 benchmark lock을 잡고 측정하며, `O_DIRECT`를 쓰지 못하면 중단한다. 임시 파일은 측정 후 삭제한다. 소스 hash, 서브모듈 revision, 장치 mount와 설정을 기록한다.

## latency의 정의

크기가 $b$ KiB이고 chunk 수가 $q$인 배치의 native I/O 시간 중앙값을 $t_{b,q}$ µs라고 하면 논리 처리량은

$$B(b,q)=\frac{qb/1024}{t_{b,q}/10^6}\quad[\mathrm{MiB/s}].$$

서브모듈은 총 읽기량에 따른 처리량의 3점 이동평균 기울기가 작아지는 구간을 찾고 그 이후 처리량을 평균한다. 그 지점을 찾지 못하면 마지막 20% 점의 처리량을 평균하며, 점이 3개 미만이면 마지막 값을 사용한다. 따라서 출력은 이 절차에 따른 처리량 추정값이며 포화의 증명은 아니다. 각 크기의 배치별 처리량과 마지막 두 배치의 차이를 함께 보존한다.

그 추정값 $\widehat B(b)$를 사용해

$$\widehat L(b)=\frac{b/1024}{\widehat B(b)}\,1000\quad[\mathrm{ms}]$$

로 크기별 latency를 계산한다. 이는 여러 chunk의 처리량에서 환산한 비용이다. 단일 read의 응답 시간이나 모델 전체 실행 시간이 아니다. 환산 전의 배치 시간도 `io_raw.csv`에 모두 남긴다.

## 그림과 산출물

1. **읽기 크기 → latency**: 전체 범위와 짧은 요청 확대, 두 측정 방향과 평균.
2. **latency / 크기**: 전체 범위의 로그 세로축과 큰 요청 구간의 선형 세로축 확대.
3. **측정 진단**: 대표 크기별 총 읽기량–처리량, 오름/내림 방향 차이, 마지막 배치 크기 증가에 따른 처리량 변화.

오름/내림 두 값의 차이는 반복 측정의 신뢰구간이 아니다. 그래프를 그릴 때 두 값의 범위를 그대로 표시하며 원시 10회 측정과 구분한다. 최적 길이나 breakpoint를 강제로 정하지 않고 관측한 곡선을 제시한다.
