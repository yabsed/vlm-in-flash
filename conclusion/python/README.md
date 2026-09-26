# 9개의 재현 실험

기존의 여러 실험을 **9개 `.ipynb` + 9개 수학 `.md`**로 압축했다. 수학 문서 전체가 대응 노트북에도 들어 있고, 실행 설정 → 새 계산/측정 → 표·그림 → 실행에서 확인한 값으로 이어진다. 노트북은 각각 독립적으로 Run All 할 수 있다. 다른 노트북이나 예전 실험의 결과 파일을 입력으로 사용하지 않는다.

## 실험 지도

| # | 질문 | 노트북 | 수학 문서 | 주요 검증 |
|---|---|---|---|---|
| 01 | 논문이 쓴 비율 목적은 실제로 어려운가? | [01_objective.ipynb](01_objective.ipynb) | [정의·단일 구간 정리](01_objective.md) | 완전탐색, 예산 채우기 반례 |
| 02 | 실제 length–latency는 두 직선인가? | [02_chunk_latency.ipynb](02_chunk_latency.ipynb) | [측정·포화점·경쟁 모델](02_chunk_latency.md) | 새 direct I/O, holdout, bootstrap |
| 03 | 작은 문제의 정확해와 얼마나 다른가? | [03_global_oracle.ipynb](03_global_oracle.ipynb) | [이산 Pareto와 regret](03_global_oracle.md) | 24개 입력의 모든 65,536 masks |
| 04 | 무엇을 다항시간에 정확히 풀 수 있는가? | [04_polynomial_dp.ipynb](04_polynomial_dp.ipynb) | [일반 DP·선형 DP·duality](04_polynomial_dp.md) | oracle 대조, 실행 시간, unsupported point |
| 05 | 새 실측 table 위에서 어느 방법이 유리한가? | [05_measured_frontier.ipynb](05_measured_frontier.ipynb) | [동일 importance 비교](05_measured_frontier.md) | latency–importance와 mask 그림 |
| 06 | 포화 길이 선택을 어디까지 보장할 수 있는가? | [06_saturation_guarantee.ipynb](06_saturation_guarantee.ipynb) | [latency 하한·quality bound·반례](06_saturation_guarantee.md) | smooth 입력, spike, 길이 민감도 |
| 07 | 예측 I/O 이득이 실제로 남는가? | [07_mask_walltime.ipynb](07_mask_walltime.ipynb) | [혼합 mask·병렬성·selector 비용](07_mask_walltime.md) | 전체 mask의 새 읽기, 1/4 workers |
| 08 | importance가 실제 projection 오차를 보장하는가? | [08_importance_error.ipynb](08_importance_error.ipynb) | [cross term·오차 상한](08_importance_error.md) | 원본 모델의 새 forward, 36 masks |
| 09 | 층마다 sparsity를 달리 배분하면 어떤가? | [09_global_allocation.ipynb](09_global_allocation.ipynb) | [전역 coverage·유한 후보 정확해](09_global_allocation.md) | 새 3-layer activation, 512 조합/방법 |

## 핵심 수학적 구분

[논문 §3.2.1](https://arxiv.org/html/2511.18692v1#S3.SS2.SSS1)에는 목적함수가 있다. 명시된 것은 `I(M)/L(M)` 최대화와 `|M| ≤ R`이다. “목적함수가 없다” 대신 다음을 검증한다.

1. **비율 목적:** 최대 run들의 비율의 가중평균이므로 최고 단일 구간이 전역 최적해다. prefix sum으로 `O(nR)`에 풀린다. 예산을 끝까지 채우는 greedy는 일반적으로 이 해가 아니다.
2. **가산 scalarization:** `max I−λL`은 일반 table에서 `O(n²)`, plateau two-line 모델에서는 monotone deque로 `O(n)`에 정확히 풀린다.
3. **quality 제약:** scalarization의 정확해를 구하는 것과 coverage 제약의 모든 최적점을 구하는 것은 다르다. 이산 unsupported point와 dual certificate를 구분한다.
4. **포화 길이:** 실제 plateau와 `R`의 나눗셈 조건 아래 latency 하한을 달성한다. arbitrary importance에 대해 품질까지 거의 최적이라는 보장은 없으며 명시적 반례가 있다.

위 정리는 가정과 증명을 문서화한 본 프로젝트의 유도다. SSD·LLM 실험의 경험적 관찰이나 논문 저자의 정리로 대체하지 않는다.

## 실행

저장소 루트에서:

```bash
python -m pip install -r conclusion/python/requirements.txt
OPENBLAS_NUM_THREADS=1 python conclusion/python/run_all.py
```

일부만 새로 실행하려면:

```bash
OPENBLAS_NUM_THREADS=1 python conclusion/python/run_all.py 02 07
OPENBLAS_NUM_THREADS=1 python conclusion/python/check_algorithms.py
```

각 노트북을 Jupyter 호환 편집기에서 열어 Run All 해도 된다. 실행 디렉토리는 저장소 루트 또는 `conclusion/python/`로 둔다. `.ipynb`에는 실제 실행 결과와 그림이 포함되어 있고, `run_all.py`는 기존 출력과 execution count를 지운 뒤 **새 커널**에서 모든 셀을 실행한다. 마지막 셀이 표시하는 `run_id`로 해당 원시 파일을 찾을 수 있다.

필요 환경:

- Python 및 직접 의존성의 검증 버전은 [requirements.txt](requirements.txt)에 고정했다. CUDA는 필요하지 않다.
- 02·05·07·09는 Linux `O_DIRECT`, `preadv`, `findmnt`와 쓰기 가능한 디스크가 필요하다. 프로젝트 디렉토리의 filesystem을 측정한다. tmpfs와 direct I/O 실패는 중단하며 cached I/O로 바꾸지 않는다.
- 02·05·07·09의 calibration은 최대 8 MiB chunk까지 측정하므로 임시 512 MiB 파일을 생성한다. 07의 별도 mask 검증 파일은 128 MiB다. 생성한 파일은 측정 후 삭제한다. 큰 모델 메모리와 측정 파일이 공존할 수 있으므로 수 GiB의 여유 메모리와 최소 1 GiB의 디스크 여유를 둔다.
- 08·09는 로컬 원본 `SmolLM2-360M-Instruct` snapshot을 사용한다. 기본 revision은 `a10cc1512eabd3dde888204e902eca88bddb4951`이다. 기본 cache에 없으면 `VLM_CHECKPOINT=/path/to/local/snapshot`으로 같은 구조의 checkpoint를 지정한다. 다운로드를 자동으로 수행하지 않으며, checkpoint가 없을 때 실험 결과를 만들어 대신하지 않는다.
- 다른 I/O 작업과 동시에 실행하면 timing이 달라진다. 수치의 동일성이 아니라 고정 seed·동일 원본·동일 코드로 **같은 절차**를 반복할 수 있다는 의미의 재현성이다. kernel 실행에는 로컬 소켓이 필요하다.

## 새 결과와 provenance

각 실행은 `runs/<실험>-<UTC>-<UUID>/`를 새로 만든다.

- `metadata.json`: seed, Python·라이브러리 버전, 소스·수학 문서·노트북 source의 SHA256, git revision.
- `io_raw.csv`, `mask_io_raw.csv`, `selector_raw.csv`: 해당 실험의 새 원시 측정. 장치와 I/O 설정은 `hardware.json`에 기록한다.
- `model.json`: 원본 checkpoint, weights SHA256, prompt, layer, dtype. activation은 항상 다시 계산한다.
- 실험별 CSV/NPZ, 그림의 PNG/SVG, `findings.json`: 해당 실행의 출력. 어느 것도 다음 실행의 입력으로 읽지 않는다.

새 run 산출물은 로컬에 유지하되 `.gitignore`로 제외했다. 공유할 때는 해당 run 디렉토리를 함께 전달하면 원시 반복값까지 검토할 수 있다. 노트북에는 결과와 그림이 이미 포함된다. 원래 `experiments/`, 기존 결과 파일, prompt 파일은 수정하지 않았다.

이전 디렉토리에서 복사된 `runs/`도 남아 있다. 이것은 과거 실행 기록이며 노트북의 입력으로 사용하지 않는다. 이 위치에서 `run_all.py`를 실행하면 새로운 `run_id`와 `metadata.json`이 생성된다. 결과를 인용할 때에는 그 실행의 `run_id`와 노트북 출력을 함께 확인한다.

## 코드 규모와 공통부의 역할

노트북의 Python 셀은 각각 약 60–120줄이다. 공통 모듈도 개별 500줄 미만이다. 한 줄에 여러 동작을 억지로 붙이거나 예전의 큰 experiment runner를 감추어 import하지 않았다.

| 파일 | 역할 | Python 줄 수 |
|---|---|---:|
| [core.py](core.py) | oracle, paper 의사코드, 새 saturation 변형, 두 DP | 195 |
| [measure.py](measure.py) | 새 난수 파일과 direct I/O, 원시 table | 124 |
| [capture.py](capture.py) | 로컬 원본 모델의 새 activation 수집 | 56 |
| [reporting.py](reporting.py) | provenance와 그림 저장 | 52 |
| [run_all.py](run_all.py) | 새 커널 실행 | 40 |
| [check_algorithms.py](check_algorithms.py) | 독립 oracle 및 reference 대조 | 53 |

공통 코드는 수식의 반복 구현과 측정·저장 boilerplate를 나눈 것이다. 각 노트북의 실험 조건·비교·plot은 해당 파일에서 읽을 수 있다. 코드 규모를 볼 때 이 공통 520줄도 함께 계산해야 한다.

## 기존 실험 주제의 압축 관계

기존 파일의 결과를 요약하거나 가져오는 관계가 아니라, 어떤 질문을 어디에서 새로 시험하는지의 지도다. 모든 기존 heuristic을 재구현하는 대신 소수의 비교 방법과 정확 oracle에 집중했다.

| 기존 주제/실험 번호 | 새 실험 |
|---|---|
| 비율·고정 R·목적함수 정리: 04, 23, 24, 40 | 01, 04, 09 |
| 작은 exact oracle·분포·quantization: 01–03, 06, 07, 10 | 03, 04 |
| affine/two-line/lookup 모델: 11, 13, 14, 42 | 02, 04, 07 |
| 현실적 규모·frontier·coverage: 05, 16, 19, 26, 28, 31, 34, 35 | 04, 05, 09 |
| refinement·tile·빠른 selector: 08, 09, 15, 18, 20, 22, 25, 27, 29, 36–38, 43 | 05–07 |
| selector/runtime·실제 weight/error: 17, 32, 39, 41, 45 | 07, 08 |
| global R: 44 | 09 |
| 수학/설계 메모: 12, 21, 30, 33 | 01, 04, 06, 09의 정의와 증명 |

## 주장 범위

수학 모델의 정확해, synthetic 입력의 비교, 이 PC의 새 SSD 측정, 텍스트 LLM의 새 projection 오차를 각 그림과 문서에 구분했다. Jetson의 논문 수치, VLM task accuracy, 전체 inference speedup을 재현했다고 표시하지 않는다. 특히 측정 상한에서 `T/length`가 최소면 포화를 확정하지 않고, actual wall time에 가산 모델이 안 맞으면 그 오차를 그대로 보고한다.
