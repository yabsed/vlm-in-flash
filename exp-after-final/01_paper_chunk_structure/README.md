# 01. Paper가 선택한 청크의 구조

기존 activation trace에서 Paper의 다중 크기 greedy 선택을 다시 실행하고,
채택한 후보 구간과 인접 구간을 합친 실제 읽기 청크를 각각 기록한다.
실험은 선택 구조를 진단한다. 새로운 SSD I/O, 모델 추론, 정확도 측정은 수행하지 않는다.

```bash
python run_experiment.py --source-final ../../final
```

전체 504개 trace와 선택 비율 0.1–0.9를 사용한다. `--limit-per-shape 2`는
구현 확인용 작은 실행이다. 결과 디렉토리는 기본 `results/`이며,
재실행은 같은 디렉토리의 이 실험 출력 파일을 갱신한다.

세 설정을 구분한다.

| 설정 | 후보 비용 | 후보 생성 |
| --- | --- | --- |
| `paper_lookup_s99` | 기존 profile의 앞 4회 반복 중앙값 | 실험 02의 shape별 start/step/stride, 최대 길이 s99 |
| `notebook02_twoline` | 현재 Two-line (free) 모델 | 실험 02와 동일 |
| `notebook03_full_lookup` | 기존 profile 전체 반복 중앙값 | 실험 03 저장 환경의 튜닝값, profile 전체 범위 |

첫 번째 설정은 모델을 직접 점수에 넣는 데서 생길 수 있는 선택 편향을 분리하기 위한
실측 lookup 대조군이다. 세 번째는 후보 상한이 포화점보다 큰 현재 실험 03 설정이며,
논문의 포화점 상한 설정과 구분해야 한다. 모두 현재 archive의 원래 행 순서와
원래 중요도를 사용하며 hot–cold reordering은 적용하지 않는다.

주 진단 경계는 현재 Two-line (free)의 `boundary_kib`다. `s99_kib`와
실험 03의 peak-efficiency 경계에 대한 짧은 청크 통계도 별도로 저장한다.
길이는 KiB로 비교하므로 행 크기가 다른 projection을 같은 행 수로 비교하지 않는다.

`R_actual`을 기준으로 다음을 계산한다.

\[
D_s=\sum_C(s-|C|_{\rm KiB})_+,\quad
L_0=(b+\delta)\,\mathrm{selected\_KiB},\quad
p=\delta D_s/L_0.
\]

`p`와 `modeled_read_ms`는 모델상 진단값이며 실측 latency가 아니다.
모든 경우에 요청 행 수, 실제 행 수, 예산 미충족량을 기록한다.

출력:

- `results/REPORT.md`: 계산 결과와 해석, 그림 링크.
- `results/cases.csv`: trace × 예산 × 설정별 중요도·청크·페널티 통계.
- `results/accepted_candidates.csv.gz`: 채택 순서, 점수 순위, 후보 길이·중요도·비용,
  선택 직전 남은 예산 및 합쳐진 run ID.
- `results/merged_runs.csv.gz`: 최종 실제 읽기 구간, 구성 후보 개수, 중요도와 비용.
- `results/candidate_decisions_by_length.csv.gz`: 생성/방문/채택/예산 탈락/겹침 탈락/
  예산 충족 후 미방문 수. 미방문 후보를 탈락 후보로 취급하지 않는다.
- `results/length_distribution.csv`: 설정·shape·구간 종류·길이별 개수/행 수/중요도 합.
- `results/by_shape_budget.csv`: shape 및 예산별 평균 진단값.
- `results/geometry.csv`: row 양자화 후 실제 후보 길이와 stride.
- `results/validation.csv`, `manifest.json`: 참조 구현 비교, 모든 입력/코드 SHA256,
  실행 환경과 설정. 생성 코드와 저장된 profile 요약도 기록한다.
- `results/figures/`: PNG와 PDF 그림.

후보 순서는 upstream의 float32 prefix sum 저장·점수 계산과 stable sort를 따른다.
Numba는 같은 greedy 루프에 로깅을 추가하는 데 사용한다. 각 shape에서 두 trace,
각 9개 예산, 각 설정을 upstream `select_chunks(..., impl="torch")`와 비교한다.
모든 사례에서 후보 합집합=최종 run 합집합, 예산, 중요도 및 모델 비용 분해도 검증한다.
실행 시간은 진단 스크립트 소요 시간이며 selector 성능 측정으로 해석하지 않는다.

관련 논문: arXiv:2511.18692v1, §3.2, Algorithm 1, Figure 10/15, Appendix H/J.
짧은 청크가 적다는 결과만으로 최소 청크 길이 제약이나 tile의 최적성을 입증할 수 없다.
현재 자료는 Qwen2.5-0.5B의 세 text prompt에서 나온 trace이며 VLM 영상 정확도 실험이 아니다.
