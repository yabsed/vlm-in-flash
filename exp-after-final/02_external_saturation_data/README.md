# 02 — 외부 장치의 saturation 이후 읽기 데이터

사용자의 노트북에서 새 벤치마크를 돌리지 않고, 외부 장치에서 측정된 데이터를 확보했다.
원시 fio JSON, 논문에 인쇄된 수치, PDF 그래프에서 복원한 값을 구분한다.

## 확보한 데이터

| 자료 | 장치 | 확보 범위 | 데이터 성격 |
|---|---|---|---|
| VLMFlash v1, Figure 4(a) | Jetson AGX + Samsung 990 Pro / Jetson Nano + SK Hynix Gold P31 | AGX 2–510 KB, Nano 1–511 KB | PDF 벡터 마커에서 복원한 평균 곡선 766점; 원시 반복 측정은 아님 |
| NVMeBenchmarks / IEEE Cluster 2023 연구 저장소 | WD Ultrastar DC ZN540, Xeon 서버 | 4, 8, 16, 32, 64, 128 KiB × QD 1, 32, 64 | 원본 fio JSON 18개 |
| ADMS 2010, Tables 8·10 | Intel X25-E 64 GB, Core 2 Duo PC | 8–1024 KB | 논문 표의 단일 SSD 순차 읽기 값 8행; 원시 반복 측정은 아님 |

주요 파일:

- `results/external_saturation_evidence.png` / `.pdf`: 네 패널 비교 그림.
- `results/vlmflash_fig4a_digitized.csv`: VLMFlash 그림의 평균 마커 값.
- `results/zns_raw_read_summary.csv`: 원시 fio에서 추출한 처리량과 파생 비용.
- `results/intel_x25e_published_tables.csv`: 공개 표의 throughput 및 latency.
- `results/metrics.json`: 아래 수치의 재계산 결과.
- `raw/NVMeBenchmarks/`: 원본 JSON, 고정 Git commit과 파일 SHA manifest, MIT license.
- `raw/adms10-petrov.pdf`: 숫자 대조에 사용한 원문.

## 직접 확인되는 사실

1. VLMFlash의 배포 lookup JSON이 포화점 근처에서 끝나는 것과, 논문에 포화점 이후 측정이 없는 것은 다르다. Figure 4(a)는 약 500 KB까지 곡선을 제시한다. 350–500 KB의 평균 처리량은 AGX 약 5.354 GB/s, Nano 약 3.068 GB/s이다. 해당 구간에서 `1 / throughput`의 `(max−min)/mean`은 각각 약 4.61%, 8.43%다. 이는 크기별 평균 곡선의 변동이며, 반복 실험의 표준편차나 신뢰구간이 아니다.

2. ZN540의 QD 32에서는 64→128 KiB의 처리량이 3106.47→3104.39 MiB/s, QD 64에서는 3106.55→3103.14 MiB/s다. 변화는 각각 −0.067%, −0.110%이다. 반면 QD 1에서는 1845.21→1989.55 MiB/s로 7.82% 증가한다. 따라서 포화점은 장치뿐 아니라 병렬성에도 의존한다. 이 subset의 포화 구간에는 두 크기만 있으며, 128 KiB보다 큰 읽기를 검증하지 않는다.

3. Intel X25-E의 QD 1 순차 읽기 latency는 128, 256, 512, 1024 KB에서 각각 0.529, 1.062, 2.115, 4.249 ms다. `latency / size`는 약 4.131–4.149 µs/KB로, `(max−min)/mean`이 약 0.45%다. QD 32의 별도 throughput 실험에서는 이 네 크기 모두 252 MB/s로 인쇄되어 있다. 오래된 SATA/SLC 장치의 보조 근거이며 현대 NVMe와 성능 수치를 합쳐 fitting하지 않는다.

## 모델과의 연결 및 한계

처리량을 B(l)이라 하면, 고정된 병렬 읽기 조건의 청크당 상각 비용은
`T_amortized(l) = l / B(l)`이다. 따라서 `T_amortized(l)/l = 1/B(l)`을 보고
포화 이후의 비례 모델을 평가할 수 있다. QD>1에서 이것을 개별 요청의 응답 시간과
동일시하지 않는다. Intel Table 10의 값은 QD 1에서 실제 측정한 요청 latency다.

이 자료는 관측한 구간에서 `T(l) ≈ α l`이라는 근사를 뒷받침한다. 모든 l≥s의
정확한 등식, 이질적인 크기의 청크를 섞었을 때의 비용 가산성, tile grid에서의
importance 손실까지 증명하지는 않는다. QD·thread 수·읽기 패턴을 고정한 장치별
분석이 필요하다. s는 허용 오차를 지정해 정해야 하며, 이 보고서는 새 s를 fitting하지 않았다.

## 출처와 측정 조건

- [VLMFlash v1 Figure 4](https://arxiv.org/pdf/2511.18692v1#page=4): 128 MB 읽기, Linux direct I/O, C++ 6-thread pool. Figure 4 캡션의 장치별 threshold와 Appendix D/H의 숫자가 서로 일치하지 않는다. 이 분석은 특정 threshold 표기를 선택하지 않고 두 장치 모두 350–500 KB를 같은 분석 구간으로 사용했다. KB·GB/s는 원문 축 단위를 보존했다. `vlm_digitization_metadata.json`에 PDF SHA256과 축 좌표를 기록했다. 오차막대의 반복 측정 원본은 복원하지 않았다.
- [ZNS 논문](https://arxiv.org/html/2310.19094v1), [데이터 저장소](https://github.com/stonet-research/NVMeBenchmarks): commit `01f40f1c9228c6df7116d050e0132743dc4fb5a1`의 `data/io_uring/zns-a/lbaf0/read/*bs/1zone/*.json` 중 명시한 18개를 보존했다. 원본에는 fio 3.32, direct=1, read, io_uring, numjobs=1, scheduler=none, fixedbufs/registerfiles/sqthread_poll, 10초 warmup, 3분 runtime이 기록되어 있다. subset별 원시 설정을 우선하며 논문 전체의 반복 실험 정책을 이 파일들에 임의 적용하지 않는다. JSON 파일 자체는 장치 별명을 사용하고, ZN540 식별은 논문의 장치 설명에 따른다.
- [ADMS 2010 원문](https://www.vldb.org/archives/workshop/2010/proceedings/files/vldb_2010_workshop/ADMS_2010/adms10-petrov.pdf): Tables 8·10의 `1 SSD` / `Read` 열만 전사하고 PDF 페이지 5·6을 시각적으로 대조했다. Windows Server 2008 R2, IOMeter 2008.06.18, raw device. Section 3에서 throughput용 light workload는 1 worker/QD 32, latency용 workload는 1 worker/QD 1이다. 표에 표시된 KB·MB/s 단위와 소수점 반올림을 유지했다.

주의: 확보한 ZNS JSON의 `clat_ns.mean`에는 약 100초 수준으로 `lat_ns.mean` 및 IOPS와
일관되지 않은 값이 있다. 원본과 CSV에는 보존하지만 모델 검증이나 그림에는 사용하지 않았다.
처리량은 `bw_bytes`로 계산했다. 원본 `bw_bytes/(IOPS×bs)`의 상대 차이는 최대 약
0.0015%이며 CSV에 기록했다. 데이터 SHA 검증과 단위 일관성 확인을 분석 스크립트에 포함했다.

추가 조사 후보: [NVMe DMA-optimal limit 측정 보고서](https://github.com/davidlohr/dma-opt-clamp-report)는
Samsung PM9A3/Micron 7450의 큰 요청과 압축 원시 자료를 공개한다. 커널 요청 분할 한계,
버퍼, 장치 상태를 바꾸는 실험이 포함되어 있어 이번 고정 조건 분석에는 넣지 않았다.
보고서의 큰 요청 성능은 장치별로 다르므로 무한히 평탄한 tail을 일반 법칙으로 가정하면 안 된다.

## 재현

Python의 numpy, matplotlib, PyMuPDF가 필요하다. 이 명령은 기존 파일만 분석한다.

```bash
python analyze.py --vlm-pdf ../../preliminary_research/2511.18692v1.pdf
```

PDF를 이동했다면 `--vlm-pdf`에 실제 경로를 지정한다. 원본 데이터 다운로드는 실행에 필요 없다.
