"""Create figures and a computed report from saved structural diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

MODES = ("paper_lookup_s99", "notebook02_twoline", "notebook03_full_lookup")
LABELS = {"paper_lookup_s99": "Measured lookup / s99 cap",
          "notebook02_twoline": "Notebook 02 / two-line",
          "notebook03_full_lookup": "Notebook 03 / full profile"}
COLORS = dict(zip(MODES, ["#0072B2", "#D55E00", "#009E73"]))


def markdown_table(frame):
    # Avoid an optional tabulate dependency.
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(row) + " |" for row in rows])


def analyze(output):
    output = Path(output)
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    manifest = json.loads((output / "manifest.json").read_text())
    cases = pd.read_csv(output / "cases.csv")
    dist = pd.read_csv(output / "length_distribution.csv")
    geometry = pd.read_csv(output / "geometry.csv")
    summary = pd.read_csv(output / "by_shape_budget.csv")
    decisions = pd.read_csv(output / "candidate_decisions_by_length.csv.gz")
    s = manifest["s_model_kib"]
    shapes = sorted(set(zip(cases.n, cases.d)))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170})

    def save(fig, name):
        fig.savefig(figdir / f"{name}.png", bbox_inches="tight")
        fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    profile = pd.read_csv(output / "profile_used.csv")
    x = profile.size_kib
    for ax in axes:
        ax.plot(x, profile.latency_ms_train_median / x * 1000, label="Measured, first 4 repeats", lw=1.4)
        ax.plot(x, profile.latency_ms_all_median / x * 1000, label="Measured, all repeats", alpha=.65)
        ax.plot(x, profile.two_line_ms / x * 1000, label="Two-line (free)", lw=1.4)
        for bound, color, label in [(s, "black", "model s"),
                                  (manifest["s99_kib"], "#CC79A7", "s99"),
                                  (manifest["peak_efficiency_kib"], "#009E73", "peak efficiency")]:
            ax.axvline(bound, color=color, ls="--", lw=.8, label=f"{label}: {bound:g} KiB")
        ax.set(xlabel="Chunk length (KiB)", ylabel="Amortized cost per KiB (us/KiB)")
    axes[0].set(xscale="log", yscale="log", title="Saved device profile (no new I/O)")
    axes[1].set(xlim=(128, float(x.max())), title="Saturation region")
    tail = profile[x >= 128]
    tail_y = np.concatenate([tail[col] / tail.size_kib * 1000 for col in
                            ["latency_ms_train_median", "latency_ms_all_median", "two_line_ms"]])
    axes[1].set_ylim(float(tail_y.min()) * .97, float(tail_y.max()) * 1.03)
    axes[0].legend(fontsize=7)
    save(fig, "01_profile")

    # Pooled selected-row weighted CDF; separate shapes instead of pooling different row sizes.
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained", sharex=True, sharey=True)
    for i, mode in enumerate(MODES):
        for j, kind in enumerate(["candidate", "run"]):
            ax = axes[i, j]
            for n, d in shapes:
                g = dist[(dist.config == mode) & (dist.n == n) & (dist.d == d) & (dist.kind == kind)].sort_values("length_over_s_model")
                ax.step(g.length_over_s_model, g.selected_rows.cumsum() / g.selected_rows.sum(),
                        where="post", label=f"{n} x {d}")
            ax.axvline(1, color="black", ls="--", lw=1)
            ax.set(xscale="log", xlim=(1/256, 64), ylim=(0, 1.02),
                   title=f"{LABELS[mode]}: {'accepted candidates' if kind == 'candidate' else 'merged read runs'}",
                   xlabel="Length / model s", ylabel="CDF weighted by selected rows")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    save(fig, "02_length_cdf")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True)
    for ax, (n, d) in zip(axes.flat, shapes):
        for mode in MODES:
            g = summary[(summary.config == mode) & (summary.n == n) & (summary.d == d)]
            ax.plot(g.fraction, g.penalty_over_baseline * 100, "o-", color=COLORS[mode], label=LABELS[mode])
        row_kib = cases[(cases.n == n) & (cases.d == d)].row_kib.iloc[0]
        boundary_fraction = s / (n * row_kib)
        if boundary_fraction > .1:
            ax.axvspan(.1, min(.9, boundary_fraction), color="gray", alpha=.12)
        ax.set(title=f"{n} x {d}", xlabel="Requested row fraction",
               ylabel="Mean model penalty / alpha R (%)", xlim=(.09, .91))
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Short-run penalty; gray = total requested bytes below s (saturated selection impossible)")
    save(fig, "03_short_run_penalty")

    fig, axes = plt.subplots(len(shapes), 2, figsize=(11, 11), layout="constrained", sharex=True, sharey=True)
    for i, (n, d) in enumerate(shapes):
        for j, field in enumerate(["run_short_model_row_fraction", "run_short_model_importance_fraction"]):
            for mode in MODES:
                g = summary[(summary.config == mode) & (summary.n == n) & (summary.d == d)]
                axes[i,j].plot(g.fraction, 100 * g[field], "o-", color=COLORS[mode], label=LABELS[mode])
            axes[i,j].set(title=f"{n} x {d}: {'rows' if j == 0 else 'importance'} in runs < s",
                          xlabel="Requested row fraction", ylabel="Mean share (%)", ylim=(-2, 102))
            axes[i,j].grid(alpha=.15)
    axes[0, 0].legend(fontsize=7)
    save(fig, "04_short_run_shares")

    fig, axes = plt.subplots(len(shapes), 3, figsize=(13, 11), layout="constrained", sharey=True)
    decision_summary = decisions.groupby(["config", "n", "d", "length_kib"], as_index=False)[
        ["generated", "visited", "selected", "rejected_budget", "rejected_overlap", "not_visited_budget_filled"]].sum()
    decision_summary["selected_per_generated"] = decision_summary.selected / decision_summary.generated
    decision_summary["selected_per_visited"] = decision_summary.selected / decision_summary.visited.replace(0, np.nan)
    decision_summary.to_csv(output / "selection_rates_by_length.csv", index=False)
    for i, (n, d) in enumerate(shapes):
        for j, mode in enumerate(MODES):
            g = decision_summary[(decision_summary.config == mode) & (decision_summary.n == n) & (decision_summary.d == d)].sort_values("length_kib")
            ax = axes[i,j]
            ax.stackplot(g.length_kib/s,
                         *[100*g[col]/g.generated for col in ["selected", "rejected_overlap", "rejected_budget", "not_visited_budget_filled"]],
                         labels=["Selected", "Overlap", "Budget", "Not visited"],
                         colors=["#0072B2", "#E69F00", "#D55E00", "#CCCCCC"], alpha=.85)
            ax.axvline(1, color="black", ls="--", lw=.8)
            ax.set(xscale="log", ylim=(0, 100), title=f"{n} x {d}\n{LABELS[mode]}",
                   xlabel="Candidate length / model s", ylabel="Share of generated candidates (%)")
    axes[0,0].legend(fontsize=7, loc="lower left")
    save(fig, "05_candidate_decisions")

    examples = json.loads((output / "mask_examples.json").read_text())
    fig, axes = plt.subplots(len(shapes), 1, figsize=(12, 8), layout="constrained")
    example_ids = []
    for ax, (n,d) in zip(axes, shapes):
        first_id = min(e["trace_id"] for e in examples if (e["n"],e["d"]) == (n,d))
        for i, mode in enumerate(MODES):
            e = next(e for e in examples if e["config"] == mode and e["trace_id"] == first_id)
            example_ids.append(e["case_id"])
            for offset, field in [(0, "accepted"), (.30, "runs")]:
                segments = [[(a, i+offset), (z, i+offset)] for a,z in e[field]]
                colors = ["#D55E00" if (z-a)*e["row_kib"] < s else "#0072B2" for a,z in e[field]]
                ax.add_collection(LineCollection(segments, colors=colors, linewidths=5))
                if field == "accepted":
                    for a,z in e[field]:
                        ax.vlines([a,z], i-.06, i+.06, color="white", lw=.5)
        ax.set(xlim=(0,n), ylim=(-.3,2.6), title=f"{n} x {d}; trace {first_id}; requested fraction 0.5",
               xlabel="Row index", yticks=[.15,1.15,2.15], yticklabels=[LABELS[m] for m in MODES])
    fig.suptitle("Each pair: accepted candidates below / merged runs above. Orange < s; blue >= s.")
    save(fig, "06_mask_examples")

    # Preserve both all-case and feasible-only summaries; R < s is not evidence against the restriction.
    rows = []
    for mode in MODES:
        for scope in ["all", "saturated_feasible", "saturated_infeasible"]:
            g = cases[cases.config == mode]
            if scope == "saturated_feasible": g = g[g.saturated_feasible]
            if scope == "saturated_infeasible": g = g[~g.saturated_feasible]
            rows.append(dict(config=mode, scope=scope, cases=len(g),
                underfilled_cases=int((g.unused_rows > 0).sum()),
                mean_fill=g.fill_fraction.mean(), mean_candidate_count=g.candidate_count.mean(),
                mean_run_count=g.run_count.mean(), mean_merge_fraction=g.merge_fraction.mean(),
                mean_short_count=g.run_short_model_count_fraction.mean(),
                mean_short_rows=g.run_short_model_row_fraction.mean(),
                mean_short_importance=g.run_short_model_importance_fraction.mean(),
                mean_penalty=g.penalty_over_baseline.mean(), median_penalty=g.penalty_over_baseline.median(),
                p95_penalty=g.penalty_over_baseline.quantile(.95),
                zero_penalty_cases=int((g.short_penalty_ms < 1e-12).sum()),
                mean_fragmentation_excess=g.fragmentation_over_minimum.mean()))
    overview = pd.DataFrame(rows)
    overview.to_csv(output / "overview.csv", index=False)
    def display_table(frame):
        return pd.DataFrame({
            "설정":frame.config, "사례":frame.cases,
            "짧은 run의 행 비중":frame.mean_short_rows.map(lambda x:f"{100*x:.2f}%"),
            "짧은 run의 중요도 비중":frame.mean_short_importance.map(lambda x:f"{100*x:.2f}%"),
            "평균 p":frame.mean_penalty.map(lambda x:f"{100*x:.2f}%"),
            "p 중앙값":frame.median_penalty.map(lambda x:f"{100*x:.2f}%"),
            "p 95백분위":frame.p95_penalty.map(lambda x:f"{100*x:.2f}%"),
            "p=0 사례":frame.zero_penalty_cases})
    all_table = markdown_table(display_table(overview[overview.scope == "all"]))
    feasible_table = markdown_table(display_table(overview[overview.scope == "saturated_feasible"]))
    budget_table = markdown_table(pd.DataFrame({
        "설정":overview[overview.scope == "all"].config,
        "미충족 사례":overview[overview.scope == "all"].underfilled_cases,
        "평균 충족률":overview[overview.scope == "all"].mean_fill.map(lambda x:f"{100*x:.3f}%"),
        "평균 채택 후보 수":overview[overview.scope == "all"].mean_candidate_count.map(lambda x:f"{x:.2f}"),
        "평균 최종 run 수":overview[overview.scope == "all"].mean_run_count.map(lambda x:f"{x:.2f}"),
        "평균 병합 감소율":overview[overview.scope == "all"].mean_merge_fraction.map(lambda x:f"{100*x:.2f}%")
    }))
    shape_rows = []
    for mode in MODES:
        for n,d in shapes:
            g = cases[(cases.config == mode) & (cases.n == n) & (cases.d == d) & cases.saturated_feasible]
            shape_rows.append({"설정":mode,"shape":f"{n} × {d}","가능 사례":len(g),
                "평균 p":f"{100*g.penalty_over_baseline.mean():.2f}%" if len(g) else "해당 없음",
                "짧은 run 행 비중":f"{100*g.run_short_model_row_fraction.mean():.2f}%" if len(g) else "해당 없음"})
    short_order_rows = []
    for mode in MODES:
        g = cases[(cases.config == mode) & cases.saturated_feasible]
        short_count = int(g.short_candidate_count.sum())
        low_remaining = int(g.short_candidate_low_remaining_count.sum())
        short_order_rows.append({"설정":mode, "채택한 짧은 후보 수":short_count,
            "선택 직전 남은 예산 < s":low_remaining,
            "비율":f"{100*low_remaining/short_count:.2f}%" if short_count else "해당 없음"})
    main = overview[(overview.config == "paper_lookup_s99") & (overview.scope == "saturated_feasible")].iloc[0]
    equivalent = manifest["reference_all_equal"]
    report = f"""# Paper가 선택한 청크의 구조 진단

실행: {manifest['completed_at']} (Asia/Seoul). 기존 {manifest['trace_count']}개 trace × 9개 선택 비율 × 3개 설정 = **{len(cases):,}사례**.

실측 lookup에 포화점 후보 상한을 적용한 주 대조군에서, 모든 청크를 포화 길이 이상으로 구성할 수 있는
{int(main.cases):,}사례의 평균 짧은-run 페널티 p는 **{100*main.mean_penalty:.2f}%**,
중앙값은 **{100*main.median_penalty:.2f}%**, 95백분위는 **{100*main.p95_penalty:.2f}%**다.
짧은 run이 차지하는 선택 행 비중은 평균 **{100*main.mean_short_rows:.2f}%**다.
이는 모델로 환산한 구조 진단이며 새로운 읽기 지연시간 측정값이 아니다.

## 설정과 경계

- `paper_lookup_s99`: profile 앞 4회 반복 중앙값으로 후보 평가. 실험 02와 같은 후보 배치 및 s99 상한.
- `notebook02_twoline`: 현재 실험 02와 동일한 Two-line 비용과 후보 배치.
- `notebook03_full_lookup`: profile 전체 반복 중앙값, 실험 03에 저장된 튜닝값과 768 KiB 범위.
  이 설정은 논문의 포화점 후보 상한과 다르다. 이 실행은 {manifest['trace_count']}개 trace 및 비율 0.1–0.9를 사용했다.

모델 경계 s = **{s:g} KiB**, s99 = **{manifest['s99_kib']:g} KiB**,
실측 peak-efficiency 경계 = **{manifest['peak_efficiency_kib']:g} KiB**.
아래 주 통계는 s={s:g} KiB를 사용하며, `cases.csv`에 나머지 두 경계의 비중도 보존했다.
후보 최대 길이는 row 양자화 때문에 설정값과 다를 수 있다. 정확한 후보 길이는 `geometry.csv`에 있다.

![저장된 profile](figures/01_profile.png)

## 최종 실제 읽기 청크

아래는 사례별 비중/페널티를 먼저 계산한 뒤 사례 사이에서 평균한 값이다. p의 정의는
`delta * sum(max(s - run_KiB, 0)) / (alpha * actual_selected_KiB)`이다.
청크 크기와 개수의 raw 합을 전체 trace에 바로 pooling한 통계와 구분한다.

모든 사례:

{all_table}

**총 선택량이 s 이상인 사례만**:

{feasible_table}

총 선택량 자체가 s보다 작으면 모든 청크를 s 이상으로 만드는 것은 불가능하다.
따라서 이 경우의 양의 p는 최소 청크 길이 제약의 실효성을 반박하는 근거로 쓸 수 없다.
특히 896 × 128 shape은 전체 가중치가 224 KiB라 0.1–0.9 예산에서 선택량이 모두 208 KiB보다 작다.
`cases.csv`에는 이를 구분하는 `saturated_feasible`과 `requested_saturated_feasible`을 기록했다.
또한 `fragmentation_over_minimum = L_model / T(total_selected_KiB) - 1`을 별도로 기록하여
낮은 예산의 불가피한 포화 미달 비용과 실제 분할 비용을 구분했다.

shape별, 포화 길이 구성이 가능한 사례:

{markdown_table(pd.DataFrame(shape_rows))}

![페널티와 예산](figures/03_short_run_penalty.png)

![짧은 run의 행 및 중요도 비중](figures/04_short_run_shares.png)

## 채택 후보와 최종 run의 차이

{budget_table}

채택한 후보가 인접하면 하나의 실제 읽기 run으로 합쳐진다. 후보 상한보다 긴 최종 run이 생길 수 있다.
`accepted_candidates.csv.gz`의 `merged_run_id`로 각 후보를 최종 run에 연결할 수 있다.
이 때문에 후보 길이 분포만으로 실제 읽기 페널티를 평가하면 잘못된 결론을 낼 수 있다.

![길이 누적 분포](figures/02_length_cdf.png)

위 CDF는 각 설정·shape 안에서 예산들을 합친 **선택 행 수 가중 분포**다.
세로선은 모델 s다. 큰 청크 하나가 담는 행 수를 반영하므로 청크 개수 가중 분포와 다르다.
`length_distribution.csv`에서 개수 및 중요도 가중 분포도 다시 만들 수 있다.

![후보 선택과 탈락](figures/05_candidate_decisions.png)

길이별 생성 후보를 분모로 선택·겹침 탈락·예산 탈락·미방문 비중을 표시한다.
원래 알고리즘 순서대로 예산 초과 여부를 먼저 검사하므로 예산 탈락 후보는 겹침 여부를 추가 분류하지 않는다.
예산을 채운 뒤 미방문한 후보는 평가 후 탈락한 후보와 구분한다.
`selection_rates_by_length.csv`에 생성 후보 기준 및 방문 후보 기준 선택률을 모두 저장했다.

총 선택량이 s 이상인 사례에서, 짧은 후보를 채택한 시점의 남은 예산도 집계했다.
남은 예산이 이미 s 미만이면 그 시점에는 포화 길이 후보를 추가로 선택할 수 없다.
단, 그 후보가 인접 구간에 합쳐지는지는 별도이며 이 통계만으로 짧은 최종 run의 원인을 단정하지 않는다.

{markdown_table(pd.DataFrame(short_order_rows))}

![실제 마스크 예시](figures/06_mask_examples.png)

예시는 shape별 첫 검증 trace의 예산 0.5를 사전에 고정해 선택했다. 성능이 좋은 사례를 골라내지 않았다.
각 쌍의 아래는 채택 후보, 위는 합쳐진 run이다. 주황은 s 미만, 파랑은 s 이상이다.

## 구현 검증

- 원래 upstream PyTorch 구현과 **{manifest['reference_comparisons']:,}사례**에서 마스크 및 채택 후보 비용 합을 비교: **{'모두 일치' if equivalent else '불일치 존재'}**.
- **{manifest['checked_invariant_cases']:,}사례 전체**에서 예산, 후보 합집합과 run 합집합,
  중요도 보존, 후보→run 연결, `L_model = alpha * actual_selected_KiB + delta * D_s`를 검증했다.
- stable sort 및 float32 점수 규칙을 유지했다. CPU 실행이며 선택 성능 벤치마크가 아니다.
- 입력/실험 코드/upstream 파일 SHA256와 commit은 `manifest.json`에 있다.

## 해석의 범위

이 실험이 직접 답하는 것은 Paper가 작은 구간을 얼마나 채택하고, 그 구간이 병합된 뒤 얼마나 남으며,
그 구조가 현재 비용 모델에서 얼마의 페널티를 발생시키는가이다.
짧은 청크를 금지했을 때 중요도가 얼마나 줄어드는지, tile의 경계 제약이 얼마나 손해인지,
실제 SSD 지연시간이 모델과 일치하는지는 이 실행에서 검증하지 않았다.
lookup 비용과 실제 whole-mask latency도 같다고 가정해서는 안 된다.

동일 trace의 여러 예산 및 같은 prompt에서 나온 여러 projection은 독립 표본이 아니다.
집계는 기술통계이며 독립 표본 기반 신뢰구간이나 유의성 주장을 하지 않는다.
데이터는 Qwen2.5-0.5B의 세 text prompt, 원래 행 순서의 activation trace다.
논문의 VLM 영상 입력·hot–cold reordering·정확도 평가를 재현한 결과는 아니다.

논문과의 대응: arXiv:2511.18692v1 §3.2/Algorithm 1의 선택 기준을 기록하고,
Figure 10/15의 길이 분포 분석에 포화점 정규화, 행/중요도 비중, 짧은-run 페널티,
후보 병합과 예산 미충족을 추가했다.
"""
    (output / "REPORT.md").write_text(report)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path(__file__).resolve().parent / "results")
    analyze(parser.parse_args().output)
