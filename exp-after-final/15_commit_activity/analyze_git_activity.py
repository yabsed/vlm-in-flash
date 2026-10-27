#!/usr/bin/env python3
"""Export the September/October 2026 activity of this repo and final, offline."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Matplotlib's default cache is outside the writable workspace on some machines.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vlm-git-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KST = ZoneInfo("Asia/Seoul")
START = date(2026, 9, 1)
END = date(2026, 10, 31)
REPOS = {"vlm-in-flash": ROOT, "final": ROOT / "final"}
BLUE = "#3469C2"
GREEN = "#159782"
INK = "#243247"
MUTED = "#66758A"
BG = "#F7F9FC"


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, encoding="utf-8"
    ).strip()


def get_snapshot(refresh: bool, as_of: date | None) -> dict:
    path = HERE / "snapshot.json"
    if path.exists() and not refresh:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if as_of is not None and as_of.isoformat() != snapshot["as_of"]:
            raise ValueError("Changing --as-of requires --refresh.")
        return snapshot
    now = datetime.now(KST)
    cutoff = min(as_of or now.date(), END)
    if cutoff < START or cutoff > now.date():
        raise ValueError("--as-of must be between 2026-09-01 and today (KST).")
    heads = {}
    for name, repo in REPOS.items():
        if git(repo, "rev-parse", "--is-shallow-repository") == "true":
            raise ValueError(f"{name} has incomplete (shallow) history.")
        heads[name] = git(repo, "rev-parse", "HEAD")
    snapshot = {
        "captured_at": now.isoformat(timespec="seconds"),
        "timezone": "Asia/Seoul",
        "start": START.isoformat(),
        "as_of": cutoff.isoformat(),
        "date_field": "committer_date",
        "history_scope": "all ancestors of each recorded HEAD; no nested submodules",
        "heads": heads,
    }
    return snapshot


def collect_commits(snapshot: dict) -> list[dict]:
    commits = []
    cutoff = date.fromisoformat(snapshot["as_of"])
    for name, repo in REPOS.items():
        # Read all ancestors, then filter in KST: git --since can prune history
        # with non-monotonic dates, and the original offsets may differ.
        raw = git(repo, "log", "--no-show-signature", "--no-notes",
                  "--format=%H%x00%aI%x00%cI%x00%an%x00%ae%x00%cn%x00%ce%x00%P%x00%s",
                  snapshot["heads"][name], "--")
        for line in raw.splitlines():
            sha, author_time, commit_time, author, email, committer, committer_email, parents, subject = line.split("\0", 8)
            authored = datetime.fromisoformat(author_time).astimezone(KST)
            committed = datetime.fromisoformat(commit_time).astimezone(KST)
            if not START <= committed.date() <= cutoff:
                continue
            is_bot = any("[bot]" in value.lower() for value in
                         (author, email, committer, committer_email))
            is_merge = len(parents.split()) > 1
            commits.append({
                "repository": name, "sha": sha,
                "author": author, "committer": committer,
                "authored_at_kst": authored.isoformat(),
                "committed_at_kst": committed.isoformat(),
                "date_kst": committed.date().isoformat(),
                "hour_kst": committed.hour,
                "parent_count": len(parents.split()),
                "is_bot": int(is_bot), "is_merge": int(is_merge),
                "human_non_merge": int(not is_bot and not is_merge),
                "author_committer_day_differs": int(authored.date() != committed.date()),
                "subject": subject,
            })
    return sorted(commits, key=lambda row: (row["committed_at_kst"], row["repository"], row["sha"]))


def summarize(commits: list[dict], snapshot: dict) -> tuple[list[dict], list[dict], dict]:
    cutoff = date.fromisoformat(snapshot["as_of"])
    daily = []
    for offset in range((cutoff - START).days + 1):
        day = (START + timedelta(days=offset)).isoformat()
        rows = [c for c in commits if c["date_kst"] == day]
        record = {"date_kst": day}
        for name in REPOS:
            own = [c for c in rows if c["repository"] == name]
            record[name] = len(own)
            record[f"{name}_human_non_merge"] = sum(c["human_non_merge"] for c in own)
        record.update(total=len(rows), human_non_merge=sum(c["human_non_merge"] for c in rows),
                      bot=sum(c["is_bot"] for c in rows), merge=sum(c["is_merge"] for c in rows))
        daily.append(record)
    monthly = []
    for month in ("2026-09", "2026-10"):
        days = [d for d in daily if d["date_kst"].startswith(month)]
        if not days:
            continue
        for name in (*REPOS, "combined"):
            rows = [c for c in commits if c["date_kst"].startswith(month)
                    and (name == "combined" or c["repository"] == name)]
            counts = Counter(c["date_kst"] for c in rows)
            monthly.append({
                "month": month, "repository": name, "commits": len(rows),
                "human_non_merge": sum(c["human_non_merge"] for c in rows),
                "bot": sum(c["is_bot"] for c in rows),
                "merge": sum(c["is_merge"] for c in rows),
                "active_days": len(counts), "observed_calendar_days": len(days),
                "commits_per_calendar_day": round(len(rows) / len(days), 3),
                "commits_per_active_day": round(len(rows) / len(counts), 3) if counts else 0,
            })
    longest, streak = 0, 0
    for day in daily:
        streak = streak + 1 if day["total"] else 0
        longest = max(longest, streak)
    peak = max(daily, key=lambda d: d["total"])
    summary = {
        "snapshot": snapshot,
        "total_commits": len(commits),
        "unique_hashes_across_repositories": len({c["sha"] for c in commits}),
        "human_non_merge_commits": sum(c["human_non_merge"] for c in commits),
        "bot_commits": sum(c["is_bot"] for c in commits),
        "merge_commits": sum(c["is_merge"] for c in commits),
        "active_days": sum(d["total"] > 0 for d in daily),
        "longest_active_streak_days": longest,
        "peak_day": peak["date_kst"], "peak_day_commits": peak["total"],
        "author_committer_day_differences": [c["sha"] for c in commits if c["author_committer_day_differs"]],
        "monthly": monthly,
    }
    # Check conservation across raw rows, daily rows, and monthly rows.
    assert sum(d["total"] for d in daily) == len(commits)
    assert sum(m["commits"] for m in monthly if m["repository"] == "combined") == len(commits)
    assert all(d["total"] == sum(d[name] for name in REPOS) for d in daily)
    assert len({(c["repository"], c["sha"]) for c in commits}) == len(commits)
    return daily, monthly, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    regular = Path("/usr/share/fonts/naver-nanum-gothic-fonts/NanumGothic.ttf")
    bold = regular.with_name("NanumGothicBold.ttf")
    for path in (regular, bold):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("NanumGothic", "Noto Sans CJK JP", "Noto Sans CJK KR")
                   if name in available), None)
    if family is None:
        raise RuntimeError("Install NanumGothic or Noto Sans CJK to render Korean labels.")
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.unicode_minus": False, "figure.facecolor": BG,
        "axes.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.spines.bottom": False, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 14,
        "grid.color": "#E4EAF2", "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / "figures" / f"{stem}.{suffix}", dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def overview(daily: list[dict], monthly: list[dict], summary: dict) -> None:
    fig = plt.figure(figsize=(15.5, 10))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.96, top=0.77, bottom=0.12,
                          height_ratios=[1.2, 1], hspace=0.47, wspace=0.24)
    fig.text(0.06, 0.945, "2026년 9월과 10월의 커밋 활동", fontsize=25, weight="bold")
    fig.text(0.06, 0.904, "vlm-in-flash + final  |  한국 시간(KST) · committer date 기준", color=MUTED, fontsize=12)
    metrics = [(str(summary["total_commits"]), "전체 커밋"),
               (str(summary["human_non_merge_commits"]), "봇·병합 제외"),
               (str(summary["active_days"]), "커밋한 날짜"),
               (str(summary["peak_day_commits"]), f"하루 최대 · {summary['peak_day'][5:].replace('-', '/')}")]
    for x, (value, label) in zip((0.06, 0.29, 0.52, 0.75), metrics):
        fig.text(x, 0.845, value, fontsize=29, weight="bold", color=BLUE)
        fig.text(x + 0.072, 0.852, label, fontsize=11, color=MUTED)
    dates = [date.fromisoformat(d["date_kst"]) for d in daily]
    root = [d["vlm-in-flash"] for d in daily]
    final = [d["final"] for d in daily]
    day_ax = fig.add_subplot(gs[0, :])
    day_ax.set_title("일별 커밋 수", fontsize=14)
    day_ax.bar(dates, root, width=0.78, color=BLUE, label="vlm-in-flash", zorder=3)
    day_ax.bar(dates, final, bottom=root, width=0.78, color=GREEN, label="final", zorder=3)
    day_ax.set_ylabel("커밋 수")
    day_ax.set_xlim(mdates.date2num(START) - 0.5, mdates.date2num(END) + 0.5)
    day_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    day_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    day_ax.axvline(date(2026, 10, 1), color="#AAB8CA", linestyle="--", linewidth=1)
    cutoff = date.fromisoformat(summary["snapshot"]["as_of"])
    if cutoff < END:
        day_ax.axvspan(mdates.date2num(cutoff) + 0.5, mdates.date2num(END) + 1,
                       color="#E9EDF3", zorder=0)
        day_ax.text(0.992, 0.91, "집계 이후", transform=day_ax.transAxes,
                    ha="right", color=MUTED, fontsize=9)
    peak = date.fromisoformat(summary["peak_day"])
    peak_count = summary["peak_day_commits"]
    day_ax.annotate(f"{peak:%m/%d} · {peak_count}회", (mdates.date2num(peak), peak_count),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10, weight="bold")
    day_ax.set_ylim(0, peak_count * 1.25 + 1)
    day_ax.grid(axis="y", zorder=0)
    day_ax.legend(loc="upper left", frameon=False, ncol=2)

    cum_ax = fig.add_subplot(gs[1, 0])
    cum_ax.set_title("기간 내 누적 커밋 수", fontsize=14)
    for name, color in (("vlm-in-flash", BLUE), ("final", GREEN)):
        values, total = [], 0
        for day in daily:
            total += day[name]
            values.append(total)
        cum_ax.step(dates, values, where="post", color=color, linewidth=2.4, label=name)
        cum_ax.annotate(str(total), (mdates.date2num(dates[-1]), total),
                        xytext=(5, 0), textcoords="offset points", color=color, weight="bold")
    cum_ax.set_xlim(START, END + timedelta(days=1))
    cum_ax.set_ylim(bottom=0)
    cum_ax.set_ylabel("누적 커밋 수")
    cum_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    cum_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    cum_ax.grid(axis="y")
    cum_ax.legend(loc="upper left", frameon=False)

    month_ax = fig.add_subplot(gs[1, 1])
    month_ax.set_title("월별 합계와 봇·병합 제외 수치", fontsize=14)
    positions, labels = [], []
    for i, month in enumerate(("2026-09", "2026-10")):
        by_repo = {m["repository"]: m for m in monthly if m["month"] == month}
        for j, field in enumerate(("commits", "human_non_merge")):
            x = i * 2.6 + j
            main, sub = (by_repo.get(name, {}).get(field, 0) for name in REPOS)
            month_ax.bar(x, main, color=BLUE, width=0.68, zorder=3)
            month_ax.bar(x, sub, bottom=main, color=GREEN, width=0.68, zorder=3)
            month_ax.text(x, main + sub + 5, str(main + sub), ha="center", weight="bold")
            for value, bottom in ((main, 0), (sub, main)):
                if value >= 18:
                    month_ax.text(x, bottom + value / 2, str(value), ha="center", va="center", color="white")
            positions.append(x)
            labels.append(f"{int(month[-2:])}월\n{'전체' if j == 0 else '봇·병합 제외'}")
    month_ax.set_xticks(positions, labels)
    month_ax.set_ylim(0, max(m["commits"] for m in monthly) * 1.18 + 1)
    month_ax.set_ylabel("커밋 수")
    month_ax.grid(axis="y", zorder=0)
    fig.text(0.06, 0.052, f"집계: 2026/09/01–{cutoff:%Y/%m/%d} (마지막 날은 스냅샷 시점까지)  ·  현재 HEAD의 모든 조상 커밋 포함", fontsize=10, color=MUTED)
    fig.text(0.06, 0.028, "합계에는 상위 저장소의 서브모듈 포인터 갱신도 포함됩니다. 커밋 수는 작업 시간이나 연구 성과의 직접 측정값이 아닙니다.", fontsize=9, color=MUTED)
    save_figure(fig, "01_commit_activity_overview")


def calendars(daily: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.5))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.74, bottom=0.12, wspace=0.11)
    fig.text(0.05, 0.93, "날짜별로 본 커밋 활동", fontsize=25, weight="bold")
    fig.text(0.05, 0.876, "큰 숫자는 전체 커밋 수 · 아래 숫자는 vlm-in-flash / final · 색이 진할수록 커밋이 많은 날", fontsize=11, color=MUTED)
    lookup = {d["date_kst"]: d for d in daily}
    cutoff = date.fromisoformat(summary["snapshot"]["as_of"])
    cmap = LinearSegmentedColormap.from_list("activity", ["#F1F5FB", "#BDD3F2", "#5889CB", "#24539B"])
    norm = Normalize(vmin=0, vmax=max(1, summary["peak_day_commits"]))
    for month, ax in zip((9, 10), axes):
        weeks = calendar.Calendar(firstweekday=0).monthdayscalendar(2026, month)
        month_days = [d for d in daily if d["date_kst"].startswith(f"2026-{month:02}")]
        counts = {name: sum(d[name] for d in month_days) for name in REPOS}
        active = sum(d["total"] > 0 for d in month_days)
        ax.set_title(f"{month}월  ·  {sum(counts.values())}회 / {active}일 활동", fontsize=16, pad=42)
        ax.text(0.5, 1.065, f"vlm-in-flash {counts['vlm-in-flash']}회  /  final {counts['final']}회",
                transform=ax.transAxes, ha="center", fontsize=11, color=MUTED)
        for col, label in enumerate("월화수목금토일"):
            ax.text(col + 0.5, -0.22, label, ha="center", fontsize=11,
                    color="#B45E64" if col == 6 else MUTED)
        for row, week in enumerate(weeks):
            for col, day in enumerate(week):
                if day == 0:
                    continue
                value_date = date(2026, month, day)
                observed = value_date <= cutoff
                counts_day = lookup.get(value_date.isoformat(), {})
                total = counts_day.get("total", 0)
                face = cmap(norm(total)) if observed else "#E5E9EF"
                ax.add_patch(Rectangle((col + 0.035, row + 0.035), 0.93, 0.93,
                                       facecolor=face, edgecolor="white", linewidth=0.5,
                                       hatch="///" if not observed else None))
                color = "white" if total >= summary["peak_day_commits"] * 0.72 and total else INK
                ax.text(col + 0.13, row + 0.24, str(day), fontsize=10, color=color if observed else MUTED)
                if observed:
                    ax.text(col + 0.5, row + 0.55, str(total), ha="center", va="center",
                            fontsize=20, color=color if total else "#AAB8CB", weight="bold")
                    if total:
                        ax.text(col + 0.5, row + 0.84,
                                f"{counts_day['vlm-in-flash']} / {counts_day['final']}",
                                ha="center", fontsize=9, color=color)
                else:
                    ax.text(col + 0.5, row + 0.61, "집계 전", ha="center", fontsize=9, color=MUTED)
        ax.set_xlim(0, 7)
        ax.set_ylim(len(weeks), -0.08)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.05, 0.065, f"KST · committer date 기준 · {cutoff:%Y/%m/%d} 스냅샷까지 · 봇·병합 커밋 포함", fontsize=10, color=MUTED)
    fig.text(0.05, 0.035, "0은 해당 날짜에 커밋 기록이 없음을 뜻합니다. 빗금은 집계일 이후 날짜입니다. 두 달은 같은 색상 기준을 사용합니다.", fontsize=9, color=MUTED)
    save_figure(fig, "02_commit_activity_calendar")


def write_readme(daily: list[dict], monthly: list[dict], summary: dict) -> None:
    table = []
    for month in ("2026-09", "2026-10"):
        rows = {m["repository"]: m for m in monthly if m["month"] == month}
        if not rows:
            continue
        combined = rows["combined"]
        table.append(f"| {month} | {rows['vlm-in-flash']['commits']} | {rows['final']['commits']} | "
                     f"{combined['commits']} | {combined['human_non_merge']} | {combined['active_days']} |")
    table.append(f"| 합계 | {sum(d['vlm-in-flash'] for d in daily)} | {sum(d['final'] for d in daily)} | "
                 f"{summary['total_commits']} | {summary['human_non_merge_commits']} | {summary['active_days']} |")
    snapshot = summary["snapshot"]
    cutoff = date.fromisoformat(snapshot["as_of"])
    september_burst = sum(d["total"] for d in daily if "2026-09-26" <= d["date_kst"] <= "2026-09-27")
    october_burst = sum(d["total"] for d in daily if "2026-10-26" <= d["date_kst"] <= "2026-10-28")
    oct_rows = [m for m in monthly if m["month"] == "2026-10"]
    oct_total = next((m["commits"] for m in oct_rows if m["repository"] == "combined"), 0)
    oct_final = next((m["commits"] for m in oct_rows if m["repository"] == "final"), 0)
    readme = f"""# 2026년 9월과 10월 커밋 활동 분석

`vlm-in-flash`와 `final` 서브모듈의 Git 로그를 한국 시간으로 집계하였다.
분석 범위는 **2026-09-01부터 {snapshot['as_of']}까지**이며, 마지막 날짜는
**{snapshot['captured_at']}**에 확보한 Git 기록까지 포함한다.

| 기간 | vlm-in-flash | final | 전체 커밋 | 봇·병합 제외 | 커밋한 날짜 수 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

9월에는 본 저장소에, 10월에는 `final`에 커밋 활동이 집중되어 있다.
9월 26–27일에는 합계 {september_burst}회, 10월 26–28일에는 {october_burst}회 기록되었다.
10월 커밋 중 `final` 비중은 {oct_final / oct_total * 100 if oct_total else 0:.1f}%이다.
가장 많은 날은 **{summary['peak_day']}의 {summary['peak_day_commits']}회**이며,
두 저장소를 합친 최장 연속 커밋 기록은 **{summary['longest_active_streak_days']}일**이다.
분석 종료일은 {cutoff:%Y년 %m월 %d일}이며, 월 전체와 직접 비교할 때 이 범위를 고려해야 한다.

## 그래프

![일별 커밋 수, 누적 커밋 수, 월별 비교](figures/01_commit_activity_overview.png)

위 그래프는 저장소별 전체 커밋을 표시하고 월별 패널에서 봇·병합 제외 수치도 함께 비교한다.

![9월과 10월의 커밋 달력](figures/02_commit_activity_calendar.png)

달력의 큰 숫자는 전체 커밋 수이고, 아래 숫자는 `vlm-in-flash / final` 순서다.
두 달의 색상 범위는 같다. 집계 이후 날짜는 빗금으로 구분하였다.
각 그림은 `figures/`에 PNG, SVG, PDF로 저장되어 있다.

## 집계 방법

- 각 저장소의 스냅샷 HEAD에서 도달 가능한 모든 조상 커밋을 집계하였다.
  다른 브랜치에서만 도달 가능한 기록과 `final` 안의 중첩 서브모듈 기록은 포함하지 않는다.
- Git의 **committer date**를 `Asia/Seoul`로 변환하여 날짜를 정하였다.
  author date와 날짜가 다른 커밋은 {len(summary['author_committer_day_differences'])}개이며,
  해당 SHA는 `summary.json`, 두 시각은 `data/commits.csv`에 기록하였다.
- 전체에는 병합 {summary['merge_commits']}개와 봇 {summary['bot_commits']}개가 포함된다.
  봇은 author 또는 committer의 이름·이메일에 `[bot]`이 있는 기록으로 분류하였다.
  `봇·병합 제외`는 두 조건 중 하나라도 해당하면 제외한 수치다.
  AI 도구를 이용해 작성한 사람이름의 커밋을 구분하는 지표는 아니다.
- 같은 저장소 안에서는 SHA당 한 번 집계하고, 활동일은 두 저장소에서 커밋이 하나라도 있는 날짜의 합집합이다.
  두 저장소 합계 {summary['total_commits']}개 중 서로 다른 SHA는 {summary['unique_hashes_across_repositories']}개다.
- 상위 저장소에서 `final` 포인터를 갱신한 커밋도 별도 기록으로 포함한다.
  합계는 두 저장소의 커밋 객체 수이며, 서로 독립적인 작업 횟수를 뜻하지 않는다.
- 9월의 첫 기록 이전 날짜도 달력상 0회로 표시한다.
  커밋 수는 기록의 빈도이며 작업 시간이나 연구 성과를 직접 측정하지 않는다.

## 재실행

Python 3.10 이상, Git, Matplotlib, 한글 글꼴(NanumGothic 또는 Noto Sans CJK)이 필요하다.
저장소 루트에서 다음 명령으로 **같은 HEAD와 집계 종료일**을 다시 분석한다.

```sh
python exp-after-final/15_commit_activity/analyze_git_activity.py
```

현재 체크아웃의 HEAD를 새로 반영하려면 다음과 같이 실행한다.
이 명령은 이 디렉토리의 스냅샷과 분석 산출물을 덮어쓴다.

```sh
python exp-after-final/15_commit_activity/analyze_git_activity.py --refresh --as-of 2026-10-28
```

Git 기록은 읽기만 하며 fetch, checkout, commit은 실행하지 않는다.

## 파일

- `snapshot.json`: 분석 시점, 날짜 기준, 고정한 두 저장소 HEAD
- `summary.json`: 전체 요약 지표와 월별·저장소별 집계
- `data/commits.csv`: 커밋 SHA, 작성자, 두 시각, 병합·봇 분류, 제목
- `data/daily.csv`: 커밋이 없는 날을 포함한 일별 수치
- `data/monthly.csv`: 저장소별·합산 월별 수치와 활동일 평균
- `analyze_git_activity.py`: 원본 로그 추출, 집계, 그래프 생성 코드

분석한 HEAD:

```text
vlm-in-flash {snapshot['heads']['vlm-in-flash']}
final        {snapshot['heads']['final']}
```
"""
    (HERE / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Replace pinned HEADs with current HEADs")
    parser.add_argument("--as-of", type=date.fromisoformat, help="Last observed date, inclusive (KST)")
    args = parser.parse_args()
    snapshot = get_snapshot(args.refresh, args.as_of)
    commits = collect_commits(snapshot)
    if not commits:
        raise ValueError("No commits in the requested period.")
    daily, monthly, summary = summarize(commits, snapshot)
    for folder in ("data", "figures"):
        (HERE / folder).mkdir(exist_ok=True)
    configure_plotting()
    overview(daily, monthly, summary)
    calendars(daily, summary)
    write_csv(HERE / "data" / "commits.csv", commits)
    write_csv(HERE / "data" / "daily.csv", daily)
    write_csv(HERE / "data" / "monthly.csv", monthly)
    for name, value in (("snapshot", snapshot), ("summary", summary)):
        (HERE / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(daily, monthly, summary)
    print(json.dumps({k: summary[k] for k in ("total_commits", "human_non_merge_commits", "active_days", "peak_day", "peak_day_commits")}, ensure_ascii=False, indent=2))
    print(f"Saved to {HERE}")


if __name__ == "__main__":
    main()
