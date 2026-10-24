#!/usr/bin/env python3
"""Instrument Paper Algorithm 1 without changing its candidate scores or choices."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper_chunk_mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "paper_chunk_numba"))
os.environ.setdefault("VLMFLASH_NATIVE", "0")

import numpy as np
import pandas as pd
import torch
import numba
import matplotlib
from numba import njit

GEOMETRY02 = {(896, 128): (8, 8), (896, 896): (8, 8),
              (896, 4864): (8, 8), (4864, 896): (12, 16)}
MODES = ("paper_lookup_s99", "notebook02_twoline", "notebook03_full_lookup")


@njit(cache=True)
def greedy_logged(starts, lengths, order, budget, n):
    """status 0=selected, 1=budget, 2=overlap, 3=not visited after filling."""
    mask = np.zeros(n, dtype=np.uint8)
    status = np.full(len(starts), 3, dtype=np.int8)
    accepted = np.empty(n, dtype=np.int64)
    ranks = np.empty(n, dtype=np.int64)
    remaining_before = np.empty(n, dtype=np.int64)
    remaining = budget
    count = 0
    for rank in range(len(order)):
        if remaining == 0:
            break
        idx = order[rank]
        start, length = starts[idx], lengths[idx]
        if length > remaining:
            status[idx] = 1
            continue
        overlap = False
        for pos in range(start, start + length):
            if mask[pos]:
                overlap = True
                break
        if overlap:
            status[idx] = 2
            continue
        status[idx] = 0
        accepted[count] = idx
        ranks[count] = rank + 1
        remaining_before[count] = remaining
        count += 1
        mask[start:start + length] = 1
        remaining -= length
    return mask, accepted[:count], ranks[:count], remaining_before[:count], status


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runs(mask):
    change = np.diff(np.r_[np.int8(0), mask.astype(np.int8), np.int8(0)])
    return np.column_stack((np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class CSVStream:
    def __init__(self, path, columns):
        self.handle = gzip.open(path, "wt", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(columns)

    def rows(self, rows):
        self.writer.writerows(rows)

    def close(self):
        self.handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-final", type=Path,
                        default=Path(__file__).resolve().parents[2] / "final")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--limit-per-shape", type=int, default=0)
    parser.add_argument("--verify-per-shape", type=int, default=2)
    args = parser.parse_args()
    root, output = args.source_final.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    started_at = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / "vlm-flash/src"))
    import vlmflash
    from vlmflash.policy import _window_sweep
    assert Path(vlmflash.__file__).resolve().is_relative_to(root / "vlm-flash")

    paths = {
        "traces": root / "results/02/activation_traces.json",
        "model": root / "results/01/latency_model.json",
        "profile": root / "results/01/chunk_latency_measurements.csv",
        "measurement_environment": root / "results/03/measured_environment.json",
        "upstream_policy": root / "vlm-flash/src/vlmflash/policy.py",
        "upstream_latency": root / "vlm-flash/src/vlmflash/latency.py",
        "upstream_native": root / "vlm-flash/src/vlmflash/csrc/native.cc",
        "capture_notebook": root / "results/02/02_01_capture_traces.ipynb",
        "selector_notebook": root / "results/02/02_03_selector_sweep.ipynb",
        "measurement_notebook": root / "results/03/03_01_measure_selection_latency.ipynb",
    }
    models = json.loads(paths["model"].read_text())
    model = models["models"]["Two-line (free)"]
    environment = json.loads(paths["measurement_environment"].read_text())
    archive = json.loads(paths["traces"].read_text())
    profile = pd.read_csv(paths["profile"])
    train_repeats = sorted(profile.repeat.unique())[:4]
    train = profile[profile.repeat.isin(train_repeats)].groupby("size_kib").latency_ms.median()
    complete = profile.groupby("size_kib").latency_ms.median()
    train, complete = train.sort_index(), complete.sort_index()
    s = float(model["boundary_kib"])
    s99 = float(models["s99_kib"])
    peak = float((complete / complete.index).idxmin())
    b, delta = float(model["short_slope_ms_per_kib"]), float(model["slope_change_ms_per_kib"])
    alpha = b + delta
    assert alpha > 0 and delta >= 0

    def model_cost(kib):
        kib = np.asarray(kib)
        return b * kib + delta * np.maximum(s, kib)

    tables = {
        "paper_lookup_s99": vlmflash.LatencyTable(train.to_dict()),
        "notebook02_twoline": vlmflash.LatencyTable({
            i: float(model_cost(i)) for i in range(1, int(np.ceil(max(s, s99))) + 2)
        }),
        "notebook03_full_lookup": vlmflash.LatencyTable(complete.to_dict()),
    }
    tuned = {(int(t["n"]), int(t["d"])): t["best"] for t in environment["paper_tuning"]}
    shapes = sorted({(t["n"], t["d"]) for t in archive["traces"]})
    traces, verify_ids = [], set()
    for shape in shapes:
        group = [t for t in archive["traces"] if (t["n"], t["d"]) == shape]
        if args.limit_per_shape:
            inds = np.linspace(0, len(group) - 1, min(args.limit_per_shape, len(group)), dtype=int)
            group = [group[i] for i in inds]
        traces.extend(group)
        inds = np.linspace(0, len(group) - 1, min(args.verify_per_shape, len(group)), dtype=int)
        verify_ids.update(group[i]["trace_id"] for i in inds)
    fractions = np.arange(1, 10) / 10  # Exactly the notebook 02 budget calculation.
    layout, geometry = {}, []
    for mode in MODES:
        for n, d in shapes:
            row_kib = d * archive["weight_bytes"] / 1024
            if mode == "notebook03_full_lookup":
                cfg = tuned[n, d]
                start, jump = cfg["start_kb"], cfg["jump_cap_kb"]
                end = float(complete.index.max()) + 1
            else:
                start, jump = GEOMETRY02[n, d]
                end = s99 + 1
            params = vlmflash.ChunkParams(start_kb=start, step_kb=start,
                                         jump_cap_kb=jump, end_kb=end)
            windows, cap = _window_sweep(n, row_kib, tables[mode], params)
            starts, lengths, costs, groups = [], [], [], []
            for group_id, (length, cost) in enumerate(windows):
                stride = min(length, cap)
                positions = np.arange(0, n - length + 1, stride, dtype=np.int64)
                starts.append(positions)
                lengths.append(np.full(len(positions), length, dtype=np.int64))
                costs.append(np.full(len(positions), cost, dtype=np.float64))
                groups.append(np.full(len(positions), group_id, dtype=np.int64))
                geometry.append(dict(config=mode, n=n, d=d, row_kib=row_kib,
                    start_kib=start, step_kib=start, jump_cap_kib=jump, end_kib_exclusive=end,
                    length_rows=length, length_kib=length * row_kib, stride_rows=stride,
                    candidate_count=len(positions), policy_cost_ms=cost))
            layout[mode, n, d] = (params, windows, np.concatenate(starts), np.concatenate(lengths),
                                   np.concatenate(costs), np.concatenate(groups))
    pd.DataFrame(geometry).to_csv(output / "geometry.csv", index=False)
    pd.DataFrame({"size_kib": complete.index, "latency_ms_all_median": complete.values,
                  "latency_ms_train_median": train.reindex(complete.index).values,
                  "two_line_ms": model_cost(complete.index.to_numpy())}).to_csv(
                      output / "profile_used.csv", index=False)

    common = ["case_id", "config", "trace_id", "n", "d", "fraction"]
    accepted_file = CSVStream(output / "accepted_candidates.csv.gz", common + [
        "selection_order", "candidate_id", "score_rank", "start", "end", "length_rows",
        "length_kib", "length_over_s_model", "importance", "mean_importance",
        "mean_importance_over_trace_mean", "utility_float32", "policy_cost_ms", "model_cost_ms",
        "remaining_before", "merged_run_id"])
    run_file = CSVStream(output / "merged_runs.csv.gz", common + [
        "run_id", "start", "end", "length_rows", "length_kib", "length_over_s_model",
        "importance", "mean_importance", "candidate_count", "policy_cost_ms", "model_cost_ms",
        "short_deficit_kib"])
    decision_file = CSVStream(output / "candidate_decisions_by_length.csv.gz", common + [
        "length_rows", "length_kib", "generated", "visited", "selected", "rejected_budget",
        "rejected_overlap", "not_visited_budget_filled", "mean_utility_all", "mean_utility_selected",
        "mean_density_over_trace_mean_all", "mean_density_over_trace_mean_selected"])

    cases, validations = [], []
    distributions = {}  # config, n, d, kind, length -> count, rows, importance, model penalty
    examples = []
    checked_cases = 0
    for ti, trace in enumerate(traces):
        n, d, trace_id = trace["n"], trace["d"], trace["trace_id"]
        row_kib = trace["row_bytes"] / 1024
        values = np.asarray(trace["values"], dtype=np.float32)
        assert len(values) == n and np.all(np.isfinite(values)) and np.all(values >= 0)
        assert row_kib == d * archive["weight_bytes"] / 1024
        prefix = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
        prefix32 = prefix.astype(np.float32)
        total_importance = float(prefix[-1])
        trace_mean = total_importance / n
        assert trace_mean > 0
        for mode in MODES:
            params, windows, starts, lengths, costs, group_ids = layout[mode, n, d]
            benefit32 = prefix32[starts + lengths] - prefix32[starts]
            scores = benefit32 / costs.astype(np.float32)
            order = np.argsort(-scores, kind="stable")
            benefit = prefix[starts + lengths] - prefix[starts]
            density = benefit / lengths / trace_mean
            generated = np.bincount(group_ids, minlength=len(windows))
            for fraction in fractions:
                R = max(1, int(fraction * n))
                mask, selected, ranks, before, status = greedy_logged(starts, lengths, order, R, n)
                spans = runs(mask)
                actual = int(mask.sum())
                assert 0 <= actual <= R and actual == int(lengths[selected].sum())
                rlength = spans[:, 1] - spans[:, 0]
                rkib = rlength * row_kib
                rimportance = prefix[spans[:, 1]] - prefix[spans[:, 0]]
                importance = float(rimportance.sum())
                np.testing.assert_allclose(importance, benefit[selected].sum(), rtol=1e-11, atol=1e-10)
                if len(selected):
                    # Each accepted candidate lies entirely in exactly one maximal run.
                    parent = np.searchsorted(spans[:, 1], starts[selected], side="right")
                    assert np.all(starts[selected] >= spans[parent, 0])
                    assert np.all(starts[selected] + lengths[selected] <= spans[parent, 1])
                    constituents = np.bincount(parent, minlength=len(spans))
                else:
                    parent, constituents = np.array([], dtype=int), np.zeros(len(spans), dtype=int)
                np.testing.assert_array_equal(rlength, np.bincount(
                    parent, weights=lengths[selected], minlength=len(spans)).astype(int))
                deficit = float(np.maximum(s - rkib, 0).sum())
                baseline = alpha * actual * row_kib
                penalty = delta * deficit
                modeled = float(model_cost(rkib).sum())
                minimum_possible = float(model_cost(actual * row_kib)) if actual else 0.
                np.testing.assert_allclose(modeled, baseline + penalty, rtol=1e-12, atol=1e-12)
                candidate_modeled = float(model_cost(lengths[selected] * row_kib).sum())
                assert candidate_modeled + 1e-10 >= modeled
                policy_run_costs = np.array([tables[mode].read_ms(float(k)) for k in rkib])
                case_id = len(cases)
                base = [case_id, mode, trace_id, n, d, float(fraction)]
                case = dict(zip(common, base))
                short_candidate = lengths[selected] * row_kib < s
                short_remaining = before * row_kib < s
                case.update(requested_rows=R, actual_rows=actual, unused_rows=R - actual,
                    fill_fraction=actual / R, row_kib=row_kib, importance=importance,
                    selected_kib=actual * row_kib,
                    saturated_feasible=bool(actual * row_kib >= s),
                    requested_saturated_feasible=bool(R * row_kib >= s),
                    importance_retained=importance / total_importance,
                    candidate_count=len(selected), run_count=len(spans),
                    short_candidate_count=int(short_candidate.sum()),
                    short_candidate_low_remaining_count=int((short_candidate & short_remaining).sum()),
                    short_candidate_low_remaining_fraction=float(short_remaining[short_candidate].mean()) if short_candidate.any() else np.nan,
                    generated_count=len(starts), visited_count=int(np.count_nonzero(status != 3)),
                    rejected_budget_count=int(np.count_nonzero(status == 1)),
                    rejected_overlap_count=int(np.count_nonzero(status == 2)),
                    not_visited_count=int(np.count_nonzero(status == 3)),
                    candidate_mean_rows=float(lengths[selected].mean()) if len(selected) else np.nan,
                    run_mean_rows=float(rlength.mean()) if len(rlength) else np.nan,
                    run_max_kib=float(rkib.max()) if len(rkib) else 0,
                    realized_candidate_max_kib=float(lengths.max() * row_kib),
                    merge_fraction=1 - len(spans) / len(selected) if len(selected) else np.nan,
                    short_deficit_kib=deficit, baseline_latency_ms=baseline,
                    short_penalty_ms=penalty, penalty_over_baseline=penalty / baseline if baseline else np.nan,
                    minimum_possible_model_ms=minimum_possible,
                    fragmentation_excess_ms=modeled - minimum_possible,
                    fragmentation_over_minimum=modeled / minimum_possible - 1 if minimum_possible else np.nan,
                    modeled_read_ms=modeled, policy_run_cost_ms=float(policy_run_costs.sum()),
                    policy_candidate_cost_ms=float(costs[selected].sum()),
                    merge_saving_model_ms=candidate_modeled - modeled,
                    merge_saving_policy_ms=float(costs[selected].sum() - policy_run_costs.sum()))
                for kind, kk, vv in [("candidate", lengths[selected] * row_kib, benefit[selected]),
                                      ("run", rkib, rimportance)]:
                    for name, threshold in [("model", s), ("s99", s99), ("peak", peak)]:
                        short = kk < threshold
                        case[f"{kind}_short_{name}_count_fraction"] = float(short.mean()) if len(kk) else np.nan
                        case[f"{kind}_short_{name}_row_fraction"] = float(kk[short].sum() / kk.sum()) if len(kk) else np.nan
                        case[f"{kind}_short_{name}_importance_fraction"] = float(vv[short].sum() / importance) if importance else np.nan
                cases.append(case)

                accepted_file.rows(base + [j + 1, int(idx), int(rank), int(starts[idx]),
                    int(starts[idx] + lengths[idx]), int(lengths[idx]), float(lengths[idx] * row_kib),
                    float(lengths[idx] * row_kib / s), float(benefit[idx]), float(benefit[idx] / lengths[idx]),
                    float(density[idx]), float(scores[idx]), float(costs[idx]),
                    float(model_cost(lengths[idx] * row_kib)), int(left), int(run_id)]
                    for j, (idx, rank, left, run_id) in enumerate(zip(selected, ranks, before, parent)))
                run_file.rows(base + [j, int(a), int(z), int(z-a), float(kk), float(kk/s),
                    float(iv), float(iv/(z-a)), int(cc), float(pc), float(model_cost(kk)), max(0., s-float(kk))]
                    for j, ((a, z), kk, iv, cc, pc) in enumerate(zip(spans, rkib, rimportance, constituents, policy_run_costs)))

                decision_counts = np.bincount(group_ids * 4 + status,
                    minlength=len(windows) * 4).reshape(-1, 4)
                sum_score = np.bincount(group_ids, weights=scores, minlength=len(windows))
                sum_density = np.bincount(group_ids, weights=density, minlength=len(windows))
                selected_scores = np.bincount(group_ids[selected], weights=scores[selected], minlength=len(windows))
                selected_density = np.bincount(group_ids[selected], weights=density[selected], minlength=len(windows))
                decision_file.rows(base + [int(length), float(length * row_kib), int(generated[j]),
                    int(generated[j] - dc[3]), int(dc[0]), int(dc[1]), int(dc[2]), int(dc[3]),
                    float(sum_score[j]/generated[j]), float(selected_scores[j]/dc[0]) if dc[0] else np.nan,
                    float(sum_density[j]/generated[j]), float(selected_density[j]/dc[0]) if dc[0] else np.nan]
                    for j, ((length, _), dc) in enumerate(zip(windows, decision_counts)))
                assert int(decision_counts.sum()) == len(starts)

                for kind, ll, vv in [("candidate", lengths[selected], benefit[selected]),
                                     ("run", rlength, rimportance)]:
                    for length in np.unique(ll):
                        matching = ll == length
                        count = int(matching.sum())
                        key = (mode, n, d, kind, int(length))
                        entry = distributions.setdefault(key, np.zeros(4))
                        entry += [count, count * int(length), vv[matching].sum(),
                                  count * delta * max(0., s - length * row_kib)]

                if trace_id in verify_ids:
                    ref = vlmflash.select_chunks(torch.from_numpy(values), R, row_kib,
                                                  tables[mode], params, impl="torch")
                    equal = np.array_equal(mask.astype(bool), ref.mask.cpu().numpy())
                    cost_equal = np.isclose(costs[selected].sum(), ref.est_cost_ms, rtol=1e-12, atol=1e-12)
                    assert equal and cost_equal, (mode, trace_id, fraction)
                    validations.append(dict(case_id=case_id, config=mode, trace_id=trace_id,
                        n=n, d=d, fraction=float(fraction), mask_equal=equal, candidate_cost_equal=bool(cost_equal)))
                if fraction == .5 and trace_id in verify_ids:
                    examples.append(dict(case_id=case_id, config=mode, trace_id=trace_id, n=n, d=d,
                        row_kib=row_kib, importance_retained=importance/total_importance,
                        penalty_over_baseline=penalty/baseline if baseline else None,
                        accepted=[[int(starts[j]), int(starts[j]+lengths[j])] for j in selected],
                        runs=spans.tolist()))
                checked_cases += 1
        if (ti + 1) % 24 == 0 or ti + 1 == len(traces):
            print(f"{ti+1}/{len(traces)} traces; {len(cases)} cases; {time.perf_counter()-start_time:.1f}s", flush=True)

    for stream in (accepted_file, run_file, decision_file):
        stream.close()
    case_df = pd.DataFrame(cases)
    case_df.to_csv(output / "cases.csv", index=False)
    pd.DataFrame(validations).to_csv(output / "validation.csv", index=False)
    distribution_rows = []
    for (mode, n, d, kind, length), entry in sorted(distributions.items()):
        distribution_rows.append(dict(config=mode, n=n, d=d, kind=kind, length_rows=length,
            length_kib=length * d * archive["weight_bytes"] / 1024,
            length_over_s_model=length * d * archive["weight_bytes"] / 1024 / s,
            count=int(entry[0]), selected_rows=int(entry[1]), importance=float(entry[2]),
            model_short_penalty_ms=float(entry[3])))
    pd.DataFrame(distribution_rows).to_csv(output / "length_distribution.csv", index=False)
    save_json(output / "mask_examples.json", examples)
    group_keys = ["config", "n", "d", "fraction"]
    numeric = [col for col in case_df.select_dtypes(include="number") if col not in group_keys + ["case_id", "trace_id"]]
    aggregate = case_df.groupby(group_keys)[numeric].mean()
    aggregate["case_count"] = case_df.groupby(group_keys).size()
    aggregate.reset_index().to_csv(output / "by_shape_budget.csv", index=False)
    save_json(output / "cost_model_snapshot.json", models)
    save_json(output / "measurement_environment_snapshot.json", environment)
    upstream_commit = subprocess.run(["git", "-C", str(root / "vlm-flash"), "rev-parse", "HEAD"],
                                     text=True, capture_output=True, check=True).stdout.strip()
    manifest = dict(
        started_at=started_at, completed_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        elapsed_seconds=time.perf_counter()-start_time, source_final=str(root),
        inputs={k: {"path":str(p), "sha256":sha256(p)} for k,p in paths.items()},
        experiment_code={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        upstream_commit=upstream_commit, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
        numba=numba.__version__, matplotlib=matplotlib.__version__,
        torch_threads=torch.get_num_threads(), sort_device="cpu", new_io_measurement=False,
        new_model_inference=False, trace_count=len(traces), archive_trace_count=len(archive["traces"]),
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        trace_context="three text prompts; archived raw mean-absolute input activations; original row order",
        importance_normalization="none; float32 upstream inputs; float64 diagnostic sums",
        fractions=fractions.tolist(), configs=list(MODES), case_count=len(cases),
        checked_invariant_cases=checked_cases, reference_comparisons=len(validations),
        reference_all_equal=bool(all(v["mask_equal"] and v["candidate_cost_equal"] for v in validations)),
        verify_trace_ids=sorted(verify_ids), s_model_kib=s, s99_kib=s99, peak_efficiency_kib=peak,
        alpha_ms_per_kib=alpha, delta_ms_per_kib=delta,
        profile_train_repeats=[int(r) for r in train_repeats],
        config_descriptions={
            "paper_lookup_s99":"training-profile lookup; notebook02 geometry and s99 cap",
            "notebook02_twoline":"exact notebook02 cost table and geometry",
            "notebook03_full_lookup":"all-repeat lookup; recorded notebook03 tuning and full profile cap; all archived traces, fractions .1-.9",
        },
        limitation="Model diagnostic only; no SSD timing or model-quality evaluation. Traces and budgets are related observations.",
    )
    save_json(output / "manifest.json", manifest)
    from analyze import analyze
    analyze(output)
    print(f"Completed: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
