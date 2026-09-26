"""Small, inspectable selectors; costs are indexed by run length, with cost[0] = 0."""
from collections import deque

import numpy as np


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return edges.reshape(-1, 2)


def latency(mask, cost):
    spans = runs(mask)
    return float(cost[spans[:, 1] - spans[:, 0]].sum())


def topk(v, r):
    mask = np.zeros(len(v), bool)
    mask[np.argsort(-v, kind="stable")[:r]] = True
    return mask


def ratio_opt(v, r, cost):
    """Exact nonempty max I/L under |M| <= r: one interval suffices."""
    prefix = np.r_[0., np.cumsum(v)]
    best, span = -np.inf, (0, 0)
    for length in range(1, min(r, len(v)) + 1):
        scores = (prefix[length:] - prefix[:-length]) / cost[length]
        start = int(scores.argmax())
        if scores[start] > best:
            best, span = scores[start], (start, start + length)
    mask = np.zeros(len(v), bool)
    mask[slice(*span)] = True
    return mask


def paper(v, r, cost, s, jump=1, start=1, step=1):
    """Algorithm 1 in row units, stable ties; return the mask, merge for evaluation."""
    prefix = np.r_[0., np.cumsum(v)]
    candidates = []
    for length in range(start, min(s, len(v)) + 1, step):
        for a in range(0, len(v) - length + 1, min(length, jump)):
            score = (prefix[a + length] - prefix[a]) / cost[length]
            candidates.append((score, a, length))
    candidates.sort(key=lambda c: -c[0])
    mask = np.zeros(len(v), bool)
    remaining = r
    for _, a, length in candidates:
        if length <= remaining and not mask[a:a + length].any():
            mask[a:a + length] = True
            remaining -= length
            if remaining == 0:
                break
    return mask


def saturation(v, r, cost, s, shifts=None):
    """Best shifted partition: top full s-tiles plus one residual interval."""
    n = len(v)
    if r == 0:
        return np.zeros(n, bool)
    s = max(1, int(s))
    if r < s:
        prefix = np.r_[0., np.cumsum(v)]
        start = int(np.argmax(prefix[r:] - prefix[:-r]))
        mask = np.zeros(n, bool)
        mask[start:start + r] = True
        return mask
    shifts = (0, s // 2) if shifts is None else shifts
    prefix = np.r_[0., np.cumsum(v)]
    choices = []
    for offset in sorted(set(shifts)):
        starts = np.arange(offset, n - s + 1, s)
        k, remainder = divmod(r, s)
        if len(starts) < k:
            continue
        weights = prefix[starts + s] - prefix[starts]
        chosen = starts[np.argsort(-weights, kind="stable")[:k]]
        mask = np.zeros(n, bool)
        for a in chosen:
            mask[a:a + s] = True
        if remainder:
            occupied = np.r_[0, np.cumsum(mask)]
            valid = occupied[remainder:] == occupied[:-remainder]
            if not valid.any():
                continue
            values = np.where(valid, prefix[remainder:] - prefix[:-remainder], -np.inf)
            a = int(values.argmax())
            mask[a:a + remainder] = True
        choices.append(mask)
    # Offset zero always fits: its unchosen cells and final tail contain the remainder.
    assert choices and all(m.sum() == r for m in choices)
    return max(choices, key=lambda m: float(v[m].sum()) / latency(m, cost))


def exhaustive(v, cost):
    """Independent vectorized oracle, O(n 2^n), limited deliberately."""
    n = len(v)
    if n > 20:
        raise ValueError("Complete enumeration is limited to n <= 20")
    masks = ((np.arange(1 << n, dtype=np.uint32)[:, None] >> np.arange(n)) & 1).astype(bool)
    totals = np.zeros(len(masks))
    length = np.zeros(len(masks), int)
    for column in range(n):
        closing = ~masks[:, column]
        totals[closing] += cost[length[closing]]
        length = np.where(closing, 0, length + 1)
    totals += cost[length]
    return masks, masks @ v, totals


def frontier(importance, costs):
    order = np.lexsort((-importance, costs))
    selected, best = [], -np.inf
    for i in order:
        if importance[i] > best + 1e-12:
            selected.append(i)
            best = importance[i]
    return np.asarray(selected, int)


def _trace(choice):
    mask = np.zeros(len(choice) - 1, bool)
    j = len(mask)
    while j:
        a = choice[j]
        if a < 0:
            j -= 1
        else:
            mask[a:j] = True
            j = max(0, a - 1)
    return mask


def lagrangian(v, cost, penalty):
    """Exact max I - penalty L, arbitrary positive run costs, O(n^2)."""
    n = len(v)
    prefix = np.r_[0., np.cumsum(v)]
    values = np.zeros(n + 1)
    choice = np.full(n + 1, -1, int)
    for j in range(1, n + 1):
        a = np.arange(j)
        scores = values[np.maximum(0, a - 1)] + prefix[j] - prefix[a] - penalty * cost[j - a]
        best = int(scores.argmax())
        values[j] = values[j - 1]
        if scores[best] > values[j]:
            values[j], choice[j] = scores[best], best
    return _trace(choice)


def two_line(n, s, b1=.03, b2=.12):
    length = np.arange(n + 1)
    cost = b2 * length + (b2 - b1) * np.maximum(s - length, 0)
    cost[0] = 0
    return cost


def linear_lagrangian(v, s, b1, b2, penalty):
    """Same exact DP for two_line(), using one sliding maximum; O(n)."""
    n = len(v)
    prefix = np.r_[0., np.cumsum(v)]
    values = np.zeros(n + 1)
    choice = np.full(n + 1, -1, int)
    short, long_best = deque(), (-np.inf, -1)
    intercept = (b2 - b1) * s
    for j in range(1, n + 1):
        a = j - 1
        score = values[max(0, a - 1)] - prefix[a] + penalty * b1 * a
        while short and short[-1][0] <= score:
            short.pop()
        short.append((score, a))
        while short and short[0][1] < j - s:
            short.popleft()
        a = j - s - 1
        if a >= 0:
            score = values[max(0, a - 1)] - prefix[a] + penalty * b2 * a
            if score > long_best[0]:
                long_best = (score, a)
        candidates = [(values[j - 1], -1)]
        candidates.append((prefix[j] - penalty * (intercept + b1 * j) + short[0][0], short[0][1]))
        candidates.append((prefix[j] - penalty * b2 * j + long_best[0], long_best[1]))
        values[j], choice[j] = max(candidates, key=lambda item: item[0])
    return _trace(choice)


def workload(n, seed=0, kind="smooth"):
    rng = np.random.default_rng(seed)
    if kind == "flat":
        return np.ones(n)
    if kind == "spiky":
        return rng.lognormal(0, 1.4, n)
    noise = rng.normal(0, .16, n)
    if kind == "smooth":
        noise += .35 * np.sin(np.arange(n) * 2 * np.pi / max(8, n / 4))
    return np.maximum(.05, 1 + noise)
