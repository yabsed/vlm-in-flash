"""Independent small-instance oracles and equivalence to the local paper port."""
import sys
from pathlib import Path

import numpy as np
import torch

import conclusion.python.ideas.core as C


def main():
    rng = np.random.default_rng(37)
    checks = 0
    for n in range(1, 10):
        for trial in range(4):
            v = rng.integers(0, 10, n).astype(float)
            cost = np.r_[0., rng.uniform(.01, 3, n)]  # deliberately nonmonotone
            masks, importance, latency = C.exhaustive(v, cost)
            for r in range(1, n + 1):
                allowed = (masks.sum(1) > 0) & (masks.sum(1) <= r)
                mask = C.ratio_opt(v, r, cost)
                assert np.isclose(v[mask].sum() / C.latency(mask, cost), (importance[allowed] / latency[allowed]).max())
                tiled = C.saturation(v, r, cost, min(3, n))
                assert tiled.sum() == r
                checks += 1
            for penalty in [0, .1, 1, 10]:
                mask = C.lagrangian(v, cost, penalty)
                assert np.isclose(v[mask].sum() - penalty * C.latency(mask, cost), (importance - penalty * latency).max())
                for s in [1, 3, n + 2]:
                    curve = C.two_line(n, s)
                    fast = C.linear_lagrangian(v, s, .03, .12, penalty)
                    slow = C.lagrangian(v, curve, penalty)
                    objective = lambda m: v[m].sum() - penalty * C.latency(m, curve)
                    assert np.isclose(objective(fast), objective(slow))
                    checks += 1
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "preliminary_research/vlm-flash/src"))
    from vlmflash.policy import ChunkParams, select_chunks
    from vlmflash.latency import LatencyTable
    for n in [9, 32, 128]:
        for jump in [1, 2, 4]:
            v = rng.integers(1, 200, n).astype(float)
            curve = C.two_line(n, 8)
            table = LatencyTable({4 * i: float(curve[i]) for i in range(1, n + 1)})
            parameters = ChunkParams(start_kb=4, end_kb=37, step_kb=4, jump_cap_kb=4 * jump)
            reference = select_chunks(torch.tensor(v), n // 2, 4, table, parameters, impl="torch").mask.numpy()
            assert np.array_equal(reference, C.paper(v, n // 2, curve, 8, jump=jump))
            checks += 1
    print(f"PASS: {checks} exhaustive/DP/budget/reference checks")


if __name__ == "__main__":
    main()
