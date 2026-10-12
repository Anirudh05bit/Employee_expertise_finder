"""Remove a set of employees and measure what happens to inter-department communication."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components  # pyrefly: ignore[missing-import]

from .routes import RouteEngine


def evaluate_removal(net, base: RouteEngine, removed) -> dict:
    removed = [int(x) for x in removed]
    alive = np.ones(net.n, bool); alive[removed] = False
    eng = RouteEngine(net.adj, net.dept, alive)

    both = base.reach & eng.dept_alive[:, None] & eng.dept_alive[None, :]   # pairs that can still be judged
    still = both & eng.reach
    n_pairs = int(both.sum())
    inc = float((eng.Dd[still] - base.Dd[still]).sum())

    coo = net.adj.tocoo()
    cross = net.dept[coo.row] != net.dept[coo.col]
    cross_kept = cross & alive[coo.row] & alive[coo.col]

    idx = np.flatnonzero(alive)
    sub = net.adj[idx][:, idx]
    lcc = lambda kind: np.bincount(connected_components(sub, directed=True, connection=kind)[1]).max()

    return {
        "removed": len(removed),
        "depts_eliminated": int((base.dept_alive & ~eng.dept_alive).sum()),
        "dept_pairs_judged": n_pairs,
        "disconnected_pairs": int((both & ~eng.reach).sum()),
        "connectivity_retained_pct": 100 * still.sum() / max(n_pairs, 1),
        "pairs_with_longer_path": int(((eng.Dd > base.Dd) & still).sum()),
        "total_path_increase": inc,
        "mean_path_increase": inc / max(int(still.sum()), 1),
        "cross_dept_edges_retained_pct": 100 * cross_kept.sum() / max(cross.sum(), 1),
        "mean_route_width": float(eng.width[still].mean()) if still.any() else 0.0,
        "fragile_pairs_pct": 100 * float((eng.width[still] == 1).mean()) if still.any() else 0.0,
        "largest_weak_cc_pct": 100 * lcc("weak") / net.n,
        "largest_strong_cc_pct": 100 * lcc("strong") / net.n,
    }


def compare_methods(net, base, rankings: dict, ks, random_trials=5, seed=0) -> pd.DataFrame:
    rows = []
    for name, order in rankings.items():
        for k in ks:
            if k <= len(order):
                rows.append({"method": name, "k": k, **evaluate_removal(net, base, order[:k])})
    rng = np.random.default_rng(seed)
    deg = np.asarray(net.adj.sum(0)).ravel() + np.asarray(net.adj.sum(1)).ravel()
    pool = np.flatnonzero(deg > 0)
    for k in ks:
        trials = pd.DataFrame([evaluate_removal(net, base, rng.choice(pool, k, replace=False))
                               for _ in range(random_trials)])
        rows.append({"method": "Random (mean)", "k": k, **trials.mean().to_dict()})
    return pd.DataFrame(rows)