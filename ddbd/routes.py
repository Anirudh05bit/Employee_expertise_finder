"""Department-level shortest-route structure - the engine behind DDBD.

For the network with some employees deleted ("alive" mask) we compute

  DA[a, v]  shortest hops from department a (best member) to employee v
  DB[b, v]  shortest hops from employee v to department b (best member)
  Dd[a, b]  shortest hops from department a to department b (min over members)

An employee v lies on a shortest a->b route  <=>  DA[a,v] + DB[b,v] == Dd[a,b].
Its *layer* on that route is DA[a,v].  Every shortest route has exactly one node per
layer, so the other on-route employees in the same layer are exactly the ones who can
replace v ("alternative employee-mediated routes"):

  alt[a,b,v] = (#on-route employees in v's layer) - 1        (-1 if v is not on the route)

alt == 0  =>  v is the only way through that layer  =>  deleting v lengthens or breaks
the department route.  alt >= 1  =>  the shortest distance is unchanged.  This is exact,
so real re-computation is only needed for the (a, b) pairs with alt == 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra  # pyrefly: ignore[missing-import]

INF = np.inf


def _removal_effects_batch(adj, alive_idx, members, Dd, items):
    """For each (e, critical_pairs): delete e, return (#pairs disconnected, total distance increase)."""
    n = adj.shape[0]
    out = []
    for e, crit in items:
        if len(crit) == 0:
            out.append((0, 0.0)); continue
        keep = alive_idx[alive_idx != e]
        sub = adj[keep][:, keep]
        pos = np.full(n, -1); pos[keep] = np.arange(len(keep))
        disc, inc = 0, 0.0
        for a in np.unique(crit[:, 0]):
            src = members[a][members[a] != e]
            dist = dijkstra(sub, directed=True, indices=pos[src], unweighted=True, min_only=True)
            for b in crit[crit[:, 0] == a, 1]:
                tgt = members[b][members[b] != e]
                new = dist[pos[tgt]].min() if len(tgt) else INF
                if np.isinf(new): disc += 1
                else: inc += new - Dd[a, b]
        out.append((disc, inc))
    return out


class RouteEngine:
    def __init__(self, adj, dept, alive=None):
        self.adj, self.dept = adj, dept
        self.n = adj.shape[0]
        self.nd = int(dept.max()) + 1
        self.alive = np.ones(self.n, bool) if alive is None else np.asarray(alive, bool).copy()
        self.alive_idx = np.flatnonzero(self.alive)
        self.members = [np.flatnonzero(self.alive & (dept == d)) for d in range(self.nd)]
        self.dept_alive = np.array([len(m) > 0 for m in self.members])
        self._build()

    # ---------------------------------------------------------------- build
    def _dept_dist(self, sub, pos):
        out = np.full((self.nd, self.n), INF)
        for d, m in enumerate(self.members):
            if len(m):
                out[d, self.alive_idx] = dijkstra(sub, directed=True, indices=pos[m],
                                                  unweighted=True, min_only=True)
        return out

    def _build(self):
        idx, n, nd = self.alive_idx, self.n, self.nd
        full = len(idx) == n
        sub = self.adj if full else self.adj[idx][:, idx]
        pos = np.full(n, -1); pos[idx] = np.arange(len(idx))
        self.DA = self._dept_dist(sub, pos)
        self.DB = self._dept_dist(sub.T.tocsr(), pos)
        Dd = np.full((nd, nd), INF)
        for b, m in enumerate(self.members):
            if len(m):
                Dd[:, b] = self.DA[:, m].min(axis=1)
        np.fill_diagonal(Dd, INF)
        self.Dd, self.reach = Dd, np.isfinite(Dd)
        self.alt = np.full((nd, nd, n), -1, np.int16)
        self.width = np.zeros((nd, nd))            # bottleneck width of each department route
        for a in range(nd):
            for b in np.flatnonzero(self.reach[a]):
                D = int(Dd[a, b])
                on = (self.DA[a] + self.DB[b]) == D
                layer = self.DA[a][on].astype(int)
                cnt = np.bincount(layer, minlength=D + 1)
                self.alt[a, b, on] = cnt[layer] - 1
                self.width[a, b] = cnt.min()

    # ------------------------------------------------------------ summaries
    def node_roles(self) -> pd.DataFrame:
        """Per employee: on how many department routes (affected) / as sole gatekeeper (critical)."""
        return pd.DataFrame({"node": np.arange(self.n),
                             "affected_pairs": (self.alt >= 0).sum(axis=(0, 1)),
                             "critical_pairs": (self.alt == 0).sum(axis=(0, 1))})

    def candidate_effects(self, cands=None, tiebreak=None, n_jobs=1) -> pd.DataFrame:
        cands = self.alive_idx if cands is None else np.asarray(cands)
        items, meta = [], []
        for e in cands:
            da = self.dept[e]
            pairs = np.argwhere(self.alt[:, :, e] >= 0)
            if len(self.members[da]) == 1:                    # e's whole department vanishes
                pairs = pairs[(pairs[:, 0] != da) & (pairs[:, 1] != da)]
            altv = self.alt[pairs[:, 0], pairs[:, 1], e] if len(pairs) else np.array([])
            crit = pairs[altv == 0]
            items.append((int(e), crit))
            meta.append((len(pairs), len(crit), altv.mean() if len(pairs) else INF))
        args = (self.adj, self.alive_idx, self.members, self.Dd)
        if n_jobs == 1 or len(items) < 50:
            res = _removal_effects_batch(*args, items)
        else:
            from joblib import Parallel, delayed  # pyrefly: ignore[missing-import]
            chunks = np.array_split(np.arange(len(items)), 8 * (n_jobs if n_jobs > 0 else 8))
            parts = Parallel(n_jobs=n_jobs)(
                delayed(_removal_effects_batch)(*args, [items[i] for i in c]) for c in chunks if len(c))
            res = [r for p in (parts or []) if p is not None for r in p]
        tb = np.zeros(self.n) if tiebreak is None else np.asarray(tiebreak, float)
        return pd.DataFrame({
            "node": cands, "dept": self.dept[cands],
            "affected_pairs": [m[0] for m in meta], "critical_pairs": [m[1] for m in meta],
            "disconnections": [r[0] for r in res], "distance_increase": [r[1] for r in res],
            "mean_alt_routes": [m[2] for m in meta], "tiebreak": tb[cands]})