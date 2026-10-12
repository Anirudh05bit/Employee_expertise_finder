"""Department Dependency Bridge Discovery (DDBD) - iterative lexicographic greedy selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from .routes import RouteEngine

SORT_COLS = ["disconnections", "distance_increase", "mean_alt_routes", "tiebreak", "node"]
SORT_ASC = [False, False, True, False, True]
#   1. most department pairs disconnected
#   2. then largest total shortest-path increase
#   3. then fewest alternative routes (mean #same-layer substitutes on affected routes)
#   4. then largest standard-centrality tie-breaker, then node index (determinism)


@dataclass
class DDBDResult:
    selected: list[int]          # node indices in DDBD order
    steps: pd.DataFrame          # winning row of every iteration
    first_round: pd.DataFrame    # single-removal effect of every employee on the intact network


def run_ddbd(net, k=10, tiebreak=None, n_jobs=1, alive0=None, verbose=True) -> DDBDResult:
    if alive0 is None:
        deg = np.asarray(net.adj.sum(0)).ravel() + np.asarray(net.adj.sum(1)).ravel()
        alive0 = deg > 0
    alive = np.asarray(alive0, bool).copy()
    selected: list[int] = []
    step_rows: list[dict] = []
    first: pd.DataFrame | None = None
    for r in range(1, k + 1):
        eng = RouteEngine(net.adj, net.dept, alive)
        df = eng.candidate_effects(None, tiebreak, n_jobs)
        if first is None:
            first = df.copy()
        best = df.sort_values(SORT_COLS, ascending=SORT_ASC).iloc[0]
        if best.affected_pairs == 0:
            break                                   # nobody is left on any department route
        row = best.to_dict()
        row.update(rank=r, node_id=int(net.node_ids[int(best.node)]),
                   reachable_pairs_before=int(eng.reach.sum()))
        step_rows.append(row)
        selected.append(int(best.node))
        alive[int(best.node)] = False
        if verbose:
            print(f"  DDBD #{r}: employee {row['node_id']} (dept {net.dept_labels[int(best.dept)]}) "
                  f"disconnects {int(best.disconnections)} pairs, +{best.distance_increase:.0f} hops, "
                  f"mean alt routes {best.mean_alt_routes:.2f}")
    steps_df = pd.DataFrame(step_rows)
    if not steps_df.empty:
        cols = ["rank", "node", "node_id", "dept", "disconnections", "distance_increase",
                "mean_alt_routes", "affected_pairs", "critical_pairs", "tiebreak",
                "reachable_pairs_before"]
        steps_df = cast(pd.DataFrame, steps_df.loc[:, cols])
    first_df = first if first is not None else pd.DataFrame()
    return DDBDResult(selected, steps_df, first_df)