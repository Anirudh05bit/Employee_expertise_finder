"""Run the full Employee Expertise Finder pipeline and write CSV/JSON results to ./outputs.

    python run_analysis.py --k 20 --jobs -1
"""
import argparse
import json
import os
import time
from typing import cast

import numpy as np
import pandas as pd

from ddbd import analysis as an
from ddbd.algorithm import run_ddbd
from ddbd.baselines import compute_centralities, top_k
from ddbd.data import load_static, load_temporal
from ddbd.evaluation import compare_methods
from ddbd.routes import RouteEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--k", type=int, default=20, help="top-K employees per method")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers for DDBD (-1 = all cores)")
    ap.add_argument("--windows", type=int, default=4, help="temporal windows (0 = skip temporal)")
    ap.add_argument("--temporal-k", type=int, default=5)
    ap.add_argument("--random-trials", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    print("[1/7] loading network")
    net = load_static(a.data_dir)
    print(f"  {net.n} employees, {net.n_edges} directed edges, {net.n_depts} departments")

    print("[2/7] centralities (Degree / Betweenness / PageRank)")
    cent = compute_centralities(net, cache=os.path.join(a.out, "centralities.csv"))
    cent.to_csv(os.path.join(a.out, "centralities.csv"), index=False)

    print("[3/7] DDBD")
    res = run_ddbd(net, k=a.k, tiebreak=cent.betweenness.to_numpy(), n_jobs=a.jobs)
    res.steps.to_csv(os.path.join(a.out, "ddbd_selected.csv"), index=False)
    res.first_round.to_csv(os.path.join(a.out, "ddbd_first_round.csv"), index=False)

    print("[4/7] comparing DDBD with Degree / Betweenness / PageRank")
    rankings = {"DDBD": res.selected,
                "Degree": top_k(cent.degree, a.k),
                "Betweenness": top_k(cent.betweenness, a.k),
                "PageRank": top_k(cent.pagerank, a.k)}
    base = RouteEngine(net.adj, net.dept)
    ks = sorted({k for k in [1, 2, 3, 5, 7, 10, 15, 20, a.k] if k <= a.k})
    comp = compare_methods(net, base, rankings, ks, a.random_trials)
    comp.to_csv(os.path.join(a.out, "comparison_metrics.csv"), index=False)
    pd.DataFrame({m: [int(net.node_ids[i]) for i in o] + [None] * (a.k - len(o))
                  for m, o in rankings.items()}).to_csv(os.path.join(a.out, "rankings.csv"), index=False)
    an.overlap_table(rankings, a.k).to_csv(os.path.join(a.out, "overlap_jaccard.csv"))

    print("[5/7] brokerage vs dependency + communities")
    roles = an.brokerage_vs_dependency(base, cent)
    roles.to_csv(os.path.join(a.out, "brokerage_dependency.csv"), index=False)
    an.rank_correlations(roles).to_csv(os.path.join(a.out, "rank_correlations.csv"), index=False)
    labels, comm_stats = an.detect_communities(net)
    pd.DataFrame({"node": np.arange(net.n), "node_id": net.node_ids, "dept": net.dept,
                  "community": labels}).to_csv(os.path.join(a.out, "communities.csv"), index=False)
    pd.DataFrame(an.dept_matrix(net), index=net.dept_labels, columns=net.dept_labels) \
        .to_csv(os.path.join(a.out, "dept_matrix.csv"))

    print("[6/7] temporal analysis")
    if a.windows > 0:
        try:
            tdf = load_temporal(net, a.data_dir)
            steps, J, persist = an.temporal_ddbd(net, tdf, a.windows, a.temporal_k, a.jobs)
            steps.to_csv(os.path.join(a.out, "temporal_ddbd.csv"), index=False)
            J.to_csv(os.path.join(a.out, "temporal_jaccard.csv"))
            persist.astype({"windows": str}).to_csv(os.path.join(a.out, "temporal_persistence.csv"), index=False)
        except FileNotFoundError as e:
            print("  skipped:", e)

    print("[7/7] summary")
    kk = a.k
    last = comp[comp.k == kk].set_index("method")
    res_cols = ["disconnected_pairs", "connectivity_retained_pct", "total_path_increase",
                "cross_dept_edges_retained_pct", "fragile_pairs_pct",
                "largest_strong_cc_pct"]
    res_df = cast(pd.DataFrame, cast(pd.DataFrame, last)[res_cols]).round(3)
    results_at_k: dict = cast(pd.DataFrame, res_df).to_dict(orient="index")
    summary = {"employees": net.n, "edges": net.n_edges, "departments": net.n_depts,
               "reachable_dept_pairs": int(base.reach.sum()),
               "structurally_dependent_employees": int((roles.role == "structurally dependent").sum()),
               "broker_but_replaceable_employees": int((roles.role == "broker but replaceable").sum()),
               **comm_stats, "k": kk,
               "ddbd_overlap_with": {m: an.jaccard(rankings["DDBD"], rankings[m]) for m in ["Degree", "Betweenness", "PageRank"]},
               "results_at_k": results_at_k,
               "runtime_sec": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print("\nNext:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()