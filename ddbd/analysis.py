"""Communities, department matrix, brokerage-vs-dependency, rank overlap, temporal DDBD."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .algorithm import run_ddbd
from .data import window_network


def detect_communities(net, seed=42):
    U = (net.adj + net.adj.T).tocsr()                       # reciprocal mail -> weight 2
    G = nx.from_scipy_sparse_array(U)
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    labels = np.zeros(net.n, int)
    for c, members in enumerate(comms):
        labels[list(members)] = c
    return labels, {"louvain_communities": len(comms),
                    "nmi_vs_departments": float(normalized_mutual_info_score(net.dept, labels)),
                    "ari_vs_departments": float(adjusted_rand_score(net.dept, labels))}


def dept_matrix(net):
    """M[a, b] = number of emails-edges from department a to department b (diagonal = internal)."""
    P = sp.csr_matrix((np.ones(net.n), (np.arange(net.n), net.dept)), shape=(net.n, net.n_depts))
    return (P.T @ net.adj @ P).toarray().astype(int)


def brokerage_vs_dependency(engine, cent: pd.DataFrame) -> pd.DataFrame:
    """Broker  = sits on many shortest department routes.
       Dependent = is the ONLY employee in some layer of at least one department route."""
    df = cent.merge(engine.node_roles(), on="node")
    thr = df.loc[df.affected_pairs > 0, "affected_pairs"].quantile(0.75)
    df["role"] = np.select(
        [df.critical_pairs > 0, df.affected_pairs >= thr, df.affected_pairs > 0],
        ["structurally dependent", "broker but replaceable", "minor route member"],
        default="not on any department route")
    return df


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def overlap_table(rankings: dict, k: int) -> pd.DataFrame:
    names = list(rankings)
    return pd.DataFrame([[jaccard(rankings[x][:k], rankings[y][:k]) for y in names] for x in names],
                        index=names, columns=names)


def rank_correlations(roles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ["degree", "betweenness", "pagerank"]:
        for target in ["critical_pairs", "affected_pairs"]:
            rho, p = spearmanr(roles[c], roles[target])
            rows.append({"centrality": c, "vs": target, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def temporal_ddbd(net, tdf, windows=4, k=5, n_jobs=1):
    cuts = np.quantile(tdf.ts.to_numpy(), np.linspace(0, 1, windows + 1))
    rows, sets = [], []
    for w in range(windows):
        last = w == windows - 1
        sel = (tdf.ts >= cuts[w]) & ((tdf.ts <= cuts[w + 1]) if last else (tdf.ts < cuts[w + 1]))
        wnet, active = window_network(net, tdf[sel])
        deg = np.asarray(wnet.adj.sum(0)).ravel() + np.asarray(wnet.adj.sum(1)).ravel()
        res = run_ddbd(wnet, k=k, tiebreak=deg, n_jobs=n_jobs, alive0=active, verbose=False)
        for _, r in res.steps.iterrows():
            rows.append({"window": w + 1, "edges": wnet.n_edges, "active_nodes": int(active.sum()),
                         **r.to_dict()})
        sets.append(res.selected)
        print(f"  window {w + 1}/{windows}: {wnet.n_edges} edges -> DDBD {[int(net.node_ids[i]) for i in res.selected]}")
    J = pd.DataFrame([[jaccard(a, b) for b in sets] for a in sets],
                     index=[f"W{i + 1}" for i in range(windows)], columns=[f"W{i + 1}" for i in range(windows)])
    steps = pd.DataFrame(rows)
    persist = (steps.groupby("node_id").window.agg(["count", lambda s: sorted(s)])
               .rename(columns={"count": "windows_selected", "<lambda_0>": "windows"})
               .sort_values("windows_selected", ascending=False).reset_index())
    return steps, J, persist