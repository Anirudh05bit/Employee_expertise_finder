"""Conventional centrality baselines (Degree, Betweenness, PageRank)."""
from __future__ import annotations

import os
from typing import cast

import networkx as nx
import numpy as np
import pandas as pd


def compute_centralities(net, cache=None) -> pd.DataFrame:
    A = net.adj
    out_deg = np.asarray(A.sum(1)).ravel()
    in_deg = np.asarray(A.sum(0)).ravel()
    if cache and os.path.exists(cache):
        df = pd.read_csv(cache)
        if len(df) == net.n and np.allclose(df.degree, in_deg + out_deg):
            return df
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    print("  computing betweenness (exact, ~1 min for EU-Core)...")
    btw: dict[int, float] = cast(dict[int, float], nx.betweenness_centrality(G))
    pr: dict[int, float] = cast(dict[int, float], nx.pagerank(G))
    df = pd.DataFrame({"node": np.arange(net.n), "node_id": net.node_ids, "dept": net.dept,
                       "in_degree": in_deg, "out_degree": out_deg, "degree": in_deg + out_deg,
                       "betweenness": [btw[i] for i in range(net.n)],
                       "pagerank": [pr[i] for i in range(net.n)]})
    if cache:
        df.to_csv(cache, index=False)
    return df


def top_k(score, k):
    score = np.asarray(score, float)
    return [int(i) for i in np.lexsort((np.arange(len(score)), -score))[:k]]