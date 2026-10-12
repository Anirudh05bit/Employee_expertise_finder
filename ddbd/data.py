from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class OrgNetwork:
    adj: sp.csr_matrix        # adj[u, v] = 1  <=>  u emailed v   (directed, unweighted)
    dept: np.ndarray          # department index (0..n_depts-1) for every node
    node_ids: np.ndarray      # original SNAP ids
    dept_labels: np.ndarray   # original department ids, position = dept index

    @property
    def n(self) -> int:
        return int(self.adj.shape[0]) if self.adj.shape is not None else 0

    @property
    def n_depts(self) -> int:
        return len(self.dept_labels)

    @property
    def n_edges(self) -> int:
        return self.adj.nnz


def _read(data_dir, name):
    for p in (os.path.join(data_dir, name), os.path.join(data_dir, name + ".gz")):
        if os.path.exists(p):
            return pd.read_csv(p, sep=r"\s+", header=None, compression="infer")
    raise FileNotFoundError(f"{name}(.gz) not found in {data_dir}. Run: python download_data.py")


def build_adj(src, dst, n) -> sp.csr_matrix:
    keep = src != dst                                   # drop self-loops
    a = sp.coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n)).tocsr()
    a.data[:] = 1.0                                     # collapse duplicate mails
    return cast(sp.csr_matrix, a)


def load_static(data_dir="data") -> OrgNetwork:
    labels = _read(data_dir, "email-Eu-core-department-labels.txt")
    edges = _read(data_dir, "email-Eu-core.txt")
    node_ids = np.sort(labels[0].to_numpy())
    idx_map = {node_id: i for i, node_id in enumerate(node_ids)}
    raw = labels.set_index(0)[1].loc[node_ids].to_numpy()
    dept_labels, dept = np.unique(raw, return_inverse=True)
    src = edges[0].map(idx_map.get)
    dst = edges[1].map(idx_map.get)
    ok = src.notna() & dst.notna()
    src_arr = np.asarray(src[ok], dtype=int)
    dst_arr = np.asarray(dst[ok], dtype=int)
    adj = build_adj(src_arr, dst_arr, len(node_ids))
    return OrgNetwork(adj, dept, node_ids, dept_labels)


def load_temporal(net: OrgNetwork, data_dir="data") -> pd.DataFrame:
    """Temporal edges mapped onto the static node index space (columns src, dst, ts)."""
    t = _read(data_dir, "email-Eu-core-temporal.txt")
    idx_map = {node_id: i for i, node_id in enumerate(net.node_ids)}
    out = pd.DataFrame({"src": t[0].map(idx_map.get), "dst": t[1].map(idx_map.get), "ts": t[2]}).dropna()
    return out.astype({"src": int, "dst": int, "ts": np.int64}).sort_values("ts").reset_index(drop=True)


def window_network(net: OrgNetwork, tdf: pd.DataFrame):
    """Network of one time window + mask of nodes that were active in it."""
    adj = build_adj(tdf["src"].to_numpy(), tdf["dst"].to_numpy(), net.n)
    active = (np.asarray(adj.sum(0)).ravel() + np.asarray(adj.sum(1)).ravel()) > 0
    return OrgNetwork(adj, net.dept, net.node_ids, net.dept_labels), active