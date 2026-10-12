"""Download the SNAP EU-Core files into ./data  (or build a synthetic stand-in).

    python download_data.py              # real SNAP data
    python download_data.py --synthetic  # same file format, random data (for testing offline)
"""
import argparse
import os
import urllib.request

import numpy as np

BASE = "https://snap.stanford.edu/data/"
FILES = ["email-Eu-core.txt.gz",
         "email-Eu-core-department-labels.txt.gz",
         "email-Eu-core-temporal.txt.gz"]


def download(out):
    os.makedirs(out, exist_ok=True)
    for f in FILES:
        dest = os.path.join(out, f)
        if os.path.exists(dest):
            print("have", f); continue
        print("downloading", f)
        urllib.request.urlretrieve(BASE + f, dest)


def synthetic(out, n=1005, nd=42, m=25000, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    sizes = rng.pareto(1.2, nd) + 1
    dept = rng.choice(nd, n, p=sizes / sizes.sum())
    dept[:nd] = np.arange(nd)                       # every dept non-empty
    act = rng.pareto(1.5, n) + 0.3; act /= act.sum()
    src = rng.choice(n, m, p=act)
    dst = np.empty(m, int)
    by = [np.flatnonzero(dept == d) for d in range(nd)]
    for i, s in enumerate(src):
        dst[i] = rng.choice(by[dept[s]]) if rng.random() < 0.75 else rng.integers(n)
    ok = src != dst
    edges = np.unique(np.c_[src[ok], dst[ok]], axis=0)
    np.savetxt(os.path.join(out, "email-Eu-core.txt"), edges, fmt="%d")
    np.savetxt(os.path.join(out, "email-Eu-core-department-labels.txt"),
               np.c_[np.arange(n), dept], fmt="%d")
    reps = rng.integers(1, 20, len(edges))
    t = np.repeat(np.arange(len(edges)), reps)
    ts = np.sort(rng.integers(0, 800 * 86400, len(t)))
    np.savetxt(os.path.join(out, "email-Eu-core-temporal.txt"),
               np.c_[edges[t], ts], fmt="%d")
    print("synthetic data written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    synthetic(a.out) if a.synthetic else download(a.out)