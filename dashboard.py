"""Interactive dashboard.   streamlit run dashboard.py   (run run_analysis.py first)"""
import json
import os
from typing import Any, TypedDict, cast

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px  # pyrefly: ignore[missing-import]
import plotly.graph_objects as go  # pyrefly: ignore[missing-import]
import streamlit as st  # pyrefly: ignore[missing-import]

from ddbd.data import load_static
from ddbd.evaluation import evaluate_removal
from ddbd.routes import RouteEngine

st.set_page_config(page_title="Employee Expertise Finder - DDBD", layout="wide")
st.title("Employee Expertise Finder - Department Dependency Bridge Discovery")
st.caption("Structural information-flow roles in the EU-Core email network. "
           "The dataset has no skill data, so this identifies communication dependency, not technical expertise.")

DATA = st.sidebar.text_input("Data folder", "data")
OUT = st.sidebar.text_input("Results folder", "outputs")
if not os.path.exists(os.path.join(OUT, "summary.json")):
    st.error("No results found - run `python run_analysis.py` first."); st.stop()


class Results(TypedDict):
    summary: dict[str, Any]
    steps: pd.DataFrame
    comp: pd.DataFrame
    rank: pd.DataFrame
    roles: pd.DataFrame
    corr: pd.DataFrame
    comm: pd.DataFrame
    dm: pd.DataFrame
    overlap: pd.DataFrame
    t_steps: pd.DataFrame | None
    t_j: pd.DataFrame | None
    t_p: pd.DataFrame | None


@st.cache_data
def load_results(out: str) -> Results:
    r = lambda f, **kw: pd.read_csv(os.path.join(out, f), **kw) if os.path.exists(os.path.join(out, f)) else pd.DataFrame()
    with open(os.path.join(out, "summary.json"), "r") as f:
        summary_data = json.load(f)
    return {
        "summary": summary_data,
        "steps": r("ddbd_selected.csv"),
        "comp": r("comparison_metrics.csv"),
        "rank": r("rankings.csv"),
        "roles": r("brokerage_dependency.csv"),
        "corr": r("rank_correlations.csv"),
        "comm": r("communities.csv"),
        "dm": r("dept_matrix.csv", index_col=0),
        "overlap": r("overlap_jaccard.csv", index_col=0),
        "t_steps": r("temporal_ddbd.csv") if os.path.exists(os.path.join(out, "temporal_ddbd.csv")) else None,
        "t_j": r("temporal_jaccard.csv", index_col=0) if os.path.exists(os.path.join(out, "temporal_jaccard.csv")) else None,
        "t_p": r("temporal_persistence.csv") if os.path.exists(os.path.join(out, "temporal_persistence.csv")) else None,
    }


@st.cache_resource
def load_net(data):
    net = load_static(data)
    return net, RouteEngine(net.adj, net.dept)


@st.cache_data
def layout(data):
    net, _ = load_net(data)
    G = nx.from_scipy_sparse_array(net.adj + net.adj.T)
    return nx.spring_layout(G, seed=7, iterations=60)


R = load_results(OUT)
net, base = load_net(DATA)
S = R["summary"]
methods = list(R["rank"].columns)
tabs = st.tabs(["Overview", "DDBD results", "Method comparison", "Brokerage vs dependency",
                "Network explorer", "Removal simulator", "Temporal"])

with tabs[0]:
    c = st.columns(5)
    c[0].metric("Employees", S["employees"]); c[1].metric("Directed edges", S["edges"])
    c[2].metric("Departments", S["departments"]); c[3].metric("Louvain communities", S["louvain_communities"])
    c[4].metric("NMI communities vs depts", f"{S['nmi_vs_departments']:.2f}")
    c = st.columns(3)
    c[0].metric("Structurally dependent employees", S["structurally_dependent_employees"])
    c[1].metric("Brokers but replaceable", S["broker_but_replaceable_employees"])
    c[2].metric("Reachable department pairs", S["reachable_dept_pairs"])
    st.subheader("Department-to-department email volume")
    dm = R["dm"].to_numpy().copy(); np.fill_diagonal(dm, 0)
    st.plotly_chart(px.imshow(np.log1p(dm), labels={"color": "log(1+edges)"}, aspect="auto"),
                    width="stretch")

with tabs[1]:
    st.subheader("DDBD selection order")
    st.dataframe(R["steps"], width="stretch")
    st.plotly_chart(px.bar(R["steps"], x="rank", y=["disconnections", "distance_increase"], barmode="group",
                           title="Effect of removing each DDBD pick (after the previous picks are gone)"),
                    width="stretch")
    st.markdown("**Selection rule (lexicographic):** most department-pair disconnections -> largest total "
                "shortest-path increase -> fewest alternative routes -> betweenness tie-break.")

with tabs[2]:
    comp = R["comp"]
    metric = st.selectbox("Metric", [c for c in comp.columns if c not in ("method", "k")], index=3)
    st.plotly_chart(px.line(comp, x="k", y=metric, color="method", markers=True), width="stretch")
    st.subheader("Top-K lists (original employee ids)"); st.dataframe(R["rank"])
    st.subheader("Overlap (Jaccard) of top-K lists"); st.dataframe(R["overlap"].round(2))
    st.subheader("Spearman correlation: centralities vs dependency"); st.dataframe(R["corr"].round(4))

with tabs[3]:
    roles = R["roles"]
    st.plotly_chart(px.scatter(roles, x="betweenness", y="critical_pairs", color="role", hover_data=["node_id", "dept", "degree", "affected_pairs"],
                               title="High betweenness does not imply sole-gatekeeper status"), width="stretch")
    st.write(roles.role.value_counts().rename("employees").to_frame().T)
    st.dataframe(roles.sort_values(["critical_pairs", "affected_pairs"], ascending=False).head(50))

with tabs[4]:
    pos = layout(DATA)
    pick = st.selectbox("Highlight top-K of", methods); kx = st.slider("K", 1, len(R["rank"]), 10)
    ids = set(R["rank"][pick].dropna().astype(int).head(kx))
    hi = {i for i in range(net.n) if int(net.node_ids[i]) in ids}
    coo = net.adj.tocoo(); rng = np.random.default_rng(0)
    pick_e = rng.choice(len(coo.row), min(4000, len(coo.row)), replace=False)
    ex, ey = [], []
    for e in pick_e:
        (u, v) = coo.row[e], coo.col[e]
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line={"width": 0.3, "color": "#aaa"}, hoverinfo="none"))
    allx = np.array([pos[i] for i in range(net.n)]); sz = 4 + 20 * np.asarray(net.adj.sum(0)).ravel() / max(1, net.adj.sum(0).max())
    other = [i for i in range(net.n) if i not in hi]
    fig.add_trace(go.Scatter(x=allx[other, 0], y=allx[other, 1], mode="markers", name="employees",
                             marker={"size": 4, "color": net.dept[other], "colorscale": "Turbo"},
                             text=[f"id {net.node_ids[i]} - dept {net.dept_labels[net.dept[i]]}" for i in other]))
    h = sorted(hi)
    fig.add_trace(go.Scatter(x=allx[h, 0], y=allx[h, 1], mode="markers", name=f"{pick} top-{kx}",
                             marker={"size": 14, "color": "red", "line": {"width": 2, "color": "black"}},
                             text=[f"id {net.node_ids[i]} - dept {net.dept_labels[net.dept[i]]}" for i in h]))
    fig.update_layout(height=700, xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
    st.caption("Edges: random sample of 4000. Node colour = department.")

with tabs[5]:
    st.write("Remove employees and see the damage to inter-department communication.")
    mode = st.radio("Choose", ["Top-K of a method", "Pick employees"], horizontal=True)
    if mode == "Top-K of a method":
        pick = st.selectbox("Method", methods, key="sim_m"); kx = st.slider("K", 1, len(R["rank"]), 5, key="sim_k")
        sim_ids = list(R["rank"][pick].dropna().astype(int).head(kx))
    else:
        sim_ids = st.multiselect("Employee ids", [int(x) for x in net.node_ids])
    if sim_ids:
        where = {int(v): i for i, v in enumerate(net.node_ids)}
        m = evaluate_removal(net, base, [where[i] for i in sim_ids])
        st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})

with tabs[6]:
    if R["t_steps"] is None:
        st.info("Temporal results not available.")
    else:
        st.dataframe(R["t_steps"], width="stretch")
        st.subheader("Jaccard similarity of DDBD picks between windows")
        st.plotly_chart(px.imshow(R["t_j"], zmin=0, zmax=1, text_auto=cast(Any, ".2f")), width="stretch")
        st.subheader("Employees selected in more than one window"); st.dataframe(R["t_p"])