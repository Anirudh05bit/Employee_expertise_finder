# Employee Expertise Finder: Department Dependency Bridge Discovery (DDBD)

A Social Network Analysis (SNA) system designed to discover critical cross-department communication bridges and information-flow dependencies within organization email networks (analyzed on the SNAP EU-Core email dataset).

Unlike conventional centrality measures (Degree, Betweenness, PageRank) that prioritize high-volume intra-departmental hubs, **DDBD (Department Dependency Bridge Discovery)** uncovers structural gatekeepers whose removal fragments inter-department communication.

---

## Key Features

- **Department Dependency Bridge Discovery (DDBD)**: Iterative lexicographic greedy selection maximizing disconnected department pairs and path stretch while minimizing substitution alternatives.
- **Baseline Comparisons**: Benchmarks against Degree Centrality, Betweenness Centrality, PageRank, and Random removals.
- **Brokerage vs. Dependency Analysis**: Classifies employee network roles into *Structurally Dependent*, *Broker but Replaceable*, *Minor Route Member*, and *Not on Department Route*.
- **Temporal Stability**: Measures persistence and Jaccard similarity of bridge roles across temporal windows.
- **Interactive Streamlit Dashboard**: Web UI with 7 interactive modules including an interactive network explorer and real-time removal simulator.

---

## Project Structure

```text
├── ddbd/
│   ├── __init__.py
│   ├── algorithm.py      # DDBD greedy selection algorithm
│   ├── analysis.py       # Communities, brokerage vs dependency, rank overlap, temporal DDBD
│   ├── baselines.py      # Degree, Betweenness, and PageRank centralities
│   ├── data.py           # Network data loaders & graph constructions
│   ├── evaluation.py     # Removal effect evaluation metrics & comparisons
│   └── routes.py         # RouteEngine shortest department path computations
├── dashboard.py          # Interactive Streamlit dashboard
├── run_analysis.py       # Full CLI analysis pipeline
├── download_data.py      # Script to download SNAP EU-Core data or generate synthetic stand-in
├── requirements.txt      # Python dependencies
├── pyrefly.toml          # Type checker configuration
└── outputs/              # Generated analysis results (CSV/JSON)
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Anirudh05bit/Employee_expertise_finder.git
   cd Employee_expertise_finder
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data**:
   ```bash
   # Download real SNAP EU-Core dataset:
   python download_data.py

   # Or generate synthetic test data offline:
   python download_data.py --synthetic
   ```

---

## Usage

### 1. Run Analysis Pipeline (CLI)
Run the full analysis pipeline to compute centralities, DDBD order, method comparisons, and temporal metrics:
```bash
python run_analysis.py --k 10 --jobs -1
```

**Options**:
- `--k <int>`: Top-K employees to discover (default: `20`).
- `--jobs <int>`: Worker processes for parallel computation (`-1` for all CPU cores).
- `--windows <int>`: Temporal windows to analyze (`0` to skip temporal).
- `--out <dir>`: Directory where output CSV/JSON files are saved (default: `outputs`).

### 2. Launch Interactive Dashboard
Launch the Streamlit visualization app:
```bash
streamlit run dashboard.py
```
Open your browser at `http://localhost:8501` to view:
- **Overview**: High-level network KPIs and department-to-department email traffic heatmaps.
- **DDBD Results**: Step-by-step gatekeeper selection order and removal impacts.
- **Method Comparison**: Comparative curves showing communication damage vs baseline centralities.
- **Brokerage vs Dependency**: Scatter plots distinguishing volume brokers from sole gatekeepers.
- **Network Explorer**: Interactive force-directed network graph highlighting top-K individuals.
- **Removal Simulator**: Real-time simulation of network damage when custom employees are removed.
- **Temporal Analysis**: Stability and persistence of bridge roles over time windows.

---

## Core Findings

- **Zero Overlap with Standard Centrality**: Top-10 DDBD gatekeepers have **0% overlap** with top-10 Degree, Betweenness, or PageRank nodes.
- **Severe Inter-Department Impact**: Removing top-10 DDBD employees disconnects **164 department pairs** (causing ~10% loss of global reachability), whereas removing top-10 Degree or Betweenness nodes disconnects **0 department pairs**.
