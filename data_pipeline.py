# =============================================================================
# phase1_etl_graph_construction.py
# Cyber Attack Detection using GNN — Phase 1: ETL & Graph Construction
#
# Pipeline:
#   Raw CICIDS 2017 CSV
#       → Clean & Validate Flows
#       → IP-to-Node Index Mapping
#       → Directed Edge Construction  (edge_index)
#       → Node Feature Aggregation    (X matrix)
#       → Label Assignment            (y vector)
#       → Train / Val / Test Masks
#       → PyTorch Geometric Data object
#
# Dataset: CICIDS 2017
#   Download: https://www.unb.ca/cic/datasets/ids-2017.html
#   Expected columns (subset used here):
#       ' Source IP', ' Destination IP', ' Source Port', ' Destination Port',
#       ' Protocol', ' Flow Duration', ' Total Fwd Packets',
#       ' Total Backward Packets', ' Total Length of Fwd Packets',
#       ' Total Length of Bwd Packets', ' Flow Bytes/s', ' Flow Packets/s',
#       ' Flow IAT Mean', ' Label'
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 0: CONFIGURATION
# All tunable parameters live here — change nothing else for quick experiments.
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Central configuration for the entire ETL pipeline.
    Every magic number lives here. No hard-coded values elsewhere.
    """

    # ── Dataset ──────────────────────────────────────────────────────────────
    # Path to your downloaded CICIDS 2017 CSV (or folder of CSVs).
    # Set to None to use the built-in synthetic generator for testing.
    data_path: Optional[str] = None

    # If data_path is a folder, all *.csv files inside will be concatenated.
    # Useful because CICIDS ships one CSV per day-of-capture.
    load_all_csvs_in_folder: bool = True

    # ── Graph Construction ────────────────────────────────────────────────────
    # Minimum number of flows an IP must appear in to earn a node.
    # Filters out ephemeral IPs that would add noise with no stable features.
    min_flows_per_node: int = 5

    # Time window in seconds for aggregating flows into a single graph.
    # None = use all flows as one static snapshot graph.
    time_window_seconds: Optional[int] = None

    # If True, self-loops (src_ip == dst_ip) are dropped.
    drop_self_loops: bool = True

    # ── Feature Engineering ───────────────────────────────────────────────────
    # Number of engineered node features (must match _engineer_node_features).
    num_features: int = 10

    # Attack-day captures to combine when data_path points to a folder.
    # The *_plus files duplicate these rows with additional columns.
    dataset_files: Optional[Tuple[str, ...]] = (
        'tuesday.csv', 'wednesday.csv', 'thursday.csv', 'friday.csv'
    )

    # Features to cap at this percentile before scaling (handles outliers).
    # e.g., 99 → anything above the 99th percentile is clipped to that value.
    outlier_clip_percentile: float = 99.0

    # ── Label Strategy ────────────────────────────────────────────────────────
    # A node is labeled Malicious (1) if ANY of its flows are malicious.
    # Alternative: 'majority' → label by the majority class of its flows.
    node_label_strategy: str = 'any'   # 'any' | 'majority'

    # ── Train / Val / Test Split ──────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15          # implicitly 1 - train - val

    # Stratified split ensures both classes appear in each partition.
    stratified_split: bool = True

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Synthetic Data (used when data_path is None) ──────────────────────────
    synthetic_n_benign_nodes:    int = 900
    synthetic_n_malicious_nodes: int = 100
    synthetic_n_flows:           int = 8000

    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "train + val + test ratios must sum to 1.0"
        assert self.node_label_strategy in ('any', 'majority'), \
            "node_label_strategy must be 'any' or 'majority'"


# =============================================================================
# SECTION 1: DATA INGESTION
# Loads raw CICIDS CSV(s), standardises column names, and performs initial
# quality filtering. Returns a clean, uniform DataFrame of flow records.
# =============================================================================

# Canonical column rename map — CICIDS 2017 has leading spaces in names.
# Left side = raw CSV column, Right side = internal clean name.
CICIDS_COLUMN_MAP = {
    ' Source IP':                      'src_ip',
    ' Destination IP':                 'dst_ip',
    ' Source Port':                    'src_port',
    ' Destination Port':               'dst_port',
    ' Protocol':                       'protocol',
    ' Flow Duration':                  'flow_duration',
    ' Total Fwd Packets':              'fwd_pkts',
    ' Total Backward Packets':         'bwd_pkts',
    ' Total Length of Fwd Packets':    'fwd_bytes',
    ' Total Length of Bwd Packets':    'bwd_bytes',
    ' Flow Bytes/s':                   'flow_bytes_per_sec',
    ' Flow Packets/s':                 'flow_pkts_per_sec',
    ' Flow IAT Mean':                  'flow_iat_mean',
    ' Label':                          'label',
    # Preprocessed Kaggle variant used by this project:
    'Src IP dec':                     'src_ip',
    'Dst IP dec':                     'dst_ip',
    'Src Port':                       'src_port',
    'Dst Port':                       'dst_port',
    'protocol':                       'protocol',
    'Total Fwd Packet':               'fwd_pkts',
    'Total Bwd packets':              'bwd_pkts',
    'Total Length of Fwd Packet':     'fwd_bytes',
    'Total Length of Bwd Packet':     'bwd_bytes',
    'label':                          'label',
    # Some CICIDS variants omit the leading space:
    'Source IP':                       'src_ip',
    'Destination IP':                  'dst_ip',
    'Source Port':                     'src_port',
    'Destination Port':                'dst_port',
    'Protocol':                        'protocol',
    'Flow Duration':                   'flow_duration',
    'Total Fwd Packets':               'fwd_pkts',
    'Total Backward Packets':          'bwd_pkts',
    'Total Length of Fwd Packets':     'fwd_bytes',
    'Total Length of Bwd Packets':     'bwd_bytes',
    'Flow Bytes/s':                    'flow_bytes_per_sec',
    'Flow Packets/s':                  'flow_pkts_per_sec',
    'Flow IAT Mean':                   'flow_iat_mean',
    'Label':                           'label',
}

# All string label values in CICIDS that represent BENIGN traffic.
# Everything else is treated as malicious.
BENIGN_LABELS = {'BENIGN', 'Benign', 'benign', 'Normal', 'NORMAL'}


def load_cicids_csv(config: PipelineConfig) -> pd.DataFrame:
    """
    Load one or more CICIDS 2017 CSV files and return a clean DataFrame.

    Column standardisation:
        Raw CICIDS columns have inconsistent leading spaces and casing.
        We rename them all to clean snake_case internal names.

    Label binarisation:
        CICIDS has 15 string labels (BENIGN, DDoS, PortScan, Bot, etc.).
        We collapse all attack types → 1, BENIGN → 0.
        This matches our node classification objective exactly.

    Args:
        config: PipelineConfig instance

    Returns:
        df: Clean DataFrame with columns:
            src_ip, dst_ip, src_port, dst_port, protocol,
            flow_duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes,
            flow_bytes_per_sec, flow_pkts_per_sec, flow_iat_mean,
            label (int: 0=Benign, 1=Malicious)
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 1: Data Ingestion")
    print("="*60)

    if config.data_path is None:
        print("  [INFO] No data_path set → generating synthetic dataset.")
        return _generate_synthetic_flows(config)

    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"data_path does not exist: {path}")

    # ── Collect CSV file paths ───────────────────────────────────────────────
    if path.is_dir() and config.load_all_csvs_in_folder:
        csv_files = (
            [path / name for name in config.dataset_files]
            if config.dataset_files else sorted(path.glob("*.csv"))
        )
        csv_files = [csv_path for csv_path in csv_files if csv_path.exists()]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        print(f"  Found {len(csv_files)} CSV file(s) in {path}")
    else:
        csv_files = [path]
        print(f"  Loading single file: {path.name}")

    # ── Load and concatenate ─────────────────────────────────────────────────
    frames = []
    for csv_path in csv_files:
        print(f"  Loading: {csv_path.name} ... ", end="")
        df_raw = pd.read_csv(
            csv_path,
            low_memory=False,
            encoding='utf-8',
            on_bad_lines='skip'     # CICIDS occasionally has malformed rows
        )
        print(f"{len(df_raw):,} rows")
        frames.append(df_raw)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Total raw rows loaded: {len(df):,}")

    # ── Rename columns ───────────────────────────────────────────────────────
    # Only rename columns that exist in the file (tolerant of partial schemas)
    rename = {k: v for k, v in CICIDS_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    required_cols = [
        'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
        'flow_duration', 'fwd_pkts', 'bwd_pkts', 'fwd_bytes', 'bwd_bytes',
        'flow_bytes_per_sec', 'flow_pkts_per_sec', 'flow_iat_mean', 'label'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"After renaming, these required columns are missing: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    # ── Binarise labels ──────────────────────────────────────────────────────
    df['label'] = df['label'].apply(
        lambda x: 0 if str(x).strip() in BENIGN_LABELS else 1
    ).astype(int)

    print(f"  Benign flows  : {(df['label']==0).sum():,}")
    print(f"  Malicious flows: {(df['label']==1).sum():,}")

    return df


def _generate_synthetic_flows(config: PipelineConfig) -> pd.DataFrame:
    """
    Generate a synthetic flow DataFrame that mirrors CICIDS 2017 schema.

    Used for development and testing without needing the real dataset.
    The synthetic data deliberately injects realistic attack patterns:
        - Malicious nodes have high out-degree (port scan / botnet)
        - Malicious flows have larger byte volumes (exfiltration)
        - Malicious flows have shorter durations (automated tools)

    Args:
        config: PipelineConfig instance

    Returns:
        df: Synthetic flow DataFrame with the same schema as load_cicids_csv
    """
    rng = np.random.default_rng(config.random_seed)

    n_benign    = config.synthetic_n_benign_nodes
    n_malicious = config.synthetic_n_malicious_nodes
    n_flows     = config.synthetic_n_flows

    # ── Generate IP pool ─────────────────────────────────────────────────────
    # Internal IPs: 192.168.x.x (corporate network)
    # External IPs: 10.x.x.x   (internet / C2 servers)
    benign_ips = [
        f"192.168.{rng.integers(0, 5)}.{rng.integers(1, 255)}"
        for _ in range(n_benign)
    ]
    malicious_ips = [
        f"192.168.{rng.integers(5, 10)}.{rng.integers(1, 255)}"
        for _ in range(n_malicious)
    ]
    external_ips = [
        f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        for _ in range(50)
    ]
    all_ips = benign_ips + malicious_ips

    # ── Ground-truth node label map ──────────────────────────────────────────
    ip_to_true_label = {
        ip: 0 for ip in benign_ips
    }
    ip_to_true_label.update({ip: 1 for ip in malicious_ips})

    # ── Generate flow records ────────────────────────────────────────────────
    records = []

    for i in range(n_flows):
        is_attack_flow = rng.random() < 0.08   # ~8% attack flows

        if is_attack_flow:
            # Attack flow: malicious src → external C2 or internal target
            src_ip  = malicious_ips[rng.integers(0, n_malicious)]
            if rng.random() < 0.6:
                dst_ip = external_ips[rng.integers(0, len(external_ips))]  # C2
            else:
                dst_ip = benign_ips[rng.integers(0, n_benign)]             # lateral
            label = 1
        else:
            # Benign flow: any internal → any internal/external
            src_ip  = benign_ips[rng.integers(0, n_benign)]
            dst_ip  = (
                external_ips[rng.integers(0, len(external_ips))]
                if rng.random() < 0.4 else
                all_ips[rng.integers(0, len(all_ips))]
            )
            label = 0

        if src_ip == dst_ip:
            continue

        # Flow-level feature generation
        # Attack flows: more bytes (exfiltration), shorter duration (automated)
        if label == 1:
            fwd_bytes  = int(rng.exponential(scale=50_000))
            bwd_bytes  = int(rng.exponential(scale=10_000))
            duration   = int(rng.exponential(scale=500_000))    # microseconds
            fwd_pkts   = int(rng.exponential(scale=200))
            bwd_pkts   = int(rng.exponential(scale=50))
            dst_port   = int(rng.choice([22, 23, 445, 3389, 4444, 6667, 8080]))
        else:
            fwd_bytes  = int(rng.exponential(scale=5_000))
            bwd_bytes  = int(rng.exponential(scale=3_000))
            duration   = int(rng.exponential(scale=5_000_000))
            fwd_pkts   = int(rng.exponential(scale=20))
            bwd_pkts   = int(rng.exponential(scale=15))
            dst_port   = int(rng.choice([80, 443, 53, 8080, 25, 587]))

        duration = max(duration, 1)  # avoid division by zero

        records.append({
            'src_ip':             src_ip,
            'dst_ip':             dst_ip,
            'src_port':           int(rng.integers(1024, 65535)),
            'dst_port':           dst_port,
            'protocol':           int(rng.choice([6, 17, 1])),  # TCP/UDP/ICMP
            'flow_duration':      duration,
            'fwd_pkts':           fwd_pkts,
            'bwd_pkts':           bwd_pkts,
            'fwd_bytes':          fwd_bytes,
            'bwd_bytes':          bwd_bytes,
            'flow_bytes_per_sec': (fwd_bytes + bwd_bytes) / (duration / 1e6 + 1e-9),
            'flow_pkts_per_sec':  (fwd_pkts  + bwd_pkts)  / (duration / 1e6 + 1e-9),
            'flow_iat_mean':      float(rng.exponential(scale=100_000)),
            'label':              label,
        })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df):,} synthetic flows")
    print(f"  Benign flows   : {(df['label']==0).sum():,}")
    print(f"  Malicious flows: {(df['label']==1).sum():,}")

    return df


# =============================================================================
# SECTION 2: DATA CLEANING
# Handles NaN, Inf, out-of-range values, and optional self-loop removal.
# Returns a validated DataFrame safe for graph construction.
# =============================================================================

def clean_flow_dataframe(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Clean and validate the raw flow DataFrame.

    Operations performed (in order):
        1. Drop rows with NaN in any required column
        2. Replace ±Inf with NaN then drop (common in rate features)
        3. Clip numerical features at the configured percentile
        4. Drop self-loops (src_ip == dst_ip) if configured
        5. Drop nodes below the minimum flow count threshold
        6. Reset index for downstream integer indexing

    Args:
        df    : Raw flow DataFrame from load_cicids_csv
        config: PipelineConfig instance

    Returns:
        df_clean: Cleaned DataFrame
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 2: Data Cleaning")
    print("="*60)

    n_before = len(df)

    # ── Step 1: Drop NaN rows ────────────────────────────────────────────────
    df = df.dropna()
    print(f"  Dropped {n_before - len(df):,} rows with NaN values")

    # ── Step 2: Replace Inf ──────────────────────────────────────────────────
    n_before = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"  Dropped {n_before - len(df):,} rows with Inf values")

    # ── Step 3: Clip outliers at configured percentile ───────────────────────
    clip_cols = [
        'flow_duration', 'fwd_pkts', 'bwd_pkts',
        'fwd_bytes', 'bwd_bytes',
        'flow_bytes_per_sec', 'flow_pkts_per_sec', 'flow_iat_mean'
    ]
    for col in clip_cols:
        if col in df.columns:
            cap = df[col].quantile(config.outlier_clip_percentile / 100.0)
            df[col] = df[col].clip(upper=cap)

    print(f"  Clipped outliers at {config.outlier_clip_percentile}th percentile")

    # ── Step 4: Drop self-loops ──────────────────────────────────────────────
    if config.drop_self_loops:
        n_before = len(df)
        df = df[df['src_ip'] != df['dst_ip']]
        print(f"  Dropped {n_before - len(df):,} self-loop flows (src_ip == dst_ip)")

    # ── Step 5: Drop infrequent nodes ───────────────────────────────────────
    # Count how many flows each IP appears in (as either src or dst)
    all_ips_series = pd.concat([df['src_ip'], df['dst_ip']])
    ip_flow_counts = all_ips_series.value_counts()
    active_ips     = set(ip_flow_counts[ip_flow_counts >= config.min_flows_per_node].index)

    n_before = len(df)
    df = df[df['src_ip'].isin(active_ips) & df['dst_ip'].isin(active_ips)]
    print(f"  Dropped {n_before - len(df):,} flows involving low-activity nodes "
          f"(< {config.min_flows_per_node} flows)")

    # ── Step 6: Reset index ──────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    print(f"\n  Final clean flow count : {len(df):,}")
    print(f"  Unique source IPs      : {df['src_ip'].nunique():,}")
    print(f"  Unique destination IPs : {df['dst_ip'].nunique():,}")

    return df


# =============================================================================
# SECTION 3: IP → NODE INDEX MAPPING
# Every unique IP address in the cleaned flow table becomes a graph node.
# We build a bijective mapping: IP string ↔ integer index [0, N-1].
# =============================================================================

def build_node_index(df: pd.DataFrame) -> Dict[str, int]:
    """
    Build a deterministic IP-address → integer-index mapping.

    Why this matters:
        PyTorch Geometric's edge_index tensor uses integer indices [0, N-1],
        not string IP addresses. This mapping is the bridge between the
        human-readable flow table and the mathematical graph representation.

    Determinism:
        We sort IPs alphabetically before assigning indices.
        This guarantees the same IP always gets the same index,
        making the mapping reproducible across runs.

    Args:
        df: Cleaned flow DataFrame

    Returns:
        ip_to_idx: Dict mapping each IP string to a unique integer index
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 3: Node Index Construction")
    print("="*60)

    # Union of all IPs that appear as src OR dst
    all_ips  = sorted(set(df['src_ip'].tolist() + df['dst_ip'].tolist()))
    ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}

    print(f"  Total unique nodes (IPs) : {len(ip_to_idx):,}")
    print(f"  Node index range         : [0, {len(ip_to_idx)-1}]")

    # Preview first 5 mappings
    print("\n  [Sample Mapping]")
    for ip, idx in list(ip_to_idx.items())[:5]:
        print(f"    {ip}  →  node {idx}")

    return ip_to_idx


# =============================================================================
# SECTION 4: DIRECTED EDGE CONSTRUCTION
# Each flow record (src_ip, dst_ip) becomes one directed edge in the graph.
# Multiple flows between the same pair create one edge with aggregated weight.
#
# Mathematical representation:
#   edge_index ∈ ℤ^(2 × E)   — COO sparse format
#   edge_attr  ∈ ℝ^(E × D)   — per-edge feature vector
#
# COO format: edge_index[0, e] is the source node of edge e,
#             edge_index[1, e] is the destination node of edge e.
# =============================================================================

def build_edge_index(
    df: pd.DataFrame,
    ip_to_idx: Dict[str, int],
    config: PipelineConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert flow records into a directed edge_index tensor (COO format).

    Edge Aggregation Strategy:
        Multiple flows between the same (src_ip, dst_ip) pair are collapsed
        into a single directed edge. The edge features capture the aggregate
        statistics of all flows along that link. This mirrors how real network
        monitoring works — you care about the traffic pattern on a link,
        not each individual packet.

    Directional semantics:
        (A → B) and (B → A) are stored as TWO SEPARATE EDGES.
        edge_index[0] = source node indices
        edge_index[1] = target node indices
        This is the fundamental asymmetry that enables detecting:
            - Exfiltration  (large bytes flowing OUT)
            - C2 beaconing  (many small flows going OUT to one external IP)
            - Lateral spread (flows going INTO many internal machines)

    Edge Features (per directed link):
        [0] flow_count       — number of flows on this directed link
        [1] mean_fwd_bytes   — avg bytes sent in the flow direction
        [2] mean_bwd_bytes   — avg bytes received in the flow direction
        [3] mean_duration    — avg flow duration (microseconds)
        [4] mean_pkts_per_sec— avg packet rate
        [5] malicious_ratio  — fraction of flows on this link that are malicious

    Args:
        df        : Cleaned flow DataFrame
        ip_to_idx : IP-to-index mapping from build_node_index
        config    : PipelineConfig instance

    Returns:
        edge_index: LongTensor of shape [2, E]
        edge_attr : FloatTensor of shape [E, 6]
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 4: Directed Edge Construction")
    print("="*60)

    # ── Map IPs to integer indices ───────────────────────────────────────────
    df = df.copy()
    df['src_idx'] = df['src_ip'].map(ip_to_idx)
    df['dst_idx'] = df['dst_ip'].map(ip_to_idx)

    # Drop any rows where mapping failed (shouldn't happen after clean step)
    df = df.dropna(subset=['src_idx', 'dst_idx'])
    df['src_idx'] = df['src_idx'].astype(int)
    df['dst_idx'] = df['dst_idx'].astype(int)

    # ── Aggregate per directed link (src_idx, dst_idx) ───────────────────────
    print("  Aggregating flows per directed link...")

    edge_df = df.groupby(['src_idx', 'dst_idx'], sort=False).agg(
        flow_count        = ('label',              'count'),
        mean_fwd_bytes    = ('fwd_bytes',          'mean'),
        mean_bwd_bytes    = ('bwd_bytes',          'mean'),
        mean_duration     = ('flow_duration',      'mean'),
        mean_pkts_per_sec = ('flow_pkts_per_sec',  'mean'),
        malicious_ratio   = ('label',              'mean'),  # mean of 0/1 = fraction
    ).reset_index()

    print(f"  Unique directed edges    : {len(edge_df):,}")
    print(f"  Multi-flow edges         : "
          f"{(edge_df['flow_count'] > 1).sum():,} "
          f"({100*(edge_df['flow_count']>1).mean():.1f}%)")

    # ── Build edge_index tensor [2, E] ───────────────────────────────────────
    # Row 0 = source node indices, Row 1 = destination node indices
    edge_index = torch.tensor(
        np.stack([
            edge_df['src_idx'].values,
            edge_df['dst_idx'].values
        ]),
        dtype=torch.long
    )

    # ── Build edge_attr tensor [E, 6] ────────────────────────────────────────
    edge_attr_np = edge_df[[
        'flow_count',
        'mean_fwd_bytes',
        'mean_bwd_bytes',
        'mean_duration',
        'mean_pkts_per_sec',
        'malicious_ratio',
    ]].values.astype(np.float32)

    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)

    print(f"\n  edge_index shape         : {list(edge_index.shape)}")
    print(f"  edge_attr shape          : {list(edge_attr.shape)}")
    print(f"\n  [Edge Attribute Columns]")
    edge_cols = ['flow_count','mean_fwd_bytes','mean_bwd_bytes',
                 'mean_duration','mean_pkts_per_sec','malicious_ratio']
    for i, col in enumerate(edge_cols):
        print(f"    [{i}] {col}")

    return edge_index, edge_attr


# =============================================================================
# SECTION 5: NODE FEATURE ENGINEERING
# Aggregate per-flow statistics into per-node behavioral fingerprints.
#
# For each node v, we collect all flows where v is the SOURCE (out-flows)
# and all flows where v is the DESTINATION (in-flows), then engineer
# 11 features that capture both directions of behavior.
#
# Why aggregate in both directions separately?
#   - A C2 server has many IN-flows (bots calling home) but few OUT-flows
#   - An exfiltrating host has large OUT byte volumes
#   - A port scanner has high OUT unique-destination count
#   Conflating directions would destroy these discriminative signals.
# =============================================================================

def engineer_node_features(
    df: pd.DataFrame,
    ip_to_idx: Dict[str, int],
    config: PipelineConfig
) -> torch.Tensor:
    """
    Build the node feature matrix X ∈ ℝ^(N × 11).

    Feature Vector Layout for node v:
        [0]  in_degree          — unique sources sending TO v
        [1]  out_degree         — unique destinations v sends TO
        [2]  in_bytes_total     — total bytes RECEIVED by v
        [3]  out_bytes_total    — total bytes SENT by v
        [4]  bytes_ratio        — out_bytes / (in_bytes + out_bytes + ε)
                                  → high = potential exfiltration
        [5]  in_pkt_rate_mean   — mean packet/sec of flows INTO v
        [6]  out_pkt_rate_mean  — mean packet/sec of flows FROM v
        [7]  unique_dst_ports   — unique dst ports v contacts
                                  → high = port scanning behavior
        [8]  protocol_entropy   — Shannon entropy of protocol distribution
                                  → high = diverse/evasive protocol use
        [9]  avg_flow_duration  — mean duration of all flows involving v
                                  → very short = automated/scripted attacks
        [10] malicious_flow_ratio— fraction of flows involving v that are malicious
                                   (used as a soft supervisory signal in features)

    Args:
        df        : Cleaned flow DataFrame
        ip_to_idx : IP-to-index mapping
        config    : PipelineConfig

    Returns:
        X: FloatTensor of shape [N, 11], StandardScaler-normalized
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 5: Node Feature Engineering")
    print("="*60)

    N = len(ip_to_idx)
    idx_to_ip = {v: k for k, v in ip_to_idx.items()}

    # Pre-map indices into the dataframe for fast groupby
    df = df.copy()
    df['src_idx'] = df['src_ip'].map(ip_to_idx)
    df['dst_idx'] = df['dst_ip'].map(ip_to_idx)
    df = df.dropna(subset=['src_idx', 'dst_idx'])
    df['src_idx'] = df['src_idx'].astype(int)
    df['dst_idx'] = df['dst_idx'].astype(int)

    # Initialise feature matrix with zeros
    X = np.zeros((N, config.num_features), dtype=np.float32)

    # ── [0] In-degree: unique sources sending TO each node ───────────────────
    in_degree = df.groupby('dst_idx')['src_idx'].nunique()
    for node_idx, val in in_degree.items():
        X[node_idx, 0] = val

    # ── [1] Out-degree: unique destinations each node sends TO ───────────────
    out_degree = df.groupby('src_idx')['dst_idx'].nunique()
    for node_idx, val in out_degree.items():
        X[node_idx, 1] = val

    # ── [2] Total bytes RECEIVED (in-flows) ──────────────────────────────────
    in_bytes = df.groupby('dst_idx')['fwd_bytes'].sum()
    for node_idx, val in in_bytes.items():
        X[node_idx, 2] = val

    # ── [3] Total bytes SENT (out-flows) ─────────────────────────────────────
    out_bytes = df.groupby('src_idx')['fwd_bytes'].sum()
    for node_idx, val in out_bytes.items():
        X[node_idx, 3] = val

    # ── [4] Bytes ratio: out / (in + out + ε) — exfiltration indicator ───────
    total_bytes = X[:, 2] + X[:, 3]
    X[:, 4] = X[:, 3] / (total_bytes + 1e-9)

    # ── [5] Mean packet rate of in-flows ─────────────────────────────────────
    in_pkt_rate = df.groupby('dst_idx')['flow_pkts_per_sec'].mean()
    for node_idx, val in in_pkt_rate.items():
        X[node_idx, 5] = val

    # ── [6] Mean packet rate of out-flows ────────────────────────────────────
    out_pkt_rate = df.groupby('src_idx')['flow_pkts_per_sec'].mean()
    for node_idx, val in out_pkt_rate.items():
        X[node_idx, 6] = val

    # ── [7] Unique destination ports contacted (port scan signal) ────────────
    unique_dst_ports = df.groupby('src_idx')['dst_port'].nunique()
    for node_idx, val in unique_dst_ports.items():
        X[node_idx, 7] = val

    # ── [8] Protocol entropy — Shannon H(protocol distribution) ──────────────
    #   H = -Σ p_k * log2(p_k)   for each protocol k used by node v
    #   High entropy → diverse protocols → evasion or scanning behavior
    def _protocol_entropy(group: pd.Series) -> float:
        counts = group.value_counts(normalize=True)
        return float(-np.sum(counts * np.log2(counts + 1e-9)))

    proto_entropy_src = df.groupby('src_idx')['protocol'].apply(_protocol_entropy)
    for node_idx, val in proto_entropy_src.items():
        X[node_idx, 8] = val

    # ── [9] Average flow duration (all flows involving this node) ─────────────
    # Combine in-flows and out-flows for a complete picture
    all_durations = pd.concat([
        df[['src_idx', 'flow_duration']].rename(columns={'src_idx': 'node_idx'}),
        df[['dst_idx', 'flow_duration']].rename(columns={'dst_idx': 'node_idx'}),
    ])
    avg_duration = all_durations.groupby('node_idx')['flow_duration'].mean()
    for node_idx, val in avg_duration.items():
        X[node_idx, 9] = val

    print(f"  Node feature matrix shape : {X.shape}  [N × F]")
    print(f"\n  [Feature Columns]")
    feature_names = [
        'in_degree', 'out_degree', 'in_bytes_total', 'out_bytes_total',
        'bytes_ratio', 'in_pkt_rate_mean', 'out_pkt_rate_mean',
        'unique_dst_ports', 'protocol_entropy', 'avg_flow_duration'
    ]
    for i, name in enumerate(feature_names):
        print(f"    [{i:2d}] {name:25s}  "
              f"mean={X[:,i].mean():.4f}  std={X[:,i].std():.4f}")

    # ── Clip outliers in feature matrix ──────────────────────────────────────
    for f in range(config.num_features):
        cap = np.percentile(X[:, f], config.outlier_clip_percentile)
        X[:, f] = np.clip(X[:, f], a_min=None, a_max=cap)

    # ── StandardScaler normalisation ─────────────────────────────────────────
    # GNNs are sensitive to feature scale — normalise to μ=0, σ=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    print(f"\n  Applied StandardScaler normalisation")
    print(f"  Feature matrix NaN check  : {np.isnan(X_scaled).sum()} NaNs")
    print(f"  Feature matrix Inf check  : {np.isinf(X_scaled).sum()} Infs")

    # Final safety: replace any remaining NaN/Inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.tensor(X_scaled, dtype=torch.float)


# =============================================================================
# SECTION 6: NODE LABEL ASSIGNMENT
# Assign binary ground-truth labels to each node.
#
# Strategy options (set in PipelineConfig.node_label_strategy):
#   'any'      → node is Malicious if ANY flow it participates in is malicious
#   'majority' → node is Malicious if >50% of its flows are malicious
#
# 'any' is the conservative (security-focused) default:
#   False Negatives (missing an attacker) are more costly than
#   False Positives (flagging a benign node for review).
# =============================================================================

def assign_node_labels(
    df: pd.DataFrame,
    ip_to_idx: Dict[str, int],
    config: PipelineConfig
) -> torch.Tensor:
    """
    Assign binary labels {0, 1} to each node.

    Args:
        df        : Cleaned flow DataFrame
        ip_to_idx : IP-to-index mapping
        config    : PipelineConfig

    Returns:
        y: LongTensor of shape [N], values in {0, 1}
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 6: Node Label Assignment")
    print("="*60)
    print(f"  Strategy: '{config.node_label_strategy}'")

    N = len(ip_to_idx)
    y = np.zeros(N, dtype=np.int64)

    df = df.copy()
    df['src_idx'] = df['src_ip'].map(ip_to_idx)
    df['dst_idx'] = df['dst_ip'].map(ip_to_idx)
    df = df.dropna(subset=['src_idx', 'dst_idx'])
    df['src_idx'] = df['src_idx'].astype(int)
    df['dst_idx'] = df['dst_idx'].astype(int)

    # Pool labels from both src and dst perspectives
    all_node_labels = pd.concat([
        df[['src_idx', 'label']].rename(columns={'src_idx': 'node_idx'}),
        df[['dst_idx', 'label']].rename(columns={'dst_idx': 'node_idx'}),
    ])

    if config.node_label_strategy == 'any':
        # Label = 1 if the node appears in ANY malicious flow
        mal_nodes = all_node_labels[all_node_labels['label'] == 1]['node_idx'].unique()
        y[mal_nodes] = 1

    elif config.node_label_strategy == 'majority':
        # Label = 1 if >50% of flows involving this node are malicious
        mal_ratio = all_node_labels.groupby('node_idx')['label'].mean()
        for node_idx, ratio in mal_ratio.items():
            if ratio > 0.5:
                y[node_idx] = 1

    y_tensor = torch.tensor(y, dtype=torch.long)
    n_malicious = int((y_tensor == 1).sum())
    n_benign    = int((y_tensor == 0).sum())
    imbalance   = n_benign / max(n_malicious, 1)

    print(f"\n  Node label distribution:")
    print(f"    Benign    (0): {n_benign:,}  ({100*n_benign/N:.1f}%)")
    print(f"    Malicious (1): {n_malicious:,}  ({100*n_malicious/N:.1f}%)")
    print(f"    Imbalance ratio (Benign:Malicious): {imbalance:.1f}:1")

    if n_malicious == 0:
        raise ValueError(
            "No malicious nodes found! Check your label assignment strategy "
            "or verify the dataset contains attack flows."
        )

    return y_tensor


# =============================================================================
# SECTION 7: TRAIN / VAL / TEST MASK CONSTRUCTION
# Creates boolean masks over the node set for three-way stratified splitting.
#
# Why masks instead of separate graphs?
#   In transductive learning (our setting), the GNN sees the FULL graph
#   during training (including test nodes' structure and features).
#   The masks only control which nodes contribute to the LOSS function.
#   This is standard practice in graph ML — it lets the model use the
#   full graph topology for message passing, which improves embeddings.
# =============================================================================

def build_masks(
    y: torch.Tensor,
    config: PipelineConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build stratified train / val / test boolean masks.

    Stratification ensures both classes (Benign, Malicious) appear in
    each partition in proportion to their overall frequency.
    Without stratification, the tiny malicious class could be entirely
    absent from the validation or test set, making evaluation meaningless.

    Args:
        y     : Label tensor [N]
        config: PipelineConfig

    Returns:
        train_mask: BoolTensor [N], True for training nodes
        val_mask  : BoolTensor [N], True for validation nodes
        test_mask : BoolTensor [N], True for test nodes
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 7: Train / Val / Test Mask Construction")
    print("="*60)

    N = len(y)
    indices = np.arange(N)
    labels  = y.numpy()

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - config.train_ratio),
        stratify=labels if config.stratified_split else None,
        random_state=config.random_seed
    )

    # Second split: val vs test from the remaining pool
    val_size_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size_adjusted),
        stratify=labels[temp_idx] if config.stratified_split else None,
        random_state=config.random_seed
    )

    # Build boolean mask tensors
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    # Verify: no node appears in two partitions
    assert (train_mask & val_mask).sum()  == 0, "Train/Val overlap!"
    assert (train_mask & test_mask).sum() == 0, "Train/Test overlap!"
    assert (val_mask   & test_mask).sum() == 0, "Val/Test overlap!"

    def _partition_stats(mask: torch.Tensor, name: str):
        n = mask.sum().item()
        n_mal = (y[mask] == 1).sum().item()
        n_ben = (y[mask] == 0).sum().item()
        print(f"    {name:10s}: {n:5,} nodes  "
              f"[Benign={n_ben:,}, Malicious={n_mal:,}]  "
              f"({100*n/N:.1f}% of total)")

    print(f"  Stratified split: {config.stratified_split}")
    _partition_stats(train_mask, "Train")
    _partition_stats(val_mask,   "Val")
    _partition_stats(test_mask,  "Test")

    return train_mask, val_mask, test_mask


# =============================================================================
# SECTION 8: PYTORCH GEOMETRIC DATA OBJECT ASSEMBLY
# Packages all tensors into a single torch_geometric.data.Data object —
# the standard container for graph ML in PyG.
# =============================================================================

def assemble_pyg_data(
    x:          torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr:  torch.Tensor,
    y:          torch.Tensor,
    train_mask: torch.Tensor,
    val_mask:   torch.Tensor,
    test_mask:  torch.Tensor,
    ip_to_idx:  Dict[str, int],
) -> Data:
    """
    Assemble all computed tensors into a PyTorch Geometric Data object.

    The Data object is the universal currency of PyG.
    It holds all graph components and can be passed directly
    to any PyG model, dataloader, or transform.

    Stored attributes:
        data.x          : Node feature matrix         [N, F]
        data.edge_index : Directed edge indices       [2, E]  (COO format)
        data.edge_attr  : Edge feature matrix         [E, 6]
        data.y          : Node labels                 [N]
        data.train_mask : Training node mask          [N] bool
        data.val_mask   : Validation node mask        [N] bool
        data.test_mask  : Test node mask              [N] bool
        data.num_nodes  : Scalar — total node count
        data.ip_to_idx  : IP→index mapping (stored as metadata)

    Args:
        All tensors from previous pipeline steps + ip_to_idx

    Returns:
        data: torch_geometric.data.Data
    """
    print("\n" + "="*60)
    print("  PHASE 1 — STEP 8: Assembling PyG Data Object")
    print("="*60)

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = y,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        num_nodes  = x.shape[0],
    )

    # Store mapping as metadata (not a tensor — stored as Python dict)
    data.ip_to_idx = ip_to_idx

    # ── Graph Statistics ─────────────────────────────────────────────────────
    N = data.num_nodes
    E = data.edge_index.shape[1]

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │        PyG Data Object Summary          │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Nodes (N)           : {N:>15,}     │")
    print(f"  │  Edges (E, directed) : {E:>15,}     │")
    print(f"  │  Avg degree          : {E/N:>15.2f}     │")
    print(f"  │  Node features (F)   : {data.num_node_features:>15}     │")
    print(f"  │  Edge features       : {data.edge_attr.shape[1]:>15}     │")
    print(f"  │  x shape             : {str(list(data.x.shape)):>15}  │")
    print(f"  │  edge_index shape    : {str(list(data.edge_index.shape)):>15}  │")
    print(f"  │  edge_attr shape     : {str(list(data.edge_attr.shape)):>15}  │")
    print(f"  │  y shape             : {str(list(data.y.shape)):>15}  │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Train nodes         : {data.train_mask.sum().item():>15,}     │")
    print(f"  │  Val nodes           : {data.val_mask.sum().item():>15,}     │")
    print(f"  │  Test nodes          : {data.test_mask.sum().item():>15,}     │")
    print(f"  └─────────────────────────────────────────┘")

    # ── Validation checks ────────────────────────────────────────────────────
    print(f"\n  [Validation Checks]")
    assert data.x.shape[0] == N, "x row count != num_nodes"
    assert data.y.shape[0] == N, "y length != num_nodes"
    assert data.edge_index.max() < N, "edge_index contains out-of-range node index"
    assert not torch.isnan(data.x).any(), "NaN detected in node features"
    assert not torch.isinf(data.x).any(), "Inf detected in node features"
    print(f"  ✓  x shape valid")
    print(f"  ✓  edge_index in range [0, {N-1}]")
    print(f"  ✓  No NaN or Inf in feature matrix")
    print(f"  ✓  Masks are disjoint and complete")

    return data


# =============================================================================
# SECTION 9: MASTER PIPELINE RUNNER
# Single function that orchestrates all steps and returns the PyG Data object.
# This is the only function Phase 2 (model building) needs to call.
# =============================================================================

def run_etl_pipeline(config: Optional[PipelineConfig] = None) -> Data:
    """
    Execute the complete ETL + Graph Construction pipeline.

    This is the single entry point for Phase 1. Call this function and
    receive a fully constructed, validated PyG Data object ready for
    Phase 3's GNN model.

    Args:
        config: PipelineConfig. If None, uses defaults (synthetic data mode).

    Returns:
        data: torch_geometric.data.Data — the graph dataset
    """
    if config is None:
        config = PipelineConfig()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   CYBER ATTACK DETECTION — GNN Phase 1: ETL Pipeline   ║")
    print("╚" + "═"*58 + "╝")

    # Set global random seed for full reproducibility
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    df_raw = load_cicids_csv(config)

    # ── Step 2: Clean and validate ───────────────────────────────────────────
    df_clean = clean_flow_dataframe(df_raw, config)

    # ── Step 3: Build IP → node index mapping ────────────────────────────────
    ip_to_idx = build_node_index(df_clean)

    # ── Step 4: Construct directed edges ──────────────────────────────────────
    edge_index, edge_attr = build_edge_index(df_clean, ip_to_idx, config)

    # ── Step 5: Engineer node features ───────────────────────────────────────
    x = engineer_node_features(df_clean, ip_to_idx, config)

    # ── Step 6: Assign node labels ───────────────────────────────────────────
    y = assign_node_labels(df_clean, ip_to_idx, config)

    # ── Step 7: Build train/val/test masks ────────────────────────────────────
    train_mask, val_mask, test_mask = build_masks(y, config)

    # ── Step 8: Assemble PyG Data object ──────────────────────────────────────
    data = assemble_pyg_data(
        x, edge_index, edge_attr, y,
        train_mask, val_mask, test_mask,
        ip_to_idx
    )

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   Phase 1 Complete. PyG Data object is ready.          ║")
    print("║   Pass `data` to Phase 2 (Dataset) and Phase 3 (GNN)  ║")
    print("╚" + "═"*58 + "╝\n")

    return data


# =============================================================================
# SECTION 10: UTILITY — SAVE / LOAD DATA OBJECT
# =============================================================================

def save_graph_data(data: Data, output_path: str = "graph_data.pt"):
    """
    Persist the PyG Data object to disk using torch.save.

    Saves the fully constructed graph — avoids re-running the
    entire ETL pipeline every time you restart a training session.

    Args:
        data       : PyG Data object from run_etl_pipeline
        output_path: File path for the saved .pt file
    """
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Graph saved to '{output_path}'  ({size_mb:.2f} MB)")


def load_graph_data(input_path: str = "graph_data.pt") -> Data:
    """
    Load a previously saved PyG Data object from disk.

    Args:
        input_path: Path to the .pt file saved by save_graph_data

    Returns:
        data: PyG Data object
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No saved graph found at '{input_path}'")

    data = torch.load(input_path, weights_only=False)
    print(f"  Graph loaded from '{input_path}'")
    print(f"  Nodes: {data.num_nodes:,}  |  Edges: {data.edge_index.shape[1]:,}")
    return data


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":

    # ── Option A: Synthetic data (no download required, runs immediately) ─────
    config = PipelineConfig(
        data_path="data",                 # downloaded CICIDS CSV folder
        load_all_csvs_in_folder=True,
        min_flows_per_node=3,
        node_label_strategy='any',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    # ── Option B: Real CICIDS 2017 data ───────────────────────────────────────
    # Uncomment and set your path after downloading:
    #
    # config = PipelineConfig(
    #     data_path="/path/to/cicids2017/",   # folder containing day CSVs
    #     load_all_csvs_in_folder=True,
    #     min_flows_per_node=5,
    #     node_label_strategy='any',
    #     train_ratio=0.70,
    #     val_ratio=0.15,
    #     test_ratio=0.15,
    #     random_seed=42,
    # )

    # ── Run the pipeline ──────────────────────────────────────────────────────
    data = run_etl_pipeline(config)

    # ── Save for reuse in Phase 3/4 ──────────────────────────────────────────
    save_graph_data(data, "cyber_graph_data.pt")

    # ── Quick preview of the assembled tensors ────────────────────────────────
    print("\n[Final Tensor Preview]")
    print(f"  data.x          : {data.x.shape}   dtype={data.x.dtype}")
    print(f"  data.edge_index : {data.edge_index.shape}  dtype={data.edge_index.dtype}")
    print(f"  data.edge_attr  : {data.edge_attr.shape}   dtype={data.edge_attr.dtype}")
    print(f"  data.y          : {data.y.shape}   dtype={data.y.dtype}")
    print(f"  data.train_mask : {data.train_mask.sum().item()} True values")
    print(f"  data.val_mask   : {data.val_mask.sum().item()} True values")
    print(f"  data.test_mask  : {data.test_mask.sum().item()} True values")

    # ── Confirm it's loadable for Phase 3 ────────────────────────────────────
    print("\n[Load Verification]")
    data_reloaded = load_graph_data("cyber_graph_data.pt")
    assert data_reloaded.num_nodes == data.num_nodes
    print("  ✓  Save/load verified. Phase 1 is complete.")
    print(f"\n  ➜  Pass `data` to phase3_model.py → build_model(data)")
