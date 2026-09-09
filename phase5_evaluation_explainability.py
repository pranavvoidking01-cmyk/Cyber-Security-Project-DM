import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import k_hop_subgraph

from train_gnn import HybridGNN


DEFAULT_MODEL = "best_cyber_gnn.pth"
DEFAULT_GRAPH = "cyber_graph_data.pt"
DEFAULT_OUTPUT = "explainability_report.png"
DEFAULT_JSON_OUTPUT = "forensic_graph.json"
FEATURE_NAMES = [
    "In-degree", "Out-degree", "In-bytes", "Out-bytes", "Bytes ratio",
    "In packet rate", "Out packet rate", "Unique destination ports",
    "Protocol entropy", "Flow duration",
]


class AttentionHybridGNN(HybridGNN):
    def forward_with_attention(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=False)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=False)
        x, attention = self.conv3(
            x, edge_index, return_attention_weights=True
        )
        x = F.elu(x)
        logits = self.classifier(x).squeeze(-1)
        return logits, attention[0], attention[1]


def load_artifacts(
    model_path: str, graph_path: str, device: torch.device
) -> tuple[AttentionHybridGNN, Data]:
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run train_gnn.py first."
        )

    data = torch.load(graph_path, map_location=device, weights_only=False)
    model = AttentionHybridGNN(data.num_node_features).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, data.to(device)


def evaluate_test_set(data: Data, logits: torch.Tensor) -> dict[str, Any]:
    mask = data.test_mask
    labels = data.y[mask].cpu().numpy()
    probabilities = torch.sigmoid(logits[mask]).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics: dict[str, Any] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float((predictions == labels).mean()),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "pr_auc": (
            average_precision_score(labels, probabilities)
            if np.unique(labels).size == 2 else float("nan")
        ),
    }
    print("\nTest metrics")
    for name in ("accuracy", "precision", "recall", "f1", "pr_auc"):
        print(f"  {name:10s}: {metrics[name]:.4f}")
    print(f"  confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\n" + classification_report(
        labels, predictions, labels=[0, 1],
        target_names=["Benign", "Malicious"], zero_division=0,
    ))
    return metrics


def select_target(data: Data, logits: torch.Tensor) -> int:
    probabilities = torch.sigmoid(logits).cpu()
    test_indices = data.test_mask.cpu().nonzero(as_tuple=True)[0]
    true_positives = test_indices[(data.y.cpu()[test_indices] == 1) &
                                  (probabilities[test_indices] >= 0.5)]
    candidates = true_positives
    if len(candidates) == 0:
        candidates = test_indices[data.y.cpu()[test_indices] == 1]
    if len(candidates) == 0:
        candidates = data.y.cpu().nonzero(as_tuple=True)[0]
    if len(candidates) == 0:
        raise ValueError("The graph contains no malicious nodes.")
    target = candidates[probabilities[candidates].argmax()].item()
    print(f"Selected target node: {target} (P(malicious)={probabilities[target]:.4f})")
    return target


def extract_attention_subgraph(
    data: Data,
    target: int,
    attention_edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    hops: int,
) -> dict[str, Any]:
    subset, sub_edges, mapping, _ = k_hop_subgraph(
        target, hops, data.edge_index, relabel_nodes=True,
        num_nodes=data.num_nodes, directed=True,
    )
    attention = attention_weights.mean(dim=-1).cpu().numpy()
    lookup: dict[tuple[int, int], float] = {}
    for index in range(attention_edge_index.shape[1]):
        key = (attention_edge_index[0, index].item(), attention_edge_index[1, index].item())
        lookup[key] = float(attention[index])

    edge_attention = np.zeros(sub_edges.shape[1], dtype=np.float32)
    for index in range(sub_edges.shape[1]):
        key = (subset[sub_edges[0, index]].item(), subset[sub_edges[1, index]].item())
        edge_attention[index] = lookup.get(key, 0.0)

    return {
        "subset": subset.cpu().numpy(),
        "edges": sub_edges.cpu().numpy(),
        "labels": data.y[subset].cpu().numpy(),
        "features": data.x[subset].cpu().numpy(),
        "attention": edge_attention,
        "target_local": mapping.item(),
        "target_global": target,
    }


def export_forensic_graph(
    subgraph: dict[str, Any],
    data: Data,
    logits: torch.Tensor,
    metrics: dict[str, Any],
    output_path: str,
    hops: int,
) -> None:
    """Export the selected subgraph in the format used by the frontend."""
    ip_to_idx = getattr(data, "ip_to_idx", {})
    idx_to_ip = {index: ip for ip, index in ip_to_idx.items()}
    probabilities = torch.sigmoid(logits).cpu().numpy()
    subset = subgraph["subset"]
    labels = subgraph["labels"]
    features = subgraph["features"]

    nodes = []
    for local_index, global_index in enumerate(subset):
        node_id = str(idx_to_ip.get(int(global_index), global_index))
        nodes.append({
            "id": node_id,
            "node_index": int(global_index),
            "label": "Malicious" if labels[local_index] == 1 else "Benign",
            "prediction": "Malicious" if probabilities[global_index] >= 0.5 else "Benign",
            "attack_confidence": float(probabilities[global_index]),
            "features": {
                name: float(features[local_index, feature_index])
                for feature_index, name in enumerate(FEATURE_NAMES)
            },
        })

    attention = subgraph["attention"]
    max_attention = float(attention.max()) if len(attention) else 0.0
    edges = []
    for edge_index in range(subgraph["edges"].shape[1]):
        source_local = int(subgraph["edges"][0, edge_index])
        target_local = int(subgraph["edges"][1, edge_index])
        source_global = int(subset[source_local])
        target_global = int(subset[target_local])
        score = float(attention[edge_index])
        edges.append({
            "source": str(idx_to_ip.get(source_global, source_global)),
            "target": str(idx_to_ip.get(target_global, target_global)),
            "attention_score": score,
            "attention_normalized": score / (max_attention + 1e-9),
        })

    payload = {
        "target_node": str(idx_to_ip.get(subgraph["target_global"], subgraph["target_global"])),
        "hops": int(hops),
        "nodes": nodes,
        "edges": edges,
        "metrics": {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        },
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    print(f"Frontend graph JSON saved to: {output_path}")


def plot_report(
    subgraph: dict[str, Any], metrics: dict[str, Any], data: Data,
    logits: torch.Tensor, output_path: str,
) -> None:
    subset = subgraph["subset"]
    edges = subgraph["edges"]
    attention = subgraph["attention"]
    target_local = subgraph["target_local"]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(subset)))
    edge_rows = np.argsort(attention)[::-1][:40]
    for index in edge_rows:
        graph.add_edge(int(edges[0, index]), int(edges[1, index]), weight=float(attention[index]))

    position = nx.spring_layout(graph, seed=42, weight="weight")
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    figure.suptitle("Cyber GNN Explainability Report", fontsize=16, fontweight="bold")

    colors = ["#e74c3c" if label == 1 else "#4285f4" for label in subgraph["labels"]]
    colors[target_local] = "#ff1744"
    sizes = [900 if index == target_local else 350 for index in range(len(subset))]
    widths = [1 + 5 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, position, ax=axes[0], arrows=True, width=widths, alpha=0.7,
                           edge_color="#e67e22", arrowsize=12)
    nx.draw_networkx_nodes(graph, position, ax=axes[0], node_color=colors, node_size=sizes)
    nx.draw_networkx_labels(graph, position, ax=axes[0],
                            labels={target_local: f"TARGET\n{subset[target_local]}"},
                            font_color="white", font_weight="bold")
    axes[0].set_title(f"2-hop attention subgraph ({len(graph)} nodes)")
    axes[0].axis("off")
    axes[0].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff1744", label="Malicious", markersize=10),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4285f4", label="Benign", markersize=10),
    ], loc="lower left")

    target_features = subgraph["features"][target_local]
    axes[1].barh(np.arange(len(FEATURE_NAMES)), target_features, color="#e67e22")
    axes[1].set_yticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES, fontsize=8)
    axes[1].set_title("Target node feature fingerprint")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].axis("off")
    probability = float(torch.sigmoid(logits[subgraph["target_global"]]).cpu())
    summary = (
        f"Target node: {subgraph['target_global']}\n"
        f"P(malicious): {probability:.2%}\n\n"
        f"Test PR-AUC: {metrics['pr_auc']:.4f}\n"
        f"Test F1: {metrics['f1']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n\n"
        f"Confusion matrix\n"
        f"TN={metrics['tn']}  FP={metrics['fp']}\n"
        f"FN={metrics['fn']}  TP={metrics['tp']}"
    )
    axes[2].text(0.05, 0.95, summary, va="top", fontsize=12,
                 bbox={"boxstyle": "round", "facecolor": "#f4f6f7", "edgecolor": "#bdc3c7"})
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Explainability report saved to: {output_path}")


def run_phase5(
    model_path: str,
    graph_path: str,
    output_path: str,
    json_path: str,
    hops: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = load_artifacts(model_path, graph_path, device)
    with torch.no_grad():
        logits, attention_edges, attention_weights = model.forward_with_attention(
            data.x, data.edge_index
        )
    metrics = evaluate_test_set(data, logits)
    target = select_target(data, logits)
    subgraph = extract_attention_subgraph(
        data, target, attention_edges.cpu(), attention_weights.cpu(), hops
    )
    export_forensic_graph(subgraph, data, logits, metrics, json_path, hops)
    plot_report(subgraph, metrics, data, logits, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and explain the trained cyber GNN.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--json", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()
    run_phase5(args.model, args.graph, args.output, args.json, args.hops)
