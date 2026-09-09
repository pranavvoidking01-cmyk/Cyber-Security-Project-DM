import argparse
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_gnn import HybridGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "pr_auc": average_precision_score(labels, probabilities),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def run_cross_validation(
    graph_path: str = "cyber_graph_data.pt",
    folds: int = 5,
    epochs: int = 40,
    seed: int = 42,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(graph_path, map_location=device, weights_only=False).to(device)
    labels = data.y.cpu().numpy()
    indices = np.arange(data.num_nodes)
    positive_count = int(labels.sum())
    if positive_count < folds:
        raise ValueError(
            f"Need at least {folds} malicious nodes for {folds}-fold CV; found {positive_count}."
        )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results = []
    print(f"Device: {device} | {folds}-fold stratified cross-validation")

    for fold, (train_validation_idx, test_idx) in enumerate(splitter.split(indices, labels), 1):
        train_idx, validation_idx = train_test_split(
            train_validation_idx,
            test_size=0.15,
            stratify=labels[train_validation_idx],
            random_state=seed + fold,
        )
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        validation_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        validation_mask[validation_idx] = True
        test_mask[test_idx] = True

        train_labels = data.y[train_mask]
        positives = int(train_labels.sum().item())
        negatives = int(train_labels.numel() - positives)
        model = HybridGNN(data.num_node_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / max(positives, 1)], device=device)
        )
        best_score = -float("inf")
        best_state = None

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[train_mask], data.y.float()[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_probabilities = torch.sigmoid(model(data.x, data.edge_index)[validation_mask]).cpu().numpy()
            validation_labels = data.y[validation_mask].cpu().numpy()
            score = average_precision_score(validation_labels, validation_probabilities)
            if score > best_score:
                best_score = score
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_probabilities = torch.sigmoid(model(data.x, data.edge_index)[test_mask]).cpu().numpy()
        result = metrics(data.y[test_mask].cpu().numpy(), test_probabilities)
        fold_results.append(result)
        print(
            f"Fold {fold}: test malicious={int(labels[test_idx].sum())} | "
            f"PR-AUC={result['pr_auc']:.4f} | F1={result['f1']:.4f} | "
            f"Precision={result['precision']:.4f} | Recall={result['recall']:.4f}"
        )

    print("\nCross-validation mean +/- standard deviation")
    for name in ("pr_auc", "f1", "precision", "recall"):
        values = [result[name] for result in fold_results]
        print(f"{name:10s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stratified k-fold GNN evaluation.")
    parser.add_argument("--graph", default="cyber_graph_data.pt")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    run_cross_validation(args.graph, args.folds, args.epochs)
