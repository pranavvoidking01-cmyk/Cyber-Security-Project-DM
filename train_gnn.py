import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv


class HybridGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.conv3 = GATConv(hidden_channels // 2, 16, heads=2, concat=True)
        self.classifier = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.classifier(x).squeeze(-1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = {
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }
    if np.unique(labels).size == 2:
        metrics["pr_auc"] = average_precision_score(labels, probabilities)
    else:
        metrics["pr_auc"] = float("nan")
    return metrics


def train_and_evaluate(
    data_path: str = "cyber_graph_data.pt",
    model_path: str = "best_cyber_gnn.pth",
    epochs: int = 150,
    patience: int = 20,
    seed: int = 42,
) -> None:
    _set_seed(seed)
    data = torch.load(data_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    train_labels = data.y[data.train_mask]
    positive_count = int((train_labels == 1).sum().item())
    negative_count = int((train_labels == 0).sum().item())
    if positive_count == 0:
        raise ValueError("The training mask contains no malicious nodes.")

    pos_weight_value = negative_count / positive_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = HybridGNN(data.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    y_float = data.y.float()

    print(f"Device: {device}")
    print(f"Training labels: {negative_count} benign, {positive_count} malicious")
    print(f"Training pos_weight: {pos_weight_value:.2f}")

    best_val_pr_auc = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], y_float[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(data.x, data.edge_index)
            validation_loss = criterion(
                validation_logits[data.val_mask], y_float[data.val_mask]
            )
            validation_labels = data.y[data.val_mask].cpu().numpy()
            validation_probabilities = torch.sigmoid(
                validation_logits[data.val_mask]
            ).cpu().numpy()

        metrics = _binary_metrics(validation_labels, validation_probabilities)
        validation_pr_auc = metrics["pr_auc"]
        score_for_stopping = (
            validation_pr_auc
            if not np.isnan(validation_pr_auc)
            else metrics["f1"]
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | "
                f"Val Loss: {validation_loss.item():.4f} | "
                f"Val F1: {metrics['f1']:.4f} | Val PR-AUC: {validation_pr_auc:.4f}"
            )

        if score_for_stopping > best_val_pr_auc:
            best_val_pr_auc = score_for_stopping
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)
        test_labels = data.y[data.test_mask].cpu().numpy()
        test_probabilities = torch.sigmoid(test_logits[data.test_mask]).cpu().numpy()

    test_metrics = _binary_metrics(test_labels, test_probabilities)
    test_predictions = (test_probabilities >= 0.5).astype(np.int64)
    print("\nFinal test evaluation")
    print(f"Test PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"Test F1:      {test_metrics['f1']:.4f}")
    print(f"Test precision:{test_metrics['precision']:.4f}")
    print(f"Test recall:   {test_metrics['recall']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            zero_division=0,
        )
    )
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    train_and_evaluate()
