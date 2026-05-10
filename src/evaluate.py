from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    top_indices = np.argsort(scores)[::-1][:k]
    return float(labels[top_indices].mean())


def evaluate_scores(
    labels: np.ndarray,
    method_scores: dict[str, np.ndarray],
    k: int | None = None,
) -> dict[str, dict[str, float]]:
    k = int(labels.sum()) if k is None else k
    metrics: dict[str, dict[str, float]] = {}

    for method_name, scores in method_scores.items():
        metrics[method_name] = {
            "pr_auc": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "precision_at_k": precision_at_k(labels, scores, k),
        }

    return metrics
