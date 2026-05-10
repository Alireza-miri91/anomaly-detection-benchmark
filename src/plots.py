from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay


def plot_precision_recall(
    labels: np.ndarray,
    method_scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, scores in method_scores.items():
        PrecisionRecallDisplay.from_predictions(labels, scores, name=method_name, ax=ax)

    ax.set_title("Precision-Recall Curves")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_distribution(
    labels: np.ndarray,
    method_scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    method_count = len(method_scores)
    fig, axes = plt.subplots(method_count, 1, figsize=(8, 2.7 * method_count), sharex=False)

    if method_count == 1:
        axes = [axes]

    for ax, (method_name, scores) in zip(axes, method_scores.items()):
        ax.hist(scores[labels == 0], bins=30, alpha=0.70, label="Normal")
        ax.hist(scores[labels == 1], bins=30, alpha=0.70, label="Anomaly")
        ax.set_title(method_name)
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
