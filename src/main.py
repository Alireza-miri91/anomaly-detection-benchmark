from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

from data import make_synthetic_clinical_data
from evaluate import evaluate_scores
from models import run_anomaly_methods
from plots import plot_precision_recall, plot_score_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic anomaly-detection benchmark.")
    parser.add_argument("--n-samples", type=int, default=900, help="Total number of synthetic rows.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Synthetic anomaly fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    print("\nAnomaly detection benchmark")
    print("Method                 PR-AUC   ROC-AUC  Precision@k")
    print("-" * 55)

    for method_name, values in metrics.items():
        print(
            f"{method_name:<22}"
            f"{values['pr_auc']:.3f}    "
            f"{values['roc_auc']:.3f}    "
            f"{values['precision_at_k']:.3f}"
        )


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    data, labels = make_synthetic_clinical_data(
        n_samples=args.n_samples,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    method_scores = run_anomaly_methods(
        data,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    metrics = evaluate_scores(labels, method_scores)

    payload = {
        "dataset": {
            "source": "synthetic",
            "n_samples": args.n_samples,
            "n_anomalies": int(labels.sum()),
            "contamination": args.contamination,
            "random_state": args.random_state,
        },
        "metrics": metrics,
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plot_precision_recall(labels, method_scores, OUTPUT_DIR / "precision_recall.png")
    plot_score_distribution(labels, method_scores, OUTPUT_DIR / "score_distribution.png")

    print_metrics(metrics)
    print(f"\nSaved metrics to {metrics_path.relative_to(PROJECT_ROOT)}")
    print("Saved plots to outputs/precision_recall.png and outputs/score_distribution.png")


if __name__ == "__main__":
    main()
