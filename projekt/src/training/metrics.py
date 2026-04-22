from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, object]:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0/1)
    y_pred : array-like
        Predicted labels (0/1)
    y_prob : array-like, optional
        Predicted probabilities for positive class

    Returns
    -------
    dict
        accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def print_metrics(metrics: Dict[str, object], title: str = "Metrics") -> None:
    """
    Pretty print metrics to console.
    """
    print(f"\n=== {title} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    roc_auc = metrics.get("roc_auc", float("nan"))
    if isinstance(roc_auc, float):
        print(f"ROC AUC  : {roc_auc:.4f}")

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def compare_metrics(results: Dict[str, Dict[str, object]]) -> None:
    """
    Print side-by-side comparison of multiple experiment results.

    Parameters
    ----------
    results : dict
        {experiment_name: metrics_dict}
    """
    headers = ["Experiment", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    print(f"{headers[0]:<20} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>10} {headers[5]:>10}")
    print("-" * 76)

    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics.get('roc_auc', float('nan')):>10.4f}"
        )