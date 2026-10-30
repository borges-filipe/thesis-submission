from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    return {
        "TP": int(np.sum(y_pred & y_true)),
        "FP": int(np.sum(y_pred & ~y_true)),
        "TN": int(np.sum(~y_pred & ~y_true)),
        "FN": int(np.sum(~y_pred & y_true)),
    }


def metrics_at_threshold(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    c = confusion(y_true, y_score >= threshold)
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    n = len(y_true)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "yes_ratio": (tn + fn) / n if n else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def optimal_f1_threshold(y_true, y_score) -> tuple[float, float]:
    """Threshold (on y_score) maximizing the F1 for the positive class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precisions * recalls / (precisions + recalls)
    f1 = np.nan_to_num(f1, nan=0.0)
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx]), float(f1[idx])


def aurac(y_true, y_score) -> float:
    """Area under the rejection-accuracy curve"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    correct = (y_true[order] == 0).astype(float)
    k = np.arange(1, len(correct) + 1)
    accuracies = np.cumsum(correct) / k
    return float(np.mean(accuracies))


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Full metric report for a binary hallucination detector."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    report = metrics_at_threshold(y_true, y_score, threshold)
    report["roc_auc"] = float(roc_auc_score(y_true, y_score))
    report["ap"] = float(average_precision_score(y_true, y_score))
    report["aurac"] = aurac(y_true, y_score)
    report["optimal_threshold"], report["optimal_f1"] = optimal_f1_threshold(y_true, y_score)
    report["baseline_accuracy"] = float(np.mean(y_true == 0))
    return report
