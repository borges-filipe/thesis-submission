from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def load_se_pairs(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_correct, scores) from a semantic entropy jsonl file."""
    y_correct: list[bool] = []
    scores: list[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for qa in data.get("qa_json_output", []):
                if "is_correct" in qa and "semantic_entropy_score" in qa:
                    ent = qa["semantic_entropy_score"]
                    if ent is not None:
                        y_correct.append(bool(qa["is_correct"]))
                        scores.append(float(ent))
    return np.asarray(y_correct, dtype=bool), np.asarray(scores, dtype=float)


def se_report(path: str | Path) -> dict:
    """Full uncertainty-detection report for semantic entropy"""
    y_correct, scores = load_se_pairs(path)
    n = len(y_correct)
    if n == 0:
        raise ValueError(f"no (is_correct, semantic_entropy_score) pairs in {path}")

    y_incorrect = (~y_correct).astype(int)

    auroc = float(roc_auc_score(y_incorrect, scores))
    ap = float(average_precision_score(y_incorrect, scores))

    precisions, recalls, thresholds = precision_recall_curve(y_incorrect, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precisions * recalls / (precisions + recalls)
    f1 = np.nan_to_num(f1, nan=0.0)
    opt_idx = int(np.argmax(f1[:-1]))
    optimal_threshold = float(thresholds[opt_idx])

    # AURAC: keep samples by ascending entropy; accuracy = fraction correct.
    order = np.argsort(scores)
    sorted_correct = y_correct[order].astype(float)
    k = np.arange(1, n + 1)
    accuracies = np.cumsum(sorted_correct) / k
    aurac = float(np.mean(accuracies))

    # Optimal-threshold rejection stats (kept = score < threshold).
    kept_mask = scores < optimal_threshold
    num_kept = int(np.sum(kept_mask))
    opt_rejection = 1.0 - num_kept / n
    opt_accuracy = float(np.mean(y_correct[kept_mask])) if num_kept > 0 else 0.0

    return {
        "n": n,
        "auroc": auroc,
        "ap": ap,
        "aurac": aurac,
        "median_se": float(np.median(scores)),
        "baseline_accuracy": float(np.mean(y_correct)),
        "optimal_threshold": optimal_threshold,
        "optimal_f1": float(f1[opt_idx]),
        "optimal_rejection": opt_rejection,
        "optimal_accuracy": opt_accuracy,
    }
