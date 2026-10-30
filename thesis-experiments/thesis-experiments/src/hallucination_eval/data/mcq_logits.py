from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_mcq_results(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an MCQ logit result JSON """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    values = results.values() if isinstance(results, dict) else results

    y_true: list[int] = []
    y_score: list[float] = []

    for value in values:
        if isinstance(value, list) and value:
            item = value[0]
        elif isinstance(value, dict):
            item = value
        else:
            continue

        gt_labels = item.get("ground_truth_labels", {}) or {}
        confidences = item.get("confidence_visible", {}) or {}
        if not isinstance(confidences, dict):
            continue

        for obj, conf in confidences.items():
            if obj not in gt_labels:
                continue
            label = str(gt_labels[obj]).strip().lower()
            y_true.append(1 if label == "no" else 0)
            y_score.append(1.0 - float(conf))

    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)
