from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_binary_results(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an A/B logit result JSON"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    y_true: list[int] = []
    y_score: list[float] = []

    for item in results:
        label = item.get("label")
        conf = item.get("confidence_visible")
        if label is None or conf is None:
            continue
        y_true.append(1 if str(label).strip().lower() == "no" else 0)
        y_score.append(1.0 - float(conf))

    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)


def load_binary_logits(path: str | Path) -> list[dict]:
    """Return the raw results list (each item has label, logits)"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])
