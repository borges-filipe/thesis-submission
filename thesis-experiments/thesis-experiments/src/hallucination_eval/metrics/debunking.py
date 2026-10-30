from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SPLITS = ("random", "popular", "adversarial")

# (name, relative subdir under {model}/{dataset}, is_float)
STRATEGIES = (
    ("regular", "full_answers/regular_decoding_temp_1", False),
    ("vcd", "full_answers/vcd", False),
    ("apc", "full_answers/apc", False),
    ("greedy", "full_answers/regular_decoding_temp_0", False),
    ("binary_decoding", "binary/binary_decoding", True),
    ("binary_vcd", "binary/binary_vcd", True),
)


def load_predictions(directory: str | Path, splits=SPLITS, is_float: bool = False):
    """Load a strategy directory and return (y_true_visible, y_pred_visible)."""
    directory = Path(directory)
    y_true: list[int] = []
    y_pred: list[int] = []

    for split in splits:
        if not directory.is_dir():
            continue
        matches = [
            f for f in os.listdir(directory)
            if split in f and f.endswith(".json")
        ]
        if not matches:
            continue
        with open(directory / matches[0], encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("results", []):
            label = str(item.get("label", "")).strip().lower()
            y_true.append(1 if label == "yes" else 0)
            if is_float:
                conf = item.get("confidence_visible", 0.0)
                y_pred.append(1 if conf >= 0.5 else 0)
            else:
                y_pred.append(1 if bool(item.get("confidence_visible", False)) else 0)

    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def f1_and_yes_ratio(y_true, y_pred) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    yes_ratio = (tp + fp) / len(y_true) if len(y_true) else 0.0
    return f1, yes_ratio


def debunking_report(results_base: str | Path, splits=SPLITS) -> dict[str, dict]:
    """F1 and yes-ratio for all six strategies under ``{model}/{dataset}``."""
    results_base = Path(results_base)
    out: dict[str, dict] = {}
    for name, rel, is_float in STRATEGIES:
        y_true, y_pred = load_predictions(results_base / rel, splits, is_float)
        f1, yes_ratio = f1_and_yes_ratio(y_true, y_pred)
        out[name] = {"f1": f1, "yes_ratio": yes_ratio, "n": int(len(y_true))}
    return out
