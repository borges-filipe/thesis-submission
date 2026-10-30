from __future__ import annotations

from pathlib import Path

import numpy as np

from hallucination_eval.data.ab_logits import load_binary_logits, load_binary_results
from hallucination_eval.data.mcq_logits import load_mcq_results
from hallucination_eval.decoding import confidence_from_logits, vcd_binary
from hallucination_eval.metrics.classification import binary_metrics


def binary_report(path: str | Path, threshold: float = 0.5) -> dict:
    """Load an A/B result JSON and return the full binary metric report."""
    y_true, y_score = load_binary_results(path)
    return binary_metrics(y_true, y_score, threshold)


def mcq_report(path: str | Path, threshold: float = 0.5) -> dict:
    """Load an MCQ result JSON and return the full binary metric report."""
    y_true, y_score = load_mcq_results(path)
    return binary_metrics(y_true, y_score, threshold)


def vcd_binary_report(
    original_path: str | Path,
    distorted_path: str | Path,
    alpha: float = 1.0,
    beta: float = 0.1,
    threshold: float = 0.5,
) -> dict:
    """Apply binary VCD+APC to two A/B logit files"""
    original = load_binary_logits(original_path)
    distorted = {d.get("question_id"): d for d in load_binary_logits(distorted_path)}

    y_true: list[int] = []
    y_score: list[float] = []
    for item in original:
        dist_item = distorted.get(item.get("question_id"))
        label = item.get("label")
        if dist_item is None or label is None:
            continue
        vcd_logits = vcd_binary(item.get("logits", {}), dist_item.get("logits", {}), alpha, beta)
        conf = confidence_from_logits(vcd_logits)
        y_true.append(1 if str(label).strip().lower() == "no" else 0)
        y_score.append(1.0 - conf)

    return binary_metrics(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), threshold)


def vcd_black_report(
    original_path: str | Path,
    black_path: str | Path,
    alpha: float = 1.0,
    beta: float = 0.1,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """VCD with black image"""
    return {
        "baseline": binary_report(original_path, threshold),
        "black_image": vcd_binary_report(original_path, black_path, alpha, beta, threshold),
    }
