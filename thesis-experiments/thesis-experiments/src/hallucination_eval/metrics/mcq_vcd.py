from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from hallucination_eval.decoding import NEG_INF, mcq_confidence_visible, sigmoid, vcd_mcq

NOTA = "None of the above"


def _level_files(directory: Path) -> dict[float, str]:

    files: dict[float, str] = {}
    for f in directory.glob("MCQ_logits_level*.json"):
        m = re.search(r"level_(\d+(?:\.\d+)?)\.json$", f.name)
        if not m:
            continue
        level = float(m.group(1))
        if level not in files:
            files[level] = str(f)
        else:
            old_dup = "level_level_" in Path(files[level]).name
            new_dup = "level_level_" in f.name
            if old_dup and not new_dup:
                files[level] = str(f)
    return files


def load_mcq_data(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pairwise_confidence(logit_obj: float, logit_nota: float) -> float:
    """``sigmoid(logit_obj - logit_nota)`` with the -inf APC convention."""
    if logit_obj == NEG_INF and logit_nota == NEG_INF:
        return 0.5
    if logit_obj == NEG_INF:
        return 0.0
    if logit_nota == NEG_INF:
        return 1.0
    return sigmoid(logit_obj - logit_nota)


def mcq_vcd_extract(
    original_data: dict,
    distorted_data: dict,
    alpha: float = 1.0,
    beta: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply VCD to original + distorted MCQ logits and extract (y_true_yes, y_prob_yes)"""
    nota = original_data.get("none_of_the_above_label", NOTA)
    orig_results = original_data.get("results", {})
    dist_results = distorted_data.get("results", {})

    y_true_yes: list[int] = []
    y_prob_yes: list[float] = []

    for img, og_groups in orig_results.items():
        dist_groups = dist_results.get(img)
        if not dist_groups:
            continue
        for og_group, dist_group in zip(og_groups, dist_groups):
            vcd_logits = vcd_mcq(
                og_group.get("average_logits_by_answer", {}),
                dist_group.get("average_logits_by_answer", {}),
                alpha,
                beta,
            )
            conf_visible = mcq_confidence_visible(vcd_logits, nota)
            gt_labels = og_group.get("ground_truth_labels", {})

            for obj, gt_val in gt_labels.items():
                if obj == nota:
                    continue
                y_true_yes.append(1 if str(gt_val).strip().lower() == "yes" else 0)
                if obj in conf_visible:
                    y_prob_yes.append(conf_visible[obj])
                else:
                    logit_obj = vcd_logits.get(obj, 0.0)
                    logit_nota = vcd_logits.get(nota, 0.0)
                    y_prob_yes.append(_pairwise_confidence(logit_obj, logit_nota))

    return np.asarray(y_true_yes, dtype=int), np.asarray(y_prob_yes, dtype=float)


def mcq_vcd_metrics(
    original_path: str | Path,
    distorted_path: str | Path,
    alpha: float = 1.0,
    beta: float = 0.2,
) -> dict[str, float]:
    """F1 (visible) and Rej F1 (hallucination) of MCQ VCD at threshold 0.5"""
    original = load_mcq_data(original_path)
    distorted = load_mcq_data(distorted_path)
    y_true_yes, y_prob_yes = mcq_vcd_extract(original, distorted, alpha, beta)

    if len(y_true_yes) == 0:
        return {"f1": float("nan"), "rej_f1": float("nan"), "n": 0}

    y_pred_yes = (y_prob_yes >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_true_yes, y_pred_yes, zero_division=0)),
        "rej_f1": float(f1_score(1 - y_true_yes, 1 - y_pred_yes, zero_division=0)),
        "n": int(len(y_true_yes)),
    }


def mcq_distortion_f1_report(
    distorted_dirs: dict[str, str | Path],
    alpha: float = 1.0,
    beta: float = 0.2,
) -> dict[str, list[tuple[float, float]]]:
    """Rej F1 (hallucination) of MCQ VCD per level, keyed by distortion name.

    ``distorted_dirs`` maps a distortion name to a directory containing
    ``MCQ_logits_level*.json`` files.  Level 0 (the original) is used as the
    contrastive baseline and excluded from the returned curves.
    """
    report: dict[str, list[tuple[float, float]]] = {}
    for name, directory in distorted_dirs.items():
        files = _level_files(Path(directory))
        if not files:
            report[name] = []
            continue
        original_path = files[min(files)]
        curve: list[tuple[float, float]] = []
        for level in sorted(files):
            if level <= 0.0:
                continue
            metrics = mcq_vcd_metrics(original_path, files[level], alpha, beta)
            curve.append((level, metrics["rej_f1"]))
        report[name] = curve
    return report
