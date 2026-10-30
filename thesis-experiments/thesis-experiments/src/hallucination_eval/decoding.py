from __future__ import annotations

import math

NEG_INF = float("-inf")


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def binary_confidence(logit_a: float, logit_b: float) -> float:
    """Confidence that the answer token is 'A' (object visible)"""
    return sigmoid(logit_a - logit_b)


def confidence_from_logits(logits: dict[str, float]) -> float:
    """Binary confidence"""
    a = logits.get("A", 0.0)
    b = logits.get("B", 0.0)
    if a == NEG_INF and b == NEG_INF:
        return 0.5  # mathematically undefined; neutral fallback
    if a == NEG_INF:
        return 0.0
    if b == NEG_INF:
        return 1.0
    return binary_confidence(a, b)


def vcd_binary(
    logits_original: dict[str, float],
    logits_distorted: dict[str, float],
    alpha: float = 1.0,
    beta: float = 0.1,
) -> dict[str, float]:
    """VCD with APC for a two-token (A/B) vocab."""
    p_a = sigmoid(logits_original.get("A", 0.0) - logits_original.get("B", 0.0))
    p_b = 1.0 - p_a
    probs = {"A": p_a, "B": p_b}
    threshold = beta * max(p_a, p_b)

    out: dict[str, float] = {}
    for letter in ("A", "B"):
        if probs[letter] < threshold:
            out[letter] = NEG_INF
        else:
            out[letter] = (1.0 + alpha) * logits_original.get(letter, 0.0) - (
                alpha * logits_distorted.get(letter, 0.0)
            )
    return out


def mcq_confidences(
    average_logits: dict[str, float],
    nota_label: str = "None of the above",
) -> dict[str, float]:
    """MCQ confidence per object via softmax against 'None of the above'"""
    if not average_logits or nota_label not in average_logits:
        return {}
    nota_logit = average_logits[nota_label]
    return {
        label: sigmoid(logit - nota_logit)
        for label, logit in average_logits.items()
        if label != nota_label
    }


def vcd_mcq(
    logits_original: dict[str, float],
    logits_distorted: dict[str, float],
    alpha: float = 1.0,
    beta: float = 0.2,
) -> dict[str, float]:
    """VCD with APC for an arbitrary number of MCQ options"""
    if not logits_original:
        return {}

    max_logit = max(logits_original.values())
    exps: dict[str, float] = {}
    total = 0.0
    for label, logit in logits_original.items():
        try:
            val = math.exp(logit - max_logit)
        except OverflowError:
            val = math.inf
        exps[label] = val
        total += val

    probs = {
        label: (val / total if total > 0 else 0.0)
        for label, val in exps.items()
    }
    max_prob = max(probs.values()) if probs else 0.0
    threshold = beta * max_prob

    out: dict[str, float] = {}
    for label, logit in logits_original.items():
        if probs.get(label, 0.0) < threshold:
            out[label] = NEG_INF
        else:
            out[label] = (1.0 + alpha) * logit - (
                alpha * logits_distorted.get(label, 0.0)
            )
    return out


def mcq_confidence_visible(
    logits: dict[str, float],
    nota_label: str = "None of the above",
) -> dict[str, float]:
    """Per-object P(visible) from MCQ logits"""
    confidences: dict[str, float] = {}
    logit_nota = logits.get(nota_label, NEG_INF)

    for label, logit in logits.items():
        if label == nota_label:
            continue
        if logit == NEG_INF and logit_nota == NEG_INF:
            confidences[label] = 0.5
        elif logit == NEG_INF:
            confidences[label] = 0.0
        elif logit_nota == NEG_INF:
            confidences[label] = 1.0
        else:
            confidences[label] = sigmoid(logit - logit_nota)
    return confidences
