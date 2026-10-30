"""Semantic-CHAIR evaluation metric"""

from __future__ import annotations

import json
from pathlib import Path


def load_chair_evaluation(path: str | Path) -> list[dict]:
    """Load a ``chair_evaluation_*.jsonl`` file into a list of samples."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def chair_metrics(samples: list[dict]) -> dict:
    """Compute CHAIR-i / CHAIR-s for original and revised cptions"""
    agg = {
        "original": {"total_objects": 0, "hallucinated_objects": 0, "total_sentences": 0, "sentences_with_hallucinations": 0},
        "revised": {"total_objects": 0, "hallucinated_objects": 0, "total_sentences": 0, "sentences_with_hallucinations": 0},
    }

    for sample in samples:
        for label, objs_key, hal_key in (
            ("original", "extracted_objects", "original_hallucinations"),
            ("revised", "revision_extracted_objects", "revised_hallucinations"),
        ):
            objs = sample.get(objs_key) or []
            hals = sample.get(hal_key) or []
            a = agg[label]
            a["total_objects"] += len(objs)
            a["hallucinated_objects"] += len(hals)
            a["total_sentences"] += 1
            if hals:
                a["sentences_with_hallucinations"] += 1

    out: dict = {}
    for label in ("original", "revised"):
        a = agg[label]
        out[label] = {
            "chair_i": (a["hallucinated_objects"] / a["total_objects"]) if a["total_objects"] else 0.0,
            "chair_s": (a["sentences_with_hallucinations"] / a["total_sentences"]) if a["total_sentences"] else 0.0,
            **a,
        }
    return out
