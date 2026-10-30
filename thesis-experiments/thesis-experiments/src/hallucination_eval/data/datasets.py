from __future__ import annotations

import json
import os
from pathlib import Path


def _read_jsonl(path: str | Path, fields: tuple[str, ...]) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out.append({k: item.get(k) for k in fields})
    return out


def load_rspope_full_answers(
    rspope_path: str | Path,
    dataset_type: str = "random",
) -> list[dict]:
    """Load RS-POPE (easy/medium/hard) for a dataset type (random/popular/adversarial)."""
    dataset: list[dict] = []
    for split in ("easy", "medium", "hard"):
        filepath = os.path.join(rspope_path, f"{dataset_type}/rspope_{dataset_type}_{split}.json")
        if not os.path.isfile(filepath):
            continue
        for item in _read_jsonl(filepath, ("question_id", "image", "text", "label")):
            dataset.append(
                {
                    "question_id": item["question_id"],
                    "image_filename": item["image"],
                    "text": item["text"],
                    "label": item["label"],
                    "dataset_source": split,
                }
            )
    return dataset


def load_pope_full_answers(
    vcd_repo_path: str | Path,
    dataset_type: str = "random",
) -> list[dict]:
    """Load POPE (coco/aokvqa/gqa) for a dataset type (random/popular/adversarial)."""
    dataset: list[dict] = []
    for split in ("coco", "aokvqa", "gqa"):
        filepath = os.path.join(
            vcd_repo_path, f"experiments/data/POPE/{split}/{split}_pope_{dataset_type}.json"
        )
        if not os.path.isfile(filepath):
            continue
        for item in _read_jsonl(filepath, ("question_id", "image", "text", "label")):
            dataset.append(
                {
                    "question_id": item["question_id"],
                    "image_filename": item["image"],
                    "text": item["text"],
                    "label": item["label"],
                    "dataset_source": split,
                }
            )
    return dataset
