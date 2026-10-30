"""Binary (A/B) visibility probing — the shared §5.1/§5.2 task.

Replaces the five near-identical per-model ``AB_logits_*.py`` scripts.  Run it
inside a model's ``uv`` environment (see ``submit_shared_job.sh``)::

    PYTHONPATH=src uv run python -m hallucination_eval.tasks.ab_logits \
        --model llava-1.5-7b --dataset AB_rspope \
        --json /path/to/rspope_adversarial_hard.json \
        --images /path/to/rspope_image/original/hard \
        --out /mnt/storagecube/filipe/out/results/llava-1.5-7b/rspope
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hallucination_eval.adapters.base import (
    adapter_binary_logits,
    binary_question,
    confidence_visible,
)
from hallucination_eval.registry import get_adapter


def load_rspope(json_path: str | Path) -> list[dict]:
    """Load RS-POPE samples from a JSONL file.

    Each line: ``{"question_id", "image", "text", "label"}``.
    """
    dataset: list[dict] = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            dataset.append(
                {
                    "question_id": sample["question_id"],
                    "image_filename": sample["image"],
                    "text": sample["text"],
                    "label": sample["label"],
                }
            )
    return dataset


def load_image(path: str) -> Any:
    from PIL import Image

    return Image.open(path)


def run(
    adapter: Any,
    samples: list[dict],
    images_dir: str | Path,
    limit: int | None = None,
    worker_id: int = 0,
    total_workers: int = 1,
) -> list[dict]:
    """Run binary probing over ``samples`` and return the result list.

    ``adapter`` is a :class:`~hallucination_eval.adapters.base.VLMAdapter`
    (usually obtained from :func:`hallucination_eval.registry.get_adapter`).
    """
    model, processor = adapter.load()

    selected = samples if total_workers <= 1 else samples[worker_id::total_workers]
    if limit is not None:
        selected = selected[:limit]

    results: list[dict] = []
    t0 = time.perf_counter()

    for idx, sample in enumerate(selected, start=1):
        image = load_image(os.path.join(str(images_dir), sample["image_filename"]))
        question = binary_question(sample["text"])
        bl = adapter_binary_logits(adapter, question, image, model, processor)
        conf = confidence_visible(bl.logits)

        results.append(
            {
                "question_id": sample["question_id"],
                "image": sample["image_filename"],
                "text": sample["text"],
                "label": sample["label"],
                "confidence_visible": conf,
                "logits": bl.logits,
                "tokens": bl.tokens,
            }
        )

        elapsed = time.perf_counter() - t0
        eta = (elapsed / idx) * (len(selected) - idx)
        print(
            f"[{idx}/{len(selected)}] elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
            flush=True,
        )

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ab-logits", description="Binary A/B probing.")
    parser.add_argument("--model", required=True, help="model key (see registry.available())")
    parser.add_argument("--dataset", default="AB_rspope")
    parser.add_argument("--json", required=True, help="dataset JSONL path")
    parser.add_argument("--images", required=True, help="directory of image files")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--total-workers", type=int, default=1)
    args = parser.parse_args(argv)

    adapter = get_adapter(args.model)
    samples = load_rspope(args.json)
    print(f"Dataset loaded with {len(samples)} samples.")

    results = run(
        adapter,
        samples,
        images_dir=args.images,
        limit=args.limit,
        worker_id=args.worker_id,
        total_workers=args.total_workers,
    )

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_worker{args.worker_id}" if args.total_workers > 1 else ""
    output_file = os.path.join(args.out, f"{args.dataset}{suffix}.json")
    output = {
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "model": adapter.model_id,
        "dataset": args.dataset,
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
