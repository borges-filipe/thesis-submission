"""Object-level visibility probing (§5.2.3 / §5.4 confidence estimation).

Ports the archived ``AB_logits_reviser_evaluation.py`` and ``AB_logits_git10m.py``.
For every object in an object-extraction JSONL, probes the model with the binary
question "Is '<object>' visible in the image?" and records the per-object
``confidence_visible``.

Run inside a model environment (see ``submit_shared_job.sh``)::

    PYTHONPATH=src uv run python -m hallucination_eval.tasks.object_confidence \
        --model rs-llava \
        --json /path/to/object-extraction.jsonl \
        --images /path/to/images \
        --out /path/to/ab_logits_masking
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from hallucination_eval.adapters.base import (
    adapter_binary_logits,
    binary_question,
    confidence_visible,
)
from hallucination_eval.registry import get_adapter


def load_object_extraction(json_path: str | Path) -> list[dict]:
    """Load an object-extraction JSONL (records: ``img_name`` + ``extracted_objects``)."""
    samples: list[dict] = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run(
    adapter: Any,
    samples: list[dict],
    images_dir: str | Path,
    limit: int | None = None,
    worker_id: int = 0,
    total_workers: int = 1,
) -> list[dict]:
    """Probe every object and return per-object confidence records."""
    from PIL import Image

    model, processor = adapter.load()

    selected = samples if total_workers <= 1 else samples[worker_id::total_workers]
    if limit is not None:
        selected = selected[:limit]

    results: list[dict] = []
    t0 = time.perf_counter()
    n_objects = 0

    for idx, sample in enumerate(selected, start=1):
        image_filename = sample.get("img_name")
        objects = sample.get("extracted_objects") or []
        if not image_filename or not objects:
            continue
        try:
            image = Image.open(os.path.join(str(images_dir), image_filename)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to load {image_filename}: {e}", flush=True)
            continue

        for obj in objects:
            question = binary_question(f"Is '{obj}' visible in the image?")
            bl = adapter_binary_logits(adapter, question, image, model, processor)
            results.append(
                {
                    "image": image_filename,
                    "object": obj,
                    "confidence_visible": confidence_visible(bl.logits),
                    "logits": bl.logits,
                    "tokens": bl.tokens,
                }
            )
            n_objects += 1

        elapsed = time.perf_counter() - t0
        eta = (elapsed / idx) * (len(selected) - idx)
        print(
            f"[{idx}/{len(selected)}] objects={n_objects} elapsed={elapsed:.1f}s ETA={eta:.1f}s",
            flush=True,
        )

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="object-confidence", description="Object-level A/B visibility probing.")
    parser.add_argument("--model", required=True, help="model key (see registry.available())")
    parser.add_argument("--json", required=True, help="object-extraction JSONL path")
    parser.add_argument("--images", required=True, help="directory of image files")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--dataset", default="object_confidence", help="output filename prefix")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N images")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--total-workers", type=int, default=1)
    args = parser.parse_args(argv)

    adapter = get_adapter(args.model)
    samples = load_object_extraction(args.json)
    print(f"Dataset loaded with {len(samples)} images.")

    results = run(
        adapter,
        samples,
        images_dir=args.images,
        limit=args.limit,
        worker_id=args.worker_id,
        total_workers=args.total_workers,
    )

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, f"{args.dataset}_worker_{args.worker_id}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(results)} object confidences to {out_file}")


if __name__ == "__main__":
    main()
