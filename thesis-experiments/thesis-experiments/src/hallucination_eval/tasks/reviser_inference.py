from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from hallucination_eval.adapters.reviser import ReviserAdapter


def resolve_image(
    sample: dict[str, Any],
    images_dir: str,
    image_field: str,
    image_suffix: str,
    search_subdirs: list[str] | None,
) -> str | None:
    key = str(sample.get(image_field, "") or "")
    if not key:
        return None
    if search_subdirs:
        for sub in search_subdirs:
            path = os.path.join(images_dir, sub, f"{key}{image_suffix}")
            if os.path.exists(path):
                return path
        return None
    return os.path.join(images_dir, f"{key}{image_suffix}")


def run(
    adapter: ReviserAdapter,
    samples: list[dict[str, Any]],
    images_dir: str,
    image_field: str = "img_name",
    caption_field: str = "generated_caption",
    image_suffix: str = "",
    search_subdirs: list[str] | None = None,
    batch_size: int = 16,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    model, processor = adapter.load()
    results: list[dict[str, Any]] = []
    batch: list[tuple[Any, str, dict[str, Any]]] = []

    def flush() -> None:
        if not batch:
            return
        import PIL.Image

        items = [
            (PIL.Image.open(path).convert("RGB"), caption) for path, caption, _ in batch
        ]
        revised = adapter.generate_batch(items, model, processor)
        for (_path, _caption, sample), rev in zip(batch, revised):
            sample["revised_caption"] = rev
            results.append(sample)
        batch.clear()

    selected = samples if limit is None else samples[:limit]
    t0 = time.perf_counter()

    for sample in selected:
        path = resolve_image(
            sample, images_dir, image_field, image_suffix, search_subdirs
        )
        if not path or not os.path.exists(path):
            continue
        caption = sample.get(caption_field, "") or ""
        if not caption:
            continue
        batch.append((path, caption, sample))
        if len(batch) >= batch_size:
            flush()
            elapsed = time.perf_counter() - t0
            print(
                f"  [{len(results)}/{len(selected)}] elapsed: {elapsed:.1f}s",
                flush=True,
            )

    flush()
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="reviser-inference"
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-field", default="img_name")
    parser.add_argument("--caption-field", default="generated_caption")
    parser.add_argument("--image-suffix", default="")
    parser.add_argument("--search-subdirs", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    with open(args.jsonl, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(samples)} samples.")

    search = args.search_subdirs.split(",") if args.search_subdirs else None
    adapter = ReviserAdapter()
    results = run(
        adapter,
        samples,
        args.images,
        image_field=args.image_field,
        caption_field=args.caption_field,
        image_suffix=args.image_suffix,
        search_subdirs=search,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    os.makedirs(args.out, exist_ok=True)
    output_file = os.path.join(
        args.out,
        f"revised_captions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved: {output_file} ({len(results)} revised captions)")


if __name__ == "__main__":
    main()
