from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def manual_eval_stats(path: str | Path) -> dict:
    """Aggregate a manual-evaluations JSONL into object/image hallucination stats"""
    total_objects = 0
    not_visible_objects = 0
    image_objects: dict[str, list[bool]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image = record.get("image")
            object_visible = record.get("object_visible")
            if image is None or object_visible is None:
                continue
            total_objects += 1
            if not object_visible:
                not_visible_objects += 1
            image_objects[image].append(bool(object_visible))

    total_images = len(image_objects)
    images_with_not_visible = sum(1 for flags in image_objects.values() if not all(flags))

    return {
        "total_objects": total_objects,
        "not_visible_objects": not_visible_objects,
        "total_images": total_images,
        "images_with_not_visible": images_with_not_visible,
        "object_hallucination_rate": (not_visible_objects / total_objects) if total_objects else 0.0,
        "image_hallucination_rate": (images_with_not_visible / total_images) if total_images else 0.0,
    }
