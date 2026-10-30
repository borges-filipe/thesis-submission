from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mask_captions(objects: list[dict], confidences: list[dict], threshold: float = 0.5) -> list[dict]:
    """Return copies of objects with low-confidence objects masked to [IDK]"""
    conf_by_image: dict[str, dict[str, dict]] = {}
    for item in confidences:
        image = item.get("image")
        obj = item.get("object")
        if image and obj:
            conf_by_image.setdefault(image, {})[obj] = item

    out: list[dict] = []
    for sample in objects:
        sample = dict(sample)
        sample["low_confidence_objects"] = []
        for obj in sample.get("extracted_objects", []):
            match = conf_by_image.get(sample.get("img_name"), {}).get(obj)
            if match and match.get("confidence_visible", 1.0) < threshold:
                sample["low_confidence_objects"].append(obj)
        for obj in sample["low_confidence_objects"]:
            sample["generated_caption"] = sample.get("generated_caption", "").replace(obj, "[IDK]").strip()
        out.append(sample)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mask-captions")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--confidences", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    objects = load_jsonl(args.objects)
    confidences = load_jsonl(args.confidences)
    masked = mask_captions(objects, confidences, args.threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in masked:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(masked)} masked captions to {out_path}")


if __name__ == "__main__":
    main()
