from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from hallucination_eval.adapters.base import adapter_mcq_logits
from hallucination_eval.decoding import mcq_confidences
from hallucination_eval.registry import get_adapter

NOTA = "None of the above"
DEFAULT_STEM = "What object is visible in the provided image?"


def extract_object_from_question(text: str) -> str:
    """Extract the object name from the standard RSPOPE question format."""
    match = re.search(r"Is there a(?:n)?\s+(.+)\s+in the image\?", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def load_grouped_rspope(json_path: str | Path) -> list[dict[str, Any]]:
    """Load RS-POPE and group all objects queried for the same image"""
    grouped: dict[str, dict[str, Any]] = {}
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            img_name = sample["image"]
            obj_name = extract_object_from_question(sample["text"])
            label = sample["label"]

            entry = grouped.setdefault(
                img_name,
                {"image_filename": img_name, "objects": [], "ground_truth_labels": {}},
            )
            if obj_name not in entry["objects"]:
                entry["objects"].append(obj_name)
            entry["ground_truth_labels"][obj_name] = label

    return list(grouped.values())


def build_groups(ground_truth_labels: dict[str, str], rng: random.Random) -> list[list[str]]:
    """Split objects into visible/not-visible groups"""
    visible = [obj for obj, label in ground_truth_labels.items() if str(label).lower() == "yes"]
    not_visible = [obj for obj, label in ground_truth_labels.items() if str(label).lower() == "no"]

    if len(visible) >= 3 and len(not_visible) >= 3:
        return [
            rng.sample(visible, 3) + rng.sample(not_visible, 2),
            rng.sample(not_visible, 3) + rng.sample(visible, 2),
        ]

    all_objs = visible + not_visible
    if len(all_objs) >= 5:
        return [rng.sample(all_objs, 5)]
    return [all_objs]


def build_permutations(
    groups: list[list[str]],
    run_all_permutations: bool,
    max_permutations_per_sample: int | None,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    """Expand the groups into the list of option permutations to evaluate"""
    if not run_all_permutations:
        return [tuple(g) for g in groups]

    object_perms: list[tuple[str, ...]] = []
    for group in groups:
        object_perms.extend(list(itertools.permutations(group)))
    rng.shuffle(object_perms)  # interleave the groups during evaluation
    if max_permutations_per_sample is not None:
        object_perms = object_perms[:max_permutations_per_sample]
    return object_perms


def run(
    adapter: Any,
    samples: list[dict[str, Any]],
    images_dir: str | Path,
    run_all_permutations: bool = True,
    max_permutations_per_sample: int | None = 1000,
    seed: int | None = None,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run MCQ probing over grouped samples"""
    rng = random.Random(seed)
    model, processor = adapter.load()

    results: dict[str, list[dict[str, Any]]] = {}
    selected = samples if limit is None else samples[:limit]
    t0 = time.perf_counter()

    for sample_idx, sample in enumerate(selected, start=1):
        image_filename = sample["image_filename"]
        objects = sample["objects"]
        gt_labels = sample["ground_truth_labels"]

        groups = build_groups(gt_labels, rng)
        object_perms = build_permutations(
            groups, run_all_permutations, max_permutations_per_sample, rng
        )
        print(
            f"[sample {sample_idx}/{len(selected)}] image={image_filename} | "
            f"objects={len(objects)} | permutations={len(object_perms)}",
            flush=True,
        )

        image_path = os.path.join(str(images_dir), image_filename)
        try:
            with _open_image(image_path) as image:
                permutation_results: list[dict[str, Any]] = []
                answer_to_logits: dict[str, list[float]] = defaultdict(list)

                for perm_idx, obj_perm in enumerate(object_perms, start=1):
                    if run_all_permutations and perm_idx % 10 == 0:
                        print(f"  - permutation {perm_idx}/{len(object_perms)}", flush=True)

                    options = list(obj_perm) + [NOTA]
                    option_letters = [ascii_uppercase[i] for i in range(len(options))]
                    option_map = dict(zip(option_letters, options))

                    logits, selected_tokens = adapter_mcq_logits(
                        adapter, options, image, model, processor
                    )

                    for letter, logit in logits.items():
                        answer_to_logits[option_map[letter]].append(float(logit))

                    permutation_results.append(
                        {
                            "permutation_index": perm_idx,
                            "options": option_map,
                            "none_of_the_above_letter": option_letters[-1],
                            "raw_logits": {k: float(v) for k, v in logits.items()},
                            "selected_tokens": selected_tokens,
                        }
                    )

            sample_result: dict[str, Any] = {
                "image_filename": image_filename,
                "objects_available": objects,
                "groups_tested": groups,
                "ground_truth_labels": gt_labels,
                "num_permutations": len(permutation_results),
                "permutation_results": permutation_results,
            }

            if run_all_permutations:
                avg_logits = {
                    answer: (sum(vals) / len(vals))
                    for answer, vals in answer_to_logits.items()
                }
                sample_result["average_logits_by_answer"] = avg_logits
                sample_result["confidence_visible"] = mcq_confidences(avg_logits, NOTA)

            results[image_filename] = [sample_result]

        except FileNotFoundError:
            print(f"Warning: Image {image_filename} not found at {image_path}. Skipping.")
            continue

        elapsed = time.perf_counter() - t0
        eta = (elapsed / sample_idx) * (len(selected) - sample_idx)
        print(f"[{sample_idx}/{len(selected)}] elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    return results


def _open_image(path: str) -> Any:
    from PIL import Image

    return Image.open(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mcq-logits")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="RSPOPE_MCQ")
    parser.add_argument("--json", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--no-permutations", action="store_true")
    parser.add_argument("--max-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    adapter = get_adapter(args.model)
    samples = load_grouped_rspope(args.json)
    print(f"Dataset loaded. Grouped into {len(samples)} unique images.")

    results = run(
        adapter,
        samples,
        images_dir=args.images,
        run_all_permutations=not args.no_permutations,
        max_permutations_per_sample=args.max_permutations,
        seed=args.seed,
        limit=args.limit,
    )

    os.makedirs(args.out, exist_ok=True)
    output_file = os.path.join(args.out, args.filename or f"MCQ_logits_{args.model}_{args.dataset}.json")
    output = {
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "model": adapter.model_id,
        "dataset": args.dataset,
        "run_all_permutations": not args.no_permutations,
        "max_permutations_per_sample": args.max_permutations,
        "none_of_the_above_label": NOTA,
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
