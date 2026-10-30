from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from transformers import LogitsProcessorList

from hallucination_eval.adapters.base import parse_yes_no, single_word_question
from hallucination_eval.generation import APCLogitsProcessor, VCDLogitsProcessor
from hallucination_eval.registry import get_adapter

STRATEGY_DIRS = {
    "regular": "full_answers/regular_decoding_temp_1",
    "greedy": "full_answers/regular_decoding_temp_0",
    "apc": "full_answers/apc",
    "vcd": "full_answers/vcd",
}


def _load_image(path: str) -> Any:
    from PIL import Image

    return Image.open(path)


def generate_answer(
    adapter: Any,
    strategy: str,
    text: str,
    image_orig: Any,
    image_dist: Any,
    model: Any,
    processor: Any,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> str:
    """Generate one answer under the requested decoding strategy"""
    question = single_word_question(text)
    if strategy == "regular":
        return adapter.generate(
            question, image_orig, model, processor,
            do_sample=True, temperature=1.0, max_new_tokens=16,
        )
    if strategy == "greedy":
        return adapter.generate(
            question, image_orig, model, processor,
            do_sample=False, max_new_tokens=16,
        )
    if strategy == "apc":
        return adapter.generate(
            question, image_orig, model, processor,
            do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=16,
            logits_processor=LogitsProcessorList([APCLogitsProcessor(beta=beta)]),
        )
    if strategy == "vcd":
        return adapter.generate(
            question, [image_orig, image_dist], model, processor,
            do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=16,
            logits_processor=LogitsProcessorList([VCDLogitsProcessor(alpha=alpha, beta=beta)]),
        )
    raise ValueError(f"unknown strategy: {strategy}")


def run(
    adapter: Any,
    strategy: str,
    samples: list[dict[str, Any]],
    images_dir_orig: str,
    images_dir_dist: str,
    limit: int | None = None,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    """Run one strategy over samples"""
    model, processor = adapter.load()

    results: list[dict[str, Any]] = []
    selected = samples if limit is None else samples[:limit]
    t0 = time.perf_counter()

    for idx, sample in enumerate(selected, start=1):
        qid = sample["question_id"]
        text = sample["text"]
        label = sample["label"]

        orig_path = os.path.join(images_dir_orig, sample["image_filename"])
        if not os.path.exists(orig_path):
            continue

        image_orig = _load_image(orig_path)
        image_dist = None
        if strategy == "vcd":
            dist_path = os.path.join(images_dir_dist, sample["image_filename"])
            if not os.path.exists(dist_path):
                continue
            image_dist = _load_image(dist_path)

        answer = generate_answer(adapter, strategy, text, image_orig, image_dist, model, processor)
        try:
            visible = parse_yes_no(answer)
        except ValueError:
            visible = False

        results.append(
            {
                "question_id": qid,
                "image": sample["image_filename"],
                "text": text,
                "label": label,
                "confidence_visible": visible,
                "answer": answer,
                "dataset_source": sample.get("dataset_source"),
            }
        )

        elapsed = time.perf_counter() - t0
        if idx % 10 == 0 or idx == 1:
            print(
                f"  [{idx}/{len(selected)}] elapsed: {elapsed:.1f}s | "
                f"ETA: {elapsed / idx * (len(selected) - idx):.1f}s",
                flush=True,
            )

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="full-answers", description="Full-answer decoding strategies.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["rspope", "pope"], default="rspope")
    parser.add_argument("--type", default="random")
    parser.add_argument("--strategy", choices=list(STRATEGY_DIRS) + ["all"], default="all")
    parser.add_argument("--images-orig", required=True)
    parser.add_argument("--images-dist", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from hallucination_eval.data.datasets import (
        load_pope_full_answers,
        load_rspope_full_answers,
    )

    if args.dataset == "rspope":
        samples = load_rspope_full_answers(
            "/mnt/storagecube/filipe/rspope/rspope_evaluation", args.type
        )
    else:
        samples = load_pope_full_answers("/mnt/storagecube/filipe/VCD", args.type)
    print(f"Dataset loaded with {len(samples)} samples.")

    adapter = get_adapter(args.model)
    strategies = list(STRATEGY_DIRS) if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        results = run(
            adapter, strategy, samples,
            images_dir_orig=args.images_orig,
            images_dir_dist=args.images_dist,
            limit=args.limit,
        )
        out_dir = os.path.join(args.out, STRATEGY_DIRS[strategy])
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, f"{args.type}.json")
        output = {
            "created_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "model": adapter.model_id,
            "dataset": args.type,
            "results": results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
