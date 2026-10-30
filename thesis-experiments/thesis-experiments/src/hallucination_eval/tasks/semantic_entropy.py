from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable

from hallucination_eval.registry import get_adapter


def entailment_prompt(question: str, text1: str, text2: str) -> str:
    return (
        f"We are evaluating answers to the visual question: '{question}' based on the provided image.\n"
        f"Here are two possible answers:\n"
        f"Possible Answer 1: {text1}\n"
        f"Possible Answer 2: {text2}\n\n"
        "Based on the visual evidence in the image and the context of the question, "
        "does Possible Answer 1 semantically entail Possible Answer 2?\n"
        "Respond ONLY with 'entailment', 'contradiction', or 'neutral'."
    )


def correctness_prompt(question: str, reference: str, predicted: str) -> str:
    return (
        f"We are assessing the quality of answers to the following visual question: '{question}' "
        f"based on the provided image.\n"
        f"The expected answer is: {reference}\n"
        f"The proposed answer is: {predicted}\n\n"
        "Based on the visual evidence in the image and the context of the question, "
        "does the proposed answer mean the same as the expected answer?\n"
        "Respond ONLY with 'yes' or 'no'."
    )


def extract_question_answer_map(conversations: list[dict]) -> dict[str, str]:
    """Map each human question to the immediately following GPT answer."""
    qa_map: dict[str, str] = {}
    turns = conversations or []
    for i, turn in enumerate(turns):
        if turn.get("from") != "human":
            continue
        question = turn.get("value", "").replace("<image>", "").strip()
        if not question:
            continue
        for j in range(i + 1, len(turns)):
            nxt = turns[j]
            if nxt.get("from") == "gpt":
                qa_map[question] = nxt.get("value", "")
                break
    return qa_map


def calculate_discrete_semantic_entropy(clusters: list[list[str]], total_samples: int) -> float:
    """SE = -sum P(C) * ln P(C)"""
    se = 0.0
    for cluster in clusters:
        p_c = len(cluster) / total_samples
        if p_c > 0:
            se -= p_c * math.log(p_c)
    return se


def cluster_answers(
    sampled_generations: list[str],
    judge: Callable[[str, str], bool],
) -> list[list[str]]:
    """Bidirectional-entailment clustering."""
    clusters: list[list[str]] = []

    for ans in sampled_generations:
        clean_ans = ans.strip().lower()

        # 1. exact string match (avoids LLM calls)
        matched = False
        for cluster in clusters:
            if any(clean_ans == item.strip().lower() for item in cluster):
                cluster.append(ans)
                matched = True
                break
        if matched:
            continue

        # 2. visually-grounded bidirectional entailment
        found = False
        for cluster in clusters:
            rep = cluster[0]
            if judge(ans, rep) and judge(rep, ans):
                cluster.append(ans)
                found = True
                break

        # 3. start a new cluster
        if not found:
            clusters.append([ans])

    return clusters


def judge_entailment(adapter, image, question, text1, text2, model, processor) -> bool:
    out = adapter.generate(
        entailment_prompt(question, text1, text2), image, model, processor,
        do_sample=False, temperature=0.1, max_new_tokens=10,
    )
    return "entailment" in out.lower()


def judge_correctness(adapter, image, question, reference, predicted, model, processor) -> bool:
    out = adapter.generate(
        correctness_prompt(question, reference, predicted), image, model, processor,
        do_sample=False, temperature=0.1, max_new_tokens=5,
    )
    return "yes" in out.lower()


def _build_image_cache(image_dirs: list[str]) -> dict[str, str]:
    cache: dict[str, str] = {}
    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            continue
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    cache.setdefault(file, os.path.join(root, file))
    return cache


def run(
    adapter,
    input_dir: str,
    image_dirs: list[str],
    output_dir: str,
    output_tag: str = "DDFAV",
    limit: int | None = None,
) -> str:
    """Run SE computation"""
    model, processor = adapter.load()
    image_cache = _build_image_cache(image_dirs)
    print(f"Cached {len(image_cache)} image paths.")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"qa_semantic_entropy_{output_tag}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not input_files:
        print(f"[Error] No JSONL files found in {input_dir}")
        sys.exit(1)

    processed = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for input_file in input_files:
            with open(input_file, encoding="utf-8") as infile:
                lines = infile.readlines()

            for line in lines:
                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                img_name = sample.get("image")
                qa_list = sample.get("qa_json_output", [])
                qa_map = extract_question_answer_map(sample.get("conversations", []))

                if not qa_list:
                    outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    continue

                image_path = image_cache.get(img_name or "")
                if not image_path or not os.path.exists(image_path):
                    outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    continue

                from PIL import Image

                image = Image.open(image_path).convert("RGB")

                for qa_item in qa_list:
                    question = qa_item.get("question")
                    expected = qa_item.get("answer") or qa_map.get(question)
                    responses = qa_item.get("model_responses", {}) or {}
                    best_prediction = responses.get("best_generation")
                    sampled = responses.get("sampled_generations", []) or []

                    if expected and best_prediction:
                        qa_item["is_correct"] = judge_correctness(
                            adapter, image, question, expected, best_prediction, model, processor
                        )
                    else:
                        qa_item["is_correct"] = None

                    if question and sampled:
                        judge = lambda t1, t2: judge_entailment(
                            adapter, image, question, t1, t2, model, processor
                        )
                        clusters = cluster_answers(sampled, judge)
                        qa_item["semantic_entropy_score"] = calculate_discrete_semantic_entropy(
                            clusters, len(sampled)
                        )
                        qa_item["semantic_clusters"] = clusters

                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
                outfile.flush()
                processed += 1
                if limit is not None and processed >= limit:
                    break

    print(f"Saved: {output_file}")
    return output_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="semantic-entropy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--image-dir", action="append", required=True, help="repeatable image dirs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="DDFAV")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    adapter = get_adapter(args.model)
    run(adapter, args.input_dir, args.image_dir, args.out, args.tag, args.limit)


if __name__ == "__main__":
    main()
