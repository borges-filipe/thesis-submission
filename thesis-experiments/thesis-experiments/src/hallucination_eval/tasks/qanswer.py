from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from hallucination_eval.registry import get_adapter

VQA_PROMPT = (
    "Answer the following visual question based on the image. "
    "Keep your answer as concise as possible.\nQuestion: {question}\nAnswer:"
)


def extract_questions(conversations: list[dict]) -> list[str]:
    """Return every human question, skipping the first (generic description) turn."""
    human_turns = [t for t in (conversations or []) if t.get("from") == "human"]
    questions: list[str] = []
    for turn in human_turns[1:]:
        value = turn.get("value", "").replace("<image>", "").strip()
        if value:
            questions.append(value)
    return questions


def answer_question(adapter, image, question: str, model, processor) -> dict:
    prompt = VQA_PROMPT.format(question=question)
    best = adapter.sample(
        prompt, image, model, processor,
        num_return_sequences=1, do_sample=True, temperature=0.1, max_new_tokens=50,
    )[0]
    sampled = adapter.sample(
        prompt, image, model, processor,
        num_return_sequences=10, do_sample=True, temperature=1.0,
        top_p=0.9, top_k=50, max_new_tokens=50,
    )
    return {"best_generation": best, "sampled_generations": sampled}


def run(
    adapter,
    samples: list[dict],
    image_dir: str,
    output_dir: str,
    tag: str = "ddfav",
    limit: int | None = None,
) -> str:
    model, processor = adapter.load()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"qa_answers_{tag}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    selected = samples if limit is None else samples[:limit]
    t0 = time.perf_counter()

    with open(output_file, "w", encoding="utf-8") as outfile:
        for idx, sample in enumerate(selected, start=1):
            img_name = sample.get("image")
            if not img_name:
                continue
            image_path = os.path.join(image_dir, img_name)
            if not os.path.exists(image_path):
                continue

            questions = extract_questions(sample.get("conversations", []))
            if not questions:
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            qa_output: list[dict[str, Any]] = []
            for question in questions:
                try:
                    qa_output.append({"question": question, "model_responses": answer_question(adapter, image, question, model, processor)})
                except Exception as e:  # noqa: BLE001 - mirror the original's per-QA error handling
                    print(f"[Error] q={question!r}: {e}")
                    qa_output.append({"question": question, "model_responses": None, "error": str(e)})

            result = dict(sample)
            result["dataset"] = tag
            result["qa_json_output"] = qa_output
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            outfile.flush()

            elapsed = time.perf_counter() - t0
            if idx % 10 == 0 or idx == 1:
                print(f"  [{idx}/{len(selected)}] elapsed: {elapsed:.1f}s", flush=True)

    print(f"Saved: {output_file}")
    return output_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="qanswer", description="11-answer sampling.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=True, help="DDFAV/caption dataset JSON (a list)")
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="ddfav")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    with open(args.json, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Dataset loaded with {len(samples)} samples.")

    adapter = get_adapter(args.model)
    run(adapter, samples, args.images, args.out, args.tag, args.limit)


if __name__ == "__main__":
    main()
