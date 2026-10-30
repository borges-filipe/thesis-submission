from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from hallucination_eval.registry import get_adapter

PROMPT2 = (
    "You are an expert dataset creator building a visual evaluation benchmark for Vision-Language Models. "
    "Your task is to generate visual questions based on the provided image caption and extracted objects.\n\n"
    "### CRITICAL RULES ###\n"
    "1. STRICT ANTI-LEAKAGE: The question must test the model, not give away the answer. "
    "You absolutely cannot use ANY word from the object's name. You must also exclude plurals, root words, "
    "and direct synonyms. For example, if the object is 'rocky terrain', you cannot use 'rock', 'rocks', "
    "'rocky', or 'terrain'. Describe the visual properties instead (e.g., 'hard, uneven geological formations').\n"
    "2. NO META-QUESTIONS: Do not generate questions about the image medium, source, or camera equipment "
    "(e.g., satellites, drones, cameras, photographs). If an object is a medium, skip it entirely.\n"
    "3. GROUNDING: Base your questions strictly on the visual information provided in the caption. "
    "Do not hallucinate outside context.\n"
    "4. SPECIFICITY: Questions must be highly specific so that the target object is the single, "
    "unambiguous correct answer.\n\n"
    "### OUTPUT FORMAT ###\n"
    "You must output a JSON array of objects. To ensure you do not leak the answer, you must first list "
    "the root words you are forbidden from using. Follow this exact structure:\n"
    "[\n"
    "  {\n"
    '    "answer": "rocky terrain",\n'
    '    "forbidden_roots": ["rock", "rocks", "rocky", "terrain", "terrains"],\n'
    '    "question": "What hard, uneven geological formations cover parts of the landscape?"\n'
    "  }\n"
    "]"
)


def prompt1(caption: str) -> str:
    return (
        "Extract all the objects visible in the image based on its caption. "
        'Make sure no objects are too vague (eg. "structures").\n\n'
        f"Caption: {caption}"
    )


def parse_generation_response(response: str) -> Any:
    """Strip markdown code fences and parse the JSON array; fall back to raw output."""
    cleaned = re.sub(r"^```json\s*", "", response.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_model_error_output": response}


def process_sample(adapter, sample: dict[str, Any], model, processor) -> dict[str, Any]:
    caption = sample.get("caption", "")
    image = sample["image"]

    messages1 = [{"role": "user", "content": [{"type": "text", "text": prompt1(caption)}]}]
    response1 = adapter.chat_generate(messages1, [], model, processor, max_new_tokens=256)

    messages2 = [
        messages1[0],
        {"role": "assistant", "content": [{"type": "text", "text": response1}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT2},
            ],
        },
    ]
    response2 = adapter.chat_generate(messages2, [image], model, processor, max_new_tokens=1024)
    qa_dict = parse_generation_response(response2)

    return {
        "img_name": sample.get("img_name", ""),
        "caption": caption,
        "extracted_objects": response1,
        "qa_json_output": qa_dict,
    }


def run(adapter, samples, output_dir: str, limit: int | None = None) -> str:
    model, processor = adapter.load()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"qa_generation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    t0 = time.perf_counter()
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            if limit is not None and idx > limit:
                break
            try:
                image = sample["image"]
                raw_img_name = sample.get("img_name")
                img_filename = os.path.basename(raw_img_name) if raw_img_name else f"test_image_{idx}.jpg"
                image.save(os.path.join(output_dir, img_filename))

                result = process_sample(adapter, sample, model, processor)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001 - keep going like the original
                print(f"[Error] sample {idx}: {e}")

            if idx % 10 == 0 or idx == 1:
                elapsed = time.perf_counter() - t0
                print(f"  [{idx}] elapsed: {elapsed:.1f}s", flush=True)

    print(f"Saved: {output_file}")
    return output_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="qgen")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--arrow", required=True, help="Git-10M .arrow shard path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)

    from datasets import Dataset

    dataset = Dataset.from_file(args.arrow)
    dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Dataset limited to {len(dataset)} samples.")

    adapter = get_adapter(args.model)
    run(adapter, dataset, args.out, args.limit)


if __name__ == "__main__":
    main()
