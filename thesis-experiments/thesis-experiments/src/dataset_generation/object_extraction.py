import json
import os
import re
import ast
import time
from datetime import datetime, timezone

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor

# Configuration (override via env vars)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
DATASET_PATH = os.environ.get("DATASET_PATH", "/mnt/storagecube/filipe/Git-10M-old/train/data-00000-of-00787.arrow")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction")
CACHE_DIR = os.environ.get("CACHE_DIR", "/data/filipe/hf-cache")

_MODEL_CACHE = {}
_PROCESSOR_CACHE = {}


def get_model_and_processor() -> tuple:
    """
    Lazy-load model and processor once.
    """
    if MODEL_ID not in _MODEL_CACHE or MODEL_ID not in _PROCESSOR_CACHE:
        print(f"Loading model and processor: {MODEL_ID}...")

        _MODEL_CACHE[MODEL_ID] = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        _PROCESSOR_CACHE[MODEL_ID] = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

    return _MODEL_CACHE[MODEL_ID], _PROCESSOR_CACHE[MODEL_ID]


def generate_text(model, processor, inputs, max_new_tokens: int) -> str:
    """
    Helper function to generate text from inputs and decode the output.
    """
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()


def process_sample(sample, model, processor) -> dict:
    """
    Executes the two-prompt sequence on a single dataset sample.
    """
    caption = sample.get("caption", "")
    image = sample["image"] 
    prompt1 = (
        f"Extract all the objects visible in the image based on its caption. "
        f'Make sure no objects are too vague (eg. "structures").\n\n'
        f'Answer should strictly follow the list format: ["object_1", "object_2", ..., "object_n"]'
        f"Caption: {caption}"
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt1}],
        }
    ]

    text_turn1 = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs_turn1 = processor(text=[text_turn1], return_tensors="pt").to(model.device)

    response1 = generate_text(model, processor, inputs_turn1, max_new_tokens=256)



    match = re.search(r'\[.*\]', response1)

    if match:
        list_string = match.group(0)
        print("Extracted string:", list_string)
    else:
        print("No list found.")


    try:

        object_list = ast.literal_eval(list_string)

    except (ValueError, SyntaxError) as e:
        print("Error parsing the list:", e)

    return {
        "img_name": sample.get("img_name", ""),
        "caption": caption,
        "extracted_objects": object_list,
    }


if __name__ == "__main__":
    # to measure total runtime
    experiment_start = time.perf_counter()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_filename = f"qa_generation_TEST_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Loading single shard from: {DATASET_PATH}")

    dataset = Dataset.from_file(DATASET_PATH)

    TEST_SIZE = 3 

    dataset = dataset.select(range(min(TEST_SIZE, len(dataset))))
    print(f"Dataset limited to {len(dataset)} samples for testing.")
    # ---------------------------------

    model, processor = get_model_and_processor()

    print(f"\n{'=' * 60}")
    print(f"Beginning object extraction")
    print(f"{'=' * 60}")

    experiment_start = time.perf_counter()


    with open(output_file, "w", encoding="utf-8") as f:
        for sample_idx, sample in enumerate(dataset, start=1):
            try:

                image = sample["image"]
                raw_img_name = sample.get("img_name")

                # Use the actual filename if it exists, otherwise generate a placeholder
                if raw_img_name:
                    img_filename = os.path.basename(raw_img_name)
                else:
                    img_filename = f"test_image_{sample_idx}.jpg"

                image_save_path = os.path.join(OUTPUT_DIR, img_filename)
                image.save(image_save_path)
                # -----------------------

                result_data = process_sample(sample, model, processor)


                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                f.flush()

            except Exception as e:
                print(
                    f"\n[Error] Failed processing sample {sample_idx} ({sample.get('img_name')}): {e}"
                )
                continue

            total_elapsed = time.perf_counter() - experiment_start
            avg_time = total_elapsed / sample_idx
            eta = avg_time * (len(dataset) - sample_idx)

            print(
                f"[{sample_idx}/{len(dataset)}] "
                f"elapsed: {total_elapsed:.1f}s | ETA: {eta:.1f}s | Saved: {img_filename}"
            )

    print(f"\nTest Completed! Results saved to: {output_file}")
    print(f"Test images saved to: {OUTPUT_DIR}")
    print(
        f"Total elapsed time: {total_elapsed:.1f}s which in hours and minutes is {total_elapsed / 3600:.1f}h / {total_elapsed / 60:.1f}m"
    )
