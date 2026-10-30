import json
import os
import time
from datetime import datetime, timezone
from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
INPUT_FILE = os.environ.get("INPUT_FILE", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/test_dataset_raw.json")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/images")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/captions")
CACHE_DIR = os.environ.get("CACHE_DIR", "/data/filipe/hf-cache")

_MODEL_CACHE = {}
_PROCESSOR_CACHE = {}


def get_model_and_processor() -> tuple:
    """Lazy-load model and processor once."""
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


def caption_image(model, processor, image: Image.Image) -> dict:
    """Generates a caption for a single image"""
    prompt = (
        "You are a Remote Sensing image expert. Write a detailed visual caption of the provided satellite image. Your caption should have approximately 3 sentences or 60 words."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs = processor(
        text=[text_input], images=[image], return_tensors="pt"
    ).to(model.device)

    input_len = inputs.input_ids.shape[1]
    
    results = {}

    with torch.no_grad():
        sampled_gen_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            num_return_sequences=10
        )

    sampled_tokens = sampled_gen_ids[0, input_len:]
    decoded_ans = processor.decode(sampled_tokens, skip_special_tokens=True).strip()
        
    results["caption"] = decoded_ans

    return results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_filename = f"caption_TEST_SPLIT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)

    model, processor = get_model_and_processor()

    print(f"\n{'=' * 60}")
    print(f"Beginning Caption Pipeline")
    print(f"Reading from: {INPUT_FILE}")
    print(f"{'=' * 60}")

    experiment_start = time.perf_counter()
    processed_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        samples = json.load(infile)
        
    total_samples = len(samples)

    with open(output_file, "w", encoding="utf-8") as outfile:
        
        for sample_idx, sample_data in enumerate(samples, start=1):
            try:
                img_name = sample_data.get("img_name")

                image_path = os.path.join(IMAGE_DIR, img_name)
                
                if not os.path.exists(image_path):
                    print(f"[Warning] Image not found for sample {sample_idx}: {image_path}")
                    outfile.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                    continue
                    
                image = Image.open(image_path).convert("RGB")
                
                caption = caption_image(model, processor, image)
                
                sample_data["generated_caption"] = caption["caption"]

                outfile.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                outfile.flush()
                
                processed_count += 1

            except Exception as e:
                print(f"\n[Error] Failed processing sample {sample_idx}: {e}")
                continue

            total_elapsed = time.perf_counter() - experiment_start
            avg_time = total_elapsed / processed_count if processed_count > 0 else 0
            eta = avg_time * (total_samples - processed_count)

            print(
                f"[{sample_idx}/{total_samples}] "
                f"elapsed: {total_elapsed:.1f}s | ETA: {eta:.1f}s | Processed: {img_name}"
            )

    print(f"\nCompleted! Results saved to: {output_file}")
    total_elapsed = time.perf_counter() - experiment_start
    print(
        f"Total elapsed time: {total_elapsed:.1f}s which in hours and minutes is {total_elapsed / 3600:.1f}h / {total_elapsed / 60:.1f}m"
    )