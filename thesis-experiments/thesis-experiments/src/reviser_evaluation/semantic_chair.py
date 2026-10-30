import json
import os
import time
from datetime import datetime, timezone
import pandas as pd
from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
TEST_FILE = os.environ.get("TEST_FILE", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/extracted_objects/revised_captions-object-extraction-20260722_091425.jsonl")
GT_FILE = os.environ.get("GT_FILE", "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/combined_object_extraction.parquet")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/images")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/evaluation")
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


def check_semantic_equivalence(model, processor, image: Image.Image, obj1: str, obj2: str) -> bool:
    """Prompts the VLM to check if two objects are semantically equivalent given the image"""
    prompt = (
        f"Are the objects '{obj1}' and '{obj2}' semantically equivalent or referring to the same entity "
        f"in the context of the provided image? Answer strictly with 'Yes' or 'No'."
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

    with torch.no_grad():
        sampled_gen_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    sampled_tokens = sampled_gen_ids[0, input_len:]
    decoded_ans = processor.decode(sampled_tokens, skip_special_tokens=True).strip().lower()

    return "yes" in decoded_ans


def is_hallucinated(obj: str, gt_objects: list[str], image: Image.Image, model, processor) -> bool:
    """
    Checks if an object is hallucinated by comparing it against all ground truth objects
    Returns True if hallucinated, False if a match is found
    """
    # Safe check for both native lists and numpy arrays
    if gt_objects is None or len(gt_objects) == 0:
        return True

    # 1. Fast path: Exact string match (case-insensitive)
    for gt_obj in gt_objects:
        if str(obj).strip().lower() == str(gt_obj).strip().lower():
            return False

    # 2. VLM path: Check semantic equivalence for remaining ground truth objects
    for gt_obj in gt_objects:
        is_equivalent = check_semantic_equivalence(model, processor, image, obj, gt_obj)
        if is_equivalent:
            return False

    return True


def load_ground_truth(parquet_path: str) -> dict:
    """Loads ground truth objects into a dictionary indexed by img_name"""
    print(f"Loading ground truth data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    gt_dict = {}
    for _, row in df.iterrows():
        # Cast to list to avoid numpy boolean ambiguity issues
        objs = row.get("extracted_objects", [])
        gt_dict[row["img_name"]] = list(objs) if objs is not None else []
    print(f"Loaded ground truth for {len(gt_dict)} images.")
    return gt_dict


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"chair_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)

    model, processor = get_model_and_processor()
    gt_dict = load_ground_truth(GT_FILE)

    print(f"\n{'=' * 60}")
    print(f"Beginning CHAIR Evaluation Pipeline")
    print(f"Reading from: {TEST_FILE}")
    print(f"{'=' * 60}")

    experiment_start = time.perf_counter()

    with open(TEST_FILE, "r", encoding="utf-8") as infile:
        samples = [json.loads(line.strip()) for line in infile if line.strip()]

    total_samples = len(samples)

    # Metrics accumulators
    eval_metrics = {
        "original": {
            "total_objects": 0,
            "hallucinated_objects": 0,
            "total_sentences": 0,
            "sentences_with_hallucinations": 0,
        },
        "revised": {
            "total_objects": 0,
            "hallucinated_objects": 0,
            "total_sentences": 0,
            "sentences_with_hallucinations": 0,
        }
    }

    with open(output_file, "w", encoding="utf-8") as outfile:
        for sample_idx, sample_data in enumerate(samples, start=1):
            try:
                img_name = sample_data.get("img_name")
                o_s = sample_data.get("extracted_objects", [])
                o_rs = sample_data.get("revision_extracted_objects", [])
                o_t = gt_dict.get(img_name, [])

         
                o_s = o_s if o_s is not None else []
                o_rs = o_rs if o_rs is not None else []

                image_path = os.path.join(IMAGE_DIR, img_name)
                
                if not os.path.exists(image_path):
                    print(f"[Warning] Image not found for sample {sample_idx}: {image_path}")
                    continue
                    
                image = Image.open(image_path).convert("RGB")

               
                sample_data["original_hallucinations"] = []
                has_orig_hal = False
                for obj in o_s:
                    eval_metrics["original"]["total_objects"] += 1
                    if is_hallucinated(obj, o_t, image, model, processor):
                        eval_metrics["original"]["hallucinated_objects"] += 1
                        sample_data["original_hallucinations"].append(obj)
                        has_orig_hal = True
                
                eval_metrics["original"]["total_sentences"] += 1
                if has_orig_hal:
                    eval_metrics["original"]["sentences_with_hallucinations"] += 1

                sample_data["revised_hallucinations"] = []
                has_rev_hal = False
                for obj in o_rs:
                    eval_metrics["revised"]["total_objects"] += 1
                    if is_hallucinated(obj, o_t, image, model, processor):
                        eval_metrics["revised"]["hallucinated_objects"] += 1
                        sample_data["revised_hallucinations"].append(obj)
                        has_rev_hal = True
                
                eval_metrics["revised"]["total_sentences"] += 1
                if has_rev_hal:
                    eval_metrics["revised"]["sentences_with_hallucinations"] += 1

                outfile.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                outfile.flush()

            except Exception as e:
                print(f"\n[Error] Failed processing sample {sample_idx}: {e}")
                continue

            total_elapsed = time.perf_counter() - experiment_start
            avg_time = total_elapsed / sample_idx
            eta = avg_time * (total_samples - sample_idx)

            print(
                f"[{sample_idx}/{total_samples}] "
                f"elapsed: {total_elapsed:.1f}s | ETA: {eta:.1f}s | Processed: {img_name}"
            )

    def calc_chair(metrics):
        chair_i = (metrics["hallucinated_objects"] / metrics["total_objects"]) if metrics["total_objects"] > 0 else 0
        chair_s = (metrics["sentences_with_hallucinations"] / metrics["total_sentences"]) if metrics["total_sentences"] > 0 else 0
        return chair_i, chair_s

    orig_chair_i, orig_chair_s = calc_chair(eval_metrics["original"])
    rev_chair_i, rev_chair_s = calc_chair(eval_metrics["revised"])

    print(f"\n{'=' * 60}")
    print("FINAL CHAIR EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print("Original Captions (o_s):")
    print(f"  CHAIR_i (Instance Level): {orig_chair_i:.4f} ({eval_metrics['original']['hallucinated_objects']}/{eval_metrics['original']['total_objects']})")
    print(f"  CHAIR_s (Sentence Level): {orig_chair_s:.4f} ({eval_metrics['original']['sentences_with_hallucinations']}/{eval_metrics['original']['total_sentences']})")
    print("\nRevised Captions (o_R(s)):")
    print(f"  CHAIR_i (Instance Level): {rev_chair_i:.4f} ({eval_metrics['revised']['hallucinated_objects']}/{eval_metrics['revised']['total_objects']})")
    print(f"  CHAIR_s (Sentence Level): {rev_chair_s:.4f} ({eval_metrics['revised']['sentences_with_hallucinations']}/{eval_metrics['revised']['total_sentences']})")
    
    total_elapsed = time.perf_counter() - experiment_start
    print(f"\nCompleted! Detailed results saved to: {output_file}")
    print(f"Total elapsed time: {total_elapsed / 3600:.1f}h / {total_elapsed / 60:.1f}m")