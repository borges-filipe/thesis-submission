from tracemalloc import start
import json
import os
import re
import ast
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

# Configuration
MODEL_ID = "Qwen/Qwen3.5-9B"
# DATASET_PATH = "/mnt/storagecube/filipe/Git-10M-old/train/data-00000-of-00787.arrow"
OUTPUT_DIR = "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_co_occurrence_objects"
CACHE_DIR = "/data/filipe/hf-cache"
BATCH_SIZE = 512  # Adjust based on your GPU VRAM

_MODEL_CACHE = {}
_PROCESSOR_CACHE = {}

start_shard = 0 
n_shards = 200
end_shard = start_shard + n_shards - 1
arrow_files = range(start_shard, end_shard + 1, 1)

def get_model_and_processor() -> tuple:
    """
    Lazy-load model and processor with optimal inference settings.
    """
    if MODEL_ID not in _MODEL_CACHE or MODEL_ID not in _PROCESSOR_CACHE:
        print(f"Loading model and processor: {MODEL_ID}...")

        _MODEL_CACHE[MODEL_ID] = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", # Optimization: Enable Flash Attention
            device_map="auto",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        
        # Optimization: Left padding is required for batch generation in causal LMs
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            
        _PROCESSOR_CACHE[MODEL_ID] = processor

    return _MODEL_CACHE[MODEL_ID], _PROCESSOR_CACHE[MODEL_ID]


@torch.inference_mode() # Optimization: Faster than torch.no_grad()
def process_batch(batch_samples: List[Dict[str, Any]], model, processor) -> List[dict]:
    """
    Executes the prompt sequence on a batch of dataset samples simultaneously.
    """
    texts = []
    
    # 1. Prepare all prompts in the batch
    for sample in batch_samples:
        caption = sample.get("caption", "")
        prompt = (
            f'List three other objects that you think are most likely to appear with the objects in the Remote Sensing scene described below:'
            f"Caption: {caption}"
            f'Output in strict accordance with the following format: ["Object one", "Object two", "Object three"]'
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        texts.append(text)

    # 2. Tokenize and pad the batch
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)

    # 3. Generate outputs for the entire batch
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    # 4. Trim prompt tokens and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    responses = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # 5. Parse outputs
    results = []
    for sample, response_text in zip(batch_samples, responses):
        object_list = []
        # Added re.DOTALL in case the model generates newlines inside the brackets
        match = re.search(r'\[.*\]', response_text.strip(), re.DOTALL) 

        if match:
            try:
                object_list = ast.literal_eval(match.group(0))
            except (ValueError, SyntaxError) as e:
                pass # Optionally log parsing errors here

        sample["co_occurrence_objects"] = object_list
        results.append(sample)

    return results


def chunk_dataset(dataset, batch_size):
    """Yield successive n-sized chunks from the dataset."""
    for i in range(0, len(dataset), batch_size):
        yield [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"{start_shard}-{end_shard}-co-occurrence-object-extraction-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)

    experiment_start = time.perf_counter()
    for i in arrow_files:
        DATASET_PATH = f"/mnt/storagecube/filipe/Git-10M-old/train/data-{str(i).zfill(5)}-of-00787.arrow"
        print(f"\nAttempting to load shard from: {DATASET_PATH}")
        
        # --- SHARD CORRUPTION HANDLING ---
        try:
            dataset = Dataset.from_file(DATASET_PATH)
        except Exception as e:
            print(f"[Error] Failed to load shard {i}. The file may be corrupted or missing. Skipping to next shard.\nDetails: {e}")
            continue

        # --- FIX: REMOVE IMAGE COLUMN ---
        if "image" in dataset.column_names:
            dataset = dataset.remove_columns("image")
            print("Successfully dropped 'image' column to prevent JSON serialization errors.")

        

        model, processor = get_model_and_processor()

        print(f"\n{'=' * 60}")
        print(f"Beginning object extraction (Batch Size: {BATCH_SIZE})")
        print(f"{'=' * 60}")

        shard_start = time.perf_counter()
        processed_count = 0

        with open(output_file, "a", encoding="utf-8") as f:
            # Optimization: Process dataset in batches
            for batch in chunk_dataset(dataset, BATCH_SIZE):
                
                # Save sample location for efficient retrieval later (optional)
                for j, sample in enumerate(batch):
                    sample["dataset_shard"] = processed_count + j

                try:
                    # GPU Execution
                    batch_results = process_batch(batch, model, processor)

                    # Write batch to disk
                    for result_data in batch_results:
                        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                    f.flush()

                except Exception as e:
                    print(f"\n[Error] Failed processing batch ending at index {processed_count + len(batch)} in shard {i}: {e}")
                    
                processed_count += len(batch)
                total_elapsed = time.perf_counter() - shard_start
                
                # Simple ETA calculation
                avg_time_per_sample = total_elapsed / processed_count
                eta = avg_time_per_sample * (len(dataset) - processed_count)

                print(
                    f"shard{i}: [{processed_count}/{len(dataset)}] "
                    f"elapsed: {total_elapsed:.1f}s | ETA for shard {i}: {eta:.1f}s"
                )

        total_elapsed = time.perf_counter() - shard_start
        print(f"\nShard {i} Completed! Results appended to: {output_file}")
        print(f"Elapsed time for shard {i}: {total_elapsed:.1f}s ({total_elapsed / 3600:.2f}h / {total_elapsed / 60:.2f}m)")