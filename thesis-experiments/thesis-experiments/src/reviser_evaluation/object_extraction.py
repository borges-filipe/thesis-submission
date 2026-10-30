import ast
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
INPUT_FILE = os.environ.get("INPUT_FILE", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/captions/caption_TEST_SPLIT_20260721_135031.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/extracted_objects")
CACHE_DIR = os.environ.get("CACHE_DIR", "/data/filipe/hf-cache")
BATCH_SIZE = 512  # Adjust based on GPU VRAM

_MODEL_CACHE = {}
_PROCESSOR_CACHE = {}


def get_model_and_processor() -> tuple:
    """Lazy-load model and processor with optimal inference settings"""
    if MODEL_ID not in _MODEL_CACHE or MODEL_ID not in _PROCESSOR_CACHE:
        print(f"Loading model and processor: {MODEL_ID}...")

        _MODEL_CACHE[MODEL_ID] = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", 
            device_map="auto",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        

        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            
        _PROCESSOR_CACHE[MODEL_ID] = processor

    return _MODEL_CACHE[MODEL_ID], _PROCESSOR_CACHE[MODEL_ID]


@torch.inference_mode()  
def process_batch(batch_samples: List[Dict[str, Any]], model, processor) -> List[dict]:
    """Executes the prompt sequence on a batch """
    texts = []
    
    #  Prepare all prompts in the batch
    for sample in batch_samples:
        # Target the newly generated caption instead of the ground truth
        caption = sample.get("generated_caption", "")
        prompt = (
            f"You are an expert Remote Sensing dataset creator building a visual evaluation benchmark for Vision-Language Models."
            f"Extract all the objects visible in the image based on its caption."
            f'Make sure no objects are too vague (e.g. "structures", "landscape", "terrain").'
            f'Include only objects that are visible in the image. Do not include objects relating to how an image was captured (e.g. "satellite", "drone").'
            f'Answer should strictly follow the list format: ["object_1", "object_2", ..., "object_n"]\n'
            f"Caption: {caption}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        texts.append(text)

    # Tokenize and pad the batch
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)

    # Generate outputs for the entire batch
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    #Trim prompt tokens and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    responses = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Parse outputs
    results = []
    for sample, response_text in zip(batch_samples, responses):
        object_list = []
        match = re.search(r'\[.*\]', response_text.strip(), re.DOTALL) 

        if match:
            try:
                object_list = ast.literal_eval(match.group(0))
            except (ValueError, SyntaxError) as e:
                pass 
        sample["extracted_objects"] = object_list
        results.append(sample)

    return results


def chunk_dataset(dataset, batch_size):
    """Yield successive n-sized chunks from the dataset."""
    for i in range(0, len(dataset), batch_size):
        yield [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"test-split-object-extraction-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n{'=' * 60}")
    print(f"Loading dataset from: {INPUT_FILE}")
    print(f"{'=' * 60}")
    
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            dataset = [json.loads(line.strip()) for line in f if line.strip()]
        print(f"Successfully loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"[Error] Failed to load input file. Details: {e}")
        exit(1)

    model, processor = get_model_and_processor()

    print(f"\n{'=' * 60}")
    print(f"Beginning object extraction (Batch Size: {BATCH_SIZE})")
    print(f"{'=' * 60}")

    experiment_start = time.perf_counter()
    processed_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for batch in chunk_dataset(dataset, BATCH_SIZE):
            try:
    
                batch_results = process_batch(batch, model, processor)

                for result_data in batch_results:
                    f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                f.flush()

            except Exception as e:
                print(f"\n[Error] Failed processing batch ending at index {processed_count + len(batch)}: {e}")
                
            processed_count += len(batch)
            total_elapsed = time.perf_counter() - experiment_start
            
            avg_time_per_sample = total_elapsed / processed_count
            eta = avg_time_per_sample * (len(dataset) - processed_count)

            print(
                f"[{processed_count}/{len(dataset)}] "
                f"elapsed: {total_elapsed:.1f}s | ETA: {eta:.1f}s"
            )

    total_elapsed = time.perf_counter() - experiment_start
    print(f"\nExtraction Completed! Results saved to: {output_file}")
    print(f"Total elapsed time: {total_elapsed:.1f}s ({total_elapsed / 3600:.2f}h / {total_elapsed / 60:.2f}m)")