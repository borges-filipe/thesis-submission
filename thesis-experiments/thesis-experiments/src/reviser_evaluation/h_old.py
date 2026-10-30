import json
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
CACHE_DIR = os.environ.get("CACHE_DIR", "/data/filipe/hf-cache")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512"))  # Adjust based on your GPU VRAM
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/out/results/qwen3.5-9b/inserted_hallucinations_git10m")

CO_OBJECTS_FILE = os.environ.get("CO_OBJECTS_FILE", "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_co_occurrence_objects/0-199-co-occurrence-object-extraction-20260720_133312.jsonl")
UNCERTAIN_OBJECTS_FILE = os.environ.get("UNCERTAIN_OBJECTS_FILE", "/mnt/storagecube/filipe/out/results/rs-llava/negative_samples/not_so_low_confidence_negatives/similarity_based_negatives.jsonl")

_MODEL_CACHE = {}
_PROCESSOR_CACHE = {}

def get_model_and_processor() -> tuple:
    """Lazy-load model and processor with optimal inference settings."""
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
    """Executes the adversarial insertion prompt"""
    texts = []
    
    for sample in batch_samples:
        caption = sample.get("caption", "")
        co_objs = sample.get("co_objects", [])
        uncertain_objs = sample.get("uncertain_objects", [])
        
        prompt = (
            f"Input caption: {caption}\n"
            f"co_objects list: {co_objs}\n"
            f"uncertain_objects list: {uncertain_objs}\n"
            f'Select one object from "co_objects list" and "uncertain_objects list" respectively and add it to "Input caption" to get "Output caption". (Try not to change the format)\n'
            f"Output caption:"
        )
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        texts.append(text)

    # Tokenize and generate
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    #Decode outpts
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    responses = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    #parse outputs and extract inserted objects
    results = []
    for sample, response_text in zip(batch_samples, responses):
        output_caption = response_text.strip()
        
        
        if output_caption.lower().startswith("output caption:"):
            output_caption = output_caption[len("output caption:"):].strip()
            
        sample["output_caption"] = output_caption
        
        out_cap_lower = output_caption.lower()
        orig_cap_lower = sample["caption"].lower()
        
        inserted_co = [
            obj for obj in sample["co_objects"] 
            if obj.lower() in out_cap_lower and obj.lower() not in orig_cap_lower
        ]
        
        inserted_unc = [
            obj for obj in sample["uncertain_objects"] 
            if obj.lower() in out_cap_lower and obj.lower() not in orig_cap_lower
        ]
        
        if inserted_co:
            sample["inserted_co_object"] = inserted_co[0] 
        else:
            continue

        if inserted_unc:
            sample["inserted_uncertain_object"] = inserted_unc[0]
        else:
            continue
        
        results.append(sample)

    return results

def chunk_dataset(dataset, batch_size):
    """Yield successive n-sized chunks from the dataset."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"bigger-adversarial-captions-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)

    print("Loading and merging data files...")
    merged_data = {}
    
    with open(CO_OBJECTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            merged_data[d['img_name']] = {
                "image": d['img_name'],
                "caption": d['caption'],
                "co_objects": d.get('co_occurrence_objects', [])
            }
            
    with open(UNCERTAIN_OBJECTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            img_name = d['image']
            if img_name in merged_data:
                merged_data[img_name]["uncertain_objects"] = [neg["object"] for neg in d.get("negatives", [])]
                
    dataset = [
        item for item in merged_data.values() 
        if item.get("co_objects") and item.get("uncertain_objects")
    ]
    print(f"Total samples ready for processing: {len(dataset)}")

    model, processor = get_model_and_processor()

    print(f"\n{'=' * 60}")
    print(f"Beginning adversarial caption generation (Batch Size: {BATCH_SIZE})")
    print(f"{'=' * 60}")

    start_time = time.perf_counter()
    processed_count = 0

    with open(output_file, "a", encoding="utf-8") as f:
        for batch in chunk_dataset(dataset, BATCH_SIZE):
            try:
                batch_results = process_batch(batch, model, processor)
                for result_data in batch_results:
                    f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"\n[Error] Failed processing batch ending at index {processed_count + len(batch)}: {e}")
                
            processed_count += len(batch)
            total_elapsed = time.perf_counter() - start_time
            avg_time = total_elapsed / processed_count
            eta = avg_time * (len(dataset) - processed_count)

            print(f"[{processed_count}/{len(dataset)}] elapsed: {total_elapsed:.1f}s | ETA: {eta:.1f}s")

    print(f"\nCompleted! Results saved to: {output_file}")