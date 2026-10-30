import json
import os
import re


CAPTIONS_PATH = os.environ.get("CAPTIONS_PATH", "/mnt/storagecube/filipe/rspope/generated_captions/rspope_constrained_captions_20260729_135810.jsonl")
CONFIDENCES_PATH = os.environ.get("CONFIDENCES_PATH", "/mnt/storagecube/filipe/rspope/llava_inference_results/rspope-evaluation_worker_0.jsonl")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/rspope/masked_captions")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rspope_masked_captions.jsonl")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading captions and confidences datasets...")
    
    with open(CAPTIONS_PATH, 'r', encoding='utf-8') as f:
        captions_data = [json.loads(line) for line in f if line.strip()]

    object_confidences_data = []
    with open(CONFIDENCES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            item = json.loads(line)
            if "image" in item and "object" in item:
                object_confidences_data.append(item)

    object_confidences_by_image = {}
    for item in object_confidences_data:
        image_name = item.get("image")
        object_name = item.get("object")
        if not image_name or not object_name:
            continue
        object_confidences_by_image.setdefault(image_name, {})[object_name] = item

    print("Evaluating confidences and applying masks...")

    for sample in captions_data:
        sample["low_confidence_objects"] = []
        image_name = sample.get("image")
        
        all_objects = sample.get("visible_objects", []) + sample.get("inserted_hallucinations", [])
        
        for obj in all_objects:
            matching_obj = object_confidences_by_image.get(image_name, {}).get(obj)
            if matching_obj:
                # If LLaVA is less than 50% confident the object is visible, flag it[cite: 3]
                if matching_obj.get("confidence_visible", 1.0) < 0.5:
                    sample["low_confidence_objects"].append(obj)

        masked_cap = sample.get("generated_caption", "")
        
        for obj in sample["low_confidence_objects"]:
            # Handle hyphenated RSPOPE objects by allowing either a hyphen or a space in the regex mask
            obj_pattern = obj.replace("-", "[- ]")
            
            # Mask the object with [IDK] 
            masked_cap = re.sub(obj_pattern, "[IDK]", masked_cap, flags=re.IGNORECASE)
            
        sample["masked_caption"] = masked_cap.strip()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in captions_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"\n{'=' * 60}")
    print(f"Masking Complete. Processed {len(captions_data)} captions.")
    print(f"Results safely saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()