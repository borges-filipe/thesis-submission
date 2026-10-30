import json
import os
import glob


EXTRACTED_DIR = os.environ.get("EXTRACTED_DIR", "/mnt/storagecube/filipe/rspope/DDFAV_Instruction_Set/generated_captions/extracted_objects")
LOGITS_DIR = os.environ.get("LOGITS_DIR", "/mnt/storagecube/filipe/rspope/DDFAV_Instruction_Set/generated_captions/extracted_objects/logits")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/rspope/DDFAV_Instruction_Set/generated_captions/masked_captions")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Locate all extracted objects JSONL files
    extracted_files = glob.glob(os.path.join(EXTRACTED_DIR, "*.jsonl"))
    
    if not extracted_files:
        print(f"[Warning] No .jsonl files found in {EXTRACTED_DIR}.")
        return

    print(f"\n{'=' * 60}")
    print(f"Found {len(extracted_files)} dataset(s) to process.")
    print(f"{'=' * 60}")

    for h_old_path in extracted_files:
        filename = os.path.basename(h_old_path)
        
        # Extract the dataset name prefix 
        dataset_name = filename.replace("extracted_captions_", "").replace(".jsonl", "")
        dataset_name = dataset_name.split("_")[0] 
        
        object_confidences_path = os.path.join(LOGITS_DIR, f"logits_{dataset_name}.json")
        output_file = os.path.join(OUTPUT_DIR, f"masked_captions_{dataset_name}.jsonl")
        
        if not os.path.exists(object_confidences_path):
            print(f"[Warning] Logits file missing for {dataset_name}. Expected at {object_confidences_path}")
            continue

        print(f"\nProcessing {dataset_name}...")
        print(f"  Inputs: {filename} & logits_{dataset_name}.json")

        with open(h_old_path, 'r', encoding='utf-8') as f:
            h_old_data = [json.loads(line) for line in f if line.strip()]

        with open(object_confidences_path, 'r', encoding='utf-8') as f:
            logits_json = json.load(f)
            object_confidences_data = logits_json.get("results", [])

        object_confidences_by_image = {}
        for item in object_confidences_data:
            image_name = item.get("image")
            object_name = item.get("object")
            if not image_name or not object_name:
                continue
            object_confidences_by_image.setdefault(image_name, {})[object_name] = item

        # Evaluate confidences and apply [IDK] masks
        for sample in h_old_data:
            sample["low_confidence_objects"] = []
            
            for obj in sample.get("extracted_objects", []):
                matching_obj = object_confidences_by_image.get(sample.get("image"), {}).get(obj)
                
                if matching_obj:
                    if matching_obj.get("confidence_visible") < 0.5:
                        sample["low_confidence_objects"].append(obj)

            masked_caption = sample.get("generated_caption", "")
            for obj in sample["low_confidence_objects"]:
                masked_caption = masked_caption.replace(obj, "[IDK]").strip()
            
            sample["masked_caption"] = masked_caption

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in h_old_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                
        print(f"  Saved masked output to: {output_file}")

if __name__ == "__main__":
    main()