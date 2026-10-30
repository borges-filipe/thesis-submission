import json
import os

h_old_path = os.environ.get("H_OLD_PATH", "/mnt/storagecube/filipe/RSTeller_download/gen_extracted_objects/rsteller-gen-object-extraction-20260813_162315.jsonl")
object_confidences_paths = os.environ.get(
    "OBJECT_CONFIDENCES_PATHS",
    "/mnt/storagecube/filipe/RSTeller_download/ab_logits_gen/rsteller-gen-objects_worker_0.jsonl",
).split(",")

output_dir = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/RSTeller_download/h_new")
output_file = os.path.join(output_dir, "h_new.jsonl")

os.makedirs(output_dir, exist_ok=True)

# Load h_old captions with extracted objects
with open(h_old_path, 'r', encoding='utf-8') as f:
    h_old_data = [json.loads(line) for line in f if line.strip()]

# Load object confidences from all worker files (skip metadata header lines)
object_confidences_data = []
for path in object_confidences_paths:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            if 'total_samples' in item:
                continue
            object_confidences_data.append(item)

print(f"Loaded {len(h_old_data)} caption samples")
print(f"Loaded {len(object_confidences_data)} object confidence entries from {len(object_confidences_paths)} workers")

# image_key: {object_name: confidence_data}}
object_confidences_by_image = {}
for item in object_confidences_data:
    image_key = item.get("image")
    object_name = item.get("object")
    if not image_key or not object_name:
        continue
    object_confidences_by_image.setdefault(image_key, {})[object_name] = item

low_conf_count = 0
total_objects = 0

for sample in h_old_data:
    sample["low_confidence_objects"] = []
    image_key = sample.get("key")
    for obj in sample.get("extracted_objects", []):
        total_objects += 1
        matching_obj = object_confidences_by_image.get(image_key, {}).get(obj)
        if matching_obj:
            if matching_obj.get("confidence_visible", 0) < 0.5:
                sample["low_confidence_objects"].append(obj)
                low_conf_count += 1

    # Replace low-confidence objects with [IDK]
    for obj in sample["low_confidence_objects"]:
        sample["generated_caption"] = sample["generated_caption"].replace(obj, "[IDK]").strip()

with open(output_file, 'w', encoding='utf-8') as f:
    for sample in h_old_data:
        f.write(json.dumps(sample) + "\n")

print(f"Total objects: {total_objects}")
print(f"Low-confidence objects masked: {low_conf_count} ({low_conf_count/total_objects*100:.1f}%)")
print(f"Saved: {output_file}")
