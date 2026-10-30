import json
import os

h_old_path = os.environ.get("H_OLD_PATH", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/extracted_objects/test-split-object-extraction-20260722_080652.jsonl")
object_confidences_path = os.environ.get("OBJECT_CONFIDENCES_PATH", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/ab_logits_masking/git10m-reviser-test-split_worker_0.jsonl")

output_dir = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/h_new")
output_file = os.path.join(output_dir, "h_new.jsonl")

os.makedirs(output_dir, exist_ok=True)

with open(h_old_path, 'r', encoding='utf-8') as f:
    h_old_data = [json.loads(line) for line in f if line.strip()]

with open(object_confidences_path, 'r', encoding='utf-8') as f:
    object_confidences_data = [json.loads(line) for line in f if line.strip()]

object_confidences_by_image = {}
for item in object_confidences_data:
    image_name = item.get("image")
    object_name = item.get("object")
    if not image_name or not object_name:
        continue
    object_confidences_by_image.setdefault(image_name, {})[object_name] = item

for sample in h_old_data:
    sample["low_confidence_objects"] = []
    for obj in sample.get("extracted_objects", []):
        matching_obj = object_confidences_by_image.get(sample.get("img_name"), {}).get(obj)
        if matching_obj:
            if matching_obj.get("confidence_visible") < 0.5:
                sample["low_confidence_objects"].append(obj)

    for obj in sample["low_confidence_objects"]:
        sample["generated_caption"] = sample["generated_caption"].replace(obj, "[IDK]").strip()

with open(output_file, 'w', encoding='utf-8') as f:
    for sample in h_old_data:
        f.write(json.dumps(sample) + "\n")


