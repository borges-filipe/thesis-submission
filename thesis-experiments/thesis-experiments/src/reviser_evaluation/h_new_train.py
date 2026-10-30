import json
import os

h_old_path = os.environ.get("H_OLD_PATH", "/mnt/storagecube/filipe/out/results/qwen3.5-9b/inserted_hallucinations_git10m/bigger-adversarial-captions-20260721_084450.jsonl")

output_dir = os.environ.get("OUTPUT_DIR", "/mnt/storagecube/filipe/out/results/qwen3.5-9b/inserted_hallucinations_git10m/")
output_file = os.path.join(output_dir, "h_new.jsonl")

with open(h_old_path, 'r', encoding='utf-8') as f:
    h_old_data = [json.loads(line) for line in f if line.strip()]

for sample in h_old_data:
    if sample["inserted_uncertain_object"]:
        sample["output_caption"] = sample["output_caption"].replace(sample["inserted_uncertain_object"], "[IDK]").strip()

with open(output_file, 'w', encoding='utf-8') as f:
    for sample in h_old_data:
        f.write(json.dumps(sample) + "\n")


