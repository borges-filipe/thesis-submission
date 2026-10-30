import json
import datetime

logits_files =[
    "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m.json",
    "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git10m_logits/git10m.json",
    "/mnt/storagecube/filipe/out/results/qwen3-8b/git10m_logits/git10m.json"
]

output_file = "average_logits.json"

# Create new file where the logits from all three models will be averaged and stored
# 2. Load the confidence and object data from the JSON file
with open(logits_files[0], 'r', encoding='utf-8') as f:
    logits_data_0 = json.load(f)

results_0 = logits_data_0.get("results", [])

for logits_file in logits_files[1:]:
    with open(logits_file, 'r', encoding='utf-8') as f:
        logits_data = json.load(f)
    results = logits_data.get("results", [])

    # Average the confidence_visible values for each object across the three models
    for i in range(len(results_0)):
        results_0[i]["confidence_visible"] += results[i]["confidence_visible"]

# Divide by the number of models to get the average
for result in results_0:
    result["confidence_visible"] /= len(logits_files)

# Save the averaged results to a new JSON file
logits_data_0["results"] = results_0
logits_data_0["model"] = "average_of_qwen3.5-9b_qwen3-8b_rs-llava"
logits_data_0["created_at"] = datetime.datetime.now().isoformat()

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(logits_data_0, f, ensure_ascii=False, indent=4)
