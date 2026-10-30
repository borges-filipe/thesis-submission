import json
import os
from datasets import Dataset

JSONL_PATH = os.environ.get("JSONL_PATH", "/mnt/storagecube/filipe/out/results/qwen3.5-9b/inserted_hallucinations_git10m/h_new.jsonl")
ARROW_PATH = os.environ.get("ARROW_PATH", "/mnt/storagecube/filipe/Git-10M/train/data-00000-of-00787.arrow")
ARROW_PATH_2 = os.environ.get("ARROW_PATH_2", "/mnt/storagecube/filipe/Git-10M/train/data-00001-of-00787.arrow")
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "5000"))

IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_JSON = os.path.join(OUT_DIR, "dataset.json")

os.makedirs(IMG_DIR, exist_ok=True)

print("Loading Arrow dataset...")
hf_dataset = Dataset.from_file(ARROW_PATH)
hf_dataset_2 = Dataset.from_file(ARROW_PATH_2)

print("Building global image index...")
img_name_to_idx = {name: idx for idx, name in enumerate(hf_dataset['img_name'])}
img_name_to_idx_2 = {name: idx for idx, name in enumerate(hf_dataset_2['img_name'])}

print("Processing JSONL...")
llava_data = []
INSTRUCTION = (
    "You are a remote sensing image caption reviser. Revise the following image description to ensure "
    "it strictly matches the visual content. First, resolve any \"[IDK]\" placeholders by replacing them "
    "with the correct objects from the image or deleting them entirely. Second, verify all other explicitly "
    "named objects in the text and remove any items that are not physically present in the visual scene."
)

with open(JSONL_PATH, 'r') as f:
    lines = f.readlines()[:MAX_SAMPLES]

for i, line in enumerate(lines):
    data = json.loads(line)
    img_name = data['image']
    hallucinated_caption = data['output_caption']
    ground_truth_caption = data['caption']

    if img_name not in img_name_to_idx:
        if img_name not in img_name_to_idx_2:
            print(f"Warning: {img_name} not found in either Arrow file. Skipping.")
            continue

        idx = img_name_to_idx_2[img_name]
        img_obj = hf_dataset_2[idx]['image']
    else:
        idx = img_name_to_idx[img_name]
        img_obj = hf_dataset[idx]['image']

    out_img_path = os.path.join(IMG_DIR, img_name)
    img_obj.save(out_img_path)

    # Format for LLaVA
    llava_data.append({
        "id": f"lure_{i}",
        "image": img_name,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{INSTRUCTION}\n\nDescription to revise: {hallucinated_caption}"
            },
            {
                "from": "gpt",
                "value": ground_truth_caption
            }
        ]
    })

with open(OUT_JSON, 'w') as f:
    json.dump(llava_data, f, indent=2)

print(f"Successfully prepared {len(llava_data)} samples. Data saved to {OUT_DIR}")