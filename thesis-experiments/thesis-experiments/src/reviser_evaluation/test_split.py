import json
import os
from datasets import Dataset


ARROW_PATH = os.environ.get("ARROW_PATH", "/mnt/storagecube/filipe/Git-10M/train/data-00002-of-00787.arrow")
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/storagecube/filipe/out/lure_finetune_data/test_split")

IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_JSON = os.path.join(OUT_DIR, "test_dataset_raw.json")

# Ensure output directories exist
os.makedirs(IMG_DIR, exist_ok=True)

def main():
    print(f"Loading 3rd Arrow shard: {ARROW_PATH}")
    try:
        hf_dataset = Dataset.from_file(ARROW_PATH)
    except Exception as e:
        print(f"Error loading Arrow file: {e}")
        return

    num_samples = min(1500, len(hf_dataset))
    print(f"Extracting {num_samples} samples...")

    test_data = []

    for i in range(num_samples):
        sample = hf_dataset[i]
        
        metadata = {key: value for key, value in sample.items() if key != "image"}
        
        img_name = sample["img_name"]
        img_obj = sample["image"]

        out_img_path = os.path.join(IMG_DIR, img_name)
        try:
            img_obj.save(out_img_path)
        except Exception as e:
            print(f"Failed to save image {img_name}: {e}")
            continue

        test_data.append(metadata)

        if (i + 1) % 250 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

    with open(OUT_JSON, 'w', encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"\nSuccessfully prepared {len(test_data)} test samples.")
    print(f"Images saved to: {IMG_DIR}")
    print(f"Metadata saved to: {OUT_JSON}")

if __name__ == "__main__":
    main()