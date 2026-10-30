import pandas as pd

def sort_parquet():
    file_path = "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/combined_object_extraction.parquet"
    
    # 1. Load the combined parquet
    df = pd.read_parquet(file_path)
    
    # 2. Sort by the shard integer to group images by their respective Arrow files
    df_sorted = df.sort_values(by="dataset_shard")
    
    # 3. Save the sorted dataset
    output_path = file_path.replace(".parquet", "_sorted.parquet")
    df_sorted.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Sorted parquet saved to: {output_path}")

if __name__ == "__main__":
    sort_parquet()