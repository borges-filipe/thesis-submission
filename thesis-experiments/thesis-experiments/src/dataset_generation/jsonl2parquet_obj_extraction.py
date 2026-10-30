import pandas as pd

def combine_jsonl_to_parquet():
    # 1. Define the input file paths
    file_paths = [
        "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/0-199-object-extraction-20260701_124048.jsonl",
        "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/200-399-object-extraction-20260701_124246.jsonl",
        "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/400-599-object-extraction-20260701_135220.jsonl"
    ]
    
    # Define the output path (saving in the same directory)
    output_path = "/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/combined_object_extraction.parquet"
    
    dataframes = []
    
    # 2. Read each JSONL file and append to the list
    for file_path in file_paths:
        try:
            # lines=True is required for parsing .jsonl files
            df = pd.read_json(file_path, lines=True)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    # 3. Concatenate all dataframes into a single dataframe
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # 4. Save the combined dataframe to a Parquet file
        combined_df.to_parquet(output_path, engine='pyarrow', index=False)
        
        # 5. Print the total number of samples
        total_samples = len(combined_df)
        print(f"Success! The final parquet file contains {total_samples} samples.")
        print(f"Saved to: {output_path}")
    else:
        print("No data was loaded. Please check the file paths.")

if __name__ == "__main__":
    combine_jsonl_to_parquet()