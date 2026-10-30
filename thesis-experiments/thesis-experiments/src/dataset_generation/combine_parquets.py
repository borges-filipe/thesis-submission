import pandas as pd

file_extraction = '/mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/combined_object_extraction_sorted.parquet'
file_logits = '/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m_combined_sorted.parquet'
output_file = '/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m_combined_sorted_all_columns.parquet'

print("Loading dataframes...")
# Read the files
df_ext = pd.read_parquet(file_extraction, engine='pyarrow')
df_logits = pd.read_parquet(file_logits, engine='pyarrow')

print("Merging dataframes...")

merged_df = df_logits.merge(
    df_ext, 
    left_on='image', 
    right_on='img_name', 
    how='left'
)


merged_df = merged_df.drop(columns=['img_name'])

print("Saving to parquet...")
merged_df.to_parquet(output_file, engine='pyarrow', index=False)
print(f"Successfully saved merged dataset to: {output_file}")