import json
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

input_files = [
    "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m_worker_0.jsonl",
    "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m_worker_1.jsonl",
    "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m_worker_2.jsonl"
]
output_file = "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m_combined_sorted.parquet"
temp_file = output_file + ".tmp" # Temporary file for unsorted data

def combine_and_sort(inputs, output, chunk_size=250000):
    writer = None
    
    print("--- Phase 1: Streaming JSONL to Temporary Parquet ---")
    for file_path in inputs:
        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            
            # Skip metadata header
            f.readline()
            
            records = []
            for line in f:
                records.append(json.loads(line))
                
                if len(records) >= chunk_size:
                    table = pa.Table.from_pylist(records)
                    if writer is None:
                        writer = pq.ParquetWriter(temp_file, table.schema)
                    writer.write_table(table)
                    records = [] 
            
            if records:
                table = pa.Table.from_pylist(records)
                if writer is None:
                    writer = pq.ParquetWriter(temp_file, table.schema)
                writer.write_table(table)
                
    if writer is not None:
        writer.close()
        
    print("\n--- Phase 2: Loading and Sorting Data ---")
    print("Loading temporary Parquet file into memory...")
    
    # Read the unsorted parquet file into a PyArrow Table
    full_table = pq.read_table(temp_file)
    
    print("Sorting by 'confidence_visible' (smallest first)...")
    # Generate the sorted indices based on the target column
    sort_indices = pc.sort_indices(full_table, sort_keys=[("confidence_visible", "ascending")])
    
    # Apply the sorted indices to create a new ordered table
    sorted_table = full_table.take(sort_indices)
    
    print("Writing sorted data to final Parquet file...")
    # Write the globally sorted table to the final destination
    pq.write_table(sorted_table, output)
    
    import os
    os.remove(temp_file)
    
    print(f"\nSuccess! Sorted Parquet saved to: {output}")

if __name__ == "__main__":
    combine_and_sort(input_files, output_file)