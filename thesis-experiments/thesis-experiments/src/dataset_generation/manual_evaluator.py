import streamlit as st
import pandas as pd
import json
import os
import glob
import math
import numpy as np
from datasets import Dataset, concatenate_datasets
import streamlit.components.v1 as components

# --- Configuration ---
PARQUET_FILE = "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/git10m_combined_sorted_all_columns.parquet"
ARROW_DIR = "/mnt/storagecube/filipe/Git-10M/train"
OUTPUT_FILE = "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/manual_evaluations.jsonl"
PROGRESS_FILE = "/mnt/storagecube/filipe/out/results/rs-llava/git10m_logits/eval_progress.txt"

st.set_page_config(layout="wide", page_title="VLM Object Evaluator")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def get_google_maps_url(image_filename):
    """Extracts z, x, y directly from filenames like Google-11-11_351_771.jpg"""
    try:
        if not image_filename or not isinstance(image_filename, str):
            return None
            
        # Strip the extension and split by underscore
        # "Google-11-11_351_771.jpg" -> ["Google-11-11", "351", "771"]
        clean_name = image_filename.replace(".jpg", "").replace(".jpeg", "")
        parts = clean_name.split("_")
        
        if len(parts) != 3:
            return None
            
        # Parse the components
        y = int(parts[2])
        x = int(parts[1])
        # Get the last number from the "Google-11-11" prefix
        z = int(parts[0].split("-")[-1])
        
        # Standard Slippy map tilename math
        n = 2.0 ** z
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        
        return f"https://www.google.com/maps?q={lat_deg},{lon_deg}"
        
    except Exception as e:
        print(f"Failed to parse coordinates from {image_filename}: {e}")
        return None

@st.cache_resource
def load_datasets():
    df = pd.read_parquet(PARQUET_FILE)
    
    arrow_files = sorted(glob.glob(os.path.join(ARROW_DIR, "*.arrow")))
    arrow_datasets = [Dataset.from_file(f) for f in arrow_files]
    global_ds = concatenate_datasets(arrow_datasets)
    
    img_names_series = global_ds.data.column("img_name").to_pandas()
    clean_names = img_names_series.str.split('/').str[-1]
    index_map = dict(zip(clean_names, range(len(clean_names))))
    
    return df, global_ds, index_map

def get_current_index():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_current_index(index):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(index))

def save_evaluation(row_dict, is_visible):
    row_dict["object_visible"] = is_visible
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_dict, cls=NumpyEncoder) + "\n")
        
    st.session_state.current_idx += 1
    save_current_index(st.session_state.current_idx)
    st.rerun()

def undo_last_evaluation():
    if st.session_state.current_idx > 0:
        st.session_state.current_idx -= 1
        save_current_index(st.session_state.current_idx)
        
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if lines:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    f.writelines(lines[:-1])
                    
        st.rerun()

def main():
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = get_current_index()

    with st.spinner("Loading Parquet and Arrow datasets into memory..."):
        df, global_ds, index_map = load_datasets()
    
    total_samples = len(df)
    current_idx = st.session_state.current_idx
    
    if current_idx >= total_samples:
        st.success("🎉 You have evaluated all samples in the dataset!")
        return

    row = df.iloc[current_idx]
    row_dict = row.to_dict()
    
    image_filename = row_dict.get("image")
    target_object = row_dict.get("object")
    confidence = row_dict.get("confidence_visible")

    st.progress(current_idx / total_samples, text=f"Evaluated: {current_idx} / {total_samples}")
    
    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>Object: {target_object}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Confidence Visible: {confidence:.4f}</h3>", unsafe_allow_html=True)
    
    absolute_idx = index_map.get(os.path.basename(image_filename))
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if absolute_idx is not None:
            try:
                img = global_ds[absolute_idx]['image']
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image {image_filename}: {e}")
        else:
            st.error(f"Image {image_filename} not found in Arrow index.")

        st.write("") 
        b_col1, b_col2 = st.columns(2)
        
        with b_col1:
            if st.button("❌ False (Not Visible) [Left Arrow]", use_container_width=True, type="secondary"):
                save_evaluation(row_dict, False)
        
        with b_col2:
            if st.button("✅ True (Visible) [Right Arrow]", use_container_width=True, type="primary"):
                save_evaluation(row_dict, True)

        if st.session_state.current_idx > 0:
            if st.button("⏪ Undo Last Evaluation [Backspace]", use_container_width=True):
                undo_last_evaluation()
                
        map_url = get_google_maps_url(image_filename)
        if map_url:
            st.link_button("🗺️ Open Image Location in Google Maps", map_url, use_container_width=True)
                
    with st.expander("Show all sample metadata"):
        st.json(row_dict)

    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.addEventListener('keydown', function(e) {
            if (['INPUT', 'TEXTAREA'].includes(doc.activeElement.tagName)) return;

            const buttons = Array.from(doc.querySelectorAll('button'));
            
            if (e.key === 'ArrowLeft') {
                const falseBtn = buttons.find(b => b.innerText.includes('False'));
                if(falseBtn) falseBtn.click();
            } else if (e.key === 'ArrowRight') {
                const trueBtn = buttons.find(b => b.innerText.includes('True'));
                if(trueBtn) trueBtn.click();
            } else if (e.key === 'Backspace') {
                const undoBtn = buttons.find(b => b.innerText.includes('Undo'));
                if(undoBtn) {
                    undoBtn.click();
                    e.preventDefault(); 
                }
            }
        });
        </script>
        """,
        height=0,
        width=0,
    )

if __name__ == "__main__":
    main()