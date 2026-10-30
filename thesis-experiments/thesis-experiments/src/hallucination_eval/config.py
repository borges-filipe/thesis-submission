from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("HALLUCINATION_RESULTS_ROOT", str(REPO_ROOT / "results"))
)

# HuggingFace cache directory.
HF_CACHE_DIR = os.environ.get("HALLUCINATION_HF_CACHE", "/data/filipe/hf-cache")

# Dataset roots.
DATASETS = {
    "pope": Path(
        os.environ.get("HALLUCINATION_DATA_POPE", "/mnt/storagecube/filipe/POPE_Exact_VCD")
    ),
    "rspope": Path(
        os.environ.get("HALLUCINATION_DATA_RSPOPE", "/mnt/storagecube/filipe/rspope/rspope_evaluation")
    ),
    "lid": Path(os.environ.get("HALLUCINATION_DATA_LID", "/mnt/storagecube/filipe/lid")),
    "ddfav": Path(
        os.environ.get("HALLUCINATION_DATA_DDFAV", "/mnt/storagecube/filipe/rspope/DDFAV_Instruction_Set")
    ),
    "git10m": Path(os.environ.get("HALLUCINATION_DATA_GIT10M", "/mnt/storagecube/filipe/Git-10M")),
}

MODEL_IDS = {
    "llava-1.5-7b": "liuhaotian/llava-v1.5-7b",
    "rs-llava-7b": "BigData-KSU/RS-llava-v1.5-7b-LoRA",
    "geochat-7b": "MBZUAI/geochat-7B",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}
