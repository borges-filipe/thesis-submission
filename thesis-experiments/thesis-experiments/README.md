# On Detecting Hallucinations in Vision-Language Model Generated Captions for Remote Sensing Imagery

This repository contains the code necessary to reproduce the experiments performed in the Master Thesis 'On Detecting Hallucinations in Vision-Language Model Generated Captions for Remote Sensing Imagery'. 

## Reproducing the experiments

Some of the steps in these experiments are more resource intensive than others. These steps were grouped with two different labels:

1. **Generate**: Run a task inside a model's environment requiring GPU (uses SLURM by default).
2. **Metric**: Recompute the thesis numbers locally from pre-computed outputs (no GPU). 

For convenience, the pre-computed data for some experiments is included in this repo.  The experiments that can be run directly are labeled with **data-included**.

Additionally, the dataset of manually evaluated hallucinations is included under `$BASE/rs-llava/git10m_logits/manual_evaluations.jsonl`.


---

### Setup (run once)

```bash
# 1. Create a virtual environment and install the metric dependencies.
#    (Requires uv: https://docs.astral.sh/uv/)
uv venv
uv pip install numpy scikit-learn

# 2. Export the paths used by every Metric command below.
export BASE="$PWD/results"            # bundled pre-computed outputs
export PY="$PWD/.venv/bin/python"     # the venv created in step 1
export PYTHONPATH="$PWD/src"          # the code lives here
```

Every **Metric** command below assumes these three variables are exported.

**Generate** commands run via `submit_shared_job.sh`.

### Models

| key | HF model | `venvs/` dir |
|---|---|---|
| `llava-1.5-7b` | liuhaotian/llava-v1.5-7b | `llava-1.5-7b` |
| `rs-llava` | BigData-KSU/RS-llava-v1.5-7b-LoRA | `rs-llava` |
| `geochat` | MBZUAI/geochat-7B | `geochat` |
| `qwen3-8b` | Qwen/Qwen3-VL-8B-Instruct | `qwen3-8b` |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B | `qwen3.5-9b` |

---

### Datasets

Download each dataset once (replace the `<HF:…>` placeholders with the real
HuggingFace repo IDs):

| Dataset | Download |
|---|---|
| RS-POPE | https://github.com/HaodongLi2024/rspope/commit/d0d0b00fcdeef8a7a7f9af5ebbbe8ea2937b83f6 |
| LID | https://huggingface.co/datasets/NLIP-lab/LIDgi |
| POPE | https://github.com/DAMO-NLP-SG/VCD/tree/master/experiments/data/POPE |
| DDFAV | https://github.com/HaodongLi2024/rspope/tree/master/DDFAV_Instruction%20Set |
| Git-10M | https://huggingface.co/datasets/lcybuaa/Git-10M |
| GAIA | https://huggingface.co/datasets/azavras/GAIA |


#### Distorted images

Generate the distorted-image sets from the
clean RS-POPE / LID / POPE images:

```bash
# VCD forward-diffusion noise 
venvs/image-distortion/bin/python src/image_distortion/distort_diffusion.py
# ColorMAE patch masking (color_mae_red/blue/green/purple)
venvs/image-distortion/bin/python src/image_distortion/distort_colormae.py
# LID distortions
venvs/image-distortion/bin/python src/image_distortion/distort_lid.py
# POPE distortions
venvs/image-distortion/bin/python src/image_distortion/distort_pope.py
```

---

### Binary baseline  **data-included** 

**Data**: RS-POPE or LID.

**Generate** (one job per model, via SLURM):

```bash
./submit_shared_job.sh llava-1.5-7b ab --full
./submit_shared_job.sh rs-llava     ab --full
./submit_shared_job.sh qwen3-8b     ab --full
./submit_shared_job.sh qwen3.5-9b   ab --full
./submit_shared_job.sh geochat      ab --full
```

**Metric** (RS-POPE):

```bash
$PY -m hallucination_eval.cli report "$BASE/llava-1.5-7b/rspope/AB_rspope.json"
$PY -m hallucination_eval.cli report "$BASE/rs-llava/rspope/AB_rspope.json"
$PY -m hallucination_eval.cli report "$BASE/qwen3-8b/rspope/AB_rspope.json"
$PY -m hallucination_eval.cli report "$BASE/qwen3.5-9b/rspope/AB_rspope.json"
$PY -m hallucination_eval.cli report "$BASE/geochat/rspope/AB_rspope.json"
```

**Metric** (LID):

```bash
$PY -m hallucination_eval.cli report "$BASE/llava-1.5-7b/lid/binary/logits/AB_LID_balanced.json"
$PY -m hallucination_eval.cli report "$BASE/rs-llava/lid/binary/logits/AB_LID_balanced.json"
$PY -m hallucination_eval.cli report "$BASE/qwen3-8b/lid/AB_LID_balanced.json"
$PY -m hallucination_eval.cli report "$BASE/qwen3.5-9b/lid/AB_LID_balanced.json"
$PY -m hallucination_eval.cli report "$BASE/geochat/lid/binary/logits/AB_LID_balanced.json"
```

---

### MCQ baseline  **data-included** 

**Generate** (one job per model):

```bash
./submit_shared_job.sh llava-1.5-7b mcq --full
./submit_shared_job.sh rs-llava     mcq --full
./submit_shared_job.sh qwen3-8b     mcq --full
./submit_shared_job.sh qwen3.5-9b   mcq --full
./submit_shared_job.sh geochat      mcq --full
```

**Metric** (RS-POPE):

```bash
$PY -m hallucination_eval.cli mcq "$BASE/llava-1.5-7b/rspope/mcq/logits/MCQ_logits_llava-1.5-7b_RSPOPE_MCQ.json"
$PY -m hallucination_eval.cli mcq "$BASE/rs-llava/rspope/mcq/logits/MCQ_logits_llava_RSPOPE_MCQ.json"
$PY -m hallucination_eval.cli mcq "$BASE/qwen3-8b/rspope/mcq/logits/MCQ_logits_qwen3_RSPOPE_MCQ.json"
$PY -m hallucination_eval.cli mcq "$BASE/qwen3.5-9b/rspope/mcq/logits/MCQ_logits_qwen3.5_RSPOPE_MCQ.json"
$PY -m hallucination_eval.cli mcq "$BASE/geochat/rspope/mcq/logits/MCQ_logits_geochat_RSPOPE_MCQ.json"
```

**Metric** (LID):

```bash
$PY -m hallucination_eval.cli mcq "$BASE/llava-1.5-7b/lid/mcq/logits_v2/MCQ_logits_llava_LID_MCQ_level_0.00.json"
$PY -m hallucination_eval.cli mcq "$BASE/rs-llava/lid/mcq/logits_v2/MCQ_logits_llava_LID_MCQ_level_0.00.json"
$PY -m hallucination_eval.cli mcq "$BASE/qwen3-8b/lid/mcq/logits_v2/MCQ_logits_qwen3_LID_MCQ_level_0.00.json"
$PY -m hallucination_eval.cli mcq "$BASE/qwen3.5-9b/lid/mcq/logits_v2/MCQ_logits_qwen3.5_LID_MCQ.json"
$PY -m hallucination_eval.cli mcq "$BASE/geochat/lid/mcq/logits/MCQ_logits_geochat_LID_MCQ.json"
```

---

### Binary VCD **data-included**



**Data**: RS-POPE with diffusion-noise distorted images.

**Generate**:

```bash
# original 
./submit_shared_job.sh llava-1.5-7b ab --full
./submit_shared_job.sh rs-llava     ab --full
./submit_shared_job.sh qwen3-8b     ab --full
./submit_shared_job.sh qwen3.5-9b   ab --full
./submit_shared_job.sh geochat      ab --full

# distorted
./submit_shared_job.sh llava-1.5-7b ab-dist diffusion 999 complete
./submit_shared_job.sh rs-llava     ab-dist diffusion 999 complete
./submit_shared_job.sh qwen3-8b     ab-dist diffusion 999 complete
./submit_shared_job.sh qwen3.5-9b   ab-dist diffusion 999 complete
./submit_shared_job.sh geochat      ab-dist diffusion 999 complete
```

**Metric** (RS-POPE):

```bash
$PY -m hallucination_eval.cli vcd "$BASE/llava-1.5-7b/rspope/AB_rspope.json" "$BASE/llava-1.5-7b/rspope/diffusion_noise/level_999.json"
$PY -m hallucination_eval.cli vcd "$BASE/rs-llava/rspope/AB_rspope.json" "$BASE/rs-llava/rspope/diffusion_noise/level_999.json"
$PY -m hallucination_eval.cli vcd "$BASE/qwen3-8b/rspope/AB_rspope.json" "$BASE/qwen3-8b/rspope/diffusion_noise/level_999.json"
$PY -m hallucination_eval.cli vcd "$BASE/qwen3.5-9b/rspope/AB_rspope.json" "$BASE/qwen3.5-9b/rspope/diffusion_noise/level_999.json"
$PY -m hallucination_eval.cli vcd "$BASE/geochat/rspope/AB_rspope.json" "$BASE/geochat/rspope/diffusion_noise/level_level_999.json"
```

**Metric** (LID):

```bash
$PY -m hallucination_eval.cli vcd "$BASE/llava-1.5-7b/lid/binary/logits/AB_LID_balanced.json" "$BASE/llava-1.5-7b/lid/all_noise/diffusion_noise/level_999.0.json"
$PY -m hallucination_eval.cli vcd "$BASE/rs-llava/lid/binary/logits/AB_LID_balanced.json" "$BASE/rs-llava/lid/all_noise/diffusion_noise/level_999.0.json"
$PY -m hallucination_eval.cli vcd "$BASE/qwen3-8b/lid/AB_LID_balanced.json" "$BASE/qwen3-8b/lid/all_noise/diffusion_noise/level_999.0.json"
$PY -m hallucination_eval.cli vcd "$BASE/qwen3.5-9b/lid/AB_LID_balanced.json" "$BASE/qwen3.5-9b/lid/all_noise/diffusion_noise/level_999.0.json"
$PY -m hallucination_eval.cli vcd "$BASE/geochat/lid/binary/logits/AB_LID_balanced.json" "$BASE/geochat/lid/all_noise/diffusion_noise/level_999.0.json"
```


#### VCD with black images **data-included** 

**Metric**:

```bash
$PY -m hallucination_eval.cli vcd-black "$BASE/llava-1.5-7b/rspope/AB_rspope.json" "$BASE/llava-1.5-7b/rspope/color_mae_masking/blue/level_1.0.json"
$PY -m hallucination_eval.cli vcd-black "$BASE/rs-llava/rspope/AB_rspope.json" "$BASE/rs-llava/rspope/color_mae_masking/blue/level_1.0.json"
$PY -m hallucination_eval.cli vcd-black "$BASE/qwen3-8b/rspope/AB_rspope.json" "$BASE/qwen3-8b/rspope/color_mae_masking/blue/level_1.0.json"
$PY -m hallucination_eval.cli vcd-black "$BASE/qwen3.5-9b/rspope/AB_rspope.json" "$BASE/qwen3.5-9b/rspope/color_mae_masking/blue/level_1.0.json"
$PY -m hallucination_eval.cli vcd-black "$BASE/geochat/rspope/AB_rspope.json" "$BASE/geochat/rspope/color_mae_masking/blue/level_level_1.0.json"
```

---

### Distortion F1 **data-included**

**Data**: RS-POPE and all distortion types.

**Generate**:

```bash
for m in llava-1.5-7b rs-llava qwen3-8b qwen3.5-9b geochat; do
  ./submit_shared_job.sh "$m" ab-dist speckle 1.0
  ./submit_shared_job.sh "$m" ab-dist diffusion 999 complete
  ./submit_shared_job.sh "$m" ab-dist color_mae_blue 1.0
  ./submit_shared_job.sh "$m" ab-dist color_mae_red 1.0
  ./submit_shared_job.sh "$m" ab-dist color_mae_green 1.0
  ./submit_shared_job.sh "$m" ab-dist color_mae_purple 1.0
done
```

**Metric**:

```bash
for m in llava-1.5-7b rs-llava qwen3-8b qwen3.5-9b geochat; do
  $PY -m hallucination_eval.cli distortion "$BASE/$m/rspope/AB_rspope.json" \
    --dir speckle="$BASE/$m/rspope/speckle_noise" \
    --dir diffusion="$BASE/$m/rspope/diffusion_noise" \
    --dir color_mae_red="$BASE/$m/rspope/color_mae_masking/red" \
    --dir color_mae_blue="$BASE/$m/rspope/color_mae_masking/blue" \
    --dir color_mae_green="$BASE/$m/rspope/color_mae_masking/green" \
    --dir color_mae_purple="$BASE/$m/rspope/color_mae_masking/purple"
done
```

**Metric** (LID):

```bash
$PY -m hallucination_eval.cli distortion "$BASE/llava-1.5-7b/lid/binary/logits/AB_LID_balanced.json" \
  --dir speckle="$BASE/llava-1.5-7b/lid/all_noise/speckle_noise" \
  --dir diffusion="$BASE/llava-1.5-7b/lid/all_noise/diffusion_noise" \
  --dir color_mae_red="$BASE/llava-1.5-7b/lid/all_noise/color_mae_masking/red" \
  --dir color_mae_blue="$BASE/llava-1.5-7b/lid/all_noise/color_mae_masking/blue" \
  --dir color_mae_green="$BASE/llava-1.5-7b/lid/all_noise/color_mae_masking/green" \
  --dir color_mae_purple="$BASE/llava-1.5-7b/lid/all_noise/color_mae_masking/purple"

$PY -m hallucination_eval.cli distortion "$BASE/rs-llava/lid/binary/logits/AB_LID_balanced.json" \
  --dir speckle="$BASE/rs-llava/lid/all_noise/speckle_noise" \
  --dir diffusion="$BASE/rs-llava/lid/all_noise/diffusion_noise" \
  --dir color_mae_red="$BASE/rs-llava/lid/all_noise/color_mae_masking/red" \
  --dir color_mae_blue="$BASE/rs-llava/lid/all_noise/color_mae_masking/blue" \
  --dir color_mae_green="$BASE/rs-llava/lid/all_noise/color_mae_masking/green" \
  --dir color_mae_purple="$BASE/rs-llava/lid/all_noise/color_mae_masking/purple"

$PY -m hallucination_eval.cli distortion "$BASE/qwen3-8b/lid/AB_LID_balanced.json" \
  --dir speckle="$BASE/qwen3-8b/lid/all_noise/speckle_noise" \
  --dir diffusion="$BASE/qwen3-8b/lid/all_noise/diffusion_noise" \
  --dir color_mae_red="$BASE/qwen3-8b/lid/all_noise/color_mae_masking/red" \
  --dir color_mae_blue="$BASE/qwen3-8b/lid/all_noise/color_mae_masking/blue" \
  --dir color_mae_green="$BASE/qwen3-8b/lid/all_noise/color_mae_masking/green" \
  --dir color_mae_purple="$BASE/qwen3-8b/lid/all_noise/color_mae_masking/purple"

$PY -m hallucination_eval.cli distortion "$BASE/qwen3.5-9b/lid/AB_LID_balanced.json" \
  --dir speckle="$BASE/qwen3.5-9b/lid/all_noise/speckle_noise" \
  --dir diffusion="$BASE/qwen3.5-9b/lid/all_noise/diffusion_noise" \
  --dir color_mae_red="$BASE/qwen3.5-9b/lid/all_noise/color_mae_masking/red" \
  --dir color_mae_blue="$BASE/qwen3.5-9b/lid/all_noise/color_mae_masking/blue" \
  --dir color_mae_green="$BASE/qwen3.5-9b/lid/all_noise/color_mae_masking/green" \
  --dir color_mae_purple="$BASE/qwen3.5-9b/lid/all_noise/color_mae_masking/purple"

$PY -m hallucination_eval.cli distortion "$BASE/geochat/lid/binary/logits/AB_LID_balanced.json" \
  --dir speckle="$BASE/geochat/lid/all_noise/speckle_noise" \
  --dir diffusion="$BASE/geochat/lid/all_noise/diffusion_noise" \
  --dir color_mae_red="$BASE/geochat/lid/all_noise/color_mae_masking/red" \
  --dir color_mae_blue="$BASE/geochat/lid/all_noise/color_mae_masking/blue" \
  --dir color_mae_green="$BASE/geochat/lid/all_noise/color_mae_masking/green" \
  --dir color_mae_purple="$BASE/geochat/lid/all_noise/color_mae_masking/purple"
```

---

### MCQ+VCD (data not included)

**Generate**:

```bash
for m in llava-1.5-7b rs-llava qwen3-8b qwen3.5-9b geochat; do
  ./submit_shared_job.sh "$m" mcq-dist speckle 1.0
  ./submit_shared_job.sh "$m" mcq-dist diffusion 999 complete
  ./submit_shared_job.sh "$m" mcq-dist color_mae_blue 1.0
  ./submit_shared_job.sh "$m" mcq-dist color_mae_red 1.0
  ./submit_shared_job.sh "$m" mcq-dist color_mae_green 1.0
  ./submit_shared_job.sh "$m" mcq-dist color_mae_purple 1.0
done
```

**Metric**:

```bash
for m in llava-1.5-7b rs-llava qwen3-8b qwen3.5-9b geochat; do
  $PY -m hallucination_eval.cli mcq-vcd \
    --dir color_mae_blue="$BASE/$m/rspope/mcq/logits/color_mae_blue" \
    --dir color_mae_green="$BASE/$m/rspope/mcq/logits/color_mae_green" \
    --dir color_mae_purple="$BASE/$m/rspope/mcq/logits/color_mae_purple" \
    --dir color_mae_red="$BASE/$m/rspope/mcq/logits/color_mae_red" \
    --dir diffusion="$BASE/$m/rspope/mcq/logits/diffusion" \
    --dir speckle="$BASE/$m/rspope/mcq/logits/speckle"
done
```

---

### Full-answer decoding strategies **data-included**

**Generate**:

```bash
./submit_shared_job.sh llava-1.5-7b fa --full
```

**Metric** (POPE and RS-POPE):

```bash
$PY -m hallucination_eval.cli debunking "$BASE/llava-1.5-7b/pope"
$PY -m hallucination_eval.cli debunking "$BASE/llava-1.5-7b/rspope"
```

---

### Manual evaluation **data-included** 


```bash
# 1. Extract objects from Git-10M captions (Qwen3.5)
venvs/dataset-generation/bin/python src/dataset_generation/object_extraction.py
# 2. Object-extraction JSONL -> parquet
venvs/dataset-generation/bin/python src/dataset_generation/jsonl2parquet_obj_extraction.py
# 3. Sort objects by shard
venvs/dataset-generation/bin/python src/dataset_generation/sort_parquet.py
# 4. Object-level visibility confidence (RS-LLaVA binary probing)
./submit_shared_job.sh rs-llava obj-conf \
  /mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/qa_generation_TEST_20260622_125314.jsonl \
  /mnt/storagecube/filipe/out/results/qwen3.5-9b/git_object_extraction/pics \
  /mnt/storagecube/filipe/out/results/rs-llava/git10m_logits \
  git10m
# 5. Logits JSONL -> parquet
venvs/dataset-generation/bin/python src/dataset_generation/jsonl2parquet_logits.py
# 6. Merge object extraction + logits
venvs/dataset-generation/bin/python src/dataset_generation/combine_parquets.py
# 7. Average logits across models (optional)
venvs/dataset-generation/bin/python src/dataset_generation/average_logits.py
# 8. Manual annotation (Streamlit app)
venvs/dataset-generation/bin/python -m streamlit run src/dataset_generation/manual_evaluator.py
# 9. Stats -> tab:manual_evaluation_low_conf (see Metric below)
```

**Metric**:

```bash
$PY -m hallucination_eval.cli manual-eval \
  "$BASE/rs-llava/git10m_logits/manual_evaluations.jsonl"
```

---

### Semantic entropy **data-included** 

**Data**: DDFAV (VQA) + Git-10M (captions).

Three stages:

```bash
# 1. 11-answer sampling
./submit_shared_job.sh llava-1.5-7b se --full
# 2. caption -> question generation
./submit_shared_job.sh qwen3.5-9b qgen --full
# 3. entailment clustering + SE: runs inside `se --full`
```

**Metric**:

```bash
# DDFAV (VQA)
$PY -m hallucination_eval.cli se "$(ls "$BASE"/llava-1.5-7b/git_vqa_semantic_entropy_ddfav/qa_semantic_entropy_DDFAV_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/rs-llava/git_vqa_semantic_entropy_ddfav/qa_semantic_entropy_DDFAV_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/qwen3-8b/git_vqa_semantic_entropy_ddfav/qa_semantic_entropy_DDFAV_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/qwen3.5-9b/git_vqa_semantic_entropy_ddfav/qa_semantic_entropy_DDFAV_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/geochat/git_vqa_semantic_entropy_ddfav/qa_semantic_entropy_DDFAV_*.jsonl | sort | tail -1)"

# Git-10M (captions)
$PY -m hallucination_eval.cli se "$(ls "$BASE"/llava-1.5-7b/git_vqa_semantic_entropy/qa_semantic_entropy_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/rs-llava/git_vqa_semantic_entropy/qa_semantic_entropy_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/qwen3-8b/git_vqa_semantic_entropy/qa_semantic_entropy_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/qwen3.5-9b/git_vqa_semantic_entropy/qa_semantic_entropy_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli se "$(ls "$BASE"/geochat/git_vqa_semantic_entropy/qa_semantic_entropy_*.jsonl | sort | tail -1)"
```

---

### LURE caption reviser (train and deploy) (data not included)

**Data**: Git-10M (train/test) + GAIA / RSTeller (evaluation)
#### Data preparation (masked captions)

```bash
# 1. Extract a test split from Git-10M
venvs/reviser-evaluation/bin/python src/reviser_evaluation/test_split.py
# 2. Generate new captions
venvs/reviser-evaluation/bin/python src/reviser_evaluation/generate_captions.py
# 3. Extract objects from the captions
venvs/reviser-evaluation/bin/python src/reviser_evaluation/object_extraction.py
# 4. Object-level visibility confidence
./submit_shared_job.sh rs-llava obj-conf \
  /mnt/storagecube/filipe/out/lure_finetune_data/test_split/extracted_objects/test-split-object-extraction-20260722_080652.jsonl \
  /mnt/storagecube/filipe/out/lure_finetune_data/test_split/images \
  /mnt/storagecube/filipe/out/lure_finetune_data/test_split/ab_logits_masking \
  git10m-reviser-test-split
# 5. Mask low-confidence objects
$PY -m hallucination_eval.tasks.mask_captions \
  --objects /mnt/storagecube/filipe/out/lure_finetune_data/test_split/extracted_objects/test-split-object-extraction-20260722_080652.jsonl \
  --confidences /mnt/storagecube/filipe/out/lure_finetune_data/test_split/ab_logits_masking/git10m-reviser-test-split_worker_0.jsonl \
  --out /mnt/storagecube/filipe/out/lure_finetune_data/test_split/h_new/h_new.jsonl

### Training data (Git-10M inserted hallucinations)

# 6. Insert a co-object + an uncertain object into each caption
venvs/reviser-evaluation/bin/python src/reviser_evaluation/h_old.py
# 7. Mask the inserted uncertain object
venvs/reviser-evaluation/bin/python src/reviser_evaluation/h_new_train.py
# 8. Assemble the LURE fine-tune dataset 
venvs/reviser-evaluation/bin/python src/reviser_evaluation/prepare_lure_data.py

### Masked captions for the evaluation datasets

# GAIA / RSTeller / RS-POPE / DDFAV
venvs/reviser-evaluation/bin/python src/reviser_evaluation/h_new_gaia.py
venvs/reviser-evaluation/bin/python src/reviser_evaluation/h_new_rsteller.py
venvs/reviser-evaluation/bin/python src/reviser_evaluation/h_new_rspope.py
venvs/reviser-evaluation/bin/python src/reviser_evaluation/ddfav_h_new.py
```

**Generate** (train + deploy):

```bash
# Train the reviser (LoRA on the merged RS-LLaVA; hours-long)
./submit_shared_job.sh rs-llava rtrain

# Deploy the reviser (batched greedy revision of masked captions)
./submit_shared_job.sh rs-llava reviser --full
```

**Metric** (Semantic-CHAIR + BLEU/CIDEr):

```bash
$PY -m hallucination_eval.cli chair \
  "$(ls /mnt/storagecube/filipe/out/lure_finetune_data/test_split/evaluation/chair_evaluation_*.jsonl | sort | tail -1)"
$PY -m hallucination_eval.cli bleu-cider GAIA
$PY -m hallucination_eval.cli bleu-cider Git-10M
$PY -m hallucination_eval.cli bleu-cider DDFAV
```

---

### Figures 

```bash
$PY -m hallucination_eval.figures --out outputs/figures
```

---

Thesis written at the Remote Sensing Image Analysis Group at Technische Universität Berlin.

Author: Filipe Alexandre Fabian Borges

Supervisor: Prof. Dr. Begüm Demir

Advisor: Marvin Beese