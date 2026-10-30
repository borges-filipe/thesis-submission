#!/bin/bash

set -euo pipefail

MODEL="${1:?Usage: $0 <model> <ab|mcq|fa|se|qgen|reviser|rtrain|ab-dist|mcq-dist|obj-conf> [--full]}"
TASK="${2:-ab}"
MODE="${3:-smoke}"

PROJECT_ROOT="/home/filipe/experiments"
RESULTS_ROOT="/mnt/storagecube/filipe/out/results"
RSPOPE_JSON="/mnt/storagecube/filipe/rspope/rspope_evaluation/adversarial/rspope_adversarial_hard.json"
RSPOPE_IMAGES="/mnt/storagecube/filipe/rspope/rspope_evaluation/rspope_image/original/hard"
RSPOPE_ROOT="/mnt/storagecube/filipe/rspope/rspope_evaluation"
FA_IMAGES_ORIG="${RSPOPE_ROOT}/rspope_image/original/complete"
FA_IMAGES_DIST="${RSPOPE_ROOT}/rspope_image/vcd_diffusion_noise/complete/level_999"
DDFAV_BASE="/mnt/storagecube/filipe/rspope/DDFAV_Instruction_Set"


declare -A VENV_DIR=(
    ["llava-1.5-7b"]="venvs/llava-1.5-7b"
    ["rs-llava"]="venvs/rs-llava"
    ["geochat"]="venvs/geochat"
    ["qwen3-8b"]="venvs/qwen3-8b"
    ["qwen3.5-9b"]="venvs/qwen3.5-9b"
)
declare -A MCQ_TAG=(
    ["llava-1.5-7b"]="llava-1.5-7b"
    ["rs-llava"]="llava"
    ["geochat"]="geochat"
    ["qwen3-8b"]="qwen3"
    ["qwen3.5-9b"]="qwen3.5"
)

declare -A DIST_IMAGES=(
    ["diffusion"]="vcd_diffusion_noise"
    ["speckle"]="speckle_noise"
    ["gaussian"]="gaussian_noise"
    ["color_mae_red"]="color_mae_masking/red"
    ["color_mae_blue"]="color_mae_masking/blue"
    ["color_mae_green"]="color_mae_masking/green"
    ["color_mae_purple"]="color_mae_masking/purple"
)
declare -A DIST_OUT=(
    ["diffusion"]="diffusion_noise"
    ["speckle"]="speckle_noise"
    ["gaussian"]="gaussian_noise"
    ["color_mae_red"]="color_mae_masking/red"
    ["color_mae_blue"]="color_mae_masking/blue"
    ["color_mae_green"]="color_mae_masking/green"
    ["color_mae_purple"]="color_mae_masking/purple"
)

PY="${PROJECT_ROOT}/${VENV_DIR[$MODEL]:?unknown model: $MODEL}/bin/python"

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party"

if [[ "$TASK" == "ab" ]]; then
    if [[ "$MODE" == "--full" ]]; then
        CMD="${PY} -m hallucination_eval.tasks.ab_logits --model ${MODEL} --dataset AB_rspope --json ${RSPOPE_JSON} --images ${RSPOPE_IMAGES} --out ${RESULTS_ROOT}/${MODEL}/rspope"
        JOBNAME="${MODEL}_ab_rspope"
    else
        CMD="HALLUCINATION_SMOKE_MODEL=${MODEL} HALLUCINATION_SMOKE_JSON=${RSPOPE_JSON} HALLUCINATION_SMOKE_IMAGES=${RSPOPE_IMAGES} HALLUCINATION_SMOKE_REF=${RESULTS_ROOT}/${MODEL}/rspope/AB_rspope.json ${PY} ${PROJECT_ROOT}/tests/gpu/smoke_ab_logits.py"
        JOBNAME="${MODEL}_ab_smoke"
    fi
elif [[ "$TASK" == "ab-dist" || "$TASK" == "mcq-dist" ]]; then

    DIST="${3:?distortion}"
    LEVEL="${4:?level}"
    SPLIT="${5:-hard}"
    IMG_DIR="${RSPOPE_ROOT}/rspope_image/${DIST_IMAGES[$DIST]:?unknown distortion: $DIST}/${SPLIT}/level_${LEVEL}"
    if [[ "$TASK" == "mcq-dist" ]]; then
        OUT_DIR="${RESULTS_ROOT}/${MODEL}/rspope/mcq/logits/${DIST}"
        CMD="${PY} -m hallucination_eval.tasks.mcq_logits --model ${MODEL} --dataset level_${LEVEL} --filename MCQ_logits_level_${LEVEL}.json --json ${RSPOPE_JSON} --images ${IMG_DIR} --out ${OUT_DIR}"
        JOBNAME="${MODEL}_mcq_${DIST}_${LEVEL}"
    else
        OUT_DIR="${RESULTS_ROOT}/${MODEL}/rspope/${DIST_OUT[$DIST]}"
        CMD="${PY} -m hallucination_eval.tasks.ab_logits --model ${MODEL} --dataset level_${LEVEL} --json ${RSPOPE_JSON} --images ${IMG_DIR} --out ${OUT_DIR}"
        JOBNAME="${MODEL}_ab_${DIST}_${LEVEL}"
    fi
elif [[ "$TASK" == "mcq" ]]; then
    MCQ_REF="${RESULTS_ROOT}/${MODEL}/rspope/mcq/logits/MCQ_logits_${MCQ_TAG[$MODEL]}_RSPOPE_MCQ.json"
    if [[ "$MODE" == "--full" ]]; then
        CMD="${PY} -m hallucination_eval.tasks.mcq_logits --model ${MODEL} --dataset RSPOPE_MCQ --json ${RSPOPE_JSON} --images ${RSPOPE_IMAGES} --out ${RESULTS_ROOT}/${MODEL}/rspope/mcq/logits"
        JOBNAME="${MODEL}_mcq_rspope"
    else
        CMD="HALLUCINATION_SMOKE_MODEL=${MODEL} HALLUCINATION_SMOKE_IMAGES=${RSPOPE_IMAGES} HALLUCINATION_SMOKE_REF=${MCQ_REF} ${PY} ${PROJECT_ROOT}/tests/gpu/smoke_mcq_logits.py"
        JOBNAME="${MODEL}_mcq_smoke"
    fi
elif [[ "$TASK" == "fa" ]]; then
    FA_REF="${RESULTS_ROOT}/${MODEL}/rspope/full_answers/regular_decoding_temp_0/random.json"
    if [[ "$MODE" == "--full" ]]; then
        CMD="${PY} -m hallucination_eval.tasks.full_answers --model ${MODEL} --dataset rspope --type random --strategy all --images-orig ${FA_IMAGES_ORIG} --images-dist ${FA_IMAGES_DIST} --out ${RESULTS_ROOT}/${MODEL}/rspope"
        JOBNAME="${MODEL}_fa_rspope"
    else
        CMD="HALLUCINATION_SMOKE_MODEL=${MODEL} HALLUCINATION_SMOKE_RSPOPE=${RSPOPE_ROOT} HALLUCINATION_SMOKE_IMAGES=${FA_IMAGES_ORIG} HALLUCINATION_SMOKE_DIST=${FA_IMAGES_DIST} HALLUCINATION_SMOKE_REF=${FA_REF} ${PY} ${PROJECT_ROOT}/tests/gpu/smoke_full_answers.py"
        JOBNAME="${MODEL}_fa_smoke"
    fi
elif [[ "$TASK" == "se" ]]; then
    QA_DIR="${RESULTS_ROOT}/${MODEL}/git_vqa_answering_ddfav"
    DDFAV_IMAGE_DIRS="--image-dir ${DDFAV_BASE}/ai-tod --image-dir ${DDFAV_BASE}/dior --image-dir ${DDFAV_BASE}/dota --image-dir ${DDFAV_BASE}/fair1m --image-dir ${DDFAV_BASE}/visdrone-2019"
    if [[ "$MODE" == "--full" ]]; then
        CMD="${PY} -m hallucination_eval.tasks.semantic_entropy --model ${MODEL} --input-dir ${QA_DIR} ${DDFAV_IMAGE_DIRS} --out ${RESULTS_ROOT}/${MODEL}/git_vqa_semantic_entropy_ddfav --tag DDFAV"
        JOBNAME="${MODEL}_se_ddfav"
    else
        CMD="HALLUCINATION_SMOKE_MODEL=${MODEL} HALLUCINATION_SMOKE_QA=${QA_DIR} ${PY} ${PROJECT_ROOT}/tests/gpu/smoke_semantic_entropy.py"
        JOBNAME="${MODEL}_se_smoke"
    fi
elif [[ "$TASK" == "qgen" ]]; then
    # Qwen3.5-only (§5.3.2 caption→question generation).
    if [[ "$MODE" == "--full" ]]; then
        CMD="${PY} -m hallucination_eval.tasks.qgen --model ${MODEL} --arrow /mnt/storagecube/filipe/Git-10M-old/train/data-00000-of-00787.arrow --out ${RESULTS_ROOT}/${MODEL}/git_vqa_generation"
        JOBNAME="${MODEL}_qgen"
    else
        CMD="HALLUCINATION_SMOKE_MODEL=${MODEL} HALLUCINATION_SMOKE_IMAGE=${DDFAV_BASE}/ai-tod/2371__600_600.png ${PY} ${PROJECT_ROOT}/tests/gpu/smoke_qgen.py"
        JOBNAME="${MODEL}_qgen_smoke"
    fi
elif [[ "$TASK" == "reviser" ]]; then
    # §5.4 LURE caption reviser (RS-LLaVA merged + LoRA).
    REV_JSONL="/mnt/storagecube/filipe/out/lure_finetune_data/test_split/h_new/h_new.jsonl"
    REV_IMAGES="/mnt/storagecube/filipe/out/lure_finetune_data/test_split/images"
    if [[ "$MODE" == "--full" ]]; then
        CMD="${PY} -m hallucination_eval.tasks.reviser_inference --jsonl ${REV_JSONL} --images ${REV_IMAGES} --image-field img_name --caption-field generated_caption --out /mnt/storagecube/filipe/out/lure_finetune_data/test_split/revised_captions"
        JOBNAME="${MODEL}_reviser"
    else
        CMD="HALLUCINATION_SMOKE_JSONL=${REV_JSONL} HALLUCINATION_SMOKE_IMAGES=${REV_IMAGES} ${PY} ${PROJECT_ROOT}/tests/gpu/smoke_reviser.py"
        JOBNAME="${MODEL}_reviser_smoke"
    fi
elif [[ "$TASK" == "rtrain" ]]; then
    CMD="${PY} -m hallucination_eval.tasks.reviser_train --data /mnt/storagecube/filipe/out/lure_finetune_data/dataset.json --images /mnt/storagecube/filipe/out/lure_finetune_data/images --merged /mnt/storagecube/filipe/models/rs-llava-merged --out /mnt/storagecube/filipe/out/results/rs-llava-lure-lora"
    JOBNAME="${MODEL}_reviser_train"
elif [[ "$TASK" == "obj-conf" ]]; then

    OBJ_JSON="${3:?object-extraction JSONL}"
    OBJ_IMAGES="${4:?images dir}"
    OBJ_OUT="${5:?output dir}"
    DATASET="${6:-object_confidence}"
    CMD="${PY} -m hallucination_eval.tasks.object_confidence --model ${MODEL} --json ${OBJ_JSON} --images ${OBJ_IMAGES} --out ${OBJ_OUT} --dataset ${DATASET}"
    JOBNAME="${MODEL}_obj_conf"
else
    echo "unknown task: ${TASK} (expected ab, ab-dist, mcq, mcq-dist, fa, se, qgen, reviser, rtrain or obj-conf)" >&2
    exit 1
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --partition=big_job
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --chdir=${PROJECT_ROOT}
#SBATCH --output=${PROJECT_ROOT}/.slurm/%j_${JOBNAME}.out

export PYTHONPATH="${PYTHONPATH}"

${CMD}
EOF

echo "Submitted ${JOBNAME} (task=${TASK}, mode=${MODE}, model=${MODEL})."
