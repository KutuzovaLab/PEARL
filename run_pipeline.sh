#!/bin/bash
# Run the pipeline on a dataset:        ./run_pipeline.sh wheat
# Use the published model:              CKPT=hf://UffeDalgas/temporary_p/best.ckpt ./run_pipeline.sh wheat
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${1:-wheat}"
CFG="configs/$DATASET.yaml"

# Build the environment.
if [ ! -d .venv_yucca ]; then
    command -v uv >/dev/null || { echo "please install uv first (https://docs.astral.sh/uv/)"; exit 1; }
    uv venv --python 3.11 .venv_yucca
    VIRTUAL_ENV=.venv_yucca uv pip install -r requirements.txt
    .venv_yucca/bin/python -m src.patch_yucca_npy
fi
PY=.venv_yucca/bin/python
BIN=.venv_yucca/bin

# Task name, patch/batch/epochs and expected image counts, read from the config.
eval "$("$PY" -m src.common --config "$CFG" --shell)"

# Yucca keeps its data under results/yucca.
export YUCCA_SOURCE="$PWD/results/yucca/source"
export YUCCA_RAW_DATA="$PWD/results/yucca/raw_data"
export YUCCA_PREPROCESSED_DATA="$PWD/results/yucca/preprocessed_data"
export YUCCA_MODELS="$PWD/results/yucca/models"
export YUCCA_RESULTS="$PWD/results/yucca/results"
mkdir -p results/yucca/{source,raw_data,preprocessed_data,models,results}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Get a model: the checkpoint or train on synthetic data once.
if [ -n "${CKPT:-}" ]; then
    SYNTH_TASK=Task043_sim14L3
    case "$CKPT" in hf://*) CKPT="$("$PY" -m src.fetch_checkpoint "$CKPT")" ;; esac
    "$PY" -m src.install_checkpoint --ckpt "$CKPT" --task "$SYNTH_TASK"
elif [ -n "${FORCE_TRAIN:-}" ] || [ ! -d "$YUCCA_MODELS/$SYNTH_TASK" ]; then
    "$PY" -m src.simulating_data_paper --config "$SYNTH_CFG" --section "$SYNTH_SEC"
    "$PY" -m src.taskconvert synthetic --src "$SYNTH_SRC" --task "$SYNTH_TASK"
    "$BIN/yucca_preprocess" -t "$SYNTH_TASK"
    "$BIN/yucca_train" -t "$SYNTH_TASK" -d 2D -m UNet \
        --epochs "$EPOCHS" --patch_size $PATCH --batch_size "$BATCH" \
        --num_workers "${WORKERS:-0}" --precision "${PRECISION:-32}" --disable_logging --new_version
fi

# images -> model predictions -> peak labels -> MS1-MS2 matching.
[ "$(ls "$EIC_DIR/images" 2>/dev/null | wc -l)" -eq "$EIC_FILES" ] || \
    "$PY" -m src.mzml_to_npy --config "$CFG" --section images
"$PY" -m src.taskconvert inference --src "$EIC_DIR/images" --task "$TASK" --prefix "$DATASET"
"$BIN/yucca_inference" -t "$TASK" -s "$SYNTH_TASK" -d 2D -m UNet \
    --version 0 --no_wandb --disable_tta --no_eval
"$PY" -m src.extracting_labels_erosion --config "$CFG" --section labels
[ "$ACQUISITION" = dda ] || "$PY" -m src.matching_MS1_MS2 --config "$CFG" --section matching
