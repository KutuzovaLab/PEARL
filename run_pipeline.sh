#!/bin/bash
set -euo pipefail

uv venv --python 3.11 .venv_yucca
VIRTUAL_ENV=.venv_yucca uv pip install -r requirements.txt

uv venv --python 3.11 .venv_dreams
VIRTUAL_ENV=.venv_dreams uv pip install git+https://github.com/pluskal-lab/DreaMS.git openpyxl

DATASET="${1:-wheat}"
WORKERS="${WORKERS:-0}"
PRECISION="${PRECISION:-32}"

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
PY="$REPO/.venv_yucca/bin/python"
BIN="$REPO/.venv_yucca/bin"
CFG="configs/$DATASET.yaml"
eval "$("$PY" -m src.common --config "$CFG" --shell)"

export YUCCA_SOURCE="$REPO/results/yucca/source"
export YUCCA_RAW_DATA="$REPO/results/yucca/raw_data"
export YUCCA_PREPROCESSED_DATA="$REPO/results/yucca/preprocessed_data"
export YUCCA_MODELS="$REPO/results/yucca/models"
export YUCCA_RESULTS="$REPO/results/yucca/results"
mkdir -p "$YUCCA_SOURCE" "$YUCCA_RAW_DATA" "$YUCCA_PREPROCESSED_DATA" "$YUCCA_MODELS" "$YUCCA_RESULTS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

# All datasets infer from the same synthetic model; train it once. FORCE_TRAIN=1 redoes it.
if [ -n "${FORCE_TRAIN:-}" ] || [ ! -d "$YUCCA_MODELS/$SYNTH_TASK" ]; then
    "$PY" -m src.simulating_data_paper --config "$SYNTH_CFG" --section "$SYNTH_SEC"
    "$PY" -m src.taskconvert synthetic --src "$SYNTH_SRC" --task "$SYNTH_TASK"
    "$BIN/yucca_preprocess" -t "$SYNTH_TASK"
    "$BIN/yucca_train" -t "$SYNTH_TASK" -d 2D -m UNet \
        --epochs "$EPOCHS" --patch_size $PATCH --batch_size "$BATCH" \
        --num_workers "$WORKERS" --precision "$PRECISION" --disable_logging --new_version
fi

# Count, not existence: a half-generated or differently-windowed EIC folder must not win.
[ "$(ls "$EIC_DIR/images" 2>/dev/null | wc -l)" -eq "$EIC_FILES" ] || \
    "$PY" -m src.mzml_to_npy --config "$CFG" --section images
"$PY" -m src.taskconvert inference --src "$EIC_DIR/images" --task "$TASK" --prefix "$DATASET"
"$PY" -m src.yucca_inference_run -t "$TASK" -s "$SYNTH_TASK" -d 2D -m UNet \
    --version 0 --no_wandb --disable_tta --no_eval

"$PY" -m src.extracting_labels_erosion --config "$CFG" --section labels
"$PY" -m src.matching_MS1_MS2 --config "$CFG" --section matching
"$PY" -m src.validate_run --config "$CFG" --section validation
