#!/bin/bash
#   python3 -m venv .venv_yucca && source .venv_yucca/bin/activate
#   pip install yucca==2.3.7
set -euxo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

TASK="Task100_synthLocal"
INFER_TASK="Task200_synthInfer"
SRC="data/simulated_data_local"
GEN_CONFIG="data/configs/config_simulating_data_paper_local.yaml"
INFER_SRC="$SRC/resolved_low/images"
N_INFER=2

export YUCCA_SOURCE="$REPO/data/yucca_data/source"
export YUCCA_RAW_DATA="$REPO/data/yucca_data/raw_data"
export YUCCA_PREPROCESSED_DATA="$REPO/data/yucca_data/preprocessed_data"
export YUCCA_MODELS="$REPO/data/yucca_data/models"
export YUCCA_RESULTS="$REPO/data/yucca_data/results"
mkdir -p "$YUCCA_SOURCE" "$YUCCA_RAW_DATA" "$YUCCA_PREPROCESSED_DATA" "$YUCCA_MODELS" "$YUCCA_RESULTS"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m src.simulating_data_paper --config "$GEN_CONFIG"

python -m src.taskconvert_synthetic --src "$SRC" --task "$TASK"

yucca_preprocess -t "$TASK"
yucca_train -t "$TASK" -d 2D -m UNet \
    --epochs 2 --train_batches_per_step 5 --val_batches_per_step 2 \
    --patch_size 64 64 --batch_size 2 \
    --precision 32 --num_workers 0 --disable_logging

python -m src.taskconvert_infer --src "$INFER_SRC" --task "$INFER_TASK" --limit "$N_INFER"
#yucca_inference -t "$INFER_TASK" -s "$TASK" -d 2D -m UNet --no_wandb --disable_tta --no_eval

echo "  model:       $YUCCA_MODELS/$TASK"
echo "  predictions: $YUCCA_RESULTS/$INFER_TASK"
echo "  python -m src.extracting_labels_erosion --config data/configs/config_extracting_labels_erosion.yaml"
echo "  python -m src.matching_MS1_MS2          --config data/configs/config_matching_MS1_MS2.yaml"
