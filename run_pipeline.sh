#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PEARL_CONFIG:-}" ]; then
    echo "run_pipeline.sh is the internal stage runner; use the pearl command instead:"
    echo "    pearl --input /path/to/mzml --output /path/to/results"
    exit 1
fi
CFG="$PEARL_CONFIG"
DATASET="$(basename "$(dirname "$CFG")")"
source ./_env.sh

CKPT="${CKPT:-hf://UffeDalgas/PEARL/best.ckpt}"

if [ -z "${KEEP_MODEL:-}" ]; then
    case "$CKPT" in hf://*) CKPT="$("$PY" -m pearl.fetch_checkpoint "$CKPT")" ;; esac
    "$PY" -m pearl.install_checkpoint --ckpt "$CKPT" --task "$SYNTH_TASK"
elif [ ! -d "$YUCCA_MODELS/$SYNTH_TASK" ]; then
    echo "KEEP_MODEL is set but there is no model at $YUCCA_MODELS/$SYNTH_TASK"
    echo "run ./train_model.sh first, or unset KEEP_MODEL to use the published checkpoint"
    exit 1
fi

"$PY" -m pearl.mzml_to_npy --config "$CFG" --section images
"$PY" -m pearl.taskconvert inference --src "$EIC_DIR/images" --task "$TASK" --prefix "$DATASET"
INFER_THREADS="${PEARL_INFER_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 1)}"
OMP_NUM_THREADS="$INFER_THREADS" OPENBLAS_NUM_THREADS="$INFER_THREADS" \
MKL_NUM_THREADS="$INFER_THREADS" \
"$BIN/yucca_inference" -t "$TASK" -s "$SYNTH_TASK" -d 2D -m UNet \
    --version 0 --no_wandb --disable_tta --no_eval
"$PY" -m pearl.extracting_labels_erosion --config "$CFG" --section labels
[ "$ACQUISITION" = dda ] || "$PY" -m pearl.matching_MS1_MS2 --config "$CFG" --section matching
"$PY" -m pearl.peak_table --config "$CFG"
