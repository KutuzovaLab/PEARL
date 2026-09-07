#!/bin/bash
# Run the pipeline on a dataset:   ./run_pipeline.sh wheat
#
# mzML -> EIC images -> model predictions -> peak labels -> MS1/MS2 matching.
# Uses the published checkpoint every time; no training involved.
#
#   CKPT=...       use a different checkpoint (hf://owner/repo/file.ckpt or a local path)
#   KEEP_MODEL=1   don't install a checkpoint; use the model already in place
#                  (i.e. one you just built with ./train_model.sh)
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${1:-wheat}"
CFG="configs/$DATASET.yaml"
source ./_env.sh

CKPT="${CKPT:-hf://UffeDalgas/temporary_p/best.ckpt}"

if [ "$DATASET" = test ] && [ "$(ls data/mzml/test/*.mzML 2>/dev/null | wc -l)" -eq 0 ]; then
    "$PY" -m src.fetch_test_data
fi

if [ -z "${KEEP_MODEL:-}" ]; then
    case "$CKPT" in hf://*) CKPT="$("$PY" -m src.fetch_checkpoint "$CKPT")" ;; esac
    "$PY" -m src.install_checkpoint --ckpt "$CKPT" --task "$SYNTH_TASK"
elif [ ! -d "$YUCCA_MODELS/$SYNTH_TASK" ]; then
    echo "KEEP_MODEL is set but there is no model at $YUCCA_MODELS/$SYNTH_TASK"
    echo "run ./train_model.sh $DATASET first, or unset KEEP_MODEL to use the published checkpoint"
    exit 1
fi

# images -> model predictions -> peak labels -> MS1-MS2 matching.
[ "$(ls "$EIC_DIR/images" 2>/dev/null | wc -l)" -eq "$EIC_FILES" ] || \
    "$PY" -m src.mzml_to_npy --config "$CFG" --section images
"$PY" -m src.taskconvert inference --src "$EIC_DIR/images" --task "$TASK" --prefix "$DATASET"
"$BIN/yucca_inference" -t "$TASK" -s "$SYNTH_TASK" -d 2D -m UNet \
    --version 0 --no_wandb --disable_tta --no_eval
"$PY" -m src.extracting_labels_erosion --config "$CFG" --section labels
[ "$ACQUISITION" = dda ] || "$PY" -m src.matching_MS1_MS2 --config "$CFG" --section matching
