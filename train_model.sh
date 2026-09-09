#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

source ./_env.sh

echo "=== synthetic data + training: $SYNTH_TASK (from $SYNTH_CFG [$SYNTH_SEC])"
"$PY" -m pearl.simulating_data_paper --config "$SYNTH_CFG" --section "$SYNTH_SEC"
"$PY" -m pearl.taskconvert synthetic --src "$SYNTH_SRC" --task "$SYNTH_TASK"
"$BIN/yucca_preprocess" -t "$SYNTH_TASK"
"$BIN/yucca_train" -t "$SYNTH_TASK" -d 2D -m UNet \
    --epochs "$EPOCHS" --patch_size $PATCH --batch_size "$BATCH" \
    --num_workers "${WORKERS:-0}" --precision "${PRECISION:-32}" --disable_logging --new_version

echo
echo "model written to $YUCCA_MODELS/$SYNTH_TASK"
echo "next:  KEEP_MODEL=1 pearl --input /path/to/mzml --output /path/to/results"
