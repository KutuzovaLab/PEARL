if [ ! -d .venv_yucca ]; then
    BASE_PY="${PYTHON:-python3.11}"
    command -v "$BASE_PY" >/dev/null || {
        echo "need Python 3.11 to build .venv_yucca (torch==2.1.2 has no wheels above 3.11)."
        echo "install it, or point PYTHON at one:  PYTHON=/path/to/python3.11 $0 $*"
        exit 1
    }
    "$BASE_PY" -m venv .venv_yucca
    .venv_yucca/bin/python -m pip install --upgrade pip
    .venv_yucca/bin/python -m pip install -e .   # requirements.txt, plus the pearl command
    .venv_yucca/bin/python -m pearl.patch_yucca_npy
fi
PY=.venv_yucca/bin/python
BIN=.venv_yucca/bin

if [ -n "${CFG:-}" ]; then
    eval "$("$PY" -m pearl.common --config "$CFG" --shell)"
else
    eval "$("$PY" -m pearl.common --train --shell)"
fi

YUCCA_ROOT="${PEARL_OUTPUT:-$PWD/results}/yucca"
export YUCCA_SOURCE="$YUCCA_ROOT/source"
export YUCCA_RAW_DATA="$YUCCA_ROOT/raw_data"
export YUCCA_PREPROCESSED_DATA="$YUCCA_ROOT/preprocessed_data"
export YUCCA_MODELS="$YUCCA_ROOT/models"
export YUCCA_RESULTS="$YUCCA_ROOT/results"
mkdir -p "$YUCCA_ROOT"/{source,raw_data,preprocessed_data,models,results}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1