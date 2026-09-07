if [ ! -d .venv_yucca ]; then
    # torch==2.1.2 has no wheels above 3.11, so pin the interpreter.
    BASE_PY="${PYTHON:-python3.11}"
    command -v "$BASE_PY" >/dev/null || {
        echo "need Python 3.11 to build .venv_yucca (torch==2.1.2 has no wheels above 3.11)."
        echo "install it, or point PYTHON at one:  PYTHON=/path/to/python3.11 $0 $*"
        exit 1
    }
    "$BASE_PY" -m venv .venv_yucca
    .venv_yucca/bin/python -m pip install --upgrade pip
    .venv_yucca/bin/python -m pip install -r requirements.txt
    .venv_yucca/bin/python -m src.patch_yucca_npy
fi
PY=.venv_yucca/bin/python
BIN=.venv_yucca/bin

eval "$("$PY" -m src.common --config "$CFG" --shell)"

export YUCCA_SOURCE="$PWD/results/yucca/source"
export YUCCA_RAW_DATA="$PWD/results/yucca/raw_data"
export YUCCA_PREPROCESSED_DATA="$PWD/results/yucca/preprocessed_data"
export YUCCA_MODELS="$PWD/results/yucca/models"
export YUCCA_RESULTS="$PWD/results/yucca/results"
mkdir -p results/yucca/{source,raw_data,preprocessed_data,models,results}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1