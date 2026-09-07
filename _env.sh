if [ ! -d .venv_yucca ]; then
    command -v uv >/dev/null || { echo "please install uv first (https://docs.astral.sh/uv/)"; exit 1; }
    uv venv --python 3.11 .venv_yucca
    VIRTUAL_ENV=.venv_yucca uv pip install -r requirements.txt
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