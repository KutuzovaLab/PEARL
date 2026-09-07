# PEARL

Peak Extraction And Representation Learner: LC-HRMS peak detection and MS1-MS2 association
by U-Net segmentation

## Layout

```
data/
  mzml/
    test/         4 mzML   388 MB   public
    wheat/       82 mzML    41 GB   63 of 82 used
    standards/    8 mzML   2.1 GB
    wastewater/  54 mzML   4.3 GB
    fluorine/    22 mzML   2.0 GB
configs/        one YAML per dataset
```

## Install

```bash
python3.11 -m venv .venv_yucca
.venv_yucca/bin/python -m pip install --upgrade pip
.venv_yucca/bin/python -m pip install -r requirements.txt
.venv_yucca/bin/python -m src.patch_yucca_npy

python3.11 -m venv .venv_dreams
.venv_dreams/bin/python -m pip install git+https://github.com/pluskal-lab/DreaMS.git openpyxl
```

## Run

```bash
./run_pipeline.sh test           # public data, downloaded on first run
./run_pipeline.sh wheat          # or wastewater | fluorine
```

`run_pipeline.sh` is inference only: mzML -> images -> model predictions -> peak
labels -> MS1/MS2 matching. It downloads the published checkpoint every time, so no
training is involved and no GPU is needed to train.

Training is a separate script, needed only to reproduce or replace that checkpoint:

```bash
./train_model.sh wheat                  # synthetic data -> train
KEEP_MODEL=1 ./run_pipeline.sh wheat    # inference
```

Individual stages. These need the `YUCCA_*` environment variables the scripts export --
see `_env.sh`.

`train_model.sh` does:

```bash
python -m src.simulating_data_paper     --config configs/synthetic.yaml --section base
python -m src.taskconvert synthetic     --src results/synthetic --task Task100_synth
yucca_preprocess -t Task100_synth
yucca_train      -t Task100_synth -d 2D -m UNet --epochs 600 --patch_size 96 528 --batch_size 44
```

`run_pipeline.sh` does:

```bash
python -m src.fetch_checkpoint           hf://UffeDalgas/temporary_p/best.ckpt
python -m src.install_checkpoint         --ckpt checkpoints/best.ckpt --task Task100_synth
python -m src.mzml_to_npy                --config configs/wheat.yaml --section images
python -m src.taskconvert inference      --src results/eic/wheat/images --task Task201_wheat --prefix wheat
yucca_inference -t Task201_wheat -s Task100_synth -d 2D -m UNet --version 0 --no_wandb --disable_tta --no_eval
python -m src.extracting_labels_erosion  --config configs/wheat.yaml --section labels
python -m src.matching_MS1_MS2           --config configs/wheat.yaml --section matching
```

`yucca_preprocess`, `yucca_train` and `yucca_inference` are console scripts from the
`yucca` package (`.venv_yucca/bin/`), not modules in `src/`.