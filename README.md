# PEARL

Peak Extraction And Representation Learner: LC-HRMS peak detection and MS1-MS2 association
by U-Net segmentation of stacked extracted-ion-chromatogram images, trained on
synthetic data.

## Layout

```
data/           
  mzml/         wheat (82 present, 63 used) · wastewater (54) · fluorine (22)
  reference/    MS-DIAL / MZmine / OpenMS result tables, standards list, injection order
  libraries/    MoNA export, MassSpecGym DreaMS embeddings, SIRIUS, parsed MoNA library
results/        
configs/        one YAML per dataset, plus the synthetic generator and the test
src/            all functionality, run as `python -m src.<module>`
validation_workflow.ipynb   the original analysis
```

## Install

```bash
uv venv --python 3.11 .venv_yucca
VIRTUAL_ENV=.venv_yucca uv pip install -r requirements.txt

uv venv --python 3.11 .venv_dreams
VIRTUAL_ENV=.venv_dreams uv pip install git+https://github.com/pluskal-lab/DreaMS.git openpyxl
```

## Run

```bash
./run_pipeline.sh test           # test run (data/mzml/standards` and `data/reference/Standards_NEG.xlsx)
./run_pipeline.sh wheat          # or standards | wastewater | fluorine
```

Each dataset uses the same model trained on syntehtic data (`FORCE_TRAIN=1` retrains it).

Individual stages:

```bash
python -m src.simulating_data_paper     --config configs/synthetic.yaml --section base
python -m src.taskconvert synthetic     --src results/synthetic --task Task100_synth
python -m src.mzml_to_npy               --config configs/wheat.yaml --section images
python -m src.taskconvert inference     --src results/eic/wheat/images --task Task201_wheat --prefix wheat
python -m src.yucca_inference_run       -t Task201_wheat -s Task100_synth -d 2D -m UNet --version 0
python -m src.extracting_labels_erosion --config configs/wheat.yaml --section labels
python -m src.matching_MS1_MS2          --config configs/wheat.yaml --section matching
python -m src.validate_run              --config configs/wheat.yaml --section validation
```

The MoNA library is built once, filtered to the compounds being validated:

```bash
python -m src.mona --export data/libraries/MoNA-export-LC-MS-MS_Positive_Mode.zip \
    --out data/libraries/mona_library.pkl --inchikeys-from results/wheat/compound_info.pkl
```

## Datasets

| Config | Acquisition | Reference | Compounds | Samples |
|---|---|---|---|---|
| `wheat` | DIA | MS-DIAL | 350 | 63 |
| `wastewater` | DDA | MZmine | 70 | 54 |
| `fluorine` | DDA  | OpenMS | 567 (of 811) | 22 |