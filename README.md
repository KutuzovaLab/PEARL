# PEARL

PEARL (**P**eak **E**xtraction **A**nd **R**epresentation **L**earner) — manual-tuning free LC-HRMS peak detection and
MS1/MS2 association enabled by deep learning.

PEARL turns raw LC-HRMS acquisitions (mzML) into a table of extracted peaks, each MS1 peak
optionally linked to the MS2 spectra that best match it. Internally, each narrow m/z slice of
the run is rendered as an image (retention time x sample), a U-Net neural network (trained via
[Yucca](https://github.com/Sllambias/yucca)) find peak segments, which are then matched between MS1 and MS2
channels.

## Contents

- [Installation](#installation)
- [Running PEARL on your own data](#running-pearl-on-your-own-data)
  - [Input data](#input-data)
  - [Configuring a dataset](#configuring-a-dataset)
  - [Output](#output)
- [Minimal example](#minimal-example)
- [Reproducing the article](#reproducing-the-article)
- [Training a new model](#training-a-new-model)
- [Tests](#tests)


## Installation

Python 3.11 is required (`torch==2.1.2` has no wheels above it).

```bash
python3.11 -m venv .venv_yucca
source .venv_yucca/bin/activate
pip install --upgrade pip
pip install -e .
```

This puts a `pearl` command on your path. Re-run `pip install -e .` after
rebuilding the environment.

## Run on your own data

```bash
pearl --input /path/to/mzml --output /path/to/results
```

PEARL detects whether the runs are DIA or DDA, reads the retention-time range
from the files, downloads the published model, and writes `<output>/peaks.csv`:

```
MS1 m/z,MS1 intensity,mz min,mz max,RT,RT min,RT max,MS2 spectrum
269.0462,905188.0,269.0,269.1,1058.9,1053.4,1063.6,225.055;41.3 151.003;12.8
```

One row per detected peak. `MS1 m/z` is the precursor mass, `RT` its retention
time in seconds, and `MS2 spectrum` the associated fragments as
`mz;intensity` pairs separated by spaces (empty for DDA, which has no MS2
images to associate).

Everything else under `<output>` belongs to that run: `eic/` (the extracted-ion
images), `clustering/` (the detected peaks), `yucca/` (the model workspace) and
the `config.yaml` PEARL generated and actually used.

## Minimal example

Nothing to prepare — this downloads four public QC runs and analyses them:

```bash
pearl --demo
```

Four replicates from [MetaboLights MTBLS1129](https://www.ebi.ac.uk/metabolights/MTBLS1129)
(~330 MB, checksum-verified) are fetched to `data/mzml/test`, and the whole
pipeline finishes in a couple of minutes, writing `results/test/peaks.csv`.

## Reproducing the article

The four datasets are not public, so the `.mzML` cannot be shipped. With the
files in place, these are the exact commands:

```bash
pearl --input data/mzml/wheat_samples --output results/wheat --name wheat \
  --acquisition dia --mz-range 50 1000 --rt-range 100 1100 --rt-bin-size 0.73 \
  --files configs/wheat_files.txt --qc-sample 230523_Batch3_090.mzML

pearl --input data/mzml/wheat_standards --output results/wheat_standards \
  --name wheat_standards --acquisition dia --mz-range 50 1200 \
  --rt-range 100 1100 --rt-bin-size 0.73 --qc-sample 230503_Batch2_13.mzML

pearl --input data/mzml/wastewater --output results/wastewater \
  --name wastewater --acquisition dda --mz-range 75 751 --rt-range 0 1000 \
  --rt-bin-size 0.45 --qc-sample QC.mzML

pearl --input data/mzml/fluorine --output results/fluorine --name fluorine \
  --acquisition dda --mz-range 90 1300 --rt-range 0 1260 --rt-bin-size 0.68 \
  --qc-sample 009_PM1_50_ugL_pos_dda_5_inj.mzML 014_PM2_50_ugL_pos_dda_5_inj.mzML \
    021_PM3_50_ugL_pos_dda_5_inj.mzML 026_PM4_50_ugL_pos_dda_5_inj26.mzML \
    033_PM5_50_ugL_pos_dda_5_inj.mzML 038_FluoroMix_50_ugL_pos_dda_5_inj.mzML
```

## Training a new model

The published checkpoint is downloaded automatically and is what the paper
used, so this is only needed to reproduce or replace it.

```bash
./train_model.sh
KEEP_MODEL=1 pearl --input /path/to/mzml --output /path/to/results
```

`train_model.sh` generates the synthetic training set from
`configs/synthetic.yaml`, converts it to a yucca task, preprocesses and trains.
The paper's model is a 2D U-Net trained for 600 epochs on 96x528 patches with
batch size 44, DiceCE loss and SGD at lr 0.001, fold 0 of 5; the checkpoint
with the lowest validation loss (epoch 557) is the one used for inference.
Training the full synthetic set needs a GPU and substantial disk.

`KEEP_MODEL=1` stops the pipeline replacing your model with the published
checkpoint.

## Tests

```bash
.venv_yucca/bin/python -m pytest
```