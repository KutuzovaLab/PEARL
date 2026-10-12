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
- [Run on your own data](#run-on-your-own-data)
  - [Parameters](#parameters)
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
`mz;intensity` pairs separated by spaces.

### Parameters

| option | default | what it does |
|---|---|---|
| `--mass-tol` | `0.1` | width in Da of the m/z slice each image covers. The published model was trained on 0.1 Da slices. |
| `--ms2-ppm` | `20` | m/z tolerance when a fragment spectrum is read from the raw files. |
| `--ms2-rt-tol` | `5` | DDA only: how far outside a peak's own RT window its fragment spectrum may lie. |
| `--dia-channels` | detected | `function` (Waters, from the `function=N` native IDs) or `mslevel`. |
| `--dia-functions` | `1 2 3` | DIA only: which functions hold MS1, MS2 and the lock mass. `0` for the lock mass if the run has none. |

Inference runs on the CPU in float32 by default, which is the slowest but most
portable setting. Four environment variables change that:

| variable | default | what it does |
|---|---|---|
| `PEARL_ACCELERATOR` | `cpu` | passed to Lightning: `mps` on Apple silicon, `gpu` on CUDA. Several times faster. |
| `PEARL_PRECISION` | `32-true` | `16-mixed` is what the model's CUDA runs used. It is faster and uses less memory, but float16 overflows on very intense peaks and loses them, so the two settings do not give the same peak list. |
| `PEARL_WINDOW_BATCH` | `8` | sliding-window patches per forward pass. Lower it if inference runs out of memory. |
| `PEARL_INFER_THREADS` | all cores | threads for the CPU maths libraries during inference. |

```bash
PEARL_ACCELERATOR=mps pearl --input /path/to/mzml --output /path/to/results
```

## Minimal example

```bash
pearl --demo
```

Four replicates from [MetaboLights MTBLS1129](https://www.ebi.ac.uk/metabolights/MTBLS1129)
are fetched to `data/mzml/test`, and the whole pipeline is executed, writing `results/test/peaks.csv`.

## Reproducing the article

The four datasets are not public, so the `.mzML` cannot be shipped. With the
files in place, these are the exact commands:

```bash
pearl --input data/mzml/wheat_samples --output results/wheat --name wheat \
  --acquisition dia --mz-range 50 1000 --mass-tol 0.1 \
  --rt-range 100 1100 --rt-bin-size 0.73 \
  --files configs/wheat_files.txt --qc-sample 230523_Batch3_090.mzML

pearl --input data/mzml/wheat_standards --output results/wheat_standards \
  --name wheat_standards --acquisition dia --mz-range 50 1200 --mass-tol 0.1 \
  --rt-range 100 1100 --rt-bin-size 0.73 --qc-sample 230503_Batch2_13.mzML

pearl --input data/mzml/wastewater --output results/wastewater \
  --name wastewater --acquisition dda --mz-range 75 751 --mass-tol 0.1 \
  --rt-range 0 1000 --rt-bin-size 0.45 --qc-sample QC.mzML

pearl --input data/mzml/fluorine --output results/fluorine --name fluorine \
  --acquisition dda --mz-range 90 1300 --mass-tol 0.1 \
  --rt-range 0 1260 --rt-bin-size 0.68 \
  --qc-sample 009_PM1_50_ugL_pos_dda_5_inj.mzML 014_PM2_50_ugL_pos_dda_5_inj.mzML \
    021_PM3_50_ugL_pos_dda_5_inj.mzML 026_PM4_50_ugL_pos_dda_5_inj26.mzML \
    033_PM5_50_ugL_pos_dda_5_inj.mzML 038_FluoroMix_50_ugL_pos_dda_5_inj.mzML
```

## Training a new model

The published checkpoint is downloaded automatically, so this is only needed to reproduce it.

```bash
./train_model.sh
KEEP_MODEL=1 pearl --input /path/to/mzml --output /path/to/results
```

`train_model.sh` generates the synthetic training set from
`configs/synthetic.yaml`, converts it to a yucca task (holding out 20% of the
images), preprocesses and trains.
The paper's model is a 2D U-Net trained for 600 epochs on 96x528 patches with
batch size 44, DiceCE loss and SGD at lr 0.001, fold 0 of 5; the checkpoint
with the lowest validation loss (epoch 556) is the one used for inference.
Training the full synthetic set needs a GPU and substantial disk space.

`KEEP_MODEL=1` stops the pipeline replacing your model with the published
checkpoint.

## Tests

```bash
.venv_yucca/bin/python -m pytest
```