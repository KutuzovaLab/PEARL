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

Requires **Python 3.11**. A GPU is not required.

```bash
python3.11 -m venv .venv_yucca
.venv_yucca/bin/python -m pip install --upgrade pip
.venv_yucca/bin/python -m pip install -r requirements.txt
.venv_yucca/bin/python -m src.patch_yucca_npy
```

## Running PEARL on your own data

### Input data

PEARL reads centroided **mzML** files. Put your files under `data/mzml/<dataset>/`, e.g.:

```
data/mzml/mydataset/sample_001.mzML
data/mzml/mydataset/sample_002.mzML
...
```

Two acquisition modes are supported, set per dataset via `images.acquisition` in the config
(see below):

- **`dia`** -- data-independent acquisition with interleaved low-energy (MS1), high-energy
  (MS2), and reference/lock-mass functions (as in Waters MSe-style acquisitions). All three
  functions are read from each file; PEARL builds an MS1 and an MS2 image per m/z slice and
  runs MS1/MS2 matching.
- **`dda`** -- data-dependent acquisition. Only MS1 survey scans are used; no MS2 image is
  built and the MS1/MS2 matching step is skipped automatically.

### Configuring a dataset

Each dataset has one YAML config in `configs/`, with three sections consumed by the three
inference stages (`configs/wheat.yaml` is a working example):

| Section | Key | Meaning |
|---|---|---|
| `images` | `mzml_path` | folder of input `.mzML` files |
| | `output_folder` | where the EIC `.npy` images are written |
| | `acquisition` | `dia` or `dda` |
| | `rt_range` | `[start, end]` retention time window to keep, in seconds |
| | `rt_bin_size` | retention-time bin width, in seconds (image height) |
| | `mz_range` | `[low, high]` m/z window to cover, in Da |
| | `mass_tol` | m/z bin width, in Da -- one image is built per bin |
| | `length_spline` | (DIA only) points either side used to spline-interpolate the reference function onto the MS1/MS2 grid |
| | `files` | optional: restrict/order which files (by basename, no extension) are read; default is all `.mzML` in `mzml_path`, alphabetical |
| `labels` | `predictions_dir` | where `yucca_inference` writes its `.nii.gz` segmentations (see the printed command in [Output](#output)) |
| | `images_raw` | the raw `.npy` images `yucca_inference` ran on (`$YUCCA_RAW_DATA/<task>/imagesTs`) |
| | `output_dir` | where extracted peak labels are written |
| | `erosion_width` | RT-axis erosion (in bins) applied before splitting touching peaks apart |
| | `sample_thres`, `label_width` | optional signal-across-samples filter; unset disables it |
| `matching` | `input_dir` | peak-label folder to match (`<labels.output_dir>/single_labels_overlap`) |
| | `k` | number of nearest MS1 neighbours kept per MS2 peak |
| | `binary` | if true, binarize label images before embedding |

To add a new dataset, copy an existing config (`fluorine.yaml` for DDA, `wheat.yaml` or
`wastewater.yaml` for DIA), point `mzml_path` at your files, and set `rt_range`/`mz_range` to
cover your acquisition. Choosing `mz_range`/`mass_tol` to give roughly the same number of
retention-time bins the model was trained with (`rt_bins` in `configs/synthetic.yaml`) gives
the most reliable segmentation.

Then run:

```bash
./run_pipeline.sh mydataset
```

`CKPT=hf://owner/repo/file.ckpt ./run_pipeline.sh mydataset` uses a different checkpoint (a
local path also works). `KEEP_MODEL=1 ./run_pipeline.sh mydataset` skips fetching a checkpoint
and uses whatever model is already installed (i.e. one you trained yourself, see
[Training a new model](#training-a-new-model)).

### Output

```
results/eic/<dataset>/images/
    MS1_<mzlo>_<mzhi>.npy, MS2_<mzlo>_<mzhi>.npy      one image per m/z bin, shape (rt_bins, n_files)

results/yucca/results/<task>/.../best/
    <dataset>_MS1_<mzlo>_<mzhi>.nii.gz                per-image segmentation mask
    <dataset>_MS2_<mzlo>_<mzhi>.nii.gz                (0 = no signal, 1 = peak, 2 = overlapping peaks)

results/clustering/<dataset>/L3_erosion/
    yucca_labels/*.png                                 mask, for a quick visual sanity check
    single_labels/*.png                                connected peak regions after erosion, one colour per peak
    single_labels_overlap/
        <dataset>_MS1_<mzlo>_<mzhi>_<NNN>.npy           one extracted peak: raw intensities (rt_bins x n_files),
        <dataset>_MS2_<mzlo>_<mzhi>_<NNN>.npy           zeroed outside the peak region; NNN indexes peaks within that m/z bin

results/clustering/<dataset>/clusters_k30/
    final_matches.csv                                  MAIN OUTPUT: for every MS2 peak label, the k nearest
                                                         MS1 peak labels (by shape) and their distance
    ms1_vectors.npz, ms1_names.npz, ms2_vectors.npz, ms2_names.npz, ms1_kdtree.pkl
                                                         the embeddings/index behind final_matches.csv
```

`final_matches.csv` has one row per (MS2 peak, candidate MS1 peak) pair:

| column | meaning |
|---|---|
| `ms2_name` | MS2 peak-label file, encodes m/z bin and retention window |
| `ms1_name` | candidate MS1 peak-label file matched to it |
| `distance` | shape-embedding distance (smaller = better match); `k` rows per `ms2_name`, sorted ascending |

For a DDA dataset there is no MS2 image and no `final_matches.csv`; `single_labels_overlap`
holds the extracted MS1 peaks only.


## Minimal example

For reviewers who want to run the full pipeline end-to-end on public data, with no data
preparation and no GPU required:

```bash
./run_pipeline.sh test
```

On first run this downloads:

- 4 public QC mzML files (~330 MB, DIA) from the
  [MetaboLights MTBLS1129](https://www.ebi.ac.uk/metabolights/MTBLS1129) study, into
  `data/mzml/test/`
- the published model checkpoint from Hugging Face

then runs the full pipeline (image conversion, segmentation, peak extraction, MS1/MS2
matching) restricted to a narrow m/z window (200-225 Da) so it finishes in a few minutes on a
laptop CPU. Results land in `results/eic/test/`, `results/yucca/results/Task900_test/...`, and
`results/clustering/test/L3_erosion/` -- see [Output](#output) for what each file is.

To fetch more of the public QC files, or point somewhere else:

```bash
.venv_yucca/bin/python -m src.fetch_test_data --n 8 --out data/mzml/test
```

## Reproducing the article

The wheat, wastewater, fluorine and standards datasets benchmarked in the article are not yet
publicly available. They are available upon request -- contact Jan H. Christensen
by email (**TODO: contact email**).

<details>
  <summary>Training a new model</summary>
## Training a new model

`run_pipeline.sh` always runs inference with a published checkpoint -- no training or GPU is
needed to reproduce the paper's predictions on new data. Training is only needed to reproduce
or replace that checkpoint, from purely synthetic, simulated LC-HRMS images (no real data
required):

```bash
./train_model.sh wheat                  # simulate training data, train a model
KEEP_MODEL=1 ./run_pipeline.sh wheat    # run inference with it, instead of the published checkpoint
```

The synthetic generator and training schedule are controlled by `configs/synthetic.yaml`
(section `base`: 600 epochs, ~100k images) or overridden per dataset config (e.g. `epochs`,
`patch`, `batch`, `synthetic_config`/`synthetic_section`). `configs/test.yaml` defines a tiny
2-epoch, 8-sample variant (`synthetic_section: synthetic`) to sanity-check the training
pipeline end-to-end in a couple of minutes before committing to a full run:

```bash
./train_model.sh test
```

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

`yucca_preprocess`, `yucca_train` and `yucca_inference` are console scripts installed by the
`yucca` package (`.venv_yucca/bin/`), not modules in `src/`. Both scripts need the `YUCCA_*`
environment variables they export via `_env.sh` (preprocessed data, model, and results
locations); run the commands above through `train_model.sh`/`run_pipeline.sh`, or `source
_env.sh` first if running a stage individually.

## Tests

```bash
.venv_yucca/bin/python -m pytest
```

<details>