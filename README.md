# PEARL

Description

## Layout

```
PEARL/
├── <script>.py            # runnable entry points (main block, --config argument)
├── src/                   # reusable functions imported by the entry points
│   └── config.py          # YAML config loader
├── configs/               # one config_<script>.yaml per entry point
├── requirements.txt
└── matching_mslevel.ipynb # exploratory notebook (unchanged)
```

## Installation

```
pip install -r requirements.txt
```

## Usage
```
python <script>.py --config configs/config_<script>.yaml
```

Examples:

```
python mzml_to_npy.py --config configs/config_mzml_to_npy.yaml
python simulating_peaks_final.py --config configs/config_simulating_peaks_final.yaml
python preparing_for_clustering.py --config configs/config_preparing_for_clustering.yaml
```

## Workflow

Synthetic data generation:
- `simulating_peaks_final.py`, `overlapping_peaks.py`

Main pipeline:
- `mzml_to_npy.py` → (Yucca inference) → `preparing_for_clustering.py` →
  `matching_MS1_MS2.py` → `calibration_curve.py`

Clustering / assessment:
- `kmeans_targets.py`, `mona.py`, `library.py`, `sirius.py`

`identification_unet.py` ?
`matching_mslevel.ipynb.py` hasn't been changed.

@TODO GitHub Actions
@TODO Configure paths in .yaml files to not be absolute
@TODO Actually run the code fully
