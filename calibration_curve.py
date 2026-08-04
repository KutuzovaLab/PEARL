"""
Usage:  python calibration_curve.py --config configs/config_calibration_curve.yaml
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import parse_config_arg
from src.calibration_curve import (
    update_file_list, load_suspects, load_sample_info,
    load_sample_concentrations, measure_suspect,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main(cfg):
    image_path = cfg["image_path"]
    clipped_path = cfg["clipped_path"]
    task_name = cfg["task_name"]
    pred_path = cfg["pred_path"]
    label_path = cfg["label_path"]
    suspect_path = cfg["suspect_path"]
    mzml_path = cfg["mzml_path"]
    selected_samples_path = cfg["selected_samples_path"]
    results_folder = cfg["results_folder"]
    rt_range = cfg["rt_range"]
    rt_bin_size = cfg["rt_bin_size"]
    rt_sd = cfg.get("rt_sd", 20)

    os.makedirs(clipped_path, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    image_names = {re.sub(r"\.png$", "", img) for img in os.listdir(image_path)}
    suspects = load_suspects(suspect_path)
    files = [f for f in os.listdir(mzml_path) if f.endswith(".mzML")]
    updated_file_list = update_file_list(files)

    info_samples = load_sample_info(suspect_path, updated_file_list)
    info_samples.to_excel(selected_samples_path, index=False)

    samples = load_sample_concentrations(suspect_path, updated_file_list)

    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)

    suspects = suspects[(suspects["RT"] >= rt_range[0]) & (suspects["RT"] <= rt_range[1])]
    print(f"Selecting {len(suspects)} compounds identified by MsDial")

    peak_areas = pd.DataFrame(
        columns=['name', 'rt', 'mz', "msdial_area", "pred_area", "file"], dtype='object'
    )

    for i in range(len(suspects)):
        suspect_row = suspects.iloc[i]
        name = suspect_row["name"]

        measured = measure_suspect(
            suspect_row, image_names, task_name, image_path, pred_path,
            label_path, clipped_path, rt_bins, rt_sd=rt_sd,
        )
        if measured is None:
            continue

        sum_int, susp_mz, susp_rt = measured

        msdial_area = samples[
            (samples["Metabolite name"] == name)
            & (samples["Average Rt(min)"] == susp_rt / 60)
        ]
        msdial_area = msdial_area.iloc[:, 2:].sort_index(axis=1).values.flatten()
        sample_names = sorted(samples.columns[2:].values)

        peak_areas = pd.concat([peak_areas, pd.DataFrame({
            'name': name, 'rt': susp_rt, 'mz': susp_mz,
            'msdial_area': msdial_area / max(msdial_area),
            'pred_area': sum_int / max(sum_int),
            'file': sample_names,
        })])

        peak_areas.plot(x='msdial_area', y='pred_area', kind='scatter', marker='o', color='b')
        plt.xlabel("MsDial peak area")
        plt.ylabel("U-net predicted peak area")
        plt.title("Calibration curve")
        plt.grid(True)
        plt.savefig(os.path.join(results_folder, f'{name}_curve.png'))
        plt.close()

    peak_areas.to_excel(os.path.join(results_folder, 'calibration_curve.xlsx'), index=False)


if __name__ == "__main__":
    config = parse_config_arg(description="Build a MsDial vs U-Net calibration curve.")
    main(config)
