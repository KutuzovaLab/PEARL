"""
Usage:  python simulating_peaks_final.py --config configs/config_simulating_peaks_final.yaml
"""

import os
from src.config import parse_config_arg
from src.simulating_peaks_final import generate_sim_images

def main(cfg):
    n_samples = cfg["n_samples"]
    n_images = cfg["n_images"]
    n_blobs_max = cfg["n_blobs_max"]
    rt_range = cfg["rt_range"]
    rt_bin_size = cfg.get("rt_bin_size", 0.6)
    peak_length = cfg.get("peak_length", 30)
    output_folder = cfg["output_folder"]

    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    generate_sim_images(
        n_samples, n_images, n_blobs_max, images_folder, labels_folder,
        rt_range, rt_bin_size=rt_bin_size, peak_length=peak_length,
    )

if __name__ == "__main__":
    config = parse_config_arg(description="Generate synthetic peak images and labels.")
    main(config)
