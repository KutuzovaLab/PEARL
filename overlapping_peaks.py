"""
Usage:  python overlapping_peaks.py --config configs/config_overlapping_peaks.yaml
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

from src.config import parse_config_arg
from src.overlapping_peaks import (
    make_overlapping_peak, find_boundary, make_image, create_mask,
)

def main(cfg):
    rt_range = cfg["rt_range"]
    peak_length = cfg["peak_length"]
    rt_bin_size = cfg["rt_bin_size"]
    n_samples = cfg["n_samples"]
    n_images = cfg["n_images"]
    rt_overlap = np.arange(cfg["rt_overlap_min"], cfg["rt_overlap_max"])

    image_path = cfg["image_path"]
    label_path = cfg["label_path"]
    check_path = cfg["check_path"]
    for p in (image_path, label_path, check_path):
        os.makedirs(p, exist_ok=True)

    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)

    for rto in rt_overlap:

        print(f"Level of overlap: {rto/peak_length}")

        for i in range(n_images):

            all_peaks = pd.DataFrame(columns=['rt_bins', 'int', 'file'], dtype='object')

            limit = [0] * n_samples

            prob_1 = random.choice([0.2, 0.6, 0.9])
            random_samples = np.random.choice([0, 1], size=n_samples, p=[1 - prob_1, prob_1])

            rt_shift = [0, 1, 2]
            length = np.repeat(peak_length, n_samples)
            shift = np.cumsum(np.random.choice(rt_shift, size=n_samples))

            rt_start = np.random.randint(len(rt_bins) - 200)

            print(f"Choice of samples: {random_samples}, rt start: {rt_start}")

            for j in range(n_samples):

                if random_samples[j] == 1:

                    rts = rt_start + shift[j]

                    df = make_overlapping_peak(rt_bin_size, int(rts), int(length[j]), rto, rt_bins)

                    limit[j] = float(find_boundary(df))

                    df["rt_bins"] = pd.cut(df["x"], bins=rt_bins, right=True)
                    new_peak = df.groupby('rt_bins')['y'].sum().reset_index()
                    new_peak['file'] = j
                    new_peak.columns = ["rt_bins", "int", "file"]
                    new_peak = new_peak[new_peak["int"] > 0]

                    all_peaks = pd.concat([all_peaks, new_peak])

            bin_labels = all_peaks["rt_bins"].cat.categories  # getting bin labels out
            zero_index = pd.cut(limit, bins=bin_labels).codes

            img = make_image(all_peaks, n_samples)
            img.save(os.path.join(image_path, f"op_image_{rto}_{i+1:02}.png"))

            label = create_mask(img, zero_index)
            label.save(os.path.join(label_path, f"op_image_{rto}_{i+1:02}.png"))

            check = Image.fromarray(np.where(np.array(label) == 1, 255, 0).astype(np.uint8))
            check.save(os.path.join(check_path, f"op_image_{rto}_{i+1:02}.png"))

            print(f"Done creating op_image_{rto}_{i+1:02}.png")


if __name__ == "__main__":
    config = parse_config_arg(description="Generate overlapping-peak images and labels.")
    main(config)
