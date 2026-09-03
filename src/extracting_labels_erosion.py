"""
    python -m src.extracting_labels_erosion --config configs/config_extracting_labels_erosion.yaml
"""

import os
import re

import numpy as np
from PIL import Image
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy import ndimage

from src.common import parse_config_arg


def label_to_rgb_pil(lbl):
    rgb = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
    rgb[lbl == 1] = [255, 255, 255]
    rgb[lbl == 2] = [255,   0,   0]
    return Image.fromarray(rgb)

def load_raw_image(images_raw, base):
    npy = os.path.join(images_raw, base + "_000.npy")
    if os.path.isfile(npy):
        return np.load(npy)
    niigz = os.path.join(images_raw, base + "_000.nii.gz")
    if os.path.isfile(niigz):
        return np.asanyarray(nib.load(niigz).dataobj, dtype=np.float32)
    raise FileNotFoundError(f"raw image for '{base}' not found as .npy or .nii.gz in {images_raw}")

def main(cfg):
    predictions_dir = cfg["predictions_dir"]
    images_raw = cfg["images_raw"]
    output_dir = cfg["output_dir"]
    task_name = cfg.get("task_name", "")
    erosion_width = cfg.get("erosion_width", 3)

    if not os.path.isdir(predictions_dir):
        raise SystemExit(f"predictions_dir does not exist: {predictions_dir}")

    labels = sorted(f for f in os.listdir(predictions_dir) if f.endswith(".nii.gz"))
    if not labels:
        raise SystemExit(
            f"no .nii.gz predictions in {predictions_dir} - check that yucca_inference ran "
            f"and that predictions_dir points at the version_N/<checkpoint> folder"
        )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Labels saved in: {output_dir}")

    label_folder = os.path.join(output_dir, "single_labels")
    overlap_folder = os.path.join(output_dir, "single_labels_overlap")
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(overlap_folder, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "yucca_labels"), exist_ok=True)

    print(f"Task: {task_name} - erosion size in pixels: ({erosion_width}, 1), number of labels: {len(labels)}")

    for k in range(len(labels)):

        if len(labels) >= 10 and k % max(1, len(labels) // 10) == 0:
            print(f"Percentage of images done: {round(k / len(labels) * 100)}", flush=True)

        lbl_name = labels[k]
        base_name = re.sub(".nii.gz", "", lbl_name)

        img = nib.load(os.path.join(predictions_dir, lbl_name)).get_fdata()

        img_raw = load_raw_image(images_raw, base_name)

        lbl3 = label_to_rgb_pil(img)
        lbl3.save(os.path.join(output_dir, "yucca_labels", re.sub("nii.gz", "png", lbl_name)))

        orig_cc = label(img, connectivity=1)
        blob_mask = orig_cc > 0

        extended_image = np.zeros_like(blob_mask, dtype=np.uint16)
        blob_i = 1

        for orig_region in regionprops(orig_cc):

            orig_component = orig_cc == orig_region.label
            eroded_component = ndimage.binary_erosion(
                orig_component, structure=np.ones((2 * erosion_width + 1, 1), dtype=bool)
            )
            eroded_cc = label(eroded_component, connectivity=1)
            n_eroded = eroded_cc.max()

            if n_eroded <= 1:
                min_y, min_rt, max_y, max_rt = orig_region.bbox
                rt_ranges = [(min_rt, max_rt)]
            else:
                rt_ranges = []
                for eroded_region in regionprops(eroded_cc):
                    min_y, min_rt, max_y, max_rt = eroded_region.bbox
                    rt_ranges.append((min_rt, max_rt))

            for min_rt, max_rt in rt_ranges:
                extended_image[:, min_rt:max_rt] = 1

        final_cc = label(extended_image, connectivity=1)

        for final_region in regionprops(final_cc):
            final_mask = final_cc == final_region.label
            empty_img = np.zeros_like(img_raw)
            empty_img[final_mask] = img_raw[final_mask]
            if np.any(empty_img != 0):
                np.save(os.path.join(overlap_folder, f"{base_name}_{str(blob_i).zfill(3)}.npy"), empty_img)
                blob_i += 1

        plt.imsave(os.path.join(label_folder, f"{base_name}.png"), final_cc, cmap="tab20")

    print(f"No. of labels created: {len(os.listdir(overlap_folder))}")
    ms1_files = [f for f in os.listdir(overlap_folder) if "MS1_" in f]
    ms2_files = [f for f in os.listdir(overlap_folder) if "MS2_" in f]
    print(f"Found {len(ms1_files)} MS1 files")
    print(f"Found {len(ms2_files)} MS2 files")

if __name__ == "__main__":
    main(parse_config_arg(description="Extract single peak labels from Yucca predictions."))
