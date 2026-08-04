
import os
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def update_file_list(files):
    updated_file_list = sorted([
        re.sub(r'_(8[0-9]|9[0-9])\.mzML$', r'_0\1.mzML', filename)
        if re.search(r'_(8[0-9]|9[0-9])\.mzML$', filename) else filename
        for filename in files
    ])
    return [re.sub(".mzML", "", item) for item in updated_file_list]


def load_suspects(suspect_path):
    all_samples = pd.read_csv(suspect_path, skiprows=4, header = 0)

    # keeping only those suspects that have %fill greater than 0.9 and they have names assigned
    suspects = all_samples.iloc[:,[1,2,3,6]]  #selecting columns with relevant info
    suspects.columns = ["RT", "mz", "name", "fill"]
    suspects.loc[:, "RT"]=suspects["RT"]*60  #rt is in min

    suspects = suspects[suspects["name"]!="Unknown"]  #choosing only compounds that were identified with names (also avoiding adducts)
    suspects = suspects[suspects["fill"]>=0.9]  # %fill is a parameter from MsDial that tells how well the identification went
    suspects = suspects[~suspects["name"].str.contains("w/o MS2", na=False)]  # using only those with assigned MS2

    return suspects


def load_sample_info(suspect_path, updated_file_list):
    info_samples = pd.read_csv(suspect_path, nrows=5, header=None)
    info_samples = info_samples.iloc[:, 31:]
    info_samples = info_samples.T
    info_samples.columns = info_samples.iloc[0]
    info_samples = info_samples[1:].reset_index(drop=True)
    info_samples = info_samples.rename(columns={"MS/MS spectrum": "File name"})
    info_samples = info_samples[
        info_samples["File name"].str.split("_").str[:3].str.join("_").isin(updated_file_list)
    ]
    return info_samples


def load_sample_concentrations(suspect_path, updated_file_list):
    all_samples = pd.read_csv(suspect_path, skiprows=4, header=0)
    sel_samples = np.append(
        np.repeat(False, 33),
        all_samples.columns[33:].str.split("_").str[:3].str.join("_").isin(updated_file_list),
    )
    sel_samples[all_samples.columns == "Metabolite name"] = True  #also giving the column with compound names, so i can select them
    sel_samples[all_samples.columns == "Average Rt(min)"] = True
    return all_samples.iloc[:, sel_samples]


def measure_suspect(suspect_row, image_names, task_name, image_path, pred_path,
                    label_path, clipped_path, rt_bins, rt_sd=20):
    susp_mz = suspect_row["mz"].item()
    susp_rt = suspect_row["RT"].item()
    name = suspect_row["name"]

    if task_name + "_" + name + "_" + "000" not in image_names:
        return None

    print(f"Converting suspect: {name} - mz = {susp_mz}, rt = {susp_rt}")

    #First I'm getting a raw image and the label and I'm overlapping them to get the pixel labeled as "pakes" and their intensities to be able to sum them
    raw_image = os.path.join(image_path, task_name) + "_" + name + "_000" + ".png"
    label_image = os.path.join(pred_path, task_name + "_" + name) + ".png"

    #@TODO Add binarize input argument so it can be binarized if it isn't.
    greyscale_img = Image.open(raw_image).convert("L")  # Ensure grayscale
    label_img = Image.open(label_image)  # already in binary
    greyscale_array = np.array(greyscale_img)  # Grayscale intensities
    label_array = np.array(label_img)  # Binary mask (0-1)

    selected_img = Image.fromarray(np.where(label_array == 1, greyscale_array, 0))
    selected_img.save(os.path.join(label_path, name) + ".png")

    label_img = Image.fromarray(np.where(label_array == 1, 255, 0).astype(np.uint8))

    susp_rt_bin = np.digitize(susp_rt, rt_bins).item() - 1
    bin_index_low = np.digitize(susp_rt - rt_sd, rt_bins).item() - 1
    bin_index_high = np.digitize(susp_rt + rt_sd, rt_bins).item() + 1

    print(f"Selected rt range for cropping: [{rt_bins[bin_index_low]}, {rt_bins[bin_index_high]}]")

    marked_img = selected_img.convert("RGB")
    draw = ImageDraw.Draw(marked_img)
    draw.line([(susp_rt_bin, 0), (susp_rt_bin, marked_img.height)], fill=(255, 0, 0), width=1)

    left, right = bin_index_low, bin_index_high
    upper, lower = 0, selected_img.height

    raw_crop = greyscale_img.crop((left, upper, right, lower))
    label_crop = label_img.crop((left, upper, right, lower))
    over_crop = selected_img.crop((left, upper, right, lower))
    marked_crop = marked_img.crop((left, upper, right, lower))

    raw_crop.save(os.path.join(clipped_path, name) + "_raw" + ".png")
    label_crop.save(os.path.join(clipped_path, name) + "_label" + ".png")
    over_crop.save(os.path.join(clipped_path, name) + "_overlay" + ".png")
    marked_crop.save(os.path.join(clipped_path, name) + "_redline" + ".png")

    int_pixels = np.array(over_crop)
    sum_int = int_pixels.sum(axis=1)

    return sum_int, susp_mz, susp_rt
