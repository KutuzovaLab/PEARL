import os
import numpy as np
import pandas as pd
from PIL import Image

def extract_mz_and_int_lists(ms2_string):
    pairs = ms2_string.split()
    mz_list = []
    int_list = []
    for pair in pairs:
        if ":" in pair:
            mz_str, int_str = pair.split(":")
            mz_list.append(float(mz_str))
            int_list.append(float(int_str))
    return mz_list, int_list

def import_suspects(suspect_path):

    all_samples = pd.read_csv(suspect_path, skiprows=4, header=0)

    # keeping only suspects with %fill >= 0.9 and names assigned
    suspects = all_samples.iloc[:, [1, 2, 3, 6, 31]]  # columns with relevant info
    suspects.columns = ["RT", "mz", "name", "fill", 'MS2']
    suspects.loc[:, "RT"] = suspects["RT"] * 60  # rt is in min

    suspects = suspects[suspects["name"] != "Unknown"]  # only named compounds (avoids adducts)
    suspects = suspects[suspects["fill"] >= 0.9]  # %fill from MsDial
    suspects = suspects[~suspects["name"].str.contains("w/o MS2", na=False)]  # only with assigned MS2
    suspects = suspects[suspects["RT"] >= 100].reset_index(drop=True)  # avoid big peaks at the beginning

    suspects[['mz_list', 'int_list']] = suspects['MS2'].apply(
        lambda x: pd.Series(extract_mz_and_int_lists(x))
    )

    return suspects

def normalize_int_list(int_list):
    max_val = max(int_list)
    return [i / max_val for i in int_list] if max_val else int_list

def noise_thres(df, thres):
    mz_over = []
    for i in range(len(df)):
        int_norm = np.array([intensity / max(df.loc[i, 'int_list']) for intensity in df.loc[i, 'int_list']])
        mz_sel = np.array(df.loc[i, 'mz_list'])[int_norm >= thres]
        mz_over.extend(mz_sel)
    mz_over = sorted(mz_over)
    mz_over = [float(x) for x in mz_over]
    return mz_over

def stack_labels(labels_path, labeled_images):
    # opening images and flattening them into 1D vectors
    all_labels = []
    for i in range(len(labeled_images)):
        label_array = np.array(
            Image.open(os.path.join(labels_path, labeled_images[i])).convert("L")
        ).astype(np.float32)
        label_flat = label_array.flatten()
        all_labels.append(label_flat)
    all_labels = np.stack(all_labels)
    print(all_labels.shape, flush=True)

    return (all_labels, labeled_images)
