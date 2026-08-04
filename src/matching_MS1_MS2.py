import os
import numpy as np
from tqdm import tqdm

def list_ms_files(path_ms_files):
    """Return (ms1_files, ms2_files) lists of .npy files in a folder."""
    all_files = [f for f in os.listdir(path_ms_files) if f.endswith(".npy")]
    ms1_files = [f for f in all_files if "MS1_" in f]
    ms2_files = [f for f in all_files if "MS2_" in f]
    print(f"Found {len(ms1_files)} MS1 files")
    print(f"Found {len(ms2_files)} MS2 files")
    return ms1_files, ms2_files


def image_projection_embedding(img_path, binary=False):
    """Embed an image as concatenated, max-normalised row and column sums."""
    arr = np.load(img_path)

    if binary:
        arr = (arr > 128).astype(int)

    row_sum = arr.sum(axis=1)  # shape (H,)
    col_sum = arr.sum(axis=0)  # shape (W,)
    vec = np.concatenate([row_sum, col_sum])

    # Normalize so max is 1
    max_val = vec.max()
    if max_val > 0:
        vec = vec / max_val

    return vec


def extract_embeddings(file_list, folder, binary=False):
    """Embed every file in ``file_list``; returns (embeddings, valid_names)."""
    embeddings = []
    valid_names = []
    for fname in tqdm(file_list):
        try:
            path = os.path.join(folder, fname)
            vec = image_projection_embedding(path, binary=binary)
            embeddings.append(vec)
            valid_names.append(fname)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    return np.array(embeddings), valid_names
