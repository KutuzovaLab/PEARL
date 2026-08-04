"""Entry point: match MS2 label images to their nearest MS1 images.

Usage:
    python matching_MS1_MS2.py --config configs/config_matching_MS1_MS2.yaml
"""

import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from src.config import parse_config_arg
from src.matching_MS1_MS2 import list_ms_files, extract_embeddings


def main(cfg):
    path_ms_files = cfg["path_ms_files"]
    binary = cfg.get("binary", False)
    K = cfg["k"]
    output_folder = cfg["output_folder"]

    ms1_files, ms2_files = list_ms_files(path_ms_files)

    # === Extract feature vectors ===
    ms1_vectors, ms1_names = extract_embeddings(ms1_files, path_ms_files, binary=binary)
    ms2_vectors, ms2_names = extract_embeddings(ms2_files, path_ms_files, binary=binary)
    print("All images embedded!")

    # === Saving results ===
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'ms1_vectors.npz'), ms1_vectors)
    np.save(os.path.join(output_folder, 'ms1_names.npz'), np.array(ms1_names))
    np.save(os.path.join(output_folder, 'ms2_vectors.npz'), ms2_vectors)
    np.save(os.path.join(output_folder, 'ms2_names.npz'), np.array(ms2_names))

    # === Nearest neighbor search ===
    tree = KDTree(ms1_vectors)
    print("Tree built!")

    dists, indices = tree.query(ms2_vectors, k=K)

    ms2_to_ms1 = pd.DataFrame([
        (ms2_names[i], ms1_names[indices[i][j]], dists[i][j])
        for i in range(len(ms2_names))
        for j in range(K)
    ], columns=["ms2_name", "ms1_name", "distance"])
    ms2_to_ms1.to_excel(os.path.join(output_folder, 'final_matches.xlsx'))

    print("Matches found and saved!")


if __name__ == "__main__":
    config = parse_config_arg(description="Match MS2 images to nearest MS1 images.")
    main(config)
