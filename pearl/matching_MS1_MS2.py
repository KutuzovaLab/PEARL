"""Cluster the MS2 peak labels onto the MS1 ones they co-elute with.

    python -m pearl.matching_MS1_MS2 --config <output>/config.yaml --section matching
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

from pearl.common import parse_config_arg
from pearl.constants import K as K_NEIGHBOURS

def image_projection_embedding(img_path):
    arr = np.load(img_path)
    row_sum = arr.sum(axis=1)
    col_sum = arr.sum(axis=0)
    vec = np.concatenate([row_sum, col_sum])


    max_val = vec.max()
    if max_val > 0:
        vec = vec / max_val
    return vec

def extract_embeddings(file_list, folder):
    embeddings = []
    valid_names = []
    for fname in tqdm(file_list):
        try:
            vec = image_projection_embedding(os.path.join(folder, fname))
            embeddings.append(vec)
            valid_names.append(fname)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    return np.array(embeddings), valid_names

def main(cfg):
    input_dir = cfg["input_dir"]
    K = K_NEIGHBOURS
    output_dir = cfg.get(
        "output_dir",
        os.path.join(os.path.dirname(os.path.normpath(input_dir)), f"clusters_k{K}"),
    )

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    ms1_files = [f for f in all_files if "MS1_" in f]
    ms2_files = [f for f in all_files if "MS2_" in f]
    print(f"Found {len(ms1_files)} MS1 files")
    print(f"Found {len(ms2_files)} MS2 files")

    if not ms1_files or not ms2_files:
        side = "MS1" if not ms1_files else "MS2"
        print(f"WARNING: no {side} label images in {input_dir}; "
              f"skipping the MS1-MS2 matching", flush=True)
        return

    ms1_vectors, ms1_names = extract_embeddings(ms1_files, input_dir)
    ms2_vectors, ms2_names = extract_embeddings(ms2_files, input_dir)
    print("All images embedded!")

    if K > len(ms1_names):
        print(f"k={K} exceeds the {len(ms1_names)} MS1 labels available; clamping k to {len(ms1_names)}")
        K = len(ms1_names)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "ms1_vectors.npz"), ms1_vectors)
    np.save(os.path.join(output_dir, "ms1_names.npz"), np.array(ms1_names))
    np.save(os.path.join(output_dir, "ms2_vectors.npz"), ms2_vectors)
    np.save(os.path.join(output_dir, "ms2_names.npz"), np.array(ms2_names))


    tree = KDTree(ms1_vectors)
    tree_path = os.path.join(output_dir, "ms1_kdtree.pkl")
    with open(tree_path, "wb") as f:
        pickle.dump(tree, f)
    print(f"Tree saved to: {tree_path}")

    dists, indices = tree.query(ms2_vectors, k=K)
    ms2_to_ms1 = pd.DataFrame([
        (ms2_names[i], ms1_names[indices[i][j]], dists[i][j])
        for i in range(len(ms2_names))
        for j in range(K)
    ], columns=["ms2_name", "ms1_name", "distance"])

    ms2_to_ms1.to_csv(os.path.join(output_dir, "final_matches.csv"), index=False)
    print("Matches found and saved!")

if __name__ == "__main__":
    main(parse_config_arg(description="Match MS2 label images to nearest MS1 via KDTree."))
