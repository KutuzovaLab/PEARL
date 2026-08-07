import os
import pandas as pd
import os
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import pickle

task_name = 'Task024_suspFinal'
version = 'sim14/L3_erosion'
folder = f"/maps/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/final results/{task_name}/clustering/{version}"
path_ms_files = os.path.join(folder, "single_labels_overlap")

# === List all PNG files ===
all_files = [f for f in os.listdir(path_ms_files) if f.endswith(".npy")]

# === Separate into MS1 and MS2 lists ===
ms1_files = [f for f in all_files if "MS1_" in f]
ms2_files = [f for f in all_files if "MS2_" in f]

# === Print summary ===
print(f"Found {len(ms1_files)} MS1 files")
print(f"Found {len(ms2_files)} MS2 files")


# === Removing images that are all 0 for some reason ===

#deleted = []

#for fname in sorted(os.listdir(path_ms_files)):

    #fpath = os.path.join(path_ms_files, fname)
    #arr = np.load(fpath)

    #if np.all(arr == 0):
        #print(fname)
        #os.remove(fpath)
        #deleted.append(fname)

#print(f"Deleted {len(deleted)} all-zero .npy files")


# === Selecting subset to make computation faster ===

# I will select only mass slices that are around nominal mass (0.0 to 0.2) for MS1

#ms1_files_sub = [f for f in ms1_files if f.split("_")[3] in {"0", "1"}]

# For MS2 I want only those that interest me aka fragments mentioned in Oskar's table

#ms2_expected = pd.read_excel("/home/frz603/DSP/matrix_standards/2dlc.xlsx")
#ms2_expected = np.array(ms2_expected.loc[ms2_expected["Name"].isin(["Luteolin", "Kaempferol"]), "Fragments (m/z)"])
#ms2_expected = np.concatenate([np.array(s.split(","), dtype=float) for s in ms2_expected])

#ms_error = 0.1

#ms_selected = []
#for i in range(len(ms2_expected)):
    #ms_selected = np.append(ms_selected, [f for f in ms2_files if float(f.split("_")[2] + '.' + f.split("_")[3]) >= (ms2_expected[i]-ms_error) and float(f.split("_")[4] + '.' + f.split("_")[5]) <= (ms2_expected[i]+ms_error)])
#print(f"Selected {len(ms_selected)} MS2 files")



# === Parameters ===
BINARY = False  # If True, convert to grayscale and binarize before sum

def image_projection_embedding(img_path):
    #img = Image.open(img_path).convert("L")  # grayscale
    arr = np.load(img_path)

    if BINARY:
        arr = (arr > 128).astype(int)

    row_sum = arr.sum(axis=1)  # shape (H,)
    col_sum = arr.sum(axis=0)  # shape (W,)
    vec = np.concatenate([row_sum, col_sum])

    # Normalize so max is 1
    max_val = vec.max()
    if max_val > 0:
        vec = vec / max_val

    return vec

def extract_embeddings(file_list, folder):
    embeddings = []
    valid_names = []
    for fname in tqdm(file_list):
        try:
            path = os.path.join(folder, fname)
            vec = image_projection_embedding(path)
            embeddings.append(vec)
            valid_names.append(fname)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    return np.array(embeddings), valid_names

# === Extract feature vectors ===
ms1_vectors, ms1_names = extract_embeddings(ms1_files, path_ms_files)
#ms2_vectors, ms2_names = extract_embeddings(ms_selected, path_ms_files)

ms2_vectors, ms2_names = extract_embeddings(ms2_files, path_ms_files)
#image_names = [f.removesuffix(".npy").replace("standardsnpy_", "") for f in ms1_files]
#image_name = [f for f in image_names if float(f.split('_')[1]+'.'+f.split('_')[2])<=stand_mz and float(f.split('_')[3]+'.'+f.split('_')[4])>=stand_mz]

#ms1_vectors = np.load('/home/frz603/DSP/matrix_standards/raw_images/clustering/kdtree/ms1_vectors.npz.npy')
#ms1_names = np.load('/home/frz603/DSP/matrix_standards/raw_images/clustering/kdtree/ms1_names.npz.npy')

print("All images embedded!")

# === Setting parameters ===
K = 30

# === Saving results ===
output_folder = os.path.join(folder, 'clusters_k'+ str(K))
os.makedirs(output_folder, exist_ok=True)
np.save(os.path.join(output_folder,'ms1_vectors.npz'), ms1_vectors)
np.save(os.path.join(output_folder,'ms1_names.npz'), np.array(ms1_names))

np.save(os.path.join(output_folder,'ms2_vectors.npz'), ms2_vectors)
np.save(os.path.join(output_folder,'ms2_names.npz'), np.array(ms2_names))

# === Nearest neighbor search ===
tree = KDTree(ms1_vectors)

tree_path = os.path.join(output_folder, "ms1_kdtree.pkl")
with open(tree_path, "wb") as f:
    pickle.dump(tree, f)

print(f"Tree saved to: {tree_path}")

# Querying MS1 tree with MS2
dists, indices = tree.query(ms2_vectors, k=K)

#ms2_to_ms1 = pd.DataFrame([(ms2_names[i], ms1_names[indices[i][0]], dists[i][0]) for i in range(len(ms2_names))])
ms2_to_ms1 = pd.DataFrame([
    (ms2_names[i], ms1_names[indices[i][j]], dists[i][j])
    for i in range(len(ms2_names))
    for j in range(K)
], columns=["ms2_name", "ms1_name", "distance"])

ms2_to_ms1.to_csv(os.path.join(output_folder, 'final_matches.csv'), index=False)

print("Matches found and saved!")




