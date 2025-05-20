import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd

# paths necessery for the running

labels_path = '/home/frz603/DSP/yucca_runs/test_01/results/Task007_allMatrix/Task003_simOver/UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best'

# getting labels and turning them into 1D np arrays

def stack_labels(labels_path, mslevel):
    # getting all the labels
    label_images = os.listdir(labels_path)
    filtered_files = [f for f in label_images if f.split('_')[1]==mslevel]
    # opening images and flattening them into 1D vectors
    all_labels = []
    for i in range(len(filtered_files)):
    #for i in range(3):
        label_array = np.array(Image.open(os.path.join(labels_path, filtered_files[i])).convert("L")).astype(np.float32)  
        label_flat = label_array.flatten()
        all_labels.append(label_flat)
    all_labels = np.stack(all_labels)
    print(all_labels.shape, flush=True)
    
    return (all_labels, filtered_files)

ms1_labels, ms1_names = stack_labels(labels_path, 'MS1')
ms2_labels, ms2_names = stack_labels(labels_path, 'MS2')

# clustering MS1 and MS2

init_centroids = ms1_labels

kmeans = KMeans(n_clusters = len(init_centroids), init = init_centroids)
kmeans.fit(ms2_labels)
print(kmeans.labels_, flush = True)

# saving the results

results = []

for i in range(len(ms2_names)):
    results.append({'MS2_image': ms2_names[i], 'Cluster': int(kmeans.labels_[i]), 'MS1_centroid': ms1_names[kmeans.labels_[i]]})

results = pd.DataFrame(results)
results.to_csv('/home/frz603/DSP/matrix_images/clustering/clusters.csv', index=False)