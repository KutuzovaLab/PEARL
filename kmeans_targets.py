"""
Usage:  python kmeans_targets.py --config configs/config_kmeans_targets.yaml
"""

import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from src.config import parse_config_arg
from src.kmeans_targets import import_suspects, noise_thres, stack_labels

def main(cfg):
    suspect_path = cfg["suspect_path"]
    images_path = cfg["images_path"]
    output_folder = cfg["output_folder"]
    target = cfg["target"]
    noise_threshold = cfg.get("noise_threshold", 0.00001)

    os.makedirs(output_folder, exist_ok=True)

    suspects = import_suspects(suspect_path)

    name_counts = suspects['name'].value_counts().reset_index()
    name_counts.columns = ['name', 'count']
    name_counts = name_counts.sort_values(by='count', ascending=False)
    name_counts = name_counts[name_counts['count'] == 1]

    susp_sel = suspects[suspects['name'].isin(target)].reset_index(drop=True)
    print(susp_sel)
    unet_labels = os.listdir(images_path)

    ms = []
    for i in range(len(susp_sel)):
        ms.append([f for f in unet_labels if f.split('_')[1] == 'MS1'
                   and float(f.split('_')[2] + '.' + f.split('_')[3]) <= susp_sel.loc[i, 'mz']
                   and susp_sel.loc[i, 'mz'] <= float(f.split('_')[4] + '.' + re.sub('.png', '', f).split('_')[5])])

    MS1_sel = np.unique(np.concatenate(ms))
    MS2_sel = [f for f in unet_labels if f.split('_')[1] == 'MS2']
    ms1_labels, ms1_names = stack_labels(images_path, MS1_sel)
    ms2_labels, ms2_names = stack_labels(images_path, MS2_sel)
    init_centroids = ms1_labels
    kmeans = KMeans(n_clusters=len(init_centroids), init=init_centroids)
    kmeans.fit(ms2_labels)
    print(kmeans.labels_, flush=True)

    results = []
    for i in range(len(ms2_names)):
        results.append({'MS2_image': ms2_names[i], 'Cluster': int(kmeans.labels_[i]),
                        'MS1_centroid': ms1_names[kmeans.labels_[i]]})

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(output_folder, 'clusters.csv'), index=False)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    distances = np.linalg.norm(ms2_labels - centroids[labels], axis=1)

    dist_centr = pd.DataFrame({'ms2_image': ms2_names, 'dist': distances, 'centroid': ms1_names[labels]})
    closest_images = dist_centr.sort_values(by=['centroid', 'dist'])

    ms2_folder = os.path.join(output_folder, 'MS2')
    os.makedirs(ms2_folder, exist_ok=True)

    for i in range(1, len(np.unique(closest_images['centroid']))):
        ms2_image_moved = np.array(
            closest_images[closest_images['centroid'] == np.unique(closest_images['centroid'])[i]]['ms2_image'][0:20]
        )
        target_path = os.path.join(ms2_folder, re.sub('.png', '', np.unique(closest_images['centroid'])[i]))
        os.makedirs(target_path, exist_ok=True)
        for k in range(len(ms2_image_moved)):
            shutil.copy(os.path.join(images_path, ms2_image_moved[k]),
                        os.path.join(target_path, ms2_image_moved[k]))

    for i in range(len(closest_images)):
        img = np.array(Image.open(os.path.join(images_path, closest_images.loc[i, 'ms2_image'])))
        img = Image.fromarray(np.where(img == 1, 255, 0).astype(np.uint8))
        img.save(os.path.join(ms2_folder, closest_images.loc[i, 'ms2_image']))

    sns.set_theme(style="whitegrid")
    closest_images['centroid'] = closest_images['centroid'].astype(str)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=closest_images, x='ms2_image', y='dist', hue='centroid')
    plt.xticks(rotation=90)
    plt.xlabel('MS2 Image')
    plt.ylabel('Distance to Centroid')
    plt.title('Closest MS2 Images to Centroids')
    plt.legend(title='Centroid', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'centroids.png'))

    os.makedirs(os.path.join(output_folder, target[0]), exist_ok=True)

    img = np.array(Image.open(os.path.join(images_path, MS1_sel[0])))
    img = Image.fromarray(np.where(img == 1, 255, 0).astype(np.uint8))
    img.save(os.path.join(output_folder, target[0], MS1_sel[0]))

    ms2_exp = np.unique(np.round(noise_thres(susp_sel, noise_threshold), 1))

    row = susp_sel.iloc[0]
    mz_values = row['mz_list']
    int_values = row['int_list']

    plt.figure(figsize=(10, 6))
    plt.vlines(mz_values, ymin=0, ymax=int_values, color='blue')
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(f"Spectrum for RT={row['RT']}, m/z={row['mz']}")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, target[0], 'MS2.png'))

    images_ms2 = []
    for i in range(len(ms2_exp)):
        mass = ms2_exp[i]
        images_ms2.extend([f for f in unet_labels if f.split('_')[1] == 'MS2'
                           and float(f.split('_')[2] + '.' + f.split('_')[3]) <= mass
                           and mass <= float(f.split('_')[4] + '.' + re.sub('.png', '', f).split('_')[5])])

    for i in range(len(images_ms2)):
        img = np.array(Image.open(os.path.join(images_path, images_ms2[i])))
        img = Image.fromarray(np.where(img == 1, 255, 0).astype(np.uint8))
        img.save(os.path.join(output_folder, target[0], images_ms2[i]))


if __name__ == "__main__":
    config = parse_config_arg(description="Cluster MS2 images against MS1 targets (K-means).")
    main(config)
