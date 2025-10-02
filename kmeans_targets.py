import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import shutil

# Functions

def extract_mz_and_int_lists(ms2_string):
    pairs = ms2_string.split()
    mz_list = []
    int_list = []
    for pair in pairs:
        if ":" in pair:
            mz_str, int_str = pair.split(":")
            mz_list.append(float(mz_str))
            int_list.append(float(int_str))  # or int(int_str) if you want pure integers
    return mz_list, int_list


def import_suspects(suspect_path):

    all_samples = pd.read_csv(suspect_path, skiprows=4, header = 0)

    # keeping only those suspects that have %fill greater than 0.9 and they have names assigned
    suspects = all_samples.iloc[:,[1,2,3,6,31]]  #selecting columns with relevant info
    suspects.columns = ["RT", "mz", "name", "fill", 'MS2']
    suspects.loc[:, "RT"]=suspects["RT"]*60  #rt is in min

    suspects = suspects[suspects["name"]!="Unknown"]  #choosing only compounds that were identified with names (also avoiding adducts)
    suspects = suspects[suspects["fill"]>=0.9]  # %fill is a parameter from MsDial that tells how well the identification went
    suspects = suspects[~suspects["name"].str.contains("w/o MS2", na=False)]  # using only those with assigned MS2
    suspects = suspects[suspects["RT"]>=100].reset_index(drop = True)   #to avoid big peaks in the beggining

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
        mz_sel = np.array(df.loc[i,'mz_list'])[int_norm>=thres]
        mz_over.extend(mz_sel)
    mz_over = sorted(mz_over)
    mz_over = [float(x) for x in mz_over]
    return mz_over

def stack_labels(labels_path, labeled_images):    
    # opening images and flattening them into 1D vectors
    all_labels = []
    for i in range(len(labeled_images)):
    #for i in range(3):
        label_array = np.array(Image.open(os.path.join(labels_path, labeled_images[i])).convert("L")).astype(np.float32)  
        label_flat = label_array.flatten()
        all_labels.append(label_flat)
    all_labels = np.stack(all_labels)
    print(all_labels.shape, flush=True)
    
    return (all_labels, labeled_images)

# Getting suspects

suspect_path = "/home/frz603/DSP/matrix_files/DK_2021_Secondary metabolites_POS.csv"
suspects = import_suspects(suspect_path)

# Checking those that have multiple ions (in source fragmentation)

name_counts = suspects['name'].value_counts().reset_index()
name_counts.columns = ['name', 'count']
name_counts = name_counts.sort_values(by='count', ascending=False)
name_counts = name_counts[name_counts['count']==1]


# I will look into one target now and get all the info on all ions

target = ['Benzimidazole']
target = ['Khellin',  'Benzimidazole', 'RIBOFLAVIN', 'TRYPTOPHAN', 'Homocystine', 'Genistein']
susp_sel = suspects[suspects['name'].isin(target)].reset_index(drop = True)
print(susp_sel)


# Now I'm grabbing all images

# MS1 - only slices correspoding to the suspect ions

images_path = '/home/frz603/DSP/yucca_runs/test_01/results/Task007_allMatrix/Task003_simOver/UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best'
unet_labels = os.listdir(images_path)

ms = []
for i in range(len(susp_sel)):
    ms.append([f for f in unet_labels if f.split('_')[1]=='MS1' and float(f.split('_')[2]+'.'+f.split('_')[3])<=susp_sel.loc[i,'mz'] and susp_sel.loc[i,'mz']<=float(f.split('_')[4]+'.'+re.sub('.png', '', f).split('_')[5])])


MS1_sel = np.unique(np.concatenate(ms))

# I'm grabbing MS2 images from all of the labels

MS2_sel = [f for f in unet_labels if f.split('_')[1]=='MS2']

# Now stacking images together to use them as 1D vectors in clustering

ms1_labels, ms1_names = stack_labels(images_path, MS1_sel)
ms2_labels, ms2_names = stack_labels(images_path, MS2_sel)

# Clustering MS1 and MS2

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

# Looking into distance between center and assigned images

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
distances = np.linalg.norm(ms2_labels - centroids[labels], axis=1)

dist_centr = pd.DataFrame({'ms2_image': ms2_names, 'dist': distances, 'centroid': ms1_names[labels]})
closest_images = dist_centr.sort_values(by=['centroid', 'dist'])

# Moving closest images so I can visually inspect them

output_folder = f'/home/frz603/DSP/matrix_images/clustering/MS2'
os.makedirs(output_folder, exist_ok=True)

for i in range(1,len(np.unique(closest_images['centroid']))):
    ms2_image_moved = np.array(closest_images[closest_images['centroid']==np.unique(closest_images['centroid'])[i]]['ms2_image'][0:20])
    target_path = os.path.join(output_folder, re.sub('.png', '', np.unique(closest_images['centroid'])[i]))
    os.makedirs(target_path, exist_ok=True)
    for k in range(len(ms2_image_moved)):
        shutil.copy(os.path.join(images_path, ms2_image_moved[k]), os.path.join(target_path, ms2_image_moved[k]))




for i in range(len(closest_images)):
    img = np.array(Image.open(os.path.join(images_path, closest_images.loc[i,'ms2_image'])))
    img = Image.fromarray(np.where(img == 1, 255, 0).astype(np.uint8))
    img.save(os.path.join(output_folder, closest_images.loc[i,'ms2_image']))

# Plotting the distribution around centroids

import matplotlib.pyplot as plt
import seaborn as sns

# Optional: set a seaborn style
sns.set_theme(style="whitegrid")

# Make sure 'centroid' is treated as a categorical variable
closest_images['centroid'] = closest_images['centroid'].astype(str)

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(data=closest_images, x='ms2_image', y='dist', hue='centroid')

# Improve readability
plt.xticks(rotation=90)
plt.xlabel('MS2 Image')
plt.ylabel('Distance to Centroid')
plt.title('Closest MS2 Images to Centroids')
plt.legend(title='Centroid', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/home/frz603/DSP/matrix_images/clustering/centroids.png')


# Checking with the expected MS2 results

# First I'm moving MS1 image to a new folder called by the name of the suspect

output_folder = '/home/frz603/DSP/matrix_images/clustering'
os.makedirs(os.path.join(output_folder, target[0]), exist_ok=True)

img = np.array(Image.open(os.path.join(images_path, MS1_sel[0])))
img = Image.fromarray(np.where(img == 1, 255, 0).astype(np.uint8))
img.save(os.path.join(output_folder, target[0], MS1_sel[0]))


# Then I'm moving MS2 images that are the highest 

ms2_exp = np.unique(np.round(noise_thres(susp_sel, 0.00001),1))

# Just plotting full MS2 to check the variability in intensities

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

# Now I'm moving all labels of MS2 that are in MsDial into the same folder to be able to visually inspect them

images_ms2 = []

for i in range(len(ms2_exp)):
    mass = ms2_exp[i]
    images_ms2.extend([f for f in unet_labels if f.split('_')[1]=='MS2' and float(f.split('_')[2]+'.'+f.split('_')[3])<=mass and mass<=float(f.split('_')[4]+'.'+re.sub('.png', '', f).split('_')[5])])

for i in range(len(images_ms2)):
    img = np.array(Image.open(os.path.join(images_path, images_ms2[i])))
    img = Image.fromarray(np.where(img == 1, 255, 0).astype(np.uint8))
    img.save(os.path.join(output_folder, target[0], images_ms2[i]))


# I'm looking into all ms2 signals to check the range

mz_list = []
int_list = []

for i in range(len(suspects)):
    mz_list.extend(suspects.loc[i, 'mz_list'])
    int_list.extend(suspects.loc[i, 'int_list'])

ms2_spectra = pd.DataFrame({'mz': mz_list, 'int': int_list})
ms2_spectra = ms2_spectra[ms2_spectra['int']>=1000]

min(mz_list)
max(ms2_spectra['mz'])