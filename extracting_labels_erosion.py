import re
import os
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy import ndimage

def label_to_rgb_pil(lbl):
    
    rgb = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
    rgb[lbl == 1] = [255, 255, 255]  # white
    rgb[lbl == 2] = [255,   0,   0]  # red
    # 0 stays black
    return Image.fromarray(rgb)


# getting labels from yucca predictions
task_name = 'Task025_standFlourID'
test_task = 'Task043_sim14L3'

img_path = f'/maps/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/yucca_runs/test_01/raw_data/{task_name}/imagesTs'
lbl_path = f'/maps/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/yucca_runs/test_01/results/{task_name}'

#labels1 = sorted(os.listdir(os.path.join(lbl_path, 'Task041_sim14L1','UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best')))
#labels2 = sorted(os.listdir(os.path.join(lbl_path, 'Task042_sim14L2','UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best')))
labels = sorted(os.listdir(os.path.join(lbl_path, test_task,'UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best')))

# folder with test images
images_raw = f"/home/frz603/DSP/yucca_runs/test_01/raw_data/{task_name}/imagesTs"

# making new folders for output
version = 'sim14'
label_number = 'L3_erosion'

clustering_folder = f'/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/final results/{task_name}/clustering/{version}/{label_number}'
os.makedirs(clustering_folder, exist_ok=True)
print(f'Labels saved in: {clustering_folder}')

label_folder = os.path.join(clustering_folder, 'single_labels')
os.makedirs(label_folder, exist_ok=True)

overlap_folder = os.path.join(clustering_folder, 'single_labels_overlap')
os.makedirs(overlap_folder, exist_ok=True)

os.makedirs(os.path.join(clustering_folder, 'yucca_labels'), exist_ok=True)

# going through all images and pulling single peak labels out (blobs out of label 3 + erosion for shoulder peaks)
erosion_width = 3

print(f'Task: {task_name} - erosion size in pixels: ({erosion_width}, 1), number of labels: {len(labels)}')

for k in range(len(labels)):

    if (k/len(labels)*100)%10==0: print(f'Percentage of images done: {k/len(labels)*100}')
    
    #susp_mz = 575.12
    #lbl_name = [f for f in labels if float(f.split('_')[2]+'.'+f.split('_')[3])<=susp_mz and float(f.split('_')[4]+'.'+re.sub('.nii.gz', '',f).split('_')[5])>=susp_mz][0]
    lbl_name = labels[k]
    base_name = re.sub(".nii.gz", "", lbl_name)
    #print(img_name)

    # opening labels to labeled images to postprocess
    img = nib.load(os.path.join(lbl_path, test_task, 'UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best', lbl_name))
    img = img.get_fdata() 

    # EIC image
    img_raw = np.load(os.path.join(images_raw, re.sub('.nii.gz', '', lbl_name)+'_000.npy'))

    # getting labels out just to inspect - not necessary to do for all images
    lbl3 = label_to_rgb_pil(img)
    lbl3.save(os.path.join(clustering_folder,'yucca_labels', re.sub('nii.gz', 'png', lbl_name)))

    # making connected images and filtering small blobs
    orig_cc = label(img, connectivity=1)
    blob_mask = orig_cc > 0

    # Saving all resulting label strips together
    extended_image = np.zeros_like(blob_mask, dtype=np.uint16)
    extended_mask = np.zeros_like(blob_mask, dtype=np.uint8)
    blob_i = 1

    for orig_region in regionprops(orig_cc):

        # One original L3 blob
        orig_component = orig_cc == orig_region.label

        # Erode only this blob
        eroded_component = ndimage.binary_erosion(orig_component, structure=np.ones((2 * erosion_width + 1, 1), dtype=bool))

        # Connected components after erosion
        eroded_cc = label(eroded_component, connectivity=1)
        n_eroded = eroded_cc.max()

        if n_eroded <= 1:

            # No split, or blob disappeared after erosion:
            # use the ORIGINAL RT range
            min_y, min_rt, max_y, max_rt = orig_region.bbox
            rt_ranges = [(min_rt, max_rt)]

        else:

            # Blob split after erosion:
            # use one RT range for every separated eroded piece
            rt_ranges = []

            for eroded_region in regionprops(eroded_cc):
                min_y, min_rt, max_y, max_rt = eroded_region.bbox
                rt_ranges.append((min_rt, max_rt))

        # Add all candidate RT strips first; do not save yet
        for min_rt, max_rt in rt_ranges:
            extended_image[:, min_rt:max_rt] = 1


    # Merge overlapping RT strips into final separate labels
    final_cc = label(extended_image, connectivity=1)

    # Save one overlap image per final merged strip
    for final_region in regionprops(final_cc):

        final_mask = final_cc == final_region.label

        empty_img = np.zeros_like(img_raw)
        empty_img[final_mask] = img_raw[final_mask]

        if np.any(empty_img != 0):

            np.save(os.path.join(overlap_folder, f"{base_name}_{str(blob_i).zfill(3)}.npy"), empty_img)
            blob_i += 1

    # Saving individual labels altogether for check
    plt.imsave(os.path.join(label_folder, f"{base_name}.png"), final_cc, cmap="tab20")


print(f'No. of labels created: {len(os.listdir(overlap_folder))}')

ms1_files = [f for f in os.listdir(overlap_folder) if "MS1_" in f]
ms2_files = [f for f in os.listdir(overlap_folder) if "MS2_" in f]

# === Print summary ===
print(f"Found {len(ms1_files)} MS1 files")
print(f"Found {len(ms2_files)} MS2 files")