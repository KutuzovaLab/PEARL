"""
Usage:  python preparing_for_clustering.py --config configs/config_preparing_for_clustering.yaml
"""

import os
import re
import numpy as np
from PIL import Image
import nibabel as nib
from src.config import parse_config_arg
from src.preparing_for_clustering import get_consecutive_subsets

def main(cfg):
    folder = cfg["folder"]
    images_raw = cfg["images_raw"]
    sample_thres = cfg["sample_thres"]
    label_width = cfg["label_width"]
    clustering_folder_base = cfg["clustering_folder_base"]

    images = sorted([f for f in os.listdir(folder) if f.endswith(".nii.gz")])

    # making new folders for output
    clustering_folder = os.path.join(clustering_folder_base, f'sn{sample_thres}_w{label_width}')
    os.makedirs(clustering_folder, exist_ok=True)

    label_folder = os.path.join(clustering_folder, 'single_labels')
    os.makedirs(label_folder, exist_ok=True)

    overlap_folder = os.path.join(clustering_folder, 'single_labels_overlap')
    os.makedirs(overlap_folder, exist_ok=True)

    os.makedirs(os.path.join(clustering_folder, 'yucca_labels'), exist_ok=True)

    # going through all images and pulling single peak labels out
    # (only if the label is present in more than sample_thres samples)
    for k in range(len(images)):

        img_name = images[k]
        print(img_name)
        img = nib.load(os.path.join(folder, img_name))
        img = img.get_fdata()

        # getting labels out just to inspect - not necessary to do for all images
        label = np.zeros(img.shape)
        label[img == 1] = 255
        label[img == 2] = 200
        label = Image.fromarray(label.astype(np.uint8), mode='L')
        label.save(os.path.join(clustering_folder, 'yucca_labels', re.sub('nii.gz', 'png', img_name)))

        # taking sums on columns and keeping pixels that are greater than sample_thres
        int_sum = img.sum(axis=0)
        int_sum = np.where(int_sum >= sample_thres, 255, 0).astype(np.uint8).reshape(1, -1)

        # selected labels in 1D visible on this image
        label_ind = np.where(int_sum[0] == 255)

        # grabbing raw images
        img_raw = np.load(os.path.join(images_raw, img_name[:-7] + '_000.npy'))

        if len(label_ind[0]) > 0:  # thresholding can leave an image as pure noise

            label_sets = get_consecutive_subsets(label_ind[0].tolist())

            label_lengths = np.array([len(sublist) for sublist in label_sets])
            label_lengths = np.where(label_lengths >= label_width)[0]

            # Saving individual labels and potentially overlapping with real images
            name_stem = re.sub('.nii.gz', '', img_name)
            for i in range(len(label_lengths)):

                # making an empty image and filling it with white on selected pixels
                img_ind = np.zeros(img.shape)
                img_ind[:, label_sets[label_lengths[i]]] = 255

                # creating images with real values in the extracted labels
                empty_img = np.zeros(img.shape)
                empty_img[:, label_sets[label_lengths[i]]] = img_raw[:, label_sets[label_lengths[i]]]

                # saving the image
                img_ind = Image.fromarray(img_ind.astype(np.uint8))
                img_ind.save(os.path.join(label_folder, f'{name_stem}_{str(i).zfill(3)}.png'))

                np.save(os.path.join(overlap_folder, f'{name_stem}_{str(i).zfill(3)}.npy'), empty_img)

    print(f'No. of labels created: {len(os.listdir(overlap_folder))}')


if __name__ == "__main__":
    config = parse_config_arg(description="Extract single-peak labels from Yucca predictions.")
    main(config)
