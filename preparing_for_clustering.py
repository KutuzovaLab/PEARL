import re
import os
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib


# fuctions

def get_consecutive_subsets(lst):
        lst = sorted(lst)
        subsets = []
        current = [lst[0]]

        for i in range(1, len(lst)):
            if lst[i] == lst[i - 1] + 1:
                current.append(lst[i])
            else:
                if len(current) > 1:
                    subsets.append(current)
                current = [lst[i]]

        # Add the last group if it's consecutive
        if len(current) > 1:
            subsets.append(current)

        return subsets

# getting labels from yucca predictions
#folder = '/home/frz603/DSP/yucca_runs/test_01/results/Task010_matrixPareto/Task003_simOver/UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best'
#folder = "/home/frz603/DSP/yucca_runs/test_01/results/Task012_smallMatrix/Task011_simBig/UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best"
#folder = "/home/frz603/DSP/yucca_runs/test_01/results/Task016_rawStand/Task015_simFinal/UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best"
folder = '/home/frz603/DSP/yucca_runs/test_01/results/Task020_standardsnpy/Task019_simFinal/UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best'
images = sorted([f for f in os.listdir(folder) if f.endswith(".nii.gz")])

# setting parameters for extraction
sample_thres = 3
label_width = 6

# making new folders for output
#images_raw = os.listdir('/home/frz603/DSP/yucca_runs/test_01/raw_data/Task010_matrixPareto/imagesTs')
images_raw = "/home/frz603/DSP/yucca_runs/test_01/raw_data/Task020_standardsnpy/imagesTs"

clustering_folder = f'/home/frz603/DSP/matrix_standards/raw_images/clustering/labels_20/sn{sample_thres}_w{label_width}'
os.makedirs(clustering_folder, exist_ok=True)

label_folder = os.path.join(clustering_folder, 'single_labels')
os.makedirs(label_folder, exist_ok=True)

overlap_folder = os.path.join(clustering_folder, 'single_labels_overlap')
os.makedirs(overlap_folder, exist_ok=True)

os.makedirs(os.path.join(clustering_folder, 'yucca_labels'), exist_ok=True)
# going through all images and pulling single peak labels out (only if the label is present in more than sample_thres)

for k in range(len(images)):

    #if (k/len(images)*100)%10==0: print(f'Percentage of images done: {k/len(images)*100}')
    
    #susp_mz = 287.05
    #img_name = [f for f in images if float(f.split('_')[2]+'.'+f.split('_')[3])<=susp_mz and float(f.split('_')[4]+'.'+re.sub('.nii.gz', '',f).split('_')[5])>=susp_mz][0]
    
    img_name = images[k]
    print(img_name)
    #img = np.array(Image.open(os.path.join(folder, img_name)))
    img = nib.load(os.path.join(folder, img_name))
    img = img.get_fdata() 

    # getting labels out just to inspect - not necessary to do for all images
    label = np.zeros(img.shape)
    label[img==1]=255
    label[img==2]=200
    label = Image.fromarray(label.astype(np.uint8), mode = 'L')
    #label = Image.fromarray(np.where(img == 1, 255, 0).astype(np.uint8), mode = 'L')
    label.save(os.path.join(clustering_folder,'yucca_labels', re.sub('nii.gz', 'png', img_name)))

    # taking sums on columns and keeping pixels that are greater than sample_thres
    int_sum = img.sum(axis = 0)
    int_sum = np.where(int_sum>=sample_thres, 255, 0).astype(np.uint8).reshape(1, -1)

    # selected labels in 1D visible on this image
    #collapsed_img = Image.fromarray(int_sum, mode = 'L')
    #collapsed_img.save(os.path.join(clustering_folder,re.sub('.png', '',img_name))+'_collapsed.png')

    # getting single labels out by looking at the indices of labels and extracting subsets of consecutive sequences
    label_ind = np.where(int_sum[0] == 255)

    # grabbing raw images
    img_raw = np.load(os.path.join(images_raw, img_name[:-7]+'_000.npy'))

    if len(label_ind[0])>0:  # it can happen because of tresholding that everything on the image is just noise (not present in enough samples)

        label_sets = get_consecutive_subsets(label_ind[0].tolist())

        label_lengths = np.array([len(sublist) for sublist in label_sets])
        label_lengths = np.where(label_lengths>=label_width)[0]
        #print(len(label_lengths))

        # Saving individual labels and potentially overlapping with real images

        for i in range(len(label_lengths)):
            
            #print(len(label_sets[i]))
            # making an ampty image and filling it with white on pixels selected for that label
            img_ind = np.zeros(img.shape)
            img_ind[:,label_sets[label_lengths[i]]] = 255

            # creating images with real values in the extracted labels
            empty_img = np.zeros(img.shape)
            empty_img[:,label_sets[label_lengths[i]]] = img_raw[:,label_sets[label_lengths[i]]]

            # saving the image
            img_ind = Image.fromarray(img_ind.astype(np.uint8))
            img_ind.save(os.path.join(label_folder, f'{re.sub('.nii.gz', '', img_name)}_{str(i).zfill(3)}.png'))

            #empty_img = Image.fromarray(empty_img.astype(np.uint8))
            np.save(os.path.join(overlap_folder, f'{re.sub('.nii.gz', '', img_name)}_{str(i).zfill(3)}.npy'), empty_img)
            

print(f'No. of labels created: {len(os.listdir(overlap_folder))}')