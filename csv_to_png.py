import numpy as np
import pandas as pd
import os
from PIL import Image
import re

# useful paths

csv_path = '/home/frz603/DSP/matrix_images/csv/MS2'
images_path = '/home/frz603/DSP/matrix_images/MS2'
os.makedirs(images_path, exist_ok=True)


# looping through all csvs and converting them to pngs

for i in range(len(os.listdir(csv_path))):

    if i%200: print(f'Image #{i}', flush=True)

    #reading csv
    ms = pd.read_csv(os.path.join(csv_path, os.listdir(csv_path)[i]), header = None).values
    
    # scaling between 0 and 255
    ms_scaled = ms / np.max(ms)*255
    ms_scaled = ms_scaled.astype(np.uint8)

    # creating image and saving
    img = Image.fromarray(ms_scaled)
    img.save(os.path.join(images_path, re.sub('.csv', '.png', re.sub('MS1_', '',os.listdir(csv_path)[i]))))

print("Done!", flush=True)