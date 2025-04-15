import pyopenms as oms
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
import warnings

#path to the images
image_path = "/home/frz603/DSP/yucca_runs/test_01/raw_data/Task002_suspMatrix/imagesTs"
image_names = {re.sub(r"\.png$", "", img) for img in os.listdir(image_path)}

clipped_path = "/home/frz603/DSP/matrix_peaks/clipped/Task003_simOver"
os.makedirs(clipped_path, exist_ok=True)


task_name = "suspMatrix"  #name of the test task
pred_path = "/home/frz603/DSP/yucca_runs/test_01/results/Task002_suspMatrix/Task003_simOver/UNet__2D/YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best"    #labels from yucca

label_path = "/home/frz603/DSP/matrix_peaks/overlay/Task003_simOver"  #folder to save detected peaks with original intensities
#os.makedirs(clipped_path)

#Importing info on suspects first

suspect_path = "/home/frz603/DSP/matrix_files/DK_2021_Secondary metabolites_POS.csv"
all_samples = pd.read_csv(suspect_path, skiprows=4, header = 0)

# keeping only those suspects that have %fill greater than 0.9 and they have names assigned
suspects = all_samples.iloc[:,[1,2,3,6]]  #selecting columns with relevant info
suspects.columns = ["RT", "mz", "name", "fill"]
suspects.loc[:, "RT"]=suspects["RT"]*60  #rt is in min

suspects = suspects[suspects["name"]!="Unknown"]  #choosing only compounds that were identified with names (also avoiding adducts)
suspects = suspects[suspects["fill"]>=0.9]  # %fill is a parameter from MsDial that tells how well the identification went
suspects = suspects[~suspects["name"].str.contains("w/o MS2", na=False)]  # using only those with assigned MS2


#Getting peak areas for samples that were used to create images
#keeping only selected samples (the ones that i have .mzml files of)
mzml_path = "/home/frz603/DSP/matrix_files"
files = [f for f in os.listdir(mzml_path) if f.endswith(".mzML")]

#matching the formats
updated_file_list = sorted([
    re.sub(r'_(8[0-9]|9[0-9])\.mzML$', r'_0\1.mzML', filename) if re.search(r'_(8[0-9]|9[0-9])\.mzML$', filename) else filename
    for filename in files
])

updated_file_list = [re.sub(".mzML", "", item) for item in updated_file_list]

#getting info on samples
info_samples = pd.read_csv(suspect_path, nrows=5, header = None)
info_samples = info_samples.iloc[:, 31:]
info_samples = info_samples.T
info_samples.columns = info_samples.iloc[0]
info_samples = info_samples[1:].reset_index(drop=True)
info_samples = info_samples.rename(columns={"MS/MS spectrum": "File name"})

info_samples = info_samples[info_samples["File name"].str.split("_").str[:3].str.join("_").isin(updated_file_list)]
info_samples.to_excel("/home/frz603/DSP/matrix_files/selected_samples.xlsx", index=False)


#selecting the sample concentrations of samples selected in .mzml form
sel_samples = np.append(np.repeat(False, 33),all_samples.columns[33:].str.split("_").str[:3].str.join("_").isin(updated_file_list))
sel_samples[all_samples.columns == "Metabolite name"]=True  #also giving the column with compound names, so i can select them
sel_samples[all_samples.columns == "Average Rt(min)"]=True
samples = all_samples.iloc[:,sel_samples]
samples.columns


# rt settings used for image formation
rt_range = [100,1200]
rt_bin_size = 0.8
rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size)

#Going through suspects and matching them with their images
suspects = suspects[(suspects["RT"]>=rt_range[0]) & (suspects["RT"]<=rt_range[1])] #selecting only those that i can actually see on the images
print(f"Selecting {len(suspects)} compounds identified by MsDial")

#preparing dataframe to store concentrations
peak_areas = pd.DataFrame(columns=['name', 'rt', 'mz', "msdial_area", "pred_area", "file"], dtype = 'object')


for i in range(len(suspects)):
#for i in np.arange(10,20):

    susp_mz = suspects.iloc[i]["mz"].item() 
    susp_rt = suspects.iloc[i]["RT"].item() 

    rt_sd = 20  #used for cropping in the smaller area around the peak (rt)

    # Open the image
    if task_name+"_"+suspects.iloc[i]["name"]+"_"+"000" in image_names: 

        print(f"Converting suspect #{i+1}: {suspects.iloc[i, suspects.columns.get_loc('name')]} - mz = {susp_mz}, rt = {susp_rt}")
        
        #First I'm getting a raw image and the label and I'm overlapping them to get the pixel labeled as "pakes" and their intensities to be able to sum them

        raw_image = os.path.join(image_path, task_name) + "_" + suspects.iloc[i]["name"]+"_000"+".png"
        label_image = os.path.join(pred_path, task_name + "_" + suspects.iloc[i]["name"])+".png"
        #overlaying raw image on the label

        # Open images
        greyscale_img = Image.open(raw_image).convert("L")  # Ensure grayscale
        #label_img = Image.open(label_image).convert("1")  # Convert to binary (0-1)
        label_img = Image.open(label_image)  # already in binary
        # Convert images to NumPy arrays
        greyscale_array = np.array(greyscale_img)  # Grayscale intensities
        label_array = np.array(label_img)  # Binary mask (0-1)

        # Select pixels where label == 1
        selected_pixels = greyscale_array[label_array == 1]  # Extract intensity values

        # Print or process selected intensities
        #print(selected_pixels)  # This will be a 1D array of intensities

        # Optional: Convert back to Pillow Image (if needed)
        selected_img = Image.fromarray(np.where(label_array == 1, greyscale_array, 0))
        selected_img.save(os.path.join(label_path, suspects.iloc[i]["name"])+".png")  # Save if needed
        
        #just preparing labels into visual image
        label_img= Image.fromarray(np.where(label_array == 1, 255, 0).astype(np.uint8))

        # Now I'm gonna crop the images to select the expected RT range for a suspect (for non-target this has to be modified)

        # cropping the image to smaller rt area and placing the red line on it where MsDial predicts center of the peak
        
        susp_rt_bin = np.digitize(susp_rt, rt_bins).item() - 1
        bin_index_low = np.digitize(susp_rt-rt_sd, rt_bins).item() - 1
        bin_index_high = np.digitize(susp_rt+rt_sd, rt_bins).item() + 1

        print(f"Selected rt range for cropping: [{rt_bins[bin_index_low]}, {rt_bins[bin_index_high]}]")

        # drawing the red line first

        # Adding the red line on the pixels that are recognized as the highest peak point in MsDial

        marked_img = selected_img.convert("RGB")  #drawing on the cropped image of overlap
        draw = ImageDraw.Draw(marked_img)
        line_color = (255, 0, 0)  # Red color in RGB
        draw.line([(susp_rt_bin, 0), (susp_rt_bin, marked_img.height)], fill=line_color, width=1)
        
        #Now I'm cropping

        # Define the clipping width range (40 to 90)
        left = bin_index_low  # Start width
        right = bin_index_high  # End width
        upper = 0  # Keep full height
        lower = selected_img.height  # Keep full height

        # Crop the image
        raw_crop = greyscale_img.crop((left, upper, right, lower))
        label_crop = label_img.crop((left, upper, right, lower))
        over_crop = selected_img.crop((left, upper, right, lower))
        marked_crop = marked_img.crop((left, upper, right, lower))

        
        # Saving all 3 images in the clipped_path by the name of the suspect
        
        raw_crop.save(os.path.join(clipped_path, suspects.iloc[i]["name"])+"_raw"+".png")  # Save if needed
        label_crop.save(os.path.join(clipped_path, suspects.iloc[i]["name"])+"_label"+".png")  # Save if needed
        over_crop.save(os.path.join(clipped_path, suspects.iloc[i]["name"])+"_overlay"+".png")  # Save if needed
        marked_crop.save(os.path.join(clipped_path, suspects.iloc[i]["name"])+"_redline"+".png")  # Save if needed

        #Now I'm taking each row of the identified "streak" and summing the intensities on the non-zero pixels to get the idea of the area (it's besically integral since the division over the rt bins is equidistant)
        
        int_pixels = np.array(over_crop) # Take only one channel
        sum_int = int_pixels.sum(axis = 1)  #taking sum across rows (each row is one sample)

        #len(sum_int)

        msdial_area = samples[(samples["Metabolite name"] == suspects.iloc[i]["name"]) & (samples["Average Rt(min)"] == susp_rt/60)] #they are multiple fragments in the table that have different rt, so i'm choosing the one with the correct rt
        msdial_area = msdial_area.iloc[:, 2:].sort_index(axis=1).values.flatten()
        sample_names = sorted(samples.columns[2:].values)
        
        peak_areas = pd.concat([peak_areas, pd.DataFrame({'name': suspects.iloc[i, suspects.columns.get_loc('name')], 'rt': susp_rt, 'mz': susp_mz, 'msdial_area': msdial_area/max(msdial_area), 'pred_area': sum_int/max(sum_int), 'file': sample_names})])
        
        peak_areas.plot(x='msdial_area', y='pred_area', kind='scatter', marker='o', color='b')
        plt.xlabel("MsDial peak area")
        plt.ylabel("U-net predicted peak area")
        plt.title("Calibration curve")
        plt.grid(True)  # Optional: Add grid
        #plt.show()
        plt.savefig(f'/home/frz603/DSP/matrix_peaks/{suspects.iloc[i, suspects.columns.get_loc('name')]}_curve.png')

peak_areas.to_excel(f'/home/frz603/DSP/matrix_peaks/calibration_curve.xlsx', index=False)




       
           


