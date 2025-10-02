import subprocess
import os
import pandas as pd
import numpy as np
from PIL import Image

def get_suspects(suspect_path):
    # keeping only those suspects that have %fill greater than 0.9 and they have names assigned
    all_samples = pd.read_csv(suspect_path, skiprows=4, header = 0)
    suspects = all_samples.iloc[:,[1,2,3,4,6,31]]  #selecting columns with relevant info
    suspects.columns = ["RT", "mz", "name", "adduct", "fill", 'MS2']
    suspects.loc[:, "RT"]=suspects["RT"]*60  #rt is in min
    suspects = suspects[suspects["name"]!="Unknown"]  #choosing only compounds that were identified with names (also avoiding adducts)
    suspects = suspects[suspects["fill"]>=0.9]  # %fill is a parameter from MsDial that tells how well the identification went
    suspects = suspects[~suspects["name"].str.contains("w/o MS2", na=False)].reset_index(drop = True)  # using only those with assigned MS2
    return suspects

def write_sirius_ms_file(output_path, compound_name, parentmass, ion_mode, ms1, ms2):
    with open(output_path, 'w') as f:
        f.write(f">{compound_name}\n")
        f.write(f">parentmass {parentmass:.4f}\n")
        f.write(f">ion {ion_mode}\n")
        f.write(">ms1\n")
        for mz, intensity in ms1:
            f.write(f"{mz:.4f} {intensity:.4f}\n")
        f.write(">ms2\n")
        for mz, intensity in ms2:
            f.write(f"{mz:.4f} {intensity:.4f}\n")


def run_sirius(sirius_path, ms_file_path, output_dir, profile="qtof"):
    cmd = [
        sirius_path,
        "-o", output_dir,
        "-i", ms_file_path,
        "-p", profile,
        "formula",
        "fingerprint"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("SIRIUS output:")
    print(result.stdout)
    print(result.stderr)

    return result.returncode

def get_comp_spectra(msdial_sp, ms2_images_path, sample_ind, int_thres):

    # MsDial spectra
    mz = []
    inten = []

    #for i in range(len(msdial_sp)):
    #for s in msdial_sp[i].split(' '):
    for s in msdial_sp.split(' '):
        #print(float(s.split(':')[1]))
        if float(s.split(':')[1])>=int_thres:
            mz.append(float(s.split(':')[0]))
            inten.append(float(s.split(':')[1]))

    ms2 = pd.DataFrame({'mz': mz, 'int': np.array(inten)/max(inten)*255, 'method': 'MsDial'})

    # U-Net spectra
    images = os.listdir(ms2_images_path)
    ms2_images = [f for f in images if len(f.split('_')) > 1 and f.split('_')[1] == 'MS2']

    mz = []
    inten = []

    for i in range(len(ms2_images)):
        eic = np.array(Image.open(os.path.join(ms2_images_path, ms2_images[i])))[sample_ind]
        img_split = ms2_images[i].split('_')
        mz.append((float(img_split[2]+'.'+img_split[3])+float(img_split[4]+'.'+img_split[5]))/2)
        inten.append(int(np.max(eic[eic>0])))

    ms2 = pd.concat([ms2, pd.DataFrame({'mz': mz, 'int': np.array(inten), 'method': 'Unet'})]).reset_index(drop = True)

    return ms2

def get_ms1(susp_mass, images_path, sample_ind):
    
    images = os.listdir(images_path)
    ms1_images = [f for f in images if (len(f.split('_')) > 1 and f.split('_')[1]=='MS1' and float(f.split('_')[2]+'.'+f.split('_')[3])<=susp_mass and float(f.split('_')[4]+'.'+f.split('_')[5])>=susp_mass)][0]

    eic = np.array(Image.open(os.path.join(images_path, ms1_images)))[sample_ind]

    return [[float(np.mean([float(ms1_images.split('_')[2]+'.'+ms1_images.split('_')[3]), float(ms1_images.split('_')[4]+'.'+ms1_images.split('_')[5])])), float(max(eic))]]




# === Example usage ===

# Getting suspects

suspect_path = "/home/frz603/DSP/matrix_files/DK_2021_Secondary metabolites_POS.csv"
suspects = get_suspects(suspect_path)

# Example data
susp_name = "Benzimidazole"
susp_mass = suspects.loc[suspects['name'] == susp_name, 'mz'].values[0]

# Getting images in the cluster
cluster_path = f'/home/frz603/DSP/matrix_images_all/clustering_pareto/clusters/{susp_name}'

# I will put info to Sirius about mass from the mass slice
tol = 0.1
mz_range = [24.8,1205.5]  #setting a fixed interval for mass to be get same mass slices across files
mass_bins = np.arange(mz_range[0], mz_range[1], tol) 

# Grabbing MS2 spectra
ms2_all = get_comp_spectra(suspects.loc[suspects['name'] == susp_name, 'MS2'].values[0], cluster_path, 15, 2) 
ms2_all.sort_values(by="mz", inplace=True)

# Input to Sirius
compound_name = susp_name
parent_mass = np.mean([mass_bins[np.min(np.where(mass_bins>=susp_mass))-1], mass_bins[np.min(np.where(mass_bins>=susp_mass))]])
ion_mode = suspects.loc[suspects['name'] == susp_name, 'adduct'].values[0]
ms1 = get_ms1(susp_mass, cluster_path, 15)
ms2 = ms2_all[ms2_all['method'] == 'Unet'][['mz', 'int']].values.tolist()

# Paths for Sirius
sirius_executable = "/home/frz603/sirius/sirius/bin/sirius"  
output_directory = "/home/frz603/DSP/matrix_images_all/sirius"

# Make sure output directory exists
os.makedirs(output_directory, exist_ok=True)
ms_file = os.path.join(output_directory, f'{susp_name}.ms')

# Write the input file
write_sirius_ms_file(ms_file, compound_name, parent_mass, ion_mode, ms1, ms2)

# Run SIRIUS
exit_code = run_sirius(sirius_executable, ms_file, output_directory)

if exit_code == 0:
    print("SIRIUS analysis complete.")
else:
    print("SIRIUS analysis failed.")