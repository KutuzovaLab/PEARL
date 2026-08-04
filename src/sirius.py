import os
import subprocess
import numpy as np
import pandas as pd
from PIL import Image

def get_suspects(suspect_path):
    all_samples = pd.read_csv(suspect_path, skiprows=4, header=0)
    suspects = all_samples.iloc[:, [1, 2, 3, 4, 6, 31]]  # columns with relevant info
    suspects.columns = ["RT", "mz", "name", "adduct", "fill", 'MS2']
    suspects.loc[:, "RT"] = suspects["RT"] * 60  # rt is in min
    suspects = suspects[suspects["name"] != "Unknown"]  # only named compounds (avoids adducts)
    suspects = suspects[suspects["fill"] >= 0.9]  # %fill from MsDial
    suspects = suspects[~suspects["name"].str.contains("w/o MS2", na=False)].reset_index(drop=True)
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
    mz = []
    inten = []

    for s in msdial_sp.split(' '):
        if float(s.split(':')[1]) >= int_thres:
            mz.append(float(s.split(':')[0]))
            inten.append(float(s.split(':')[1]))

    ms2 = pd.DataFrame({'mz': mz, 'int': np.array(inten) / max(inten) * 255, 'method': 'MsDial'})

    images = os.listdir(ms2_images_path)
    ms2_images = [f for f in images if len(f.split('_')) > 1 and f.split('_')[1] == 'MS2']

    mz = []
    inten = []

    for i in range(len(ms2_images)):
        eic = np.array(Image.open(os.path.join(ms2_images_path, ms2_images[i])))[sample_ind]
        img_split = ms2_images[i].split('_')
        mz.append((float(img_split[2] + '.' + img_split[3]) + float(img_split[4] + '.' + img_split[5])) / 2)
        inten.append(int(np.max(eic[eic > 0])))

    ms2 = pd.concat([ms2, pd.DataFrame({'mz': mz, 'int': np.array(inten), 'method': 'Unet'})]).reset_index(drop=True)

    return ms2

def get_ms1(susp_mass, images_path, sample_ind):
    images = os.listdir(images_path)
    ms1_images = [f for f in images if (len(f.split('_')) > 1 and f.split('_')[1] == 'MS1'
                  and float(f.split('_')[2] + '.' + f.split('_')[3]) <= susp_mass
                  and float(f.split('_')[4] + '.' + f.split('_')[5]) >= susp_mass)][0]

    eic = np.array(Image.open(os.path.join(images_path, ms1_images)))[sample_ind]

    return [[float(np.mean([float(ms1_images.split('_')[2] + '.' + ms1_images.split('_')[3]),
                            float(ms1_images.split('_')[4] + '.' + ms1_images.split('_')[5])])),
             float(max(eic))]]
