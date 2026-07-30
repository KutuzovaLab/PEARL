import pyopenms as oms
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import warnings
from scipy.interpolate import CubicSpline
import time
import gc


def interpolating_rt(tic_sample, length_spline):
    
    missing_indices = np.where(tic_sample['native_id'] =='function=3')[0]         

    for k in range(len(missing_indices)):
        
        ind_select = np.append(np.arange(max(missing_indices[k]-length_spline,0), missing_indices[k]), np.arange(missing_indices[k]+1, min(missing_indices[k]+length_spline+1, len(tic_sample))))
        
        tic_sel = tic_sample.iloc[ind_select]
        spline = CubicSpline(tic_sel['rt'], tic_sel['int_sum'])
        tic_sample.loc[missing_indices[k],'int_sum'] = max(float(spline(tic_sample.iloc[int(missing_indices[k])]['rt'])),0)

    return tic_sample

def making_rt_bins(ms, rt_bins):

    # cutting rt into bins
    ms = ms.sort_values(by = 'rt').reset_index(drop = True)
    bin_intervals = pd.IntervalIndex.from_breaks(rt_bins, closed='left')
    ms['bins'] = pd.cut(ms['rt'], bins = bin_intervals)

    # saving into csv file correspoding to that slice
    #check = ms.groupby('bins', observed = True)['int_sum'].max().reindex(bin_intervals, fill_value=0)
    int_bins = ms.groupby('bins', observed = True)['int_sum'].max().reindex(bin_intervals, fill_value=0).to_numpy()
    eic_ms = pd.DataFrame([int_bins])

    return eic_ms

def selecting_files(mzml_path, suspect_path):

    all_samples = pd.read_excel(suspect_path, skiprows=4, header = 0)
    #all_samples = pd.read_excel(suspect_path, skiprows=4, header = 0)

    #keeping only selected samples (the ones that i have .mzml files of)
    files = sorted([f for f in os.listdir(mzml_path) if f.endswith(".mzML")])

    #making sure the files are in order as they were injected in the instrument because I want the correct RT shift
    updated_file_list = sorted([re.sub(r'_(8[0-9]|9[0-9])\.mzML$', r'_0\1.mzML', filename) if re.search(r'_(8[0-9]|9[0-9])\.mzML$', filename) else filename for filename in files])
    updated_file_list = [re.sub(".mzML", "", item) for item in updated_file_list]
    # only for standards!!!
    #updated_file_list = [re.sub(r'_(0*)(\d+)$', r'_\2', f) if not re.search(r'_100$', f) else f for f in updated_file_list]

    #selecting the sample concentrations of samples selected in .mzml form
    sel_samples = np.append(np.repeat(False, 32),all_samples.columns[32:].str.split("_").str[:3].str.join("_").isin(updated_file_list))
    samples = all_samples.iloc[:,sel_samples]
    samples.columns

    #passing only the files that I have in the excel table
    final_files = []
    for filename in updated_file_list:
        if filename in samples.columns.str.split("_").str[:3].str.join("_"):
            final_files.append(filename)
    print(f"Final number of included samples: {len(final_files)}", flush=True)

    return final_files

def read_mzml_files(path, files, rt_bin_size, rt_range, mz_range, mass_tol):    

    # setting parameters
    rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size) 
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)  # creating all possible mass slices
    #mass_bins =  [mass_bins[[690, 1230]]][0]

    # this is just checking for a selected mass slice if it looks good
    #mass_bins = [mass_bins[np.arange(600,602)]][0]

    # saving "images" into this
    results_ms1 = np.zeros((len(mass_bins), len(files), len(rt_bins)-1), dtype=np.float32)
    results_ms2 = np.zeros((len(mass_bins), len(files), len(rt_bins)-1), dtype=np.float32)

    # looping through all files -> all mass slices -> all spectra
    for file_ind in range(len(files)):

        # creating experiment to load mzml files on
        exp = oms.MSExperiment()

        # checking time required to process each file
        start = time.time()

        #fetching raw data
        ind_file = os.path.join(path,files[file_ind])+".mzML"
        oms.MzMLFile().load(ind_file, exp)  #opening files
        df = exp.get_df(long=False) 
        df = df[['RT', 'mzarray', 'intarray']].copy()

        #getting spectrum id
        native_ids = []
        for spectrum in exp:
            native_ids.extend([spectrum.getNativeID().split(' ', 1)[0]])
        df['native_id'] = native_ids
        
        # selecting only those spectra that are in the selected RT range (in order to avoid spills in the end and beginning)
        df = df[(df['RT']>=rt_range[0]) & (df['RT']<=rt_range[1])].reset_index(drop = True)
        
        # now i'm looping through all the slices and collecting (rt, mz, int) points that belong to that slice
        for j, mb in enumerate(mass_bins):

            #print(mb)
            #mb = mass_bins[0]            
            mass_slice = [float(mb), float(mb+mass_tol)]
            print(f'Selected slice: {mass_slice}', flush = True)

            #saving all points into empty array
            mass_slice_spectra = []
            
            # looping through all spectra
            for i in range(len(df)):
                
                # getting rt + mz array + int array
                mz_value = df.iloc[i]['mzarray']  # Get the mzarray (single float)
                int_value = df.iloc[i]['intarray']  # Get the intarray (single float)

                # taking only those values in the mass slice
                ind_sel = np.where((mz_value >= mass_slice[0]) & (mz_value <= mass_slice[1]))[0]
                mz_sel = mz_value[ind_sel]
                int_sel = int_value[ind_sel]

                # saving spectra
                if len(mz_sel)==0:
                    mass_slice_spectra.append({'mz': [0], 'rt': df.loc[i,'RT'], 'int': [0], 'native_id': df.loc[i,'native_id']})
                else:
                    mass_slice_spectra.append({'mz': mz_sel, 'rt': df.loc[i,'RT'], 'int': int_sel, 'native_id': df.loc[i,'native_id']})
            
            # getting into format that i can work with
            mass_slice_spectra = pd.DataFrame(mass_slice_spectra)
            mass_slice_spectra['int_sum']=mass_slice_spectra['int'].apply(sum)
            
            # selecting MS1
            ms1 = mass_slice_spectra[mass_slice_spectra['native_id']!='function=2'].copy().reset_index(drop = True)
            #interpolating lock sprays
            ms1 = interpolating_rt(ms1, 5)
            # cutting rt into bins of size rt_bin_size
            int_final = making_rt_bins(ms1, rt_bins)
            # saving csv
            results_ms1[j, file_ind, :]=int_final
            
            # selecting ms2 now
            ms2 = mass_slice_spectra[mass_slice_spectra['native_id']!='function=1'].copy().reset_index(drop = True)
            #interpolating lock sprays
            ms2 = interpolating_rt(ms2, 5)
            # cutting rt into bins of size rt_bin_size
            int_final = making_rt_bins(ms2, rt_bins)
            # saving into csv
            results_ms2[j, file_ind, :]=int_final

            del mass_slice_spectra

        end = time.time()     
            
        print(f"file: {files[file_ind]} - {file_ind+1}/{len(files)} - time elapsed: {(end-start)/60:.2f} min", flush=True)

        del df
        del exp
        gc.collect()

    return results_ms1, results_ms2


def saving_results(ms1, ms2, output_folder, mz_range, mass_tol):

    # making sure paramters are consistent eveywhere
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)  # creating all possible mass slices
    int_thres = 1000

    for i, mb in enumerate(mass_bins):
        mb1 = str(round(mb, 1)).replace('.', '_')
        mb2 = str(round(mb+mass_tol, 1)).replace('.', '_')
        filename_ms1 = f"MS1_{mb1}_{mb2}.npy"
        #print(filename_ms1)
        np.save(os.path.join(output_folder, 'images', filename_ms1), ms1[i])
        label = (ms1[i] > int_thres).astype(np.uint8)
        np.save(os.path.join(output_folder, 'labels', filename_ms1), label)

        filename_ms2 = f"MS2_{mb1}_{mb2}.npy"
        #print(filename_ms2)
        np.save(os.path.join(output_folder, 'images', filename_ms2), ms2[i])
        label = (ms2[i] > int_thres).astype(np.uint8)
        np.save(os.path.join(output_folder, 'labels', filename_ms2), label)



#Paths to the table from MsDial with .mzml files that were used and suspects found + .mzml files

suspect_path = "new_stuff/Standards_NEG.xlsx"
mzml_path = "new_stuff/mzml_files"
output_folder = 'new_stuff/for_yucca/test_dir'
os.makedirs(output_folder, exist_ok = True)


# saving images into this folder
#images_folder = "/home/frz603/DSP/matrix_images_all/images_073"
#os.makedirs(images_folder, exist_ok = True)

# getting interesting rt ranges and mass slices
rt_range = [100,1100]
rt_bin_size = 0.73
mz_range = [50,1000]  #setting a fixed interval for mass to be get same mass slices across files
mass_tol = 0.1

# first, i'm opening mzml files and slicing, interpolating and saving EICs into .csv for each slice and all mzmls
final_files = selecting_files(mzml_path, suspect_path)

# reading spectra for all slices
print('Reading raw files...', flush = True)
ms1, ms2 = read_mzml_files(mzml_path, final_files, rt_bin_size, rt_range, mz_range, mass_tol)  

# saving individual numpys to use for yucca
saving_results(ms1, ms2, output_folder, mz_range, mass_tol)