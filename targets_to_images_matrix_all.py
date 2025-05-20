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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#functions

def interpolating_rt(tic_sample, length_spline):
    
    missing_indices = np.where(tic_sample['native_id'] =='function=3')[0]         

    for k in range(len(missing_indices)):
        
        ind_select = np.append(np.arange(max(missing_indices[k]-length_spline,0), missing_indices[k]), np.arange(missing_indices[k]+1, min(missing_indices[k]+length_spline+1, len(tic_sample))))
        
        tic_sel = tic_sample.iloc[ind_select]
        spline = CubicSpline(tic_sel['rt'], tic_sel['int_sum'])
        tic_sample.loc[missing_indices[k],'int_sum'] = max(spline(tic_sample.iloc[int(missing_indices[k])]['rt']),0)

    return tic_sample


def read_mzml_files(path, files, rt_bin_size, rt_range, mass_tol, output_folder):    

    # creating experiment to load mzml files on
    exp = oms.MSExperiment()

    # setting parameters
    rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size) 
    mz_range = [50,1000]  #setting a fixed interval for mass to be get same mass slices across files
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)  # creating all possible mass slices

    # this is just checking for a selected mass slice if it looks good
    #mass_bins = [mass_bins[1052]]

    # looping through all files -> all mass slices -> all spectra
    for file_ind in range(len(files)):

        #file_ind = 17
        # checking time required to process each file
        start = time.time()

        #fetching raw data
        ind_file = os.path.join(path,re.sub(r'_(0[0-9]{2})$', lambda m: '_' + m.group(1)[1:], files[file_ind]))+".mzML"
        oms.MzMLFile().load(ind_file, exp)  #opening files
        df = exp.get_df(long=False) 

        #getting spectrum id
        native_ids = []
        for spectrum in exp:
            native_ids.extend([spectrum.getNativeID().split(' ', 1)[0]])
        df['native_id'] = native_ids
        
        # selecting only those spectra that are in the selected RT range (in order to avoid spills in the end and beginning)
        df = df[(df['RT']>=rt_range[0]) & (df['RT']<=rt_range[1])].reset_index()
        
        # now i'm looping through all the slices and collecting (rt, mz, int) points that belong to that slice
        for mb in mass_bins:

            #mb = mass_bins[0]            
            mass_slice = [float(mb), float(mb+mass_tol)]
            #print(f'Selected slice: {mass_slice}', flush = True)

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
            #ms1 = mass_slice_spectra[mass_slice_spectra['native_id']!='function=2'].copy().reset_index(drop = True)

            #interpolating lock sprays
            #ms1 = interpolating_rt(ms1, 5)

            # cutting rt into bins of size rt_bin_size
            #ms1 = ms1.sort_values(by = 'rt').reset_index(drop = True)
            #bin_intervals = pd.IntervalIndex.from_breaks(rt_bins, closed='left')
            #ms1['bins'] = pd.cut(ms1['rt'], bins = bin_intervals)
            #ms1['bins'].value_counts()

            # saving into csv file correspoding to that slice
            #int_bins = ms1.groupby('bins')['int_sum'].sum().reindex(bin_intervals, fill_value=0).to_numpy()
            #eic_m1 = pd.DataFrame([int_bins])
            #eic_m1.to_csv(os.path.join(output_folder, 'MS1', 'MS1_' + str(round(mb,1)) + '_'+ str(round(mb+mass_tol,1)) + '.csv'), mode = 'a', index = False, header = False)

            # selecting ms2 now
            ms2 = mass_slice_spectra[mass_slice_spectra['native_id']=='function=2'].copy().reset_index(drop = True)

            # cutting rt into bins of size rt_bin_size
            ms2 = ms2.sort_values(by = 'rt').reset_index(drop = True)
            bin_intervals = pd.IntervalIndex.from_breaks(rt_bins, closed='left')
            ms2['bins'] = pd.cut(ms2['rt'], bins = bin_intervals)
            #ms1['bins'].value_counts()

            # saving into csv file correspoding to that slice
            int_bins = ms2.groupby('bins')['int_sum'].sum().reindex(bin_intervals, fill_value=0).to_numpy()
            eic_m2 = pd.DataFrame([int_bins])
            eic_m2.to_csv(os.path.join(output_folder, 'MS2', str(round(mb,1)) + '_'+ str(round(mb+mass_tol,1)) + '.csv'), mode = 'a', index = False, header = False)


        #plt.plot(np.arange(len(int_sums)), int_sums)
        #plt.plot(np.arange(len(ms1)), ms1['int_sum'])
        #plt.savefig("/home/frz603/DSP/matrix_images/check/"+str(mass_slice)+'.png')   

        end = time.time()     
    
        print(f"file: {files[file_ind]} - {file_ind+1}/{len(files)} - time elapsed: {end-start}", flush=True)


def making_images(mass_tol, output_folder):

    mz_range = [50,1000]  #setting a fixed interval for mass to be get same mass slices across files
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)  # creating all possible mass slices
    
    i = 1052
    eic = pd.read_csv(os.path.join(output_folder, 'MS1_'+ str(round(mass_bins[i],1)) + '_'+ str(round(mass_bins[i]+mass_tol,1)) + '.csv'), header = None)
    
    eic_values = eic.to_numpy()
    max_int = eic_values.max()

    eic_scaled = np.round(eic_values / max_int * 255).astype(np.uint8)

    img = Image.fromarray(eic_scaled, mode='L')
    img.save('/home/frz603/DSP/matrix_images/MS1_'+ str(round(mass_bins[i],1)) + '_'+ str(round(mass_bins[i]+mass_tol,1)) + '.png')
    


#Paths to two different files, one with samples and concentrations and the other with chemical info on targets

suspect_path = "/home/frz603/DSP/matrix_files/DK_2021_Secondary metabolites_POS.csv"
all_samples = pd.read_csv(suspect_path, skiprows=4, header = 0)

# keeping only those suspects that have %fill greater than 0.9 and they have names assigned
suspects = all_samples.iloc[:,[1,2,3,6]]  #selecting columns with relevant info
suspects.columns = ["RT", "mz", "name", "fill"]
suspects.loc[:, "RT"]=suspects["RT"]*60  #rt is in min

suspects = suspects[suspects["name"]!="Unknown"]  #choosing only compounds that were identified with names (also avoiding adducts)
suspects = suspects[suspects["fill"]>=0.9]  # %fill is a parameter from MsDial that tells how well the identification went
suspects = suspects[~suspects["name"].str.contains("w/o MS2", na=False)]  # using only those with assigned MS2
suspects = suspects[suspects["RT"]>=100]   #to avoid big peaks in the beggining
print(f"Converting {len(suspects)} compounds identified by MsDial")

#keeping only selected samples (the ones that i have .mzml files of)
mzml_path = "/home/frz603/DSP/matrix_files"
files = [f for f in os.listdir(mzml_path) if f.endswith(".mzML")]

#matching the formats
updated_file_list = sorted([
    re.sub(r'_(8[0-9]|9[0-9])\.mzML$', r'_0\1.mzML', filename) if re.search(r'_(8[0-9]|9[0-9])\.mzML$', filename) else filename
    for filename in files
])

updated_file_list = [re.sub(".mzML", "", item) for item in updated_file_list]

#selecting the sample concentrations of samples selected in .mzml form
sel_samples = np.append(np.repeat(False, 33),all_samples.columns[33:].str.split("_").str[:3].str.join("_").isin(updated_file_list))
samples = all_samples.iloc[:,sel_samples]
samples.columns

#passing only the files that I have in the excel table
final_files = []
for filename in updated_file_list:
    if filename in samples.columns.str.split("_").str[:3].str.join("_"):
        final_files.append(filename)
print(f"Final number of included samples: {len(final_files)}")

#saving images in this folder
output_folder = "/home/frz603/DSP/matrix_images/csv"
#os.makedirs(output_folder)

# getting interesting rt ranges and mass slices
rt_range = [100,1100]

# reading spectra for all slices
read_mzml_files(mzml_path, final_files, 0.8, rt_range, 0.1, output_folder)  

# making images of all slices
#making_images(0.1, output_folder)
