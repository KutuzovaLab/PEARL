import pyopenms as oms
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#functions

def read_mzml_files(path, files, ms_level, mass_slice):
    # lets extract all the raw data points in that slice
    #Loading mzml files
    exp = oms.MSExperiment()

    #empty dataframe to store values int
    df = pd.DataFrame(columns=['rt', 'mz', 'int', 'mslevel', 'file']) # initializing dataframe to hold all the raw data for the file
    
    #looping to the files and collecting (mz, rt, int) for the selected slices
    files = sorted(files)  #quite important to sort them, so we have a positive rt shift

    for file_ind in range(len(files)):

        ind_file = os.path.join(path,re.sub(r'_(0[0-9]{2})$', lambda m: '_' + m.group(1)[1:], files[file_ind]))+".mzML"
        oms.MzMLFile().load(ind_file, exp)  #opening files

        sp = exp.getSpectra() #extracting spectra for each file

        #print(f"file: {ind_file} - # of spectra: {len(sp)} - {file_ind+1}/{len(files)}", flush=True)

        for i in range(len(sp)):
        #for i in range(100):            
            
            s = sp[i]
            mslevel = s.getMSLevel() 
            #if i%200 == 0: print(f"ind: {i}, mslevel {mslevel}", flush=True)  

            if(mslevel == ms_level):

                #if i%200 == 0: print(f"ind: {i}, mslevel {mslevel}", flush=True)      
                #print(f"ind: {i}, mslevel {mslevel}")

                mz = s.get_peaks()[0]  # getting mz

                mz_sel = mz[(mz>=mass_slice[0]) & (mz<=mass_slice[1])]  #extracting MS1 for all files in that mass slice

                if len(mz_sel)!=0:

                    intensity = s.get_peaks()[1] # getting intensities
                    int_sel= intensity[(mz>=mass_slice[0]) & (mz<=mass_slice[1])]

                    rt = s.getRT()
                    #scan_no = float(re.sub("scan=", "", s.getNativeID())) 

                    df = pd.concat([df, pd.DataFrame({'rt': rt, 'mz': mz_sel, 'int': int_sel, 'mslevel': mslevel, 'file': files[file_ind]})])
    
    print(f"# of collected spectra: {len(df)}", flush=True)
    return df


def plot_eic(df, rt_range, mass_slice):

    # summing intensities for the same retention times
    tic = df.groupby(['rt', 'file'])['int'].sum().reset_index()
    
    # selecting only the desired rt range
    tic = tic[tic["rt"]>=rt_range[0]]
    tic = tic[tic["rt"]<=rt_range[1]]

    # Assuming you have defined `files` already
    files = np.unique(tic["file"])

    # Plotting individual line plots for each file
    for file_ind in range(len(files)):
        sns.lineplot(data=tic[tic["file"] == files[file_ind]], x='rt', y='int', hue='file', palette='viridis', linewidth=2)

    # Creating a FacetGrid for each file
    g = sns.FacetGrid(tic, col="file", col_wrap=2, height=4)
    g.set(yscale="linear")
    g.map(sns.lineplot, "rt", "int")  # You could add hue and palette here as well

    # Adjusting the layout
    plt.subplots_adjust(top=0.9, hspace=0.3)  # Adjust the top and vertical space (hspace)

    # Setting the titles manually
    for i, ax in enumerate(g.axes.flat):
        ax.set_title(f"File {files[i]}", fontsize=10)

    g.add_legend()  # Adding legend
    #plt.show()

    plt.savefig(f"/home/frz603/DSP/matrix_tics/ms_{np.round(mass_slice[0],2)}_{np.round(mass_slice[1],2)}_rt_{rt_range[0]}_{rt_range[1]}.png", bbox_inches='tight')

    return

def convert_mzml_to_images(mass_slice, rt_range, rt_bin_size):

    tic = mass_slice.groupby(['rt', 'file'])['int'].sum().reset_index()  # summing all intensities with same RT in the mass slice

    tic = tic[tic["rt"]>= rt_range[0]]  # subsetting to the selected rt_range
    tic = tic[tic["rt"]<= rt_range[1]]  
    
    rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size)  #binning retention time into bins of equal width rt_bin_size
    tic["bins"] = pd.cut(tic["rt"], bins = rt_bins, right=True)

    #tic_counts = tic.groupby("file")["bins"].nunique().reset_index(name="unique_bins_count")
    #repeating_bins = tic.groupby(["file", "bins"]).size().reset_index(name="count").query("count > 1")
    #tic["scaled_int"] = round(tic["int"] / (tic["int"].max()-tic["int"].min()) * 255) #changing intensity to 0 - 255 range
    #tic["scaled_int"] = round(a["int"].mean() / (tic["int"].max()-tic["int"].min()) * 255) #changing intensity to 0 - 255 range

    unique_bins = tic["bins"].unique()
    unique_samples = np.sort(tic["file"].unique())

    height, width = len(unique_samples), len(unique_bins)  
    empty_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(len(tic)):
        
        #single_bin =  repeating_bins.iloc[0]["bins"]
        #i = np.where(tic["bins"]==single_bin)
        y = np.where(unique_samples == tic.iloc[i]["file"])[0]
        x = np.where(unique_bins == tic.iloc[i]["bins"])[0]
    
        mean_intensity = tic.loc[(tic["file"] == tic.iloc[i]["file"]) & (tic["bins"] == tic.iloc[i]["bins"]), "int"].mean()
        pixel_int = round((mean_intensity / tic["int"].max()) * 255)
        empty_image[y, x] = [pixel_int, pixel_int, pixel_int]  # grey-scale image

    img = Image.fromarray(empty_image) #converting into image

    return img



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


#deciding on mass slicing
tol = 0.05
mz_range = [24.8,1205.5]  #setting a fixed interval for mass to be get same mass slices across files
mass_bins = np.arange(mz_range[0], mz_range[1], tol)  # creating all possible mass slices

#saving images in this folder
output_folder = "/home/frz603/DSP/matrix_compounds"
#os.makedirs(output_folder)

# getting interesting rt ranges and mass slices

rt_range = [100,1200]

# reading spectra for all suspects

#for i in range(len(suspects)):
for i in np.arange(9,30,1):
    #locating the mass slice that belongs to the target
    susp_mh = suspects.iloc[i, suspects.columns.get_loc('mz')]
    print(f"Converting suspect #{i+1}: {suspects.iloc[i, suspects.columns.get_loc('name')]} - mz = {susp_mh}")

    mass_slice = np.array([mass_bins[np.min(np.where(mass_bins>=susp_mh))-1], mass_bins[np.min(np.where(mass_bins>=susp_mh))]])
    print(f"Chosen mass slice: {np.round(mass_slice[0],2)}-{round(mass_slice[1],2)}")

    #extracting MS1 for all files in that mass slice
    df = read_mzml_files(mzml_path, final_files, 1, mass_slice)
    
    #checking if EIC makes sense (rt range can be selected)
    plot_eic(df, rt_range, mass_slice)

    #creating an image by summing the intensities for same RT points and stacking eics of different files next to each other
    print("Creating image...")
    rt_bin_size = 0.8
    slice_img = convert_mzml_to_images(df, rt_range, rt_bin_size)  # 0.8 refers to the bins in retention time (in this case, the diff between ms1 and ms2 scans was 0.3)
    #slice_img.show()

    print(f"Image created - saving it now into: {output_folder}")
    #saving created images
    #slice_img.save(f"{output_folder}\\{suspects.iloc[i, suspects.columns.get_loc('name')]}_{np.round(mass_slice[0],2)}_{np.round(mass_slice[1],2)}_{rt_bin_size}.png")
    slice_img.save(f"{output_folder}/{suspects.iloc[i, suspects.columns.get_loc('name')]}.png")
    