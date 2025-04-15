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

def read_mzml_files(path, ms_level, mass_slice):
    # lets extract all the raw data points in that slice
    #Loading mzml files
    exp = oms.MSExperiment()

    #empty dataframe to store values int
    df = pd.DataFrame(columns=['rt', 'mz', 'int', 'mslevel', 'file']) # initializing dataframe to hold all the raw data for the file
    
    #looping to the files and collecting (mz, rt, int) for the selected slices
    mzml_files = sorted(os.listdir(path)) #its quite important to sort them to get the right rt shift

    for file_ind in range(len(mzml_files)):

        ind_file = os.path.join(path, mzml_files[file_ind])
        oms.MzMLFile().load(ind_file, exp)  #opening files

        sp = exp.getSpectra() #extracting spectra for each file

        print(f"file: {mzml_files[file_ind]} - # of spectra: {len(sp)}", flush=True)

        for i in range(len(sp)):
        #for i in range(100):            
            
            s = sp[i]
            mslevel = s.getMSLevel() 
            if i%200 == 0: print(f"ind: {i}, mslevel {mslevel}", flush=True)  

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

                    df = pd.concat([df, pd.DataFrame({'rt': rt, 'mz': mz_sel, 'int': int_sel, 'mslevel': mslevel, 'file': mzml_files[file_ind]})])
    
    print(f"# of collected spectra: {len(df)}", flush=True)
    return df


def plot_eic(df, rt_range, mass_slice):

    # summing intensities for the same retention times
    tic = df.groupby(['scan_no', 'rt', 'file'])['int'].sum().reset_index()
    
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

    plt.savefig(f"/home/frz603/DSP/tics_cons_samples/ms_{np.round(mass_slice[0],2)}_{np.round(mass_slice[1],2)}_rt_{rt_range[0]}_{rt_range[1]}.png", bbox_inches='tight')

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

suspect_path = "/home/frz603/DSP/targets/SuspList_PubChem.xlsx"
sample_path = "/home/frz603/DSP/targets/WP2_Phase3_LC(RP)QToF_target_and_suspects_20230209.xlsx"

suspects = pd.read_excel(suspect_path)
samples = pd.read_excel(sample_path)

# keeping only those suspects that I know rt of
suspects = suspects.dropna(subset=['RT'])
suspects["RT"]=suspects["RT"]*60

#keeping only selected samples (the ones that i have .mzml files of)
mzml_path = "/home/frz603/DSP/samples_vandalf_p3"
files = ["P3" + s[:3] for s in os.listdir(mzml_path)]
samples = samples[samples["Short name"].isin(files)]

#subsetting the big table with only suspects with rt included
samples = samples[["Short name"] + suspects["name"].tolist()]

#deciding on mass slicing
tol = 0.05
mz_range = [24.8,1205.5]  #setting a fixed interval for mass to be get same mass slices across files
mass_bins = np.arange(mz_range[0], mz_range[1], tol)  # creating all possible mass slices

#useful paths
output_folder = "/home/frz603/DSP/targets"

# getting interesting rt ranges and mass slices

rt_range = [100,700]

# reading spectra for all suspects

for i in range(len(suspects)):
    #locating the mass slice that belongs to the target

    susp_mh = suspects.iloc[i, suspects.columns.get_loc('MH')]
    mass_slice = np.array([mass_bins[np.min(np.where(mass_bins>=susp_mh))-1], mass_bins[np.min(np.where(mass_bins>=susp_mh))]])
    print(f"Chosen mass slice: {np.round(mass_slice[0],2)}-{round(mass_slice[1],2)}")

    #extracting MS1 for all files in that mass slice
    df = read_mzml_files(mzml_path, 1, mass_slice)
    
    #checking if EIC makes sense (rt range can be selected)
    #plot_eic(df, rt_range, mass_slice)

    #creating an image by summing the intensities for same RT points and stacking eics of different files next to each other
    rt_bin_size = 0.8
    slice_img = convert_mzml_to_images(df, rt_range, rt_bin_size)  # 0.8 refers to the bins in retention time (in this case, the diff between ms1 and ms2 scans was 0.3)
    #slice_img.show()

    print(f"Image created - saving it now into: {output_folder}")
    #saving created images
    #slice_img.save(f"{output_folder}\\{suspects.iloc[i, suspects.columns.get_loc('name')]}_{np.round(mass_slice[0],2)}_{np.round(mass_slice[1],2)}_{rt_bin_size}.png")
    slice_img.save(f"{output_folder}/{suspects.iloc[i, suspects.columns.get_loc('name')]}.png")
    