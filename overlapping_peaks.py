import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image
import warnings
import os

#supressing  warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



# functions

def make_gaussian(rt_range_shifted, rt_bin_size, max_int):

    uni_noise = 0.05*max_int # uniform noise added on the Gaussian peak is set to be 20% of the intensity

    x = np.arange(rt_bins[rt_range_shifted[0]]+rt_bin_size/2, rt_bins[rt_range_shifted[1]]+rt_bin_size/2, rt_bin_size)

    mu = np.mean(x)      # Mean (peak location)
    sigma = 2   # Standard deviation (width of the peak)

    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)  # generating intensities of the peak (gaussian peak)
    y_normalized = y / np.max(y) * np.random.uniform(max_int-1000, max_int+1000)  # setting it to the random intensity (selected max +/- 1000)

    uniform_noise = np.random.uniform(low=0, high=uni_noise, size=len(x))  # adding uniform noise
    y_final = y_normalized + uniform_noise

    df = pd.DataFrame({'x': x, 'y': y_final})

    return df

# creating single gaussians and saving them together


def make_overlapping_peak(rt_bin_size, rt_start, peak_length, rt_overlap):

    rt_start_2 = rt_start + rt_overlap  # here the idea is to change 15 to different levels of overlap

    print(f"Start no. 1: {rt_start}, start no. 2: {rt_start_2}")

    df1 = make_gaussian([rt_start, rt_start+peak_length], rt_bin_size, np.random.uniform(7000,8000))
    df2 = make_gaussian([rt_start_2, rt_start_2+peak_length], rt_bin_size, np.random.uniform(4000,5000))

    df1["peak"]=1
    df2["peak"]=2

    df = pd.concat([df1, df2], ignore_index=True)

    return df


# Plotting

def plotting_two_gaussians(df):
    colors = {1: 'blue', 2: 'orange'}
    fig, ax = plt.subplots()

    for peak_value, group in df.groupby('peak'):
        ax.scatter(group['x'], group['y'], label=f'Peak {peak_value}', color=colors[peak_value])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    plt.savefig("/home/frz603/DSP/simulated_data_02/gaussians.png")



def plot_overlap(dp, lmt):
    
    dp['rt_mid'] = dp['rt_bins'].apply(lambda x: x.mid)

    plt.figure(figsize=(12, 6))
    plt.plot(dp['rt_mid'], dp['int'], marker='o')
    plt.xlabel('RT (midpoint)')
    plt.ylabel('Intensity')
    plt.title('Intensity vs Retention Time Bins')
    plt.axvline(x=lmt, color='gray', linestyle='--', label=f'x = {lmt:.2f}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/frz603/DSP/simulated_data_02/gaussians_final.png")

    return

# Assume df contains the full dataset with 'x', 'y', and 'peak' columns
# Separate the peaks

def find_boundary(df):
    df1 = df[df['peak'] == 1].sort_values('x')
    df2 = df[df['peak'] == 2].sort_values('x')

    # Create interpolation functions
    f1 = interp1d(df1['x'], df1['y'], kind='cubic', fill_value="extrapolate")
    f2 = interp1d(df2['x'], df2['y'], kind='cubic', fill_value="extrapolate")

    # Define difference function
    def diff(x):
        return f1(x) - f2(x)

    # Generate x range where both curves are defined
    x_min = max(df1['x'].min(), df2['x'].min())
    x_max = min(df1['x'].max(), df2['x'].max())
    x_vals = np.mean([x_min, x_max])

    # Find approximate roots
    initial_guesses = x_vals

    # Use fsolve to refine
    intersections = fsolve(diff, initial_guesses)[0]
    #intersections = sorted(set(np.round(intersections, 6)))

    print(f"Intersection: {intersections:.2f}, rt range: {x_min:.2f} - {x_max:.2f}")

    # Optional: plot
    #x_plot = np.linspace(x_min, x_max, 500)
    #plt.plot(x_plot, f1(x_plot), label='Peak 1')
    ##plt.plot(x_plot, f2(x_plot), label='Peak 2')
    #plt.axvline(x=intersections, color='gray', linestyle='--', label=f'x = {intersections:.2f}')
    #plt.legend()
    #plt.title('Intersection of Peak 1 and Peak 2')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.grid(True)
    #plt.savefig("/home/frz603/DSP/simulated_data_02/gaussians_intersection.png")

    return(intersections)

def make_image(all_peaks, n_samples):
        
        bin_labels = all_peaks["rt_bins"].cat.categories  # getting bin labels out

        width, height = len(bin_labels), n_samples  # creating an empty image with size no. of samples x no. of rt bins
        empty_image = np.zeros((height, width, 3), dtype=np.uint8)

        #preparing parameters
        unique_samples = np.arange(0,n_samples)
        max_int = np.max(all_peaks["int"])  # using for normalization (should be change to the absolute max in the real images)

        # assigning each peak value to a pixel according to rt bin and sample
        for p in range(len(all_peaks)):
            
            peak_bin = all_peaks.iloc[p]["rt_bins"]
            peak_sample = all_peaks.iloc[p]["file"]

            x = np.where(unique_samples == peak_sample)[0][0]
            y = np.where(bin_labels == peak_bin)[0][0]

            pixel_int =  max(int(np.round(np.mean(all_peaks.iloc[p]["int"]) /max_int *255)),1)
            empty_image[x, y] = [pixel_int, pixel_int, pixel_int]

        #print(f"Max. value in the np array: {empty_image.max()}", flush = True)
        img = Image.fromarray(empty_image) #converting into image
        return img

def create_mask(image, zero_index):
   
    image_array = np.array(image)

    # Convert the binary array back to an image
    bw_image = np.where(image_array > 0, 1, 0).astype(np.uint8)

    for k in range(len(zero_index)):
        if zero_index[k] != -1:
            bw_image[k,zero_index[k]] = [0,0,0]

    bw_image = Image.fromarray(bw_image)

    return bw_image


#initial set parameters

rt_range = [100,1200]
peak_length = 30

rt_bin_size = 0.6 
rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size)

max_int = np.random.uniform(1000,8000) #max intensity for the generated peak (random in the range )   

# useful paths

image_path = "/home/frz603/DSP/simulated_data_02/overlapping_peaks/images"
label_path = "/home/frz603/DSP/simulated_data_02/overlapping_peaks/labels"
check_path = "/home/frz603/DSP/simulated_data_02/overlapping_peaks/check"

#other images have these parameters:
n_samples = 25
n_images = 10
rt_overlap = np.arange(7,20)

for rto in rt_overlap:
    
    print(f"Level of overlap: {rto/peak_length}")

    for i in range(n_images):

        all_peaks = pd.DataFrame(columns=['rt_bins', 'int', 'file'], dtype = 'object')

        limit = [0]*n_samples

        prob_1 = random.choice([0.2,0.6,0.9])     
        #prob_1 = 0.8       
        random_samples = np.random.choice([0, 1], size=n_samples, p=[1-prob_1, prob_1])

        rt_shift = [0,1,2]
        length = np.repeat(peak_length, n_samples)
        shift = np.cumsum(np.random.choice(rt_shift,size = n_samples)) 

        rt_start = np.random.randint(len(rt_bins)-200)

        print(f"Choice of samples: {random_samples}, rt start: {rt_start}")

        for j in range(n_samples):

            if random_samples[j]==1:

                rts = rt_start+shift[j]
            
                df = make_overlapping_peak(rt_bin_size, int(rts), int(length[j]), rto)
            
                limit[j] = float(find_boundary(df))

                df["rt_bins"] = pd.cut(df["x"], bins = rt_bins, right=True)  # adding bins to the simulated peaks
                new_peak = df.groupby('rt_bins')['y'].sum().reset_index()
                new_peak['file']=j
                new_peak.columns=["rt_bins", "int", "file"]
                new_peak = new_peak[new_peak["int"]>0]

                #plot_overlap(new_peak, limit[j])

                all_peaks = pd.concat([all_peaks, new_peak])
        
        bin_labels = all_peaks["rt_bins"].cat.categories  # getting bin labels out
        zero_index = pd.cut(limit, bins=bin_labels).codes           

        img = make_image(all_peaks, n_samples)
        img.save(os.path.join(image_path, f"op_image_{rto}_{i+1:02}.png"))

        label = create_mask(img, zero_index)
        label.save(os.path.join(label_path, f"op_image_{rto}_{i+1:02}.png"))

        check = Image.fromarray(np.where(np.array(label) == 1, 255, 0).astype(np.uint8))
        check.save(os.path.join(check_path, f"op_image_{rto}_{i+1:02}.png"))

        print(f"Done creating op_image_{rto}_{i+1:02}.png")








   


