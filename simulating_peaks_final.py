from PIL import Image
import numpy as np
import pandas as pd
import warnings
import random
import os
#import argparse

# passing the arguments from the comand line
#parser = argparse.ArgumentParser(description="Simulate Peaks")
#parser.add_argument('--n_samples', type=int, default=50, help='Number of samples')
#parser.add_argument('--n_images', type=int, default=5, help='Number of images')
#parser.add_argument('--n_blobs_max', type=int, default=10, help='Maximum number of blobs')
#parser.add_argument('--input_path', type=str, help="Path to input file", required=True)

#args = parser.parse_args()

#print(f"images are saved in folder: {args.input_path}")

#supressing  warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# all the necessary functions
def simulate_1D_peak(rt_start, length, shift, random_samples, rt_bin_size, rt_range):
        
    #setting the parameters
    max_int = np.random.uniform(1000,8000) #max intensity for the generated peak (random in the range )    
    uni_noise = 0.05*max_int # uniform noise added on the Gaussian peak is set to be 20% of the intensity
    rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size)  #size of the rt bin (or just  considered as single rt point, doesnt matter)

    # generating peaks

    fake_peaks = pd.DataFrame(columns=['rt', 'int', 'file'], dtype = 'object')

    for i in range(len(random_samples)):
        if random_samples[i] == 1:
            
            rt_end = rt_start+int(length[i])

            rt_range_shifted = [rt_start+int(shift[i]), rt_end+int(shift[i])]
            print(f"rt_range_shift before: {rt_range_shifted}")            

            if rt_range_shifted[1]>=(len(rt_bins)-1): 
                rt_range_shifted[1] = len(rt_bins)-1  #if the peak gets "out of the frame", we set it to the end of the rt range
            print(f"rt_range_shift after: {rt_range_shifted}", flush = True)

            if rt_range_shifted[0]<rt_range_shifted[1]:
                x = np.arange(rt_bins[rt_range_shifted[0]]+rt_bin_size/2, rt_bins[rt_range_shifted[1]]+rt_bin_size/2, rt_bin_size)  # generating rt values for the peak (rt_bin_size/2 is to get the points in the middle of the existing bins)

                mu = np.mean(x)      # Mean (peak location)
                sigma = 2   # Standard deviation (width of the peak)

                y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)  # generating intensities of the peak (gaussian peak)
                y_normalized = y / np.max(y) * np.random.uniform(max_int-1000, max_int+1000)  # setting it to the random intensity (selected max +/- 1000)

                uniform_noise = np.random.uniform(low=0, high=uni_noise, size=len(x))  # adding uniform noise
                noisy_gaussian_vector = y_normalized + uniform_noise

                #print(f"Final intensity - file {i+1}: {noisy_gaussian_vector}")

                df = pd.DataFrame({"rt": x, "int": noisy_gaussian_vector, "file": i})  #saving each peak in a dataframe
                
                fake_peaks = pd.concat([fake_peaks,df], ignore_index = True)


    #rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size)
    fake_peaks["rt_bins"] = pd.cut(fake_peaks["rt"], bins = rt_bins, right=True)  # adding bins to the simulated peaks
    
    fake_peaks = fake_peaks[fake_peaks["rt_bins"].notna()] # removing those peak points that are outside of the given rt range (might happen because of the shift)

    return fake_peaks

def convert_into_image(fake_peaks, n_samples):
    
    #creating empty image of right dimensions
    bin_labels = fake_peaks["rt_bins"].cat.categories  # getting bin labels out

    width, height = len(bin_labels), n_samples  # creating an empty image with size no. of samples x no. of rt bins
    empty_image = np.zeros((height, width, 3), dtype=np.uint8)

    #preparing parameters
    unique_samples = np.arange(0,n_samples)
    max_int = np.max(fake_peaks["int"])  # using for normalization (should be change to the absolute max in the real images)

    # assigning each peak value to a pixel according to rt bin and sample
    for p in range(len(fake_peaks)):
        
        peak_bin = fake_peaks.iloc[p]["rt_bins"]
        peak_sample = fake_peaks.iloc[p]["file"]

        x = np.where(unique_samples == peak_sample)[0][0]
        y = np.where(bin_labels == peak_bin)[0][0]

        pixel_int =  max(int(np.round(np.mean(fake_peaks.iloc[p]["int"]) /max_int *255)),1)

        empty_image[x, y] = [pixel_int, pixel_int, pixel_int]

    #print(f"Max. value in the np array: {empty_image.max()}", flush = True)
    img = Image.fromarray(empty_image) #converting into image
    
    return img

def create_masks_sim(image):
    # Convert the image to a numpy array for processing
    image_array = np.array(image)

    # Convert the binary array back to an image
    bw_image_scaled = Image.fromarray(np.where(image_array > 0, 1, 0).astype(np.uint8))

    return bw_image_scaled 

def generate_sim_images(n_samples, n_images, n_blobs_max, images_folder, labels_folder, rt_range):

    #sparsity_levels = ['minimal', 'mild', 'large'] # 3 possible options of how often a single peak appears  
    
    rt_bin_size = 0.6  #fixed for now
    rt_bins = np.arange(rt_range[0], rt_range[1]+rt_bin_size, rt_bin_size)  # calculating rt bin boundaries according to the given bin size   

    peak_length = 30
    rt_shift = [0,1,2]  #step of rt bins for the shift (between 1 and max-1)

    for k in range(n_images):
        print(f"Image #{k+1}", flush = True)
        
        # choose a random number of blobs per image (max n_peaks_max)
        n_peaks = np.random.randint(1,n_blobs_max+1)  
        print(f"# of blobs: {n_peaks}", flush = True)

        # saving all generated peaks in the dataframe
        sim_peaks = pd.DataFrame(columns=['rt', 'int', 'file'], dtype = 'object')     

        # saving ends of the peaks to make sure they are not overlapping
        rts = np.array([], dtype=int)
        rte = np.array([], dtype=int)
        
        #looping to create individual peaks at different rt ranges with varying size and shift
        for i in range(n_peaks):
            
            prob_1 = random.choice([0.2,0.6,0.9])            
            random_samples = np.random.choice([0, 1], size=n_samples, p=[1-prob_1, prob_1])  # choosing randomly which samples will include the peak

            #length = np.random.choice(np.arange(peak_length[0],peak_length[1],1),n_samples)  # varying length for peaks in different samples (in bin terms as well)
            length = np.repeat(peak_length, n_samples)
            shift = np.cumsum(np.random.choice(rt_shift,size = n_samples)) # moving the peaks to simulate the retention time shift (in bin terms, not seconds)            
            rt_start = np.random.randint(len(rt_bins)-2) # choosing the location of the peak randomly (excluding the bordering bins >= max rt shift)
            rt_end = int(rt_start+length[0]+shift[len(shift)-1])  #the end of the peak thats the furthest to the right will be r_start + length of the peaks + max shift (cum. shift)

            if sum((rt_start>=rts) & (rt_start<=rte) | (rt_end>=rts) & (rt_end<=rte))==0:
                
                print(f"Simulating peak with range: {rt_start}-{rt_end}")
                
                #simulating a single peak
                sp = simulate_1D_peak(rt_start, length, shift, random_samples, rt_bin_size, rt_range)
                sim_peaks = pd.concat([sim_peaks, sp], ignore_index = True)

                # saving peak bouldaries
                rts = np.append(rts, rt_start)
                rte = np.append(rte, rt_end)
                
                print(f"New boudaries: start {rts}, end {rte}")
            else:
                print(f"Simulating peak with range: {rt_start}-{rt_end} - rejected")

        # turning all of the peaks into a single image
        sim_img = convert_into_image(sim_peaks, n_samples)        
        #sim_img.show()

        #saving image into images_folder
        filename = f"image_{k+1:02}.png"
        sim_img.save(images_folder+"/"+filename)

        #print(f"{images_folder}/{filename} - image", flush = True)

        # creating labels for peaks (1 - everything bigger than 0)
        sim_mask = create_masks_sim(sim_img)
        sim_mask.save(labels_folder+"/"+filename)
        
        #print(f"{images_folder}/{filename} - label", flush = True)

        print(f"Saved image and mask - {filename}")


#parsing the parameters
#n_samples = args.n_samples
#n_images = args.n_images
#n_blobs_max = args.n_blobs_max
#output_folder = args.input_path

n_samples = 100
n_images = 10000
n_blobs_max = 5
rt_range = [100,1200]

output_folder = "/home/frz603/DSP/simulated_data_01"

# Define the subfolder paths
images_folder = os.path.join(output_folder, "images")
labels_folder = os.path.join(output_folder, "labels")


# Create the folders if they don't exist
os.makedirs(images_folder, exist_ok=True)  # exist_ok=True prevents an error if the folder already exists
os.makedirs(labels_folder, exist_ok=True)


# Generating images and their labels
generate_sim_images(n_samples, n_images, n_blobs_max, images_folder, labels_folder, rt_range)