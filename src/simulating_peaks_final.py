import os
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def simulate_1D_peak(rt_start, length, shift, random_samples, rt_bin_size, rt_range):
    max_int = np.random.uniform(1000, 8000)  # max intensity for the generated peak (random in the range)
    uni_noise = 0.05 * max_int  # uniform noise added on the Gaussian peak is set to be 20% of the intensity
    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)  # size of the rt bin
    fake_peaks = pd.DataFrame(columns=['rt', 'int', 'file'], dtype='object')

    for i in range(len(random_samples)):
        if random_samples[i] == 1:

            rt_end = rt_start + int(length[i])

            rt_range_shifted = [rt_start + int(shift[i]), rt_end + int(shift[i])]
            print(f"rt_range_shift before: {rt_range_shifted}")

            if rt_range_shifted[1] >= (len(rt_bins) - 1):
                rt_range_shifted[1] = len(rt_bins) - 1  # if the peak gets "out of the frame", set it to the end
            print(f"rt_range_shift after: {rt_range_shifted}", flush=True)

            if rt_range_shifted[0] < rt_range_shifted[1]:
                x = np.arange(rt_bins[rt_range_shifted[0]] + rt_bin_size / 2,
                              rt_bins[rt_range_shifted[1]] + rt_bin_size / 2, rt_bin_size)

                mu = np.mean(x)      # Mean (peak location)
                sigma = 2   # Standard deviation (width of the peak)

                y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                y_normalized = y / np.max(y) * np.random.uniform(max_int - 1000, max_int + 1000)

                uniform_noise = np.random.uniform(low=0, high=uni_noise, size=len(x))
                noisy_gaussian_vector = y_normalized + uniform_noise

                df = pd.DataFrame({"rt": x, "int": noisy_gaussian_vector, "file": i})

                fake_peaks = pd.concat([fake_peaks, df], ignore_index=True)

    fake_peaks["rt_bins"] = pd.cut(fake_peaks["rt"], bins=rt_bins, right=True)  # adding bins to the simulated peaks

    fake_peaks = fake_peaks[fake_peaks["rt_bins"].notna()]  # removing peak points outside the rt range

    return fake_peaks


def convert_into_image(fake_peaks, n_samples):
    bin_labels = fake_peaks["rt_bins"].cat.categories  # getting bin labels out
    width, height = len(bin_labels), n_samples  # empty image of size (n_samples x n_rt_bins)
    empty_image = np.zeros((height, width, 3), dtype=np.uint8)

    # preparing parameters
    unique_samples = np.arange(0, n_samples)
    max_int = np.max(fake_peaks["int"])  # used for normalization

    # assigning each peak value to a pixel according to rt bin and sample
    for p in range(len(fake_peaks)):

        peak_bin = fake_peaks.iloc[p]["rt_bins"]
        peak_sample = fake_peaks.iloc[p]["file"]

        x = np.where(unique_samples == peak_sample)[0][0]
        y = np.where(bin_labels == peak_bin)[0][0]

        pixel_int = max(int(np.round(np.mean(fake_peaks.iloc[p]["int"]) / max_int * 255)), 1)

        empty_image[x, y] = [pixel_int, pixel_int, pixel_int]

    img = Image.fromarray(empty_image)  # converting into image

    return img


def create_masks_sim(image):
    image_array = np.array(image)
    bw_image_scaled = Image.fromarray(np.where(image_array > 0, 1, 0).astype(np.uint8))
    return bw_image_scaled

def generate_sim_images(n_samples, n_images, n_blobs_max, images_folder, labels_folder,
                        rt_range, rt_bin_size=0.6, peak_length=30):
    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)  # rt bin boundaries
    rt_shift = [0, 1, 2]  # step of rt bins for the shift (between 1 and max-1)

    for k in range(n_images):
        print(f"Image #{k+1}", flush=True)

        # choose a random number of blobs per image (max n_blobs_max)
        n_peaks = np.random.randint(1, n_blobs_max + 1)
        print(f"# of blobs: {n_peaks}", flush=True)

        # saving all generated peaks in the dataframe
        sim_peaks = pd.DataFrame(columns=['rt', 'int', 'file'], dtype='object')

        # saving ends of the peaks to make sure they are not overlapping
        rts = np.array([], dtype=int)
        rte = np.array([], dtype=int)

        # looping to create individual peaks at different rt ranges with varying size and shift
        for i in range(n_peaks):

            prob_1 = random.choice([0.2, 0.6, 0.9])
            random_samples = np.random.choice([0, 1], size=n_samples, p=[1 - prob_1, prob_1])

            length = np.repeat(peak_length, n_samples)
            shift = np.cumsum(np.random.choice(rt_shift, size=n_samples))  # simulate rt shift (in bin terms)
            rt_start = np.random.randint(len(rt_bins) - 2)  # location of the peak (excluding bordering bins)
            rt_end = int(rt_start + length[0] + shift[len(shift) - 1])

            if sum((rt_start >= rts) & (rt_start <= rte) | (rt_end >= rts) & (rt_end <= rte)) == 0:

                print(f"Simulating peak with range: {rt_start}-{rt_end}")

                # simulating a single peak
                sp = simulate_1D_peak(rt_start, length, shift, random_samples, rt_bin_size, rt_range)
                sim_peaks = pd.concat([sim_peaks, sp], ignore_index=True)

                # saving peak boundaries
                rts = np.append(rts, rt_start)
                rte = np.append(rte, rt_end)

                print(f"New boundaries: start {rts}, end {rte}")
            else:
                print(f"Simulating peak with range: {rt_start}-{rt_end} - rejected")

        # turning all of the peaks into a single image
        sim_img = convert_into_image(sim_peaks, n_samples)

        # saving image into images_folder
        filename = f"image_{k+1:02}.png"
        sim_img.save(images_folder + "/" + filename)

        # creating labels for peaks (1 - everything bigger than 0)
        sim_mask = create_masks_sim(sim_img)
        sim_mask.save(labels_folder + "/" + filename)

        print(f"Saved image and mask - {filename}")
