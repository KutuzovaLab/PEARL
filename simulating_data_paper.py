import numpy as np
from scipy.stats import exponnorm
import matplotlib.pyplot as plt
import random
import os

def make_peak(rt_range_shifted, max_int, p_noise, sigma, tau):
    
    x = np.arange(rt_range_shifted[0], rt_range_shifted[1], 1)
    mu = np.mean(x)

    K = tau / sigma
    clean = exponnorm.pdf(x, K=K, loc=mu, scale=sigma)
    clean = clean / clean.max() * max_int

    noise = np.random.uniform(0, p_noise * max_int, size=len(x))
    y_final = clean + noise

    return x, y_final, clean

def create_resolved_peak(n_max, max_bins_shift, max_intensity_range, int_sd, n_samples, prob_range, tau_range, used, image, label_1, label_2, label_3):

    # Placing up to n_max group of peaks in the image
    n_peaks = random.choice(np.arange(1, n_max + 1))

    for j in range(n_peaks):

        # Peak length is between 60 and 101 pixels (RT bins) - but it's regulated by sigma mostly
        peak_length = random.choice(np.arange(60, 101))

        # Checking if there is available space in the image to create a peak
        available = [x for x in range(image.shape[1] - peak_length - max_bins_shift) if x not in used]
        if not available:
            print("No available RT bins left for new peaks!")
            break
        
        # Choosing which sample get a compound and which don't
        prob_samples = random.uniform(prob_range[0], prob_range[1])
        random_samples = np.random.choice([0, 1], size=n_samples, p=[1-prob_samples, prob_samples])

        # Choosing the position of the peak group
        rt_start = random.choice(available)
        # Adding random RT shift
        rt_shift = np.random.choice(np.arange(-max_bins_shift, max_bins_shift + 1), size=n_samples)

        # Selecting random intensity for the peak 
        intr = random.uniform(max_intensity_range[0], max_intensity_range[1])
        max_int = np.clip(np.random.uniform(intr - int_sd, intr + int_sd, size=n_samples), 0, None)

        # Choosing sigma and noise (for tall and small peaks)
        if np.nanmedian(max_int) >= 5000:
            base_sigma = np.random.uniform(2, 4)
            p_noise_range = [0.01, 0.05]
        else:
            base_sigma = np.random.uniform(4, 8)
            p_noise_range = [0.05, 0.20]

        # Choosing individual sigma and tau for each peak (small variation)
        sigma = base_sigma * np.random.uniform(0.95, 1.05, size=n_samples)
        tau = sigma * np.random.uniform(tau_range[0], tau_range[1], size=n_samples)

        # Choosing noise for each peak
        p_noise = np.random.uniform(p_noise_range[0], p_noise_range[1], size=n_samples)

        # Calculating total range of each peak
        rt_starting_points = rt_start + rt_shift
        rt_ending_points = rt_starting_points + peak_length

        rt_min = int(np.min(rt_starting_points))
        rt_max = int(np.max(rt_ending_points))

        # Just checking if the peak is overlapping with anything
        if rt_min < 0 or rt_max >= image.shape[1]:
            continue

        if set(range(rt_min, rt_max)).intersection(used):
            continue

        # Saving old version of the image in case of any accidents with peak placing
        old_image = image[:, rt_min:rt_max].copy()
        old_label_1 = label_1[:, rt_min:rt_max].copy()
        old_label_2 = label_2[:, rt_min:rt_max].copy()
        old_label_3 = label_3[:, rt_min:rt_max].copy()

        # Using actual ranges for making masks
        s_starts, s_ends = [], []

        try:
            for i in range(n_samples):

                if random_samples[i] == 1:
                    
                    # Making the peak
                    rt_range = np.array([rt_starting_points[i], rt_ending_points[i]])
                    rt_idx = np.arange(rt_range[0], rt_range[1])

                    rt_peak, intensity_peak, clean_peak = make_peak(
                        rt_range_shifted=rt_range,
                        max_int=max_int[i],
                        p_noise=p_noise[i],
                        sigma=sigma[i],
                        tau=tau[i]
                    )

                    # Storing into an image
                    image[i, rt_idx] = intensity_peak

                    # Making a mask
                    threshold = 0.15 * clean_peak.max()
                    significant = clean_peak > threshold
                    
                    s_start = int(rt_peak[significant][0])
                    s_end = int(rt_peak[significant][-1]) + 1

                    label_1[i, s_start:s_end] = 1
                    s_starts.append(s_start)
                    s_ends.append(s_end)

            if s_starts:
                label_2[random_samples == 1, min(s_starts):max(s_ends)] = 1
                label_3[:, min(s_starts):max(s_ends)] = 1

            # Updating indices of used up spots
            used.update(range(rt_min, rt_max))

        except Exception as e:
            image[:, rt_min:rt_max] = old_image
            label_1[:, rt_min:rt_max] = old_label_1
            label_2[:, rt_min:rt_max] = old_label_2
            label_3[:, rt_min:rt_max] = old_label_3
            print(f"Error on peak {j+1}, reverting changes: {e}")

    return image, label_1, label_2, label_3


def add_background_noise(int_mean, int_sd, image):
   
    image[image == 0] = np.random.normal(loc=int_mean, scale=int_sd, size=len(np.where(image == 0)[1]))
    image[image<0] = 0

    return image


# Number of images
n_images = 10000
print(f'Sparse peaks - no. of images: {n_images}', flush = True)

# Set parameters
n_samples = 100
rt_bins = np.arange(0, 3500)

# Folder with images
img_path = "/maps/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/simulated_data_14/images"
os.makedirs(img_path, exist_ok=True)

lbl_path = "/maps/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/simulated_data_14/labels"
os.makedirs(os.path.join(lbl_path, 'v1'), exist_ok=True)
os.makedirs(os.path.join(lbl_path, 'v2'), exist_ok=True)
os.makedirs(os.path.join(lbl_path, 'v3'), exist_ok=True)

# Creating images
for k in range(n_images):

    # Empty images
    image = np.zeros((n_samples, len(rt_bins)))
    label_1 = np.zeros((n_samples, len(rt_bins)))
    label_2 = np.zeros((n_samples, len(rt_bins)))
    label_3 = np.zeros((n_samples, len(rt_bins)))

    # Keeping track of peaks
    used = set()

    # Big peak
    image, label_1, label_2, label_3 = create_resolved_peak(n_max = 1, max_bins_shift = random.choice([0, 1, 2]), max_intensity_range = [18000, 20000], int_sd = 10000, n_samples = n_samples, prob_range =  [0.1, 0.5], tau_range = [0.2, 0.8], used = used, image = image, label_1 = label_1, label_2 = label_2, label_3 = label_3)

    # Small peaks
    image, label_1, label_2, label_3 = create_resolved_peak(n_max = 3, max_bins_shift = random.choice([0, 1, 2]), max_intensity_range = [2000, 5000], int_sd = 500, n_samples = n_samples, prob_range = [0.1, 0.5], tau_range = [0.2, 0.8], used = used, image = image, label_1 = label_1, label_2 = label_2, label_3 = label_3)

    # Adding uniform noise outside
    image = add_background_noise(500, 200, image)

    # Saving final results
    img_name = f"{(k+70001):05d}.npy"
    np.save(os.path.join(img_path, img_name), image)
    np.save(os.path.join(os.path.join(lbl_path, 'v1'), img_name), label_1)
    np.save(os.path.join(os.path.join(lbl_path, 'v2'), img_name), label_2)
    np.save(os.path.join(os.path.join(lbl_path, 'v3'), img_name), label_3)

