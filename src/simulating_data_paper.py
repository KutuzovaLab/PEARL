import numpy as np
from scipy.special import log_ndtr
import random
import os
from src.common import parse_config_arg

def make_peak(rt_range_shifted, max_int, p_noise, sigma, tau):

    x = np.arange(rt_range_shifted[0], rt_range_shifted[1], 1)
    mu = np.mean(x)

    K = tau / sigma
    invK = 1.0 / K
    z = (x - mu) / sigma
    clean = np.exp(invK * (0.5 * invK - z) + log_ndtr(z - invK) - np.log(K)) / sigma
    clean = clean / clean.max() * max_int

    noise = np.random.uniform(0, p_noise * max_int, size=len(x))
    y_final = clean + noise

    return x, y_final, clean

def create_resolved_peak(n_max, max_bins_shift, max_intensity_range, int_sd, n_samples, prob_range, tau_range, used, image, label_1, label_2, label_3, noise_range=None):
    n_peaks = random.choice(np.arange(1, n_max + 1))

    for j in range(n_peaks):
        peak_length = random.choice(np.arange(60, 101))

        available = [x for x in range(image.shape[1] - peak_length - max_bins_shift) if x not in used]
        if not available:
            print("No available RT bins left for new peaks!")
            break
        
        prob_samples = random.uniform(prob_range[0], prob_range[1])
        random_samples = np.random.choice([0, 1], size=n_samples, p=[1-prob_samples, prob_samples])

        rt_start = random.choice(available)
        rt_shift = np.random.choice(np.arange(-max_bins_shift, max_bins_shift + 1), size=n_samples)

        intr = random.uniform(max_intensity_range[0], max_intensity_range[1])
        max_int = np.clip(np.random.uniform(intr - int_sd, intr + int_sd, size=n_samples), 0, None)

        if np.nanmedian(max_int) >= 5000:
            base_sigma = np.random.uniform(2, 4)
            p_noise_range = [0.01, 0.05]
        else:
            base_sigma = np.random.uniform(4, 8)
            p_noise_range = [0.05, 0.20]

        if noise_range is not None:
            p_noise_range = noise_range

        sigma = base_sigma * np.random.uniform(0.95, 1.05, size=n_samples)
        tau = sigma * np.random.uniform(tau_range[0], tau_range[1], size=n_samples)

        p_noise = np.random.uniform(p_noise_range[0], p_noise_range[1], size=n_samples)

        rt_starting_points = rt_start + rt_shift
        rt_ending_points = rt_starting_points + peak_length

        rt_min = int(np.min(rt_starting_points))
        rt_max = int(np.max(rt_ending_points))

        if rt_min < 0 or rt_max >= image.shape[1]:
            continue

        if not used.isdisjoint(range(rt_min, rt_max)):
            continue

        old_image = image[:, rt_min:rt_max].copy()
        old_label_1 = label_1[:, rt_min:rt_max].copy()
        old_label_2 = label_2[:, rt_min:rt_max].copy()
        old_label_3 = label_3[:, rt_min:rt_max].copy()

        s_starts, s_ends = [], []

        try:
            for i in range(n_samples):

                if random_samples[i] == 1:
                    rt_range = np.array([rt_starting_points[i], rt_ending_points[i]])
                    rt_idx = np.arange(rt_range[0], rt_range[1])

                    rt_peak, intensity_peak, clean_peak = make_peak(
                        rt_range_shifted=rt_range,
                        max_int=max_int[i],
                        p_noise=p_noise[i],
                        sigma=sigma[i],
                        tau=tau[i]
                    )

                    image[i, rt_idx] = intensity_peak
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

            used.update(range(rt_min, rt_max))

        except Exception as e:
            image[:, rt_min:rt_max] = old_image
            label_1[:, rt_min:rt_max] = old_label_1
            label_2[:, rt_min:rt_max] = old_label_2
            label_3[:, rt_min:rt_max] = old_label_3
            print(f"Error on peak {j+1}, reverting changes: {e}")

    return image, label_1, label_2, label_3

def add_background_noise(int_mean, int_sd, image):
    mask = image == 0
    image[mask] = np.random.normal(loc=int_mean, scale=int_sd, size=int(mask.sum()))
    np.clip(image, 0, None, out=image)

    return image

def generate_dataset(cfg):
    seed = cfg.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_samples = cfg["n_samples"]
    rt_bins = np.arange(0, cfg["rt_bins"])
    width = len(rt_bins)
    out_dir = cfg["out_dir"]
    label_versions = cfg.get("labels", ["v1", "v2", "v3"])
    start_index = cfg.get("start_index", 0)
    bg = cfg.get("background", {"mean": 500, "sd": 200})
    default_shift = cfg.get("max_bins_shift_choices", [0, 1, 2])
    default_tau = cfg.get("tau_range", [0.2, 0.8])

    for subset in cfg["subsets"]:
        name = subset["name"]
        n_images = subset["n_images"]
        groups = subset["groups"]

        img_dir = os.path.join(out_dir, name, "images")
        os.makedirs(img_dir, exist_ok=True)
        lbl_dirs = {}
        for v in label_versions:
            lbl_dirs[v] = os.path.join(out_dir, name, "labels", v)
            os.makedirs(lbl_dirs[v], exist_ok=True)

        print(f"[{name}] generating {n_images} images", flush=True)

        for k in range(n_images):

            image = np.zeros((n_samples, width))
            label_1 = np.zeros((n_samples, width))
            label_2 = np.zeros((n_samples, width))
            label_3 = np.zeros((n_samples, width))

            used = set()

            for g in groups:
                max_bins_shift = random.choice(g.get("max_bins_shift_choices", default_shift))
                image, label_1, label_2, label_3 = create_resolved_peak(
                    n_max=g["n_max"],
                    max_bins_shift=max_bins_shift,
                    max_intensity_range=g["intensity"],
                    int_sd=g["int_sd"],
                    n_samples=n_samples,
                    prob_range=g["occurrence"],
                    tau_range=g.get("tau_range", default_tau),
                    used=used,
                    image=image,
                    label_1=label_1,
                    label_2=label_2,
                    label_3=label_3,
                    noise_range=g.get("noise_range"),
                )

            image = add_background_noise(bg["mean"], bg["sd"], image)

            img_name = f"{(k + start_index):05d}.npy"
            np.save(os.path.join(img_dir, img_name), image)
            all_labels = {"v1": label_1, "v2": label_2, "v3": label_3}
            for v in label_versions:
                np.save(os.path.join(lbl_dirs[v], img_name), all_labels[v])

if __name__ == "__main__":
    config = parse_config_arg(description="Generate synthetic LC-MS training images from a YAML config.")
    generate_dataset(config)