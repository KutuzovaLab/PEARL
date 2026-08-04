import warnings
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def make_gaussian(rt_range_shifted, rt_bin_size, max_int, rt_bins):
    uni_noise = 0.05 * max_int  # uniform noise added on the Gaussian peak

    x = np.arange(rt_bins[rt_range_shifted[0]] + rt_bin_size / 2,
                  rt_bins[rt_range_shifted[1]] + rt_bin_size / 2, rt_bin_size)

    mu = np.mean(x)      # Mean (peak location)
    sigma = 2   # Standard deviation (width of the peak)

    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y_normalized = y / np.max(y) * np.random.uniform(max_int - 1000, max_int + 1000)

    uniform_noise = np.random.uniform(low=0, high=uni_noise, size=len(x))
    y_final = y_normalized + uniform_noise

    df = pd.DataFrame({'x': x, 'y': y_final})
    return df

def make_overlapping_peak(rt_bin_size, rt_start, peak_length, rt_overlap, rt_bins):
    rt_start_2 = rt_start + rt_overlap  # level of overlap between the two peaks
    print(f"Start no. 1: {rt_start}, start no. 2: {rt_start_2}")

    df1 = make_gaussian([rt_start, rt_start + peak_length], rt_bin_size,
                        np.random.uniform(7000, 8000), rt_bins)
    df2 = make_gaussian([rt_start_2, rt_start_2 + peak_length], rt_bin_size,
                        np.random.uniform(4000, 5000), rt_bins)

    df1["peak"] = 1
    df2["peak"] = 2

    df = pd.concat([df1, df2], ignore_index=True)

    return df


def plotting_two_gaussians(df, save_path):
    colors = {1: 'blue', 2: 'orange'}
    fig, ax = plt.subplots()

    for peak_value, group in df.groupby('peak'):
        ax.scatter(group['x'], group['y'], label=f'Peak {peak_value}', color=colors[peak_value])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    plt.savefig(save_path)


def plot_overlap(dp, lmt, save_path):

    dp['rt_mid'] = dp['rt_bins'].apply(lambda x: x.mid)

    plt.figure(figsize=(12, 6))
    plt.plot(dp['rt_mid'], dp['int'], marker='o')
    plt.xlabel('RT (midpoint)')
    plt.ylabel('Intensity')
    plt.title('Intensity vs Retention Time Bins')
    plt.axvline(x=lmt, color='gray', linestyle='--', label=f'x = {lmt:.2f}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)


def find_boundary(df):
    df1 = df[df['peak'] == 1].sort_values('x')
    df2 = df[df['peak'] == 2].sort_values('x')

    f1 = interp1d(df1['x'], df1['y'], kind='cubic', fill_value="extrapolate")
    f2 = interp1d(df2['x'], df2['y'], kind='cubic', fill_value="extrapolate")

    def diff(x):
        return f1(x) - f2(x)

    x_min = max(df1['x'].min(), df2['x'].min())
    x_max = min(df1['x'].max(), df2['x'].max())
    x_vals = np.mean([x_min, x_max])

    initial_guesses = x_vals
    intersections = fsolve(diff, initial_guesses)[0]

    print(f"Intersection: {intersections:.2f}, rt range: {x_min:.2f} - {x_max:.2f}")
    return intersections

def make_image(all_peaks, n_samples):
    bin_labels = all_peaks["rt_bins"].cat.categories
    width, height = len(bin_labels), n_samples
    empty_image = np.zeros((height, width, 3), dtype=np.uint8)
    unique_samples = np.arange(0, n_samples)
    max_int = np.max(all_peaks["int"])

    for p in range(len(all_peaks)):
        peak_bin = all_peaks.iloc[p]["rt_bins"]
        peak_sample = all_peaks.iloc[p]["file"]

        x = np.where(unique_samples == peak_sample)[0][0]
        y = np.where(bin_labels == peak_bin)[0][0]

        pixel_int = max(int(np.round(np.mean(all_peaks.iloc[p]["int"]) / max_int * 255)), 1)
        empty_image[x, y] = [pixel_int, pixel_int, pixel_int]

    img = Image.fromarray(empty_image)
    return img

def create_mask(image, zero_index):
    image_array = np.array(image)
    bw_image = np.where(image_array > 0, 1, 0).astype(np.uint8)

    for k in range(len(zero_index)):
        if zero_index[k] != -1:
            bw_image[k, zero_index[k]] = [0, 0, 0]

    bw_image = Image.fromarray(bw_image)
    return bw_image
