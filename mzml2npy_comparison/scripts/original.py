import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import gc
import time
import argparse
import numpy as np
import pandas as pd
import pyopenms as oms
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def interpolating_rt(tic_sample, length_spline):
    missing_indices = np.where(tic_sample['native_id'] == 'function=3')[0]
    for k in range(len(missing_indices)):
        ind_select = np.append(
            np.arange(max(missing_indices[k]-length_spline, 0), missing_indices[k]),
            np.arange(missing_indices[k]+1, min(missing_indices[k]+length_spline+1, len(tic_sample))))
        tic_sel = tic_sample.iloc[ind_select]
        spline = CubicSpline(tic_sel['rt'], tic_sel['int_sum'])
        tic_sample.loc[missing_indices[k], 'int_sum'] = max(
            float(spline(tic_sample.iloc[int(missing_indices[k])]['rt'])), 0)
    return tic_sample

def making_rt_bins(ms, rt_bins):
    ms = ms.sort_values(by='rt').reset_index(drop=True)
    bin_intervals = pd.IntervalIndex.from_breaks(rt_bins, closed='left')
    ms['bins'] = pd.cut(ms['rt'], bins=bin_intervals)
    int_bins = ms.groupby('bins', observed=True)['int_sum'].max().reindex(
        bin_intervals, fill_value=0).to_numpy()
    return pd.DataFrame([int_bins])

def read_mzml_files(path, files, rt_bin_size, rt_range, mz_range, mass_tol):
    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)

    results_ms1 = np.zeros((len(mass_bins), len(files), len(rt_bins)-1), dtype=np.float32)
    results_ms2 = np.zeros((len(mass_bins), len(files), len(rt_bins)-1), dtype=np.float32)

    for file_ind in range(len(files)):
        exp = oms.MSExperiment()
        start = time.time()
        oms.MzMLFile().load(os.path.join(path, files[file_ind]) + ".mzML", exp)
        df = exp.get_df(long=False)
        df = df[['RT', 'mzarray', 'intarray']].copy()

        native_ids = []
        for spectrum in exp:
            native_ids.extend([spectrum.getNativeID().split(' ', 1)[0]])
        df['native_id'] = native_ids

        df = df[(df['RT'] >= rt_range[0]) & (df['RT'] <= rt_range[1])].reset_index(drop=True)

        for j, mb in enumerate(mass_bins):
            mass_slice = [float(mb), float(mb + mass_tol)]
            mass_slice_spectra = []
            for i in range(len(df)):
                mz_value = df.iloc[i]['mzarray']
                int_value = df.iloc[i]['intarray']
                ind_sel = np.where((mz_value >= mass_slice[0]) & (mz_value <= mass_slice[1]))[0]
                mz_sel = mz_value[ind_sel]
                int_sel = int_value[ind_sel]
                if len(mz_sel) == 0:
                    mass_slice_spectra.append({'mz': [0], 'rt': df.loc[i, 'RT'], 'int': [0],
                                               'native_id': df.loc[i, 'native_id']})
                else:
                    mass_slice_spectra.append({'mz': mz_sel, 'rt': df.loc[i, 'RT'], 'int': int_sel,
                                               'native_id': df.loc[i, 'native_id']})

            mass_slice_spectra = pd.DataFrame(mass_slice_spectra)
            mass_slice_spectra['int_sum'] = mass_slice_spectra['int'].apply(sum)

            ms1 = mass_slice_spectra[mass_slice_spectra['native_id'] != 'function=2'].copy().reset_index(drop=True)
            ms1 = interpolating_rt(ms1, 5)
            results_ms1[j, file_ind, :] = making_rt_bins(ms1, rt_bins)

            ms2 = mass_slice_spectra[mass_slice_spectra['native_id'] != 'function=1'].copy().reset_index(drop=True)
            ms2 = interpolating_rt(ms2, 5)
            results_ms2[j, file_ind, :] = making_rt_bins(ms2, rt_bins)

            del mass_slice_spectra

        print(f"file: {files[file_ind]} - {file_ind+1}/{len(files)} - "
              f"time elapsed: {(time.time()-start)/60:.2f} min", flush=True)
        del df, exp
        gc.collect()

    return results_ms1, results_ms2

def saving_results(ms1, ms2, output_folder, mz_range, mass_tol):
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)
    int_thres = 1000
    for i, mb in enumerate(mass_bins):
        mb1 = str(round(mb, 1)).replace('.', '_')
        mb2 = str(round(mb+mass_tol, 1)).replace('.', '_')
        for tag, arr in (("MS1", ms1), ("MS2", ms2)):
            name = f"{tag}_{mb1}_{mb2}.npy"
            np.save(os.path.join(output_folder, 'images', name), arr[i])
            np.save(os.path.join(output_folder, 'labels', name),
                    (arr[i] > int_thres).astype(np.uint8))

def selecting_files(mzml_path, suspect_path):
    all_samples = pd.read_excel(suspect_path, skiprows=4, header=0)
    files = sorted([f for f in os.listdir(mzml_path) if f.endswith(".mzML")])
    updated_file_list = sorted([re.sub(r'_(8[0-9]|9[0-9])\.mzML$', r'_0\1.mzML', filename)
                                if re.search(r'_(8[0-9]|9[0-9])\.mzML$', filename) else filename
                                for filename in files])
    updated_file_list = [re.sub(".mzML", "", item) for item in updated_file_list]
    sel_samples = np.append(np.repeat(False, 32),
        all_samples.columns[32:].str.split("_").str[:3].str.join("_").isin(updated_file_list))
    samples = all_samples.iloc[:, sel_samples]
    final_files = []
    for filename in updated_file_list:
        if filename in samples.columns.str.split("_").str[:3].str.join("_"):
            final_files.append(filename)
    print(f"Final number of included samples: {len(final_files)}", flush=True)
    return final_files

def default_suspect(input_dir):
    return os.path.join(ROOT, "input", "mzml", "Standards_NEG.xlsx")

def run(input_dir=None, output_dir=None, suspect_path=None, rt_range=(100, 1100),
        rt_bin_size=0.73, mz_range=(50, 1200), mass_tol=0.1, limit=0):
    input_dir = input_dir or os.path.join(ROOT, "input", "mzml")
    output_dir = output_dir or os.path.join(ROOT, "output", "original")
    suspect_path = suspect_path or default_suspect(input_dir)
    files = selecting_files(input_dir, suspect_path)
    if limit:
        files = files[:limit]
    print(f"[original] {len(files)} file(s), m/z {list(mz_range)} -> {output_dir}", flush=True)
    ms1, ms2 = read_mzml_files(input_dir, files, rt_bin_size, list(rt_range),
                               list(mz_range), mass_tol)
    saving_results(ms1, ms2, output_dir, list(mz_range), mass_tol)
    return files

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=os.path.join(ROOT, "input", "mzml"))
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "output", "original"))
    ap.add_argument("--suspect-path", default=os.path.join(ROOT, "input", "mzml", "Standards_NEG.xlsx"))
    ap.add_argument("--rt-range", nargs=2, type=float, default=[100, 1100])
    ap.add_argument("--rt-bin-size", type=float, default=0.73)
    ap.add_argument("--mz-range", nargs=2, type=float, default=[50, 1200])
    ap.add_argument("--mass-tol", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="process only first N selected files (0=all)")
    a = ap.parse_args()
    t0 = time.time()
    run(a.input_dir, a.output_dir, a.suspect_path, a.rt_range, a.rt_bin_size,
        a.mz_range, a.mass_tol, a.limit)
    print(f"[original] done in {(time.time()-t0)/60:.2f} min", flush=True)