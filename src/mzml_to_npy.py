import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
import gc
import time
import numpy as np
import pandas as pd
import pyopenms as oms
from scipy.interpolate import CubicSpline

try:
    import numba as nb

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:

    @nb.njit(cache=True)
    def _fill_row_sums(lo, hi, inten, out_row):
        for b in range(lo.shape[0]):
            s = np.float32(0.0)
            for t in range(lo[b], hi[b]):
                s = np.float32(s + inten[t])
            out_row[b] = s

    @nb.njit(cache=True)
    def _rowwise_bin_max(colvals, rtbin, n_rt):
        out = np.zeros((n_rt, colvals.shape[1]), dtype=np.float64)
        for r in range(colvals.shape[0]):
            k = rtbin[r]
            if k < 0 or k >= n_rt:
                continue
            for b in range(colvals.shape[1]):
                if colvals[r, b] > out[k, b]:
                    out[k, b] = colvals[r, b]
        return out
else:

    def _fill_row_sums(lo, hi, inten, out_row):
        for b in range(lo.shape[0]):
            s = np.float32(0.0)
            for t in range(lo[b], hi[b]):
                s = np.float32(s + inten[t])
            out_row[b] = s

    def _rowwise_bin_max(colvals, rtbin, n_rt):
        out = np.zeros((n_rt, colvals.shape[1]), dtype=np.float64)
        for r in range(colvals.shape[0]):
            k = rtbin[r]
            if 0 <= k < n_rt:
                np.maximum(out[k], colvals[r], out=out[k])
        return out


def interpolating_rt(colvals, sub_rts, f3_pos, length_spline):
    n_sub = colvals.shape[0]
    for p in f3_pos:
        idx = np.concatenate((np.arange(max(p - length_spline, 0), p),
                              np.arange(p + 1, min(p + length_spline + 1, n_sub))))
        vals = CubicSpline(sub_rts[idx], colvals[idx], axis=0)(sub_rts[p])
        np.maximum(vals, 0.0, out=vals)
        colvals[p] = vals
    return colvals


def making_rt_bins(colvals, sub_rtbin, rt_bins):
    return _rowwise_bin_max(colvals, sub_rtbin, rt_bins.shape[0] - 1)


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


def read_mzml_files(path, files, rt_bin_size, rt_range, mz_range, mass_tol, length_spline=5):
    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)
    hi_vals = mass_bins + mass_tol
    n_bins, n_rt = mass_bins.shape[0], rt_bins.shape[0] - 1

    results_ms1 = np.zeros((n_bins, len(files), n_rt), dtype=np.float32)
    results_ms2 = np.zeros((n_bins, len(files), n_rt), dtype=np.float32)

    for file_ind in range(len(files)):
        start = time.time()
        exp = oms.MSExperiment()
        oms.MzMLFile().load(os.path.join(path, files[file_ind]) + ".mzML", exp)

        rts, codes, mzs, ints = [], [], [], []
        for sp in exp:
            mz, it = sp.get_peaks()
            rts.append(sp.getRT())
            fid = sp.getNativeID().split(' ', 1)[0]
            codes.append(1 if fid == 'function=1' else 2 if fid == 'function=2'
                         else 3 if fid == 'function=3' else 0)
            mzs.append(np.ascontiguousarray(mz, np.float64))
            ints.append(np.ascontiguousarray(it, np.float32))
        rts = np.asarray(rts, np.float64)
        nid = np.asarray(codes, np.int64)

        keep = np.nonzero((rts >= rt_range[0]) & (rts <= rt_range[1]))[0]
        rts, nid = rts[keep], nid[keep]
        mzs, ints = [mzs[i] for i in keep], [ints[i] for i in keep]
        rtbin = (np.searchsorted(rt_bins, rts, 'right') - 1).astype(np.int64)

        dense = np.zeros((len(rts), n_bins), dtype=np.float64)
        for s in range(len(rts)):
            mz = mzs[s]
            if mz.shape[0]:
                lo = np.searchsorted(mz, mass_bins, 'left').astype(np.int64)
                hi = np.searchsorted(mz, hi_vals, 'right').astype(np.int64)
                _fill_row_sums(lo, hi, ints[s], dense[s])

        for exclude, results in ((2, results_ms1), (1, results_ms2)):
            sub = np.nonzero(nid != exclude)[0]
            colvals = np.ascontiguousarray(dense[sub])
            f3 = np.nonzero(nid[sub] == 3)[0]
            if f3.size:
                interpolating_rt(colvals, rts[sub], f3, length_spline)
            results[:, file_ind, :] = making_rt_bins(colvals, rtbin[sub], rt_bins).T.astype(np.float32)

        print(f"file: {files[file_ind]} - {file_ind+1}/{len(files)} - "
              f"time elapsed: {(time.time()-start)/60:.2f} min", flush=True)
        del exp, dense, mzs, ints
        gc.collect()

    return results_ms1, results_ms2


def saving_results(ms1, ms2, output_folder, mz_range, mass_tol, int_thres=1000):
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)

    for i, mb in enumerate(mass_bins):
        mb1 = str(round(mb, 1)).replace('.', '_')
        mb2 = str(round(mb + mass_tol, 1)).replace('.', '_')
        filename_ms1 = f"MS1_{mb1}_{mb2}.npy"
        np.save(os.path.join(output_folder, 'images', filename_ms1), ms1[i])
        label = (ms1[i] > int_thres).astype(np.uint8)
        np.save(os.path.join(output_folder, 'labels', filename_ms1), label)

        filename_ms2 = f"MS2_{mb1}_{mb2}.npy"
        np.save(os.path.join(output_folder, 'images', filename_ms2), ms2[i])
        label = (ms2[i] > int_thres).astype(np.uint8)
        np.save(os.path.join(output_folder, 'labels', filename_ms2), label)
