import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import gc
import time
import numpy as np
import pandas as pd
import pyopenms as oms
from scipy.interpolate import CubicSpline

from src.common import parse_config_arg

try:
    import numba as nb

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

_njit = nb.njit(cache=True) if _HAVE_NUMBA else (lambda f: f)


@_njit
def _fill_row_sums(lo, hi, inten, out_row):
    for b in range(lo.shape[0]):
        s = np.float32(0.0)
        for t in range(lo[b], hi[b]):
            s = np.float32(s + inten[t])
        out_row[b] = s

if _HAVE_NUMBA:

    @_njit
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

# Reference tables are .xlsx (MS-DIAL standards) or .csv (MS-DIAL wheat).
def _read_reference_table(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, skiprows=4, header=0)
    return pd.read_csv(path, skiprows=4, header=0, low_memory=False)

# '230523_Batch3_103_Blank_13' -> '230523_Batch3_103'
def _stem3(names):
    return pd.Index(names).str.split("_").str[:3].str.join("_")

def selecting_files(mzml_path, suspect_path=None, order_path=None):
    files = sorted(f[:-5] for f in os.listdir(mzml_path) if f.endswith(".mzML"))

    if suspect_path:
        ref = _read_reference_table(suspect_path)
        covered = set(_stem3(ref.columns[32:]))
        selected = [f for f in files if f in covered]
        if len(selected) != len(files):
            print(f"selecting_files: {len(selected)} of {len(files)} mzML are covered by "
                  f"{os.path.basename(suspect_path)}", flush=True)
        files = selected

    if order_path:
        order = pd.read_excel(order_path)
        key = dict(zip(_stem3(order["File name"].astype(str)),
                       pd.to_numeric(order["Injection order"], errors="coerce")))
        known = [f for f in files if f in key]
        if known and [key[f] for f in known] != sorted(key[f] for f in known):
            print(f"selecting_files: WARNING - filename order disagrees with injection order "
                  f"in {os.path.basename(order_path)}; image rows may not be in acquisition "
                  f"sequence", flush=True)
        else:
            print(f"selecting_files: filename order confirmed as acquisition order "
                  f"({len(known)}/{len(files)} samples checked)", flush=True)

    print(f"Final number of included samples: {len(files)}", flush=True)
    return files

# Build stacked-EIC images from mzML files.
def read_mzml_files(path, files, rt_bin_size, rt_range, mz_range, mass_tol, length_spline=5,
                    acquisition="dia"):
    if acquisition not in ("dia", "dda"):
        raise ValueError(f"acquisition must be 'dia' or 'dda', got {acquisition!r}")

    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)
    hi_vals = mass_bins + mass_tol
    n_bins, n_rt = mass_bins.shape[0], rt_bins.shape[0] - 1

    results_ms1 = np.zeros((n_bins, len(files), n_rt), dtype=np.float32)
    results_ms2 = (np.zeros((n_bins, len(files), n_rt), dtype=np.float32)
                   if acquisition == "dia" else None)

    for file_ind in range(len(files)):
        start = time.time()
        exp = oms.MSExperiment()
        oms.MzMLFile().load(os.path.join(path, files[file_ind]) + ".mzML", exp)

        rts, codes, mzs, ints = [], [], [], []
        for sp in exp:
            mz, it = sp.get_peaks()
            rts.append(sp.getRT())
            if acquisition == "dia":
                fid = sp.getNativeID().split(' ', 1)[0]
                codes.append(1 if fid == 'function=1' else 2 if fid == 'function=2'
                             else 3 if fid == 'function=3' else 0)
            else:
                codes.append(sp.getMSLevel())
            mzs.append(np.ascontiguousarray(mz, np.float64))
            ints.append(np.ascontiguousarray(it, np.float32))
        rts = np.asarray(rts, np.float64)
        nid = np.asarray(codes, np.int64)

        level_ok = (nid == 1) if acquisition == "dda" else np.ones(len(rts), bool)
        keep = np.nonzero((rts >= rt_range[0]) & (rts <= rt_range[1]) & level_ok)[0]
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

        if acquisition == "dda":
            results_ms1[:, file_ind, :] = making_rt_bins(
                np.ascontiguousarray(dense), rtbin, rt_bins).T.astype(np.float32)
        else:
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

# Write one .npy image + threshold label per mass channel. ms2 may be None (DDA).
def saving_results(ms1, ms2, output_folder, mz_range, mass_tol, int_thres=1000):
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)

    for i, mb in enumerate(mass_bins):
        mb1 = str(round(mb, 1)).replace('.', '_')
        mb2 = str(round(mb + mass_tol, 1)).replace('.', '_')

        for level, arr in (("MS1", ms1), ("MS2", ms2)):
            if arr is None:
                continue
            filename = f"{level}_{mb1}_{mb2}.npy"
            np.save(os.path.join(output_folder, 'images', filename), arr[i])
            np.save(os.path.join(output_folder, 'labels', filename),
                    (arr[i] > int_thres).astype(np.uint8))

def main(cfg):
    suspect_path = cfg.get("suspect_path")
    mzml_path = cfg["mzml_path"]
    output_folder = cfg["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    rt_range = cfg["rt_range"]
    rt_bin_size = cfg["rt_bin_size"]
    mz_range = cfg["mz_range"]
    mass_tol = cfg["mass_tol"]
    length_spline = cfg.get("length_spline", 5)
    int_thres = cfg.get("int_thres", 1000)
    acquisition = cfg.get("acquisition", "dia")

    final_files = selecting_files(mzml_path, suspect_path, cfg.get("order_path"))

    print(f'Reading raw files ({acquisition.upper()})...', flush=True)
    ms1, ms2 = read_mzml_files(
        mzml_path, final_files, rt_bin_size, rt_range, mz_range, mass_tol,
        length_spline=length_spline, acquisition=acquisition,
    )

    saving_results(ms1, ms2, output_folder, mz_range, mass_tol, int_thres=int_thres)

if __name__ == "__main__":
    config = parse_config_arg(description="Convert mzML files to MS1/MS2 .npy images.")
    main(config)