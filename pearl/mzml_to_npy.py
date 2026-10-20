import os
import json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import gc
import glob
import time
import numpy as np
import pyopenms as oms
from scipy.interpolate import CubicSpline

from pearl.common import parse_config_arg, mass_tol_of, mz_decimals, slice_tag
from pearl.constants import LENGTH_SPLINE

try:
    import numba as nb

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

_njit = nb.njit(cache=True) if _HAVE_NUMBA else (lambda f: f)


@_njit
def _fill_row_sums(lo, hi, inten, out_row):
    for b in range(lo.shape[0]):
        s = 0.0
        for t in range(lo[b], hi[b]):
            s = s + inten[t]
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

def selecting_files(mzml_path, files=None):
    found = sorted(f[:-5] for f in os.listdir(mzml_path) if f.endswith(".mzML"))

    if files:
        wanted = list(dict.fromkeys(files))
        missing = [f for f in wanted if f not in set(found)]
        if missing:
            raise SystemExit(
                f"{len(missing)} file(s) listed in the config are not in {mzml_path}: "
                f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
        selected = [f for f in found if f in set(wanted)]
        print(f"selecting_files: {len(selected)} of {len(found)} mzML selected by config",
              flush=True)
        found = selected

    print(f"Final number of included samples: {len(found)}", flush=True)
    return found

DIA_FUNCTIONS = (1, 2, 3)   # Waters MSe: MS1, MS2, lock mass

def dia_codes(fids, levels, channels=None, functions=None):
    ms1_f, ms2_f, lock_f = tuple(functions) if functions else DIA_FUNCTIONS
    strategy = channels or (
        "function" if any(f.startswith("function=") for f in fids) else "mslevel")

    if strategy == "function":
        want = {f"function={int(ms1_f)}": 1, f"function={int(ms2_f)}": 2}
        if lock_f:
            want[f"function={int(lock_f)}"] = 3
        codes = [want.get(f, 0) for f in fids]
    elif strategy == "mslevel":
        codes = [1 if lv == 1 else 2 if lv == 2 else 0 for lv in levels]
    else:
        raise SystemExit(f"images.dia_channels must be 'function' or 'mslevel', got {strategy!r}")

    return np.asarray(codes, np.int64), strategy

def check_dia_channels(name, codes, strategy, fids, levels, report=True):
    n = [int((codes == c).sum()) for c in (1, 2, 3, 0)]
    if report:
        print(f"DIA channels from {strategy}: MS1 {n[0]}, MS2 {n[1]}, "
              f"lock mass {n[2]}, not imaged {n[3]}", flush=True)
        if not n[2]:
            print("  no lock-mass channel in these files - no spline interpolation applied",
                  flush=True)
    if n[0] and n[1]:
        return
    seen_ids = sorted({f for f in fids})[:6]
    seen_lv = sorted(set(int(lv) for lv in levels))
    raise SystemExit(
        f"{name}: the DIA {'MS1' if not n[0] else 'MS2'} channel is empty "
        f"({strategy} strategy). Native IDs look like {seen_ids}, MS levels {seen_lv}. "
        f"Set images.dia_channels (function|mslevel) and images.dia_functions "
        f"[MS1 MS2 LOCKMASS] for this instrument, or run with --acquisition dda.")

def load_mzml(path):
    """One mzML into memory, reporting which file the parser gave up on."""
    exp = oms.MSExperiment()
    try:
        oms.MzMLFile().load(path, exp)
    except RuntimeError as e:
        raise SystemExit(f"could not read {path} - is it a complete mzML file?\n  {e}")
    return exp

def mass_edges(mz_range, mass_tol, acquisition):
    """Lower m/z edge of every slice; each slice is [edge, edge + mass_tol], both ends inclusive."""
    edges = np.arange(mz_range[0], mz_range[1], mass_tol)
    return np.round(edges, mz_decimals(mass_tol)) if acquisition == "dda" else edges

def read_mzml_files(path, files, rt_bin_size, rt_range, mz_range, mass_tol, length_spline=5,
                    acquisition="dia", dia_channels=None, dia_functions=None):
    if acquisition not in ("dia", "dda"):
        raise ValueError(f"acquisition must be 'dia' or 'dda', got {acquisition!r}")

    rt_bins = np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)
    mass_bins = mass_edges(mz_range, mass_tol, acquisition)
    hi_vals = mass_bins + mass_tol
    n_bins, n_rt = mass_bins.shape[0], rt_bins.shape[0] - 1

    results_ms1 = np.zeros((n_bins, len(files), n_rt), dtype=np.float32)
    results_ms2 = (np.zeros((n_bins, len(files), n_rt), dtype=np.float32)
                   if acquisition == "dia" else None)

    for file_ind in range(len(files)):
        start = time.time()
        exp = load_mzml(os.path.join(path, files[file_ind]) + ".mzML")

        rts, fids, levels, mzs, ints = [], [], [], [], []
        for sp in exp:
            mz, it = sp.get_peaks()
            rts.append(sp.getRT())
            levels.append(sp.getMSLevel())
            if acquisition == "dia":
                fids.append(sp.getNativeID().split(' ', 1)[0])
            mzs.append(np.ascontiguousarray(mz, np.float64))
            ints.append(np.ascontiguousarray(it, np.float32))
        rts = np.asarray(rts, np.float64)

        if acquisition == "dia":
            nid, strategy = dia_codes(fids, levels, dia_channels, dia_functions)
            check_dia_channels(files[file_ind], nid, strategy, fids, levels,
                               report=file_ind == 0)
        else:
            nid = np.asarray(levels, np.int64)

        if acquisition == "dda":
            in_rt = (rts >= rt_bins[0]) & (rts < rt_bins[-1]) & (nid == 1)
        else:
            in_rt = (rts >= rt_range[0]) & (rts <= rt_range[1])
        keep = np.nonzero(in_rt)[0]
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
            for other, results in ((2, results_ms1), (1, results_ms2)):
                sub = np.nonzero((nid != other) & (nid != 0))[0]
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

def _stamp(cfg, files):
    img = cfg
    return {"rt_range": list(img["rt_range"]), "rt_bin_size": img["rt_bin_size"],
            "mz_range": list(img["mz_range"]), "mass_tol": mass_tol_of(img),
            "acquisition": img.get("acquisition", "dia"),
            "mz_edges": "rounded" if img.get("acquisition", "dia") == "dda" else "arange",
            "length_spline": LENGTH_SPLINE,
            "n_files": len(files), "first": files[0], "last": files[-1]}

def up_to_date(cfg, files, output_folder):
    path = os.path.join(output_folder, "params.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            have = json.load(f)
    except Exception:
        return False
    if have != _stamp(cfg, files):
        return False
    n = len(glob.glob(os.path.join(output_folder, "images", "*.npy")))
    expected = int(np.arange(cfg["mz_range"][0], cfg["mz_range"][1],
                             mass_tol_of(cfg)).shape[0]) * (1 if cfg.get("acquisition", "dia") == "dda" else 2)
    return n == expected

def saving_results(ms1, ms2, output_folder, mz_range, mass_tol):
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    mass_bins = np.arange(mz_range[0], mz_range[1], mass_tol)
    nd = mz_decimals(mass_tol)

    for i, mb in enumerate(mass_bins):
        mb1 = slice_tag(mb, nd)
        mb2 = slice_tag(mb + mass_tol, nd)

        for level, arr in (("MS1", ms1), ("MS2", ms2)):
            if arr is None:
                continue
            filename = f"{level}_{mb1}_{mb2}.npy"
            np.save(os.path.join(output_folder, 'images', filename), arr[i])

def main(cfg):
    mzml_path = cfg["mzml_path"]
    output_folder = cfg["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    rt_range = cfg["rt_range"]
    rt_bin_size = cfg["rt_bin_size"]
    mz_range = cfg["mz_range"]
    mass_tol = mass_tol_of(cfg)
    length_spline = LENGTH_SPLINE
    acquisition = cfg.get("acquisition", "dia")

    final_files = selecting_files(mzml_path, cfg.get("files"))

    if up_to_date(cfg, final_files, output_folder):
        print(f"images in {output_folder} already match this config - skipping", flush=True)
        return

    print(f'Reading raw files ({acquisition.upper()}, m/z slice {mass_tol} Da)...', flush=True)
    ms1, ms2 = read_mzml_files(
        mzml_path, final_files, rt_bin_size, rt_range, mz_range, mass_tol,
        length_spline=length_spline, acquisition=acquisition,
        dia_channels=cfg.get("dia_channels"), dia_functions=cfg.get("dia_functions"),
    )

    saving_results(ms1, ms2, output_folder, mz_range, mass_tol)
    with open(os.path.join(output_folder, "params.json"), "w") as f:
        json.dump(_stamp(cfg, final_files), f, indent=1)

if __name__ == "__main__":
    config = parse_config_arg(description="Convert mzML files to MS1/MS2 .npy images.")
    main(config)
