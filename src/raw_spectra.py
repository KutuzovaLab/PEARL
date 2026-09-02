import os
import numpy as np
import pandas as pd
from src.common import MissingInput

_COLUMNS = ["RT", "mz", "inty"]

#Return (df_ms1, df_ms2) with columns RT, mz, inty.
def load_qc_spectra(mzml_path, mz_window=None, verbose=True):
    if not os.path.exists(mzml_path):
        raise MissingInput(mzml_path, "raw MS1/MS2 spectra for precise precursor m/z", "cell 9")

    import pyopenms as oms

    if verbose:
        size_mb = os.path.getsize(mzml_path) / 1e6
        print(f"  raw spectra: parsing {os.path.basename(mzml_path)} ({size_mb:.0f} MB) ...", flush=True)

    exp = oms.MSExperiment()
    oms.MzMLFile().load(mzml_path, exp)

    rows = {1: [[], [], []], 2: [[], [], []]}
    for spec in exp:
        level = spec.getMSLevel()
        if level not in rows:
            continue
        mzs, ints = spec.get_peaks()
        if len(mzs) == 0:
            continue
        if mz_window is not None:
            keep = (mzs >= mz_window[0]) & (mzs <= mz_window[1])
            if not keep.any():
                continue
            mzs, ints = mzs[keep], ints[keep]
        rt = spec.getRT()
        rows[level][0].append(np.full(len(mzs), rt, dtype=np.float64))
        rows[level][1].append(mzs.astype(np.float64))
        rows[level][2].append(ints.astype(np.float64))

    def build(level):
        cols = rows[level]
        if not cols[0]:
            return pd.DataFrame({c: np.array([], dtype=np.float64) for c in _COLUMNS})
        return pd.DataFrame({
            "RT": np.concatenate(cols[0]),
            "mz": np.concatenate(cols[1]),
            "inty": np.concatenate(cols[2]),
        })

    df_ms1, df_ms2 = build(1), build(2)
    if verbose:
        print(f"  raw spectra: {len(df_ms1)} MS1 peaks, {len(df_ms2)} MS2 peaks")

    return df_ms1, df_ms2

# Peaks inside an m/z x RT box
def slice_box(df, mz_range, rt_range):
    m = (
        (df["mz"] >= mz_range[0]) & (df["mz"] <= mz_range[1])
        & (df["RT"] >= rt_range[0]) & (df["RT"] <= rt_range[1])
    )
    return df[m].reset_index(drop=True)
