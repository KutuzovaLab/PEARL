import argparse
import os
import zipfile
import numpy as np
import pandas as pd
from src.common import MissingInput

_RT_KEYS = {"retention time", "retention time(s)", "retention_time"}
_PRECURSOR_KEYS = {"precursor m/z", "precursor mz", "precursor_m/z"}

# Yield a binary file object for the export, transparently unzipping.
def _open_stream(path):
    if not os.path.exists(path):
        raise MissingInput(path, "MoNA reference spectra for MS2 match factor", "manuscript: Validation metrics")
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.lower().endswith(".json")]
        if not names:
            raise ValueError(f"{path} is a zip but contains no .json member: {z.namelist()}")
        return z.open(names[0]), z
    return open(path, "rb"), None

# MoNA's 'mz:intensity mz:intensity ...' string -> (mz array, intensity array).
def parse_spectrum(spectrum_str):
    mzs, ints = [], []
    for pair in str(spectrum_str).strip().split():
        if ":" not in pair:
            continue
        a, b = pair.split(":", 1)
        try:
            mzs.append(float(a)); ints.append(float(b))
        except ValueError:
            continue
    return np.asarray(mzs, dtype=float), np.asarray(ints, dtype=float)

def build_library(export_path, keep_inchikeys=None, verbose=True):
    import ijson

    keep = None
    if keep_inchikeys is not None:
        keep = {str(k).strip().upper() for k in keep_inchikeys if isinstance(k, str) and k.strip()}
        if verbose:
            print(f"  restricting MoNA to {len(keep)} InChIKeys")

    stream, zf = _open_stream(export_path)
    rows, n_seen = [], 0
    try:
        for entry in ijson.items(stream, "item", use_float=True):
            n_seen += 1
            if verbose and n_seen % 200_000 == 0:
                print(f"    {n_seen:,} records scanned, {len(rows):,} kept", flush=True)

            compounds = entry.get("compound") or []
            if not compounds:
                continue
            compound = compounds[0]

            inchikey = compound.get("inchiKey")
            if not inchikey:
                for meta in compound.get("metaData") or []:
                    if str(meta.get("name", "")).lower() == "inchikey":
                        inchikey = meta.get("value")
                        break

            if keep is not None and (not inchikey or str(inchikey).strip().upper() not in keep):
                continue

            names = compound.get("names") or []
            name = names[0].get("name") if names and isinstance(names[0], dict) else None

            precursor = rt = ms_level = precursor_type = instrument_type = None
            for meta in entry.get("metaData") or []:
                key = str(meta.get("name", "")).lower()
                val = meta.get("value")
                if key in _PRECURSOR_KEYS:
                    precursor = val
                elif key in _RT_KEYS:
                    rt = val
                elif key == "ms level":
                    ms_level = val
                elif key == "precursor type":
                    precursor_type = val
                elif key == "instrument type":
                    instrument_type = val

            spectrum_str = entry.get("spectrum") or compound.get("spectrum")
            if not spectrum_str:
                continue
            mz, inten = parse_spectrum(spectrum_str)
            if len(mz) == 0:
                continue

            rows.append({
                "inchikey": str(inchikey).strip().upper(),
                "name": name,
                "precursor_mz": pd.to_numeric(precursor, errors="coerce"),
                "rt": pd.to_numeric(rt, errors="coerce"),
                "ms_level": ms_level,
                "precursor_type": precursor_type,
                "instrument_type": instrument_type,
                "mz": mz,
                "intensity": inten,
            })
    finally:
        stream.close()
        if zf is not None:
            zf.close()

    lib = pd.DataFrame(rows, columns=["inchikey", "name", "precursor_mz", "rt", "ms_level",
                                      "precursor_type", "instrument_type", "mz", "intensity"])
    if verbose:
        print(f"  MoNA: {n_seen:,} records scanned, {len(lib):,} spectra kept "
              f"({lib['inchikey'].nunique()} distinct InChIKeys)")
    return lib

def main():
    ap = argparse.ArgumentParser(description="Build a compact MoNA reference library.")
    ap.add_argument("--export", required=True, help="MoNA export (.json or .zip).")
    ap.add_argument("--out", required=True, help="Output .pkl for the parsed library.")
    ap.add_argument("--inchikeys-from", default=None,
                    help="Optional compound_info.pkl / .xlsx whose 'inchikey' column limits the library.")
    args = ap.parse_args()

    keep = None
    if args.inchikeys_from:
        src = (pd.read_pickle(args.inchikeys_from) if args.inchikeys_from.endswith(".pkl")
               else pd.read_excel(args.inchikeys_from))
        keep = src["inchikey"].dropna().unique()

    lib = build_library(args.export, keep_inchikeys=keep)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lib.to_pickle(args.out)
    print(f"written: {args.out}  ({len(lib)} spectra)")

if __name__ == "__main__":
    main()