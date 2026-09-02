import os
import numpy as np
import pandas as pd

from src.common import MissingInput
from src.labels_io import label_rt_extent, labels_covering_mz, parse_label_name
from src.raw_spectra import slice_box
from src.reference_methods import META_COLUMNS

LIB_SOURCES = {"UNet": "unet", "MsDial": "msdial"}

COMPOUND_INFO_COLUMNS = [
    "Compound", "MS1 label", "precursor mass", "precursor int", "expected mass",
    "mz min", "mz max", "expected RT", "RT min", "RT max",
    "inchikey", "MsDial MS2", "formula", "adduct",
]

COMPARISON_COLUMNS = ["Compound", "UNet height", "Software height", "Concentration", "MS1 label"]

# Pearson r between U-Net and reference peak heights, per compound/label pair.
def pearson_per_compound(comparison_table):
    rows = []
    for (compound, label), sub in comparison_table.groupby(["Compound", "MS1 label"], sort=False):
        x = pd.to_numeric(sub["Software height"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["UNet height"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        # Pearson is undefined when either series is constant.
        r = (float(np.corrcoef(x[ok], y[ok])[0, 1])
             if ok.sum() >= 2 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0 else np.nan)
        rows.append({"Compound": compound, "MS1 label": label, "n_samples": int(ok.sum()), "pearson_r": r})
    return pd.DataFrame(rows, columns=["Compound", "MS1 label", "n_samples", "pearson_r"])

 #Return (compound_info, comparison_table, not_matched)
def match_compounds(reference, single_labels_dir, rt_bins, df_ms1, verbose=True):
    table = reference.table
    concentrations = reference.concentrations
    n_meta = len(META_COLUMNS)

    label_files = sorted(f for f in os.listdir(single_labels_dir) if f.endswith(".npy"))
    if not label_files:
        raise FileNotFoundError(
            f"no .npy single labels in {single_labels_dir} - did extracting_labels_erosion produce anything?"
        )

    info_rows, comparison_rows, unmatched_idx = [], [], []
    n_label_hits = 0

    for i in range(len(table)):
        expected_mz = table["mz"].iloc[i]
        expected_rt = table["RT"].iloc[i]
        reported_int = table.iloc[i, n_meta:].to_numpy()
        adduct = table["adduct"].iloc[i]
        compound = table["Compound"].iloc[i]

        candidates = labels_covering_mz(single_labels_dir, expected_mz, "MS1", files=label_files)
        matched = False

        for name in candidates:
            img = np.load(os.path.join(single_labels_dir, name))
            extent = label_rt_extent(img, rt_bins)
            if extent is None:
                continue
            if not (extent[0] <= expected_rt <= extent[1]):
                continue

            n_label_hits += 1
            _, _, mz_min, mz_max, _ = parse_label_name(name)
            mph = np.max(img, axis=1)  # per-sample peak height

            mass_slice = slice_box(df_ms1, [mz_min, mz_max], extent)
            if len(mass_slice) == 0:
                continue

            matched = True
            peak = int(np.argmax(mass_slice["inty"].to_numpy()))
            info_rows.append({
                "Compound": compound,
                "MS1 label": name,
                "precursor mass": mass_slice.loc[peak, "mz"],
                "precursor int": float(np.max(mass_slice["inty"])),
                "expected mass": expected_mz,
                "mz min": mz_min,
                "mz max": mz_max,
                "expected RT": expected_rt,
                "RT min": extent[0],
                "RT max": extent[1],
                "inchikey": table["inchikey"].iloc[i],
                "MsDial MS2": table["MS2"].iloc[i],
                "formula": table["formula"].iloc[i],
                "adduct": adduct,
            })

            comparison_rows.append(pd.DataFrame({
                "Compound": compound,
                "UNet height": mph,
                "Software height": reported_int,
                "Concentration": concentrations,
                "MS1 label": name,
            }))

        if not matched:
            unmatched_idx.append(i)

    compound_info = pd.DataFrame(info_rows, columns=COMPOUND_INFO_COLUMNS)
    comparison_table = (
        pd.concat(comparison_rows, ignore_index=True)
        if comparison_rows else pd.DataFrame(columns=COMPARISON_COLUMNS)
    )
    not_matched = table.iloc[unmatched_idx].reset_index(drop=True)

    if verbose:
        n = len(table)
        pct = (len(compound_info) / n * 100) if n else 0.0
        print(f"  matched {len(compound_info)} compound/label pairs from {n} compounds ({pct:.1f}%)")
        print(f"  {n_label_hits} label hits, {len(not_matched)} compounds with no match")

    return compound_info, comparison_table, not_matched

# final_matches.csv / .xlsx from matching_MS1_MS2.
def load_clusters(path):
    if not os.path.exists(path):
        raise MissingInput(path, "MS1<->MS2 cluster assignments", "cells 3/4 (`clusters`)")
    if path.endswith((".xlsx", ".xls")):
        clusters = pd.read_excel(path, header=0)
    else:
        clusters = pd.read_csv(path, header=0)
    missing = {"ms1_name", "ms2_name"} - set(clusters.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    return clusters

# MsDial's 'mz:intensity mz:intensity ...'
def parse_msdial_ms2(ms2_string):
    mzs, ints = [], []
    for token in str(ms2_string).split():
        if ":" not in token:
            continue
        mz, inten = token.split(":", 1)
        try:
            mzs.append(float(mz))
            ints.append(float(inten))
        except ValueError:
            continue
    return np.asarray(mzs, dtype=float), np.asarray(ints, dtype=float)

# Add `ms2_mz` / `ms2_int` columns to compound_info
def attach_unet_ms2(compound_info, clusters, single_labels_dir, rt_bins, df_ms2, verbose=True):
    compound_info = compound_info.copy()
    col_mz, col_int = [], []

    n_empty_box = 0
    n_missing_file = 0

    for i in range(len(compound_info)):
        ms1_name = compound_info["MS1 label"].iloc[i]
        ms2_names = sorted(clusters.loc[clusters["ms1_name"] == ms1_name, "ms2_name"].to_numpy())

        ms2_mz, ms2_int = [], []
        for name in ms2_names:
            path = os.path.join(single_labels_dir, name)
            if not os.path.exists(path):
                n_missing_file += 1
                continue
            _, _, mz_min, mz_max, _ = parse_label_name(name)
            extent = label_rt_extent(np.load(path), rt_bins)
            if extent is None:
                continue
            mass_slice = slice_box(df_ms2, [mz_min, mz_max], extent)
            if len(mass_slice) == 0:
                n_empty_box += 1
                continue
            peak = int(np.argmax(mass_slice["inty"].to_numpy()))
            ms2_mz.append(float(mass_slice["mz"].iloc[peak]))
            ms2_int.append(float(np.max(mass_slice["inty"])))

        col_mz.append(np.round(np.array(ms2_mz, dtype=float), 3))
        col_int.append(np.array(ms2_int, dtype=float))

    compound_info["ms2_mz"] = pd.Series(col_mz, index=compound_info.index, dtype=object)
    compound_info["ms2_int"] = pd.Series(col_int, index=compound_info.index, dtype=object)

    if verbose:
        sizes = [len(np.atleast_1d(x)) for x in compound_info["ms2_mz"]]
        print(f"  UNet MS2: {sum(sizes)} fragments over {len(sizes)} compound rows "
              f"(median {int(np.median(sizes)) if sizes else 0} per compound)")
        if n_empty_box:
            print(f"  note: {n_empty_box} MS2 label(s) had no raw peaks in their m/z x RT box, skipped")
        if n_missing_file:
            print(f"  note: {n_missing_file} clustered MS2 label file(s) not found on disk, skipped")

    return compound_info

LIB_SOURCES = {"UNet": "unet", "MsDial": "msdial"}

def _spectrum(mz, intensities):
    from matchms import Spectrum
    order = np.argsort(mz)
    return Spectrum(mz=np.asarray(mz, dtype=float)[order],
                    intensities=np.asarray(intensities, dtype=float)[order],
                    metadata={})

# Cosine similarity and match count between two (mz, int) frames
def get_match_factor(sp1, sp2, tol=0.1):
    from matchms.similarity import CosineGreedy
    if len(sp1) == 0 or len(sp2) == 0:
        return np.nan, 0
    score = CosineGreedy(tolerance=tol).pair(
        _spectrum(sp1["mz"].to_numpy(), sp1["int"].to_numpy()),
        _spectrum(sp2["mz"].to_numpy(), sp2["int"].to_numpy()),
    )
    return float(score["score"]), int(score["matches"])

def _unet_frame(compound_info, i, cap_mz=None):
    df = pd.DataFrame({
        "mz": np.asarray(compound_info["ms2_mz"].iloc[i], dtype=float),
        "int": np.asarray(compound_info["ms2_int"].iloc[i], dtype=float),
    })
    if cap_mz is not None and len(df):
        df = df[df["mz"] <= cap_mz]
    return df.reset_index(drop=True)

# MF (UNet_MsDial) - cosine between U-Net and MsDial MS2
def add_msdial_cosine(compound_info, tol=0.5, verbose=True):
    compound_info = compound_info.copy()
    compound_info["MF (UNet_MsDial)"] = np.nan

    n_scored = 0
    for i in range(len(compound_info)):
        mz, inten = parse_msdial_ms2(compound_info["MsDial MS2"].iloc[i])
        if len(mz) == 0:
            continue
        msdial = pd.DataFrame({"mz": mz, "int": inten})
        unet = _unet_frame(compound_info, i, cap_mz=float(np.max(msdial["mz"])) + 1)
        if unet.empty:
            continue
        mf, _ = get_match_factor(unet, msdial, tol)
        compound_info.loc[i, "MF (UNet_MsDial)"] = mf
        n_scored += 1

    if verbose:
        print(f"  cosine vs MsDial: scored {n_scored}/{len(compound_info)} rows")
    return compound_info

def load_dda_fragments(path):
    if not path or not os.path.exists(str(path)):
        raise MissingInput(
            str(path), "DDA reference spectra for precision/recall/F1", "cell 35"
        )
    dda = pd.read_csv(path, header=0)
    required = {"Average Mz", "Average Rt(min)", "dda_ms2"}
    missing = required - set(dda.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    return dda

def _parse_dda_spectrum(cell):
    pairs = [p.split(",") for p in str(cell).split(";") if "," in p]
    if not pairs:
        return pd.DataFrame(columns=["mz", "int"])
    return pd.DataFrame({
        "mz": [float(p[0]) for p in pairs],
        "int": [float(p[1]) for p in pairs],
    })

# MF and precision/recall/F1 against DDA fragments for both methods
def add_dda_metrics(compound_info, dda_frag, tol=0.1, mz_tol=0.01, verbose=True):
    compound_info = compound_info.copy()
    for col in ["MF (UNet_DDA)", "MF (MsDial_DDA)",
                "UNet precision", "MsDial precision",
                "UNet recall", "MsDial recall",
                "UNet F1", "MsDial F1"]:
        compound_info[col] = np.nan

    n_scored = 0
    for i in range(len(compound_info)):
        hit = dda_frag[
            (np.abs(compound_info["precursor mass"].iloc[i] - dda_frag["Average Mz"]) < mz_tol)
            & (dda_frag["Average Rt(min)"] * 60 >= compound_info["RT min"].iloc[i])
            & (dda_frag["Average Rt(min)"] * 60 <= compound_info["RT max"].iloc[i])
        ].reset_index(drop=True)
        if hit.empty:
            continue

        dda_ms2 = _parse_dda_spectrum(hit["dda_ms2"].iloc[0])
        if dda_ms2.empty:
            continue

        mz, inten = parse_msdial_ms2(compound_info["MsDial MS2"].iloc[i])
        msdial_ms2 = pd.DataFrame({"mz": mz, "int": inten})
        cap = float(np.max(msdial_ms2["mz"])) + 1 if len(msdial_ms2) else None
        unet_ms2 = _unet_frame(compound_info, i, cap_mz=cap)

        mf_unet, n_unet = get_match_factor(unet_ms2, dda_ms2, tol)
        mf_msdial, n_msdial = get_match_factor(msdial_ms2, dda_ms2, tol)
        compound_info.loc[i, "MF (UNet_DDA)"] = mf_unet
        compound_info.loc[i, "MF (MsDial_DDA)"] = mf_msdial

        for label, n_match, n_pred in (("UNet", n_unet, len(unet_ms2)),
                                       ("MsDial", n_msdial, len(msdial_ms2))):
            tp = n_match
            fp = n_pred - tp
            fn = len(dda_ms2) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else np.nan
            compound_info.loc[i, f"{label} precision"] = prec
            compound_info.loc[i, f"{label} recall"] = rec
            compound_info.loc[i, f"{label} F1"] = f1
        n_scored += 1

    if verbose:
        print(f"  DDA metrics: scored {n_scored}/{len(compound_info)} rows")
    return compound_info

# Precision / recall / F1 for a predicted spectrum against a reference spectrum.
def _prf(n_match, n_pred, n_ref):
    tp = n_match
    fp = n_pred - tp
    fn = n_ref - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else np.nan
    return prec, rec, f1

# Restrict MoNA to spectra acquired under comparable conditions.
def filter_library(library, instrument_pattern=None, precursor_types=None, verbose=True):
    out = library
    if instrument_pattern:
        out = out[out["instrument_type"].astype(str)
                  .str.contains(instrument_pattern, case=False, na=False, regex=True)]
    if precursor_types:
        out = out[out["precursor_type"].astype(str).isin(list(precursor_types))]
    if verbose:
        print(f"  library filter: {len(library)} -> {len(out)} spectra "
              f"({out['inchikey'].nunique()} InChIKeys)")
    return out.reset_index(drop=True)

# Match factor / precision / recall / F1 against MoNA reference spectra.
def add_library_metrics(compound_info, library, tol=0.1, verbose=True):
    compound_info = compound_info.copy()
    cols = ["MF (UNet_lib)", "MF (MsDial_lib)", "lib entry (UNet)", "lib entry (MsDial)",
            "lib same entry", "lib n candidates",
            "UNet precision (lib)", "UNet recall (lib)", "UNet F1 (lib)",
            "MsDial precision (lib)", "MsDial recall (lib)", "MsDial F1 (lib)"]
    for c in cols:
        compound_info[c] = np.nan

    by_key = {k: g for k, g in library.groupby("inchikey", sort=False)} if len(library) else {}
    n_with_ref = 0

    for i in range(len(compound_info)):
        key = compound_info["inchikey"].iloc[i]
        key = str(key).strip().upper() if isinstance(key, str) else None
        cands = by_key.get(key)
        if cands is None or not len(cands):
            continue
        n_with_ref += 1
        compound_info.loc[i, "lib n candidates"] = len(cands)

        mz, inten = parse_msdial_ms2(compound_info["MsDial MS2"].iloc[i])
        spectra = {
            "UNet": _unet_frame(compound_info, i),
            "MsDial": pd.DataFrame({"mz": mz, "int": inten}),
        }

        best = {}
        for label, pred in spectra.items():
            if pred.empty:
                continue
            scores = []
            for j in range(len(cands)):
                ref = pd.DataFrame({"mz": cands["mz"].iloc[j], "int": cands["intensity"].iloc[j]})
                mf, n_match = get_match_factor(pred, ref, tol)
                scores.append((mf if mf == mf else -1.0, j, n_match, len(ref)))
            if not scores:
                continue
            mf, j, n_match, n_ref = max(scores)
            best[label] = j
            compound_info.loc[i, f"MF ({label}_lib)"] = mf if mf >= 0 else np.nan
            compound_info.loc[i, f"lib entry ({label})"] = j
            prec, rec, f1 = _prf(n_match, len(pred), n_ref)
            compound_info.loc[i, f"{label} precision (lib)"] = prec
            compound_info.loc[i, f"{label} recall (lib)"] = rec
            compound_info.loc[i, f"{label} F1 (lib)"] = f1

        if len(best) == 2:
            compound_info.loc[i, "lib same entry"] = float(best["UNet"] == best["MsDial"])

    if verbose:
        same = compound_info["lib same entry"]
        print(f"  MoNA library: {n_with_ref}/{len(compound_info)} compounds had >=1 reference spectrum; "
              f"{int((same == 1).sum())} chose the same best entry for both methods")
    return compound_info
