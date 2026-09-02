import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from src.common import MissingInput

META_COLUMNS = ["Compound", "mz", "RT", "adduct", "formula", "inchikey", "MS2"]

@dataclass
class ReferenceTable:
    name: str
    table: pd.DataFrame
    concentrations: list
    qc_mzml: str
    rt_range: list
    rt_bin_size: float
    deviations: list = field(default_factory=list)
    undetected: pd.DataFrame = None

    @property
    def height_columns(self):
        return list(self.table.columns[len(META_COLUMNS):])

    def summary(self):
        return (
            f"[{self.name}] {len(self.table)} compounds, "
            f"{len(self.height_columns)} sample columns, "
            f"{len(self.deviations)} deviation(s) from the notebook"
        )

def _require(path, purpose, notebook_ref):
    if not os.path.exists(path):
        raise MissingInput(path, purpose, notebook_ref)
    return path

def load_standneg(cfg):
    """Matrix standards, negative mode - Task024_standNeg (notebook cell 4)."""
    deviations = []

    results_path = _require(
        cfg["results_path"], "the MsDial standards result table", "cell 4"
    )
    compounds = pd.read_excel(results_path, skiprows=4, header=0)

    suspect_list = cfg.get("suspect_list_path")
    if suspect_list and os.path.exists(suspect_list):
        all_compounds = pd.read_excel(suspect_list, sheet_name=cfg.get("suspect_list_sheet", "SSList_NEG"))
        flavonoids = all_compounds.loc[all_compounds["Class"] == "Flavonoids", "INCHIKEY"].to_numpy()
        compounds = compounds.loc[compounds["INCHIKEY"].isin(flavonoids), :].reset_index(drop=True)
    elif cfg.get("inchikey_allowlist"):
        allow = list(cfg["inchikey_allowlist"])
        compounds = compounds.loc[compounds["INCHIKEY"].isin(allow), :].reset_index(drop=True)
        deviations.append(
            f"flavonoid filter approximated by a {len(allow)}-entry inchikey allowlist from the config "
            f"(suspect screening list not available)"
        )
    else:
        deviations.append(
            "flavonoid filter SKIPPED - 'Matrix_LCMS_Suspect Screening List final.xlsx' not available, "
            "so all compounds passing the Fill% threshold are kept. Compound set is WIDER than the notebook's."
        )

    fill_min = cfg.get("fill_min", 0.7)
    compounds = compounds[compounds["Fill %"] >= fill_min].reset_index(drop=True)

    metadata = compounds.iloc[:, np.r_[3, 2, 1, 4, 10, 12, 31]].copy()
    metadata.columns = META_COLUMNS
    metadata["RT"] = metadata["RT"] * 60  # minutes -> seconds

    peak_heights = compounds.iloc[:, 32:].sort_index(axis=1)
    table = pd.concat([metadata, peak_heights], axis=1)

    # Concentrations as reported by the chemist; first file in the sequence was a blank.
    raw_conc = pd.read_excel(results_path, nrows=5, header=None).iloc[0, 32:].to_numpy()
    concentrations = [0] + [
        float(str(x).replace("Cal_", "").replace("ppm", "")) for x in raw_conc[1:]
    ]

    return _finalize("standNeg", cfg, table, concentrations, deviations)

# RT / m/z filtering, concentration alignment and packaging, shared by all loaders.
def _finalize(name, cfg, table, concentrations, deviations):
    rt_range = cfg["rt_range"]
    before = len(table)
    table = table[table["RT"] >= rt_range[0]].reset_index(drop=True)
    if before != len(table):
        deviations.append(f"dropped {before - len(table)} compound(s) with RT below the image RT range")

    # Restrict to the m/z slice EIC images were actually generated for, so that
    # "not matched" means something. null = no restriction.
    mz_window = cfg.get("mz_window")
    if mz_window:
        before = len(table)
        table = table[(table["mz"] >= mz_window[0]) & (table["mz"] <= mz_window[1])].reset_index(drop=True)
        deviations.append(
            f"restricted to m/z {mz_window[0]}-{mz_window[1]} ({len(table)} of {before} compounds) "
            f"to match the EIC images that were generated"
        )

    n_samples = len(table.columns) - len(META_COLUMNS)
    if concentrations is None:
        concentrations = [None] * n_samples      # datasets without known concentrations
    if len(concentrations) != n_samples:
        raise ValueError(
            f"[{name}] {len(concentrations)} concentrations vs {n_samples} sample columns; "
            f"the calibration curve would be misaligned"
        )

    return ReferenceTable(
        name=name,
        table=table,
        concentrations=concentrations,
        qc_mzml=_require(cfg["qc_mzml"], "raw MS1/MS2 spectra for precise m/z", "cell 9"),
        rt_range=rt_range,
        rt_bin_size=cfg["rt_bin_size"],
        deviations=deviations,
    )

# Wheat metabolomics, DIA positive
def load_suspfinal(cfg):
    deviations = []
    results_path = _require(cfg["results_path"], "the MS-DIAL wheat result table", "cell 3")
    res = pd.read_csv(results_path, header=0, skiprows=4, low_memory=False)

    batch = cfg.get("batch", "Batch3")
    keep = np.asarray(res.columns.str.contains(batch))
    keep[:32] = True                               # first 32 columns are metadata
    res = res.loc[:, keep]

    n_all = len(res)
    res = res[(res["Fill %"] >= cfg.get("fill_min", 0.8))
              & (res["Metabolite name"] != "Unknown")].reset_index(drop=True)
    # Keep only IDs confirmed by MS2, not MS1-only matches.
    res = res[~res["Metabolite name"].str.contains("w/o MS2", na=False)].reset_index(drop=True)
    deviations.append(f"kept {len(res)} of {n_all} features (Fill% >= "
                      f"{cfg.get('fill_min', 0.8)}, named, MS2-confirmed)")

    metadata = res.iloc[:, np.r_[3, 2, 1, 4, 10, 12, 31]].copy()
    metadata.columns = META_COLUMNS
    metadata["RT"] = metadata["RT"] * 60
    peak_heights = res.iloc[:, 32:].sort_index(axis=1)
    table = pd.concat([metadata, peak_heights], axis=1)

    return _finalize("suspFinal", cfg, table, None, deviations)

# Wastewater, DDA positive, MZmine reference
def load_wastewater(cfg):
    deviations = []
    results_path = _require(cfg["results_path"], "the MZmine wastewater feature table", "cell 5")
    allf = pd.read_csv(results_path, header=0, low_memory=False)

    det = allf[~allf["Annotations"].isna()].reset_index(drop=True)
    det = det.iloc[:, 1:]                       # drop 'Filename'
    det = det.rename(columns={"Annotations": "Compound", "Confirmed by stantard": "formula"})
    det["RT"] = det["RT"] * 60
    for col in ("adduct", "MS2", "inchikey"):
        det[col] = np.nan
    deviations.append(f"{len(det)} annotated compounds of {len(allf)} aligned features")

    # The feature table and the mzML folder disagree by exactly one file each way.
    mzml_files = sorted(f for f in os.listdir(cfg["mzml_dir"]) if f.endswith(".mzML"))
    for extra in set(det.columns) & set(cfg.get("drop_samples", ["T3IN1_2.mzML"])):
        det = det.drop(columns=[extra])
        deviations.append(f"dropped sample column '{extra}' - no matching mzML file")
    for missing in [f for f in mzml_files if f not in det.columns]:
        det[missing] = np.nan
        deviations.append(f"added empty sample column '{missing}' - file present but absent from the table")

    table = pd.concat([det[META_COLUMNS], det[[f for f in mzml_files if f in det.columns]]], axis=1)
    return _finalize("wastewater", cfg, table, None, deviations)

# Fluorinated standard mixes, DDA positive, OpenMS reference
def load_flourid(cfg):
    deviations = []
    detected_path = _require(cfg["results_path"], "the OpenMS fluorine result table", "cell 6")
    standards_path = _require(cfg["standards_path"], "the fluorine standards list", "cell 6")

    # This workbook holds an integer too large for pandas' float coercion; read as text.
    detected = pd.read_excel(detected_path, header=0, dtype=str)
    for c in ("m/z", "Peak area", "RT (s)"):
        detected[c] = pd.to_numeric(detected[c], errors="coerce")

    table = detected[["Name", "Formula", "m/z", "RT (s)", "mix_cur"]].copy()
    table.columns = ["Compound", "formula", "mz", "RT", "mix_cur"]
    for col in ("adduct", "inchikey", "MS2"):
        table[col] = np.nan
    peak_area = pd.to_numeric(detected["Peak area"], errors="coerce")

    mzml_files = sorted(f for f in os.listdir(cfg["mzml_dir"]) if f.endswith(".mzML"))
    sample_cols = ["_".join(f.split("_")[:3]) for f in mzml_files]
    heights = pd.DataFrame(np.nan, index=table.index, columns=sample_cols)

    # OpenMS reported a peak area only at the highest concentration level.
    ref_conc = str(cfg.get("reported_concentration", 50))
    for i in range(len(table)):
        hit = [c for c in sample_cols
               if c.split("_")[1] == table["mix_cur"].iloc[i] and c.split("_")[2] == ref_conc]
        if hit:
            heights.loc[i, hit] = peak_area.iloc[i]
    deviations.append(f"OpenMS peak areas are only reported at {ref_conc} ug/L; other levels are NaN")

    stds = pd.read_excel(standards_path, header=0, dtype=str)
    stds = stds[["name", "molecular_formula", "exact_mass", "rt_min", "mix", "conc"]].copy()
    stds.columns = ["Compound", "formula", "mz", "RT", "mix_cur", "conc"]
    stds["mz"] = pd.to_numeric(stds["mz"], errors="coerce") + 1.0073
    stds["RT"] = pd.to_numeric(stds["RT"], errors="coerce") * 60
    stds = stds[~stds["Compound"].isin(table["Compound"])].reset_index(drop=True)
    stds["adduct"] = "[M+H]+"
    deviations.append(f"{len(stds)} standards on the list were not reported by OpenMS "
                      f"(candidates for additional detections)")

    table = pd.concat([table[META_COLUMNS], heights], axis=1)
    concentrations = [int(c.split("_")[2]) if c.split("_")[2].isnumeric() else None
                      for c in sample_cols]
    ref = _finalize("flourID", cfg, table, concentrations, deviations)
    ref.undetected = stds
    return ref

def _not_yet(branch, needed):
    def loader(cfg):
        raise MissingInput(
            needed,
            f"the {branch} reference table (loader not yet enabled - see plan)",
            "cells 3/5/6",
        )
    return loader

REGISTRY = {
    "standneg": load_standneg,
    "suspfinal": load_suspfinal,
    "wastewater": load_wastewater,
    "flourid": load_flourid,
}

def load_reference(cfg):
    branch = cfg["branch"].lower()
    if branch not in REGISTRY:
        raise ValueError(f"unknown branch '{branch}'; known: {sorted(REGISTRY)}")
    return REGISTRY[branch](cfg)