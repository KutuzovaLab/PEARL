# VIRTUAL_ENV=.venv_yucca uv pip install git+https://github.com/pluskal-lab/DreaMS.git

import os
import re
import requests
import numpy as np
import pandas as pd
from src.common import MissingInput
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS

DREAMS_INSTALL_HINT = (
    "DreaMS is not installed. Use:\n"
    "    VIRTUAL_ENV=.venv_yucca uv pip install git+https://github.com/pluskal-lab/DreaMS.git"
)

# MS-DIAL 'mz:int mz:int' -> (2, n) float32 array, or None.
def parse_msdial_spectrum(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    mzs, ints = [], []
    for pair in raw.strip().split():
        parts = pair.split(":")
        if len(parts) == 2:
            try:
                mzs.append(float(parts[0])); ints.append(float(parts[1]))
            except ValueError:
                pass
    return np.array([mzs, ints], dtype=np.float32) if mzs else None

# numpy repr string (as written into the xlsx) -> 1-D float array.
def parse_array(raw):
    if isinstance(raw, (list, tuple, np.ndarray)):
        return np.asarray(raw, dtype=float)
    if not isinstance(raw, str):
        return None
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    return np.array([float(v) for v in vals], dtype=float) if vals else None

# (2, n) array from the ms2_mz / ms2_int columns.
def unet_spectrum(row):
    mz, inten = parse_array(row.get("ms2_mz")), parse_array(row.get("ms2_int"))
    if mz is None or inten is None or len(mz) == 0 or len(mz) != len(inten):
        return None
    return np.array([mz, inten], dtype=np.float32)

# InChIKey -> canonical SMILES via PubChem
def add_ground_truth_smiles(df, cache_path=None, verbose=True):
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = pd.read_pickle(cache_path)

    keys = [k for k in df["inchikey"].dropna().unique() if k not in cache]
    if verbose and keys:
        print(f"  PubChem: resolving {len(keys)} InChIKeys ({len(cache)} cached)")
    for k in keys:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{k}"
               f"/property/CanonicalSMILES/TXT")
        try:
            r = requests.get(url, timeout=30)
            cache[k] = r.text.strip() if r.status_code == 200 else None
        except Exception:
            cache[k] = None

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        pd.to_pickle(cache, cache_path)

    df = df.copy()
    df["ground_truth_SMILES"] = df["inchikey"].map(cache)
    if verbose:
        print(f"  PubChem: {df['ground_truth_SMILES'].notna().sum()}/{len(df)} rows resolved")
    return df

# Tanimoto / Dice / MCS between the true structure and each predicted structure.
def add_structural_similarity(df, sources=("msdial", "ms2cols"), mcs_timeout=5, verbose=True):
    RDLogger.DisableLog("rdApp.*")

    def fp(smiles):
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048,
                                                     useChirality=True) if mol else None

    def mcs(s1, s2):
        # NaN is a float and is truthy, so an isinstance check is required here.
        if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
            return None
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None:
            return None
        res = rdFMCS.FindMCS((m1, m2), timeout=mcs_timeout,
                             ringMatchesRingOnly=True, completeRingsOnly=True)
        n = res.numAtoms
        return 0.0 if n == 0 else n / (m1.GetNumAtoms() + m2.GetNumAtoms() - n)

    df = df.copy()
    for name in sources:
        col = f"predicted_SMILES_{name}"
        if col not in df.columns:
            continue
        gt, pred = df["ground_truth_SMILES"], df[col]
        fps = [(fp(a), fp(b)) for a, b in zip(gt, pred)]
        df[f"Tanimoto_{name}"] = [DataStructs.TanimotoSimilarity(a, b) if a and b else None
                                  for a, b in fps]
        df[f"DICE_{name}"] = [DataStructs.DiceSimilarity(a, b) if a and b else None
                              for a, b in fps]
        df[f"MCS_{name}"] = [mcs(a, b) for a, b in zip(gt, pred)]
        if verbose:
            n = df[f"Tanimoto_{name}"].notna().sum()
            print(f"  structural similarity [{name}]: scored {n}/{len(df)} rows")
    return df

# Top-1 DreaMS cosine similarity against the MassSpecGym reference embeddings.
def add_dreams_similarity(df, library_path, tmp_dir="results/dreams_tmp", verbose=True):
    try:
        from dreams.api import dreams_embeddings
        from dreams.utils.data import MSData
        from dreams.definitions import DREAMS_EMBEDDING
    except ImportError as exc:
        raise MissingInput("dreams (python package)", DREAMS_INSTALL_HINT,
                           "manuscript: Validation metrics") from exc

    if not os.path.exists(library_path):
        raise MissingInput(library_path, "MassSpecGym DreaMS reference embeddings",
                           "manuscript: Validation metrics")

    from sklearn.metrics.pairwise import cosine_similarity
    os.makedirs(tmp_dir, exist_ok=True)

    msdata_lib = MSData.load(library_path)
    embs_lib = msdata_lib[DREAMS_EMBEDDING]

    df = df.copy()
    builders = {
        "msdial": lambda r: parse_msdial_spectrum(r.get("MsDial MS2")),
        "ms2cols": unet_spectrum,
    }
    for name, build in builders.items():
        rows = []
        for idx, row in df.iterrows():
            spec = build(row)
            if spec is None:
                continue
            rows.append({"spectrum": spec,
                         "precursor_mz": pd.to_numeric(row.get("precursor mass"), errors="coerce"),
                         "scan_number": idx})
        if not rows:
            if verbose:
                print(f"  DreaMS [{name}]: no usable spectra, skipped")
            continue

        built = pd.DataFrame(rows)
        pkl = os.path.join(tmp_dir, f"{name}.pkl")
        built.to_pickle(pkl)
        embs = dreams_embeddings(MSData.from_pandas(pkl, in_mem=True))

        sims = cosine_similarity(embs, embs_lib)
        top1 = np.argmax(sims, axis=1)
        best = sims[np.arange(len(top1)), top1]

        df[f"predicted_SMILES_{name}"] = None
        df[f"DreaMS_similarity_{name}"] = np.nan
        for i, scan in enumerate(built["scan_number"]):
            df.at[scan, f"predicted_SMILES_{name}"] = msdata_lib.get_smiles(int(top1[i]))
            df.at[scan, f"DreaMS_similarity_{name}"] = float(best[i])
        if verbose:
            print(f"  DreaMS [{name}]: scored {len(built)}/{len(df)} rows")
    return df
