import json
import pandas as pd

with open("/home/frz603/DSP/mona_spectra/MoNA-export-LC-MS-MS_Positive_Mode.json", "r") as f:
    data = json.load(f)

rows = []

for entry in data:
    compounds = entry.get("compound", [])
    if not compounds:
        continue

    compound = compounds[0]  # Use the first compound in the list
    meta_data = compound.get("metaData", [])
    inchi_key = None
    precursor_mz = None
    rt = None  # retention time
    name = compound.get("name", None)  # compound name (optional)

    for meta in meta_data:
        key = meta.get("name", "").lower()
        value = meta.get("value", "")

        if key == "inchikey":
            inchi_key = value
        elif key == "precursor m/z":
            precursor_mz = value
        elif key in {"retention time", "retention time(s)", "retention_time"}:
            try:
                rt = float(value)
            except ValueError:
                rt = value  # fallback to raw value if not a float

    spectrum_str = entry.get("spectrum") or compound.get("spectrum")
    if spectrum_str:
        spectrum = [
            tuple(map(float, pair.split(":")))
            for pair in spectrum_str.strip().split()
            if ":" in pair
        ]
    else:
        spectrum = None

    rows.append({
        "inchi_key": inchi_key,
        "precursor_mz": precursor_mz,
        "rt": rt,
        "name": name,
        "spectrum": spectrum
    })

df = pd.DataFrame(rows)
#df = df.dropna(subset=["inchi_key", "spectrum"]).reset_index(drop=True)

df.to_csv("/home/frz603/DSP/mona_spectra/lc_ms_ms_positive_all.csv", index = False)
print(len(df))