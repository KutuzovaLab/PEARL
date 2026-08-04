import json
import pandas as pd

def load_mona_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def parse_mona(data):
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

    return pd.DataFrame(rows)
