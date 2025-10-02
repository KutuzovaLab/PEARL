#from matchms.importing import load_from_msp
import json
import os
import re

# reading library
msp_file = "/home/frz603/DSP/matrix_standards/SSLibrary_P.msp"
#library = list(load_from_msp(msp_file))
library_path = '/home/frz603/DSP/matrix_standards/library'
os.makedirs(library_path, exist_ok=True)

def sanitize_filename(name):
    """Replace illegal characters in filenames with underscore."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def stream_msp(filepath):
    """Generator that yields one spectrum (as dict) at a time from an MSP file."""
    spectrum = {"metadata": {}, "ms2": []}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # blank line → end of block
                if spectrum["metadata"] or spectrum["ms2"]:
                    yield spectrum
                spectrum = {"metadata": {}, "ms2": []}
            elif ":" in line:  # metadata line
                key, value = line.split(":", 1)
                spectrum["metadata"][key.strip().lower()] = value.strip()
            else:  # MS² peak line
                try:
                    mz, intensity, *rest = line.split()
                    spectrum["ms2"].append((float(mz), float(intensity)))
                except ValueError:
                    pass
        # yield last spectrum if file doesn’t end with blank line
        if spectrum["metadata"] or spectrum["ms2"]:
            yield spectrum

# Example: Inspect first 2 spectra without loading the whole file
for i, s in enumerate(stream_msp(msp_file)):
    print(f"\n--- Compound {i+1} ---")
    print("Metadata:", s["metadata"])
    print("First 5 MS² peaks:", s["ms2"][:5])
    if i >= 1:  # stop after 2 compounds
        break

print('Starting json conversion now!')
# Example: Save each spectrum as individual JSON file
for i, s in enumerate(stream_msp(msp_file)):
    name = s["metadata"].get("name", f"compound_{i}")
    safe_name = sanitize_filename(name)
    filename = f"{library_path}/{i}_{safe_name}.json"  # index guarantees uniqueness
    print(filename)
    with open(filename, "w") as f:
        json.dump(s, f)

# once the previous is done, doesnt have to be repeated
