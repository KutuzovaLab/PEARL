"""Entry point: convert an MSP spectral library into per-spectrum JSON files.

Usage:
    python library.py --config configs/config_library.yaml
"""

import json
import os

from src.config import parse_config_arg
from src.library import sanitize_filename, stream_msp


def main(cfg):
    msp_file = cfg["msp_file"]
    library_path = cfg["library_path"]
    preview = cfg.get("preview", 2)
    os.makedirs(library_path, exist_ok=True)

    # Inspect the first few spectra without loading the whole file
    for i, s in enumerate(stream_msp(msp_file)):
        print(f"\n--- Compound {i+1} ---")
        print("Metadata:", s["metadata"])
        print("First 5 MS2 peaks:", s["ms2"][:5])
        if i >= preview - 1:
            break

    print('Starting json conversion now!')
    # Save each spectrum as an individual JSON file
    for i, s in enumerate(stream_msp(msp_file)):
        name = s["metadata"].get("name", f"compound_{i}")
        safe_name = sanitize_filename(name)
        filename = f"{library_path}/{i}_{safe_name}.json"  # index guarantees uniqueness
        print(filename)
        with open(filename, "w") as f:
            json.dump(s, f)


if __name__ == "__main__":
    config = parse_config_arg(description="Convert an MSP library to per-spectrum JSON.")
    main(config)
