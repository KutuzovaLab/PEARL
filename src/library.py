import re

def sanitize_filename(name):
    """Replace illegal characters in filenames with underscore."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def stream_msp(filepath):
    """Generator that yields one spectrum (as dict) at a time from an MSP file."""
    spectrum = {"metadata": {}, "ms2": []}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # blank line -> end of block
                if spectrum["metadata"] or spectrum["ms2"]:
                    yield spectrum
                spectrum = {"metadata": {}, "ms2": []}
            elif ":" in line:  # metadata line
                key, value = line.split(":", 1)
                spectrum["metadata"][key.strip().lower()] = value.strip()
            else:  # MS2 peak line
                try:
                    mz, intensity, *rest = line.split()
                    spectrum["ms2"].append((float(mz), float(intensity)))
                except ValueError:
                    pass
        # yield last spectrum if file doesn't end with blank line
        if spectrum["metadata"] or spectrum["ms2"]:
            yield spectrum
