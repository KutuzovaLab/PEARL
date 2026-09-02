import os
import re
import numpy as np

MS_LEVELS = ("MS1", "MS2")

class LabelNameError(ValueError):
    """Raised when a label filename does not follow the expected convention."""

# Split a label filename into (prefix, ms_level, mz_min, mz_max, blob)
def parse_label_name(fname):
    stem = re.sub(r"\.npy$", "", os.path.basename(fname))
    parts = stem.split("_")

    if parts and parts[0] in MS_LEVELS:
        parts = [None] + parts

    if len(parts) not in (6, 7):
        raise LabelNameError(
            f"cannot parse '{fname}': expected 6 or 7 underscore-separated fields, got "
            f"{len(parts)} ({parts}). The case-id prefix must not contain underscores - "
            f"pass a prefix like 'standNeg' to taskconvert_infer."
        )

    prefix, ms_level = parts[0], parts[1]
    if ms_level not in MS_LEVELS:
        raise LabelNameError(f"cannot parse '{fname}': field 2 is '{ms_level}', expected one of {MS_LEVELS}")

    try:
        mz_min = float(f"{parts[2]}.{parts[3]}")
        mz_max = float(f"{parts[4]}.{parts[5]}")
    except ValueError as exc:
        raise LabelNameError(f"cannot parse m/z range from '{fname}': {exc}") from exc

    blob = int(parts[6]) if len(parts) == 7 else None
    return prefix, ms_level, mz_min, mz_max, blob


def make_rt_bins(rt_range, rt_bin_size):
    return np.arange(rt_range[0], rt_range[1] + rt_bin_size, rt_bin_size)

# rt_bins[np.min(rt_ind)] / rt_bins[np.max(rt_ind)] over np.where(img > 0)[1].
def label_rt_extent(img, rt_bins):
    if not (img > 0).any():
        return None
    rt_ind = np.unique(np.where(img > 0)[1])
    lo, hi = int(np.min(rt_ind)), int(np.max(rt_ind))
    if hi >= len(rt_bins):
        raise IndexError(
            f"label spans RT bin {hi} but rt_bins has only {len(rt_bins)} entries - "
            f"rt_range/rt_bin_size in the validation config do not match the images"
        )
    return [float(rt_bins[lo]), float(rt_bins[hi])]

# Labels whose m/z window contains `mz` (notebook cell 11's first filter).
def labels_covering_mz(folder, mz, ms_level="MS1", files=None):
    if files is None:
        files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    out = []
    for f in files:
        try:
            _, lvl, mz_min, mz_max = parse_label_name(f)[:4]
        except LabelNameError:
            continue
        if lvl == ms_level and mz_min <= mz <= mz_max:
            out.append(f)
    return out
