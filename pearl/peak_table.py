import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import csv
import glob
from collections import defaultdict

import numpy as np
import pyopenms as oms

from pearl.common import parse_config_arg
from pearl.mzml_to_npy import selecting_files


BLOB = re.compile(r"^(?P<prefix>.+)_(?P<level>MS[12])_"
                  r"(?P<a>\d+)_(?P<b>\d+)_(?P<c>\d+)_(?P<d>\d+)_(?P<idx>\d+)\.npy$")

def parse_blob(name):
    m = BLOB.match(os.path.basename(name))
    if not m:
        return None
    g = m.groupdict()
    return dict(name=os.path.basename(name), level=g["level"], idx=int(g["idx"]),
                mz_lo=float(f"{g['a']}.{g['b']}"), mz_hi=float(f"{g['c']}.{g['d']}"))

def blob_window(arr, rt_bins):
    cols = np.unique(np.nonzero(arr > 0)[1])
    if cols.size == 0:
        return None
    lo, hi = int(cols.min()), int(cols.max())
    return dict(rt_lo=float(rt_bins[lo]), rt_hi=float(rt_bins[hi]),
                sample=int(np.argmax(arr.max(axis=1))),
                intensity=float(arr.max()))

def widen_ms2(blobs, rt_bin_size):
    for b in blobs.values():
        if b["level"] == "MS2":
            b["rt_hi"] += rt_bin_size


def _wanted(level, sp, acquisition):
    if acquisition == "dia":
        fid = sp.getNativeID().split(" ", 1)[0]
        if fid.startswith("function="):
            return fid == ("function=1" if level == "MS1" else "function=2")
    return sp.getMSLevel() == (1 if level == "MS1" else 2)

def refine(mzml_path, stems, acquisition, tasks):
    if isinstance(stems, str):
        stems = [stems]
    out = {}
    for stem in stems:
        _refine_one(os.path.join(mzml_path, stem) + ".mzML", acquisition, tasks, out)
    return out


def _refine_one(path, acquisition, tasks, out):
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path, exp)
    for sp in exp:
        rt = sp.getRT()
        mz, it = sp.get_peaks()
        if mz.shape[0] == 0:
            continue
        for key, t in tasks.items():
            if not (t["rt_lo"] <= rt <= t["rt_hi"]):
                continue
            if not _wanted(t["level"], sp, acquisition):
                continue
            lo = np.searchsorted(mz, t["mz_lo"], "left")
            hi = np.searchsorted(mz, t["mz_hi"], "right")
            if hi <= lo:
                continue
            j = lo + int(np.argmax(it[lo:hi]))
            best = out.get(key)
            if best is None or it[j] > best[1]:
                out[key] = (float(mz[j]), float(it[j]))

def main(cfg):
    img = cfg["images"]
    blob_dir = os.path.join(cfg["labels"]["output_dir"], "single_labels_overlap")
    matches_csv = os.path.join(cfg["matching"]["output_dir"], "final_matches.csv")
    out_csv = cfg.get("peaks", {}).get("output") or os.path.join(
        os.path.dirname(cfg["labels"]["output_dir"]), "peaks.csv")

    acquisition = img.get("acquisition", "dia")
    rt_bins = np.arange(img["rt_range"][0],
                        img["rt_range"][1] + img["rt_bin_size"], img["rt_bin_size"])
    files = selecting_files(img["mzml_path"], img.get("files"))

    blobs = {}
    for p in sorted(glob.glob(os.path.join(blob_dir, "*.npy"))):
        b = parse_blob(p)
        if b is None:
            continue
        w = blob_window(np.load(p), rt_bins)
        if w is None:
            continue
        b.update(w)
        blobs[b["name"]] = b
    if not blobs:
        raise SystemExit(f"no blobs found in {blob_dir} - run extracting_labels_erosion first")

    widen_ms2(blobs, img["rt_bin_size"])

    ms1 = {k: v for k, v in blobs.items() if v["level"] == "MS1"}
    print(f"peak_table: {len(ms1)} MS1 peaks, {len(blobs) - len(ms1)} MS2 fragments", flush=True)

    pairs = defaultdict(list)
    if os.path.exists(matches_csv):
        with open(matches_csv) as f:
            for row in csv.DictReader(f):
                if row["ms1_name"] in blobs and row["ms2_name"] in blobs:
                    pairs[row["ms1_name"]].append(row["ms2_name"])
    elif acquisition != "dda":
        print(f"peak_table: WARNING - no {matches_csv}; MS2 column will be empty", flush=True)


    reference = cfg.get("peaks", {}).get("qc_sample")
    if reference:
        if isinstance(reference, str):
            reference = [reference]
        missing = [r for r in reference if r not in files]
        if missing:
            raise SystemExit(f"peaks.qc_sample not among the {len(files)} selected "
                             f"mzML in {img['mzml_path']}: {', '.join(missing)}")
        why = "from config"
    else:
        totals = defaultdict(float)
        for b in blobs.values():
            totals[files[b["sample"]]] += b["intensity"]
        reference = [max(totals, key=totals.get)]
        why = "auto-selected: most blob signal"
    print(f"  QC sample(s): {', '.join(reference)} ({why})", flush=True)
    refined = refine(img["mzml_path"], reference, acquisition, blobs)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    n_frag = 0
    dropped = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(["MS1 m/z", "MS1 intensity", "mz min", "mz max",
                    "RT", "RT min", "RT max", "MS2 spectrum"])
        for name in sorted(ms1):
            b = ms1[name]
            if name not in refined:

                dropped += 1
                continue
            mz, inten = refined[name]
            frags = []
            for ms2 in sorted(set(pairs.get(name, []))):
                if ms2 in refined:
                    fmz, fint = refined[ms2]
                    frags.append(f"{fmz:.3f};{fint:.1f}")
            n_frag += len(frags)
            w.writerow([f"{mz:.4f}" if np.isfinite(mz) else "",
                        f"{inten:.1f}" if np.isfinite(inten) else "",
                        f"{b['mz_lo']:.1f}", f"{b['mz_hi']:.1f}",
                        f"{(b['rt_lo'] + b['rt_hi']) / 2.0:.2f}",
                        f"{b['rt_lo']:.2f}", f"{b['rt_hi']:.2f}",
                        " ".join(frags)])

    print(f"peak_table: wrote {out_csv} ({len(ms1) - dropped} peaks, {n_frag} "
          f"fragment entries; {dropped} peaks dropped - no signal in the QC sample)",
          flush=True)

if __name__ == "__main__":
    main(parse_config_arg(description="Write the identified-peak table."))
