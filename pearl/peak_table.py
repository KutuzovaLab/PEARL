import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import csv
import glob
import bisect
from collections import defaultdict

import numpy as np

from pearl.common import parse_config_arg, mass_tol_of, mz_decimals
from pearl.constants import MS2_PPM, MS2_RT_TOL
from pearl.mzml_to_npy import load_mzml, selecting_files


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
    best, scans = {}, defaultdict(list)
    for stem in stems:
        _refine_one(os.path.join(mzml_path, stem) + ".mzML", acquisition, tasks, best, scans)
    return best, scans

def _refine_one(path, acquisition, tasks, out, scans):
    exp = load_mzml(path)
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
            if t["level"] == "MS2":
                scans[key].append((rt, float(mz[j]), float(it[j])))


def dia_fragment(entries, ref_mz, target_rt, ppm):
    tol = ppm * 1e-6 * ref_mz
    ok = [e for e in entries if abs(e[1] - ref_mz) <= tol]
    if not ok:
        return None
    rt, mz, inten = min(ok, key=lambda e: abs(e[0] - target_rt))
    return mz, inten

def _fragment_string(mz, inten):
    return " ".join(f"{m:.3f};{i:.1f}" for m, i in zip(mz, inten) if i > 0)

def iter_dda_ms2(path):
    exp = load_mzml(path)
    for i, sp in enumerate(exp):

        if sp.getMSLevel() != 2:
            continue

        precursors = sp.getPrecursors()
        if len(precursors) == 0:
            continue

        precursor_mz = precursors[0].getMZ()
        if not precursor_mz:
            continue
        mz, intensity = sp.get_peaks()

        yield {"scan": i, "precursor_mz": precursor_mz, "rt": sp.getRT(),
               "mz": mz, "intensity": intensity}


def dda_fragments(mzml_path, stems, blobs, refined, ppm, rt_tol):
    if isinstance(stems, str):
        stems = [stems]
    if not refined:
        return {}

    by_edge = defaultdict(list)
    width = 0.0
    for name, b in blobs.items():
        if name in refined:
            by_edge[b["mz_lo"]].append(name)
            width = max(width, b["mz_hi"] - b["mz_lo"])
    edges = sorted(by_edge)

    chosen = {}
    for stem in stems:
        for scan in iter_dda_ms2(os.path.join(mzml_path, stem) + ".mzML"):
            pmz, rt, mz, it = (scan["precursor_mz"], scan["rt"],
                               scan["mz"], scan["intensity"])
            if mz.shape[0] == 0:
                continue
            # the peaks whose slice could hold a mass within tolerance of this precursor
            tol = ppm * 1e-6 * pmz
            first = bisect.bisect_left(edges, pmz - width - tol)
            last = bisect.bisect_right(edges, pmz + tol)
            base = -float(it.max())
            fragments = None

            for e in range(first, last):
                for name in by_edge[edges[e]]:
                    ref_mz = refined[name][0]
                    off = abs(pmz - ref_mz)
                    if off > ppm * 1e-6 * ref_mz:
                        continue
                    b = blobs[name]
                    inside = b["rt_lo"] <= rt <= b["rt_hi"]
                    away = 0.0 if inside else min(abs(rt - b["rt_lo"]), abs(rt - b["rt_hi"]))
                    if away > rt_tol:
                        continue
                    
                    rank = (abs(rt - (b["rt_lo"] + b["rt_hi"]) / 2.0), base)
                    have = chosen.get(name)
                    if have is None or rank < have[0]:
                        if fragments is None:
                            fragments = _fragment_string(mz, it)
                        chosen[name] = (rank, fragments, off / ref_mz * 1e6, away, inside)

    return {name: v[1:] for name, v in chosen.items()}

def main(cfg):
    img = cfg["images"]
    peaks_cfg = cfg.get("peaks", {})
    blob_dir = os.path.join(cfg["labels"]["output_dir"], "single_labels_overlap")
    matches_csv = os.path.join(cfg["matching"]["output_dir"], "final_matches.csv")
    out_csv = peaks_cfg.get("output") or os.path.join(
        os.path.dirname(cfg["labels"]["output_dir"]), "peaks.csv")

    acquisition = img.get("acquisition", "dia")
    ms2_ppm = float(peaks_cfg.get("ms2_ppm", MS2_PPM))
    ms2_rt_tol = float(peaks_cfg.get("ms2_rt_tol", MS2_RT_TOL))
    nd = mz_decimals(mass_tol_of(img))
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


    reference = peaks_cfg.get("qc_sample")
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
    refined, ms2_scans = refine(img["mzml_path"], reference, acquisition, blobs)

    dda_ms2 = {}
    if acquisition == "dda":
        dda_ms2 = dda_fragments(img["mzml_path"], reference, ms1, refined,
                                ms2_ppm, ms2_rt_tol)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    n_frag = 0
    n_spectra = 0
    n_outside = 0
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
            centre = (b["rt_lo"] + b["rt_hi"]) / 2.0
            if acquisition == "dda":
                hit = dda_ms2.get(name)
                spectrum = hit[0] if hit else ""
                if hit:
                    n_spectra += 1
                    n_frag += spectrum.count(";")
                    n_outside += not hit[3]
            else:
                frags = []
                for ms2 in sorted(set(pairs.get(name, []))):
                    if ms2 not in refined:
                        continue
                    got = dia_fragment(ms2_scans.get(ms2, ()), refined[ms2][0],
                                       centre, ms2_ppm)
                    if got:
                        frags.append(f"{got[0]:.3f};{got[1]:.1f}")
                n_frag += len(frags)
                n_spectra += bool(frags)
                spectrum = " ".join(frags)
            w.writerow([f"{mz:.4f}" if np.isfinite(mz) else "",
                        f"{inten:.1f}" if np.isfinite(inten) else "",
                        f"{b['mz_lo']:.{nd}f}", f"{b['mz_hi']:.{nd}f}",
                        f"{centre:.2f}",
                        f"{b['rt_lo']:.2f}", f"{b['rt_hi']:.2f}",
                        spectrum])

    print(f"peak_table: wrote {out_csv} ({len(ms1) - dropped} peaks, {n_frag} "
          f"fragment entries; {dropped} peaks dropped - no signal in the QC sample)",
          flush=True)
    if acquisition == "dda":
        offsets = [v[1] for v in dda_ms2.values()]
        print(f"  MS2 from the raw files: {n_spectra} peaks matched a precursor within "
              f"{ms2_ppm:g} ppm ({n_outside} outside the peak's own RT window, "
              f"median offset {np.median(offsets) if offsets else 0:.1f} ppm)", flush=True)
    else:
        print(f"  MS2 from the matched fragment labels: {n_spectra} peaks "
              f"(fragments kept within {ms2_ppm:g} ppm of the label's mass)", flush=True)

if __name__ == "__main__":
    main(parse_config_arg(description="Write the identified-peak table."))