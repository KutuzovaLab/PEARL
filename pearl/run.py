import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import glob
import argparse
import subprocess

import yaml

from pearl.common import expected_eic_files
from pearl.constants import K, MASS_TOL, MS2_PPM, MS2_RT_TOL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = "hf://UffeDalgas/PEARL/best.ckpt"
DEMO_INPUT = "data/mzml/test"
DEMO_OUTPUT = "results/test"

DIA_ISOLATION_WIDTH = 5.0

def detect_acquisition(mzml_dir):
    import numpy as np
    import pyopenms as oms
    for path in sorted(glob.glob(os.path.join(mzml_dir, "*.mzML"))):
        od = oms.OnDiscMSExperiment()
        if not od.openFile(path):
            continue
        widths = []
        for sp in od.getMetaData():
            if sp.getNativeID().split(" ", 1)[0].startswith("function="):
                return "dia", "native IDs carry function="
            if sp.getMSLevel() == 2:
                for pr in sp.getPrecursors():
                    widths.append(pr.getIsolationWindowLowerOffset()
                                  + pr.getIsolationWindowUpperOffset())
        if widths:
            median = float(np.median(widths))
            if median >= DIA_ISOLATION_WIDTH:
                return "dia", f"MS2 isolation windows are {median:.1f} Da wide"
            return "dda", f"MS2 isolation windows are {median:.2f} Da wide"
    return "dda", "no MS2 spectra in any run"

def detect_rt_range(mzml_dir):
    import numpy as np
    import pyopenms as oms
    lo, hi = [], []
    for p in sorted(glob.glob(os.path.join(mzml_dir, "*.mzML"))):
        od = oms.OnDiscMSExperiment()
        if not od.openFile(p):
            continue
        rts = [sp.getRT() for sp in od.getMetaData()]
        if rts:
            lo.append(min(rts)); hi.append(max(rts))
    if not lo:
        raise SystemExit(f"could not read retention times from {mzml_dir}")
    return [float(np.floor(min(lo))), float(np.ceil(max(hi)))]

def check_runs(mzml_dir, names, flag, selected=None):
    if not names:
        return
    have = {f[:-5] for f in os.listdir(mzml_dir) if f.endswith(".mzML")}
    missing = [n for n in names if n not in have]
    if missing:
        raise SystemExit(f"{flag}: no such run in {mzml_dir}: {', '.join(missing)}")
    if selected:
        outside = [n for n in names if n not in set(selected)]
        if outside:
            raise SystemExit(f"{flag}: {', '.join(outside)} is not among the "
                             f"{len(selected)} run(s) selected by --files")

def read_file_list(files):
    if files and len(files) == 1 and files[0].lower().endswith(".txt"):
        with open(files[0]) as f:
            files = [ln.strip() for ln in f if ln.strip()]
    return [f[:-5] if f.endswith(".mzML") else f for f in files] if files else None

def apply_demo_defaults(a):
    """Public MTBLS1129 QC runs - the parameters the shipped demo is pinned to."""
    from pearl import fetch_test_data
    a.input = a.input or DEMO_INPUT
    a.output = a.output or DEMO_OUTPUT
    a.name = a.name or "test"
    a.acquisition = "dia"
    a.mz_range = [200, 225]
    a.rt_range = [0, 720]
    a.rt_bin_size = 0.73
    a.qc_sample = a.qc_sample or ["QC01"]
    a.mass_tol = MASS_TOL
    a.task_id = "900"
    if not glob.glob(os.path.join(a.input, "*.mzML")):
        print(f"demo: fetching public test data into {a.input}", flush=True)
        fetch_test_data.main([])

def build_config(a):
    name = a.name or os.path.basename(os.path.normpath(a.input))
    acquisition, why = (a.acquisition, "") if a.acquisition else detect_acquisition(a.input)
    rt_range = a.rt_range or detect_rt_range(a.input)
    task = f"Task{a.task_id}_{name}"
    out = os.path.normpath(a.output)
    pred = (f"{out}/yucca/results/{task}/Task100_synth/UNet__2D/"
            "YuccaManager__YuccaPlanner/default/kfold_5_fold_0/version_0/best")
    return name, task, {
        "task": task,
        "images": {"acquisition": acquisition, "mzml_path": a.input,
                   "output_folder": f"{out}/eic",
                   "rt_range": rt_range, "rt_bin_size": a.rt_bin_size,
                   "mz_range": list(a.mz_range), "mass_tol": a.mass_tol,
                   "files": a.files,
                   "dia_channels": a.dia_channels,
                   "dia_functions": list(a.dia_functions) if a.dia_functions else None},
        "labels": {"predictions_dir": pred,
                   "images_raw": f"{out}/yucca/raw_data/{task}/imagesTs",
                   "output_dir": f"{out}/clustering/L3_erosion",
                   "task_name": name},
        "matching": {"input_dir": f"{out}/clustering/L3_erosion/single_labels_overlap",
                     "output_dir": f"{out}/clustering/L3_erosion/clusters_k{K}"},
        "peaks": {"output": f"{out}/peaks.csv", "qc_sample": a.qc_sample,
                  "ms2_ppm": a.ms2_ppm, "ms2_rt_tol": a.ms2_rt_tol},
    }, why

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="folder of .mzML files")
    ap.add_argument("--output", help="folder for all results")
    ap.add_argument("--name", help="dataset name (default: input folder name)")
    ap.add_argument("--mz-range", nargs=2, type=float, default=[50, 1200], metavar=("LO", "HI"))
    ap.add_argument("--rt-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="seconds; default: read from the files")
    ap.add_argument("--acquisition", choices=["dia", "dda"], default=None,
                    help="default: detected from the files")
    ap.add_argument("--files", nargs="+", default=None, metavar="NAME",
                    help="run names to include (no .mzML), or one .txt file "
                         "with one name per line; default: every file in --input")
    ap.add_argument("--qc-sample", nargs="+", default=None, metavar="FILE",
                    help="QC run(s) the peak m/z and intensity are read from; "
                         "default: the run with the most signal")
    ap.add_argument("--rt-bin-size", type=float, default=0.73)
    ap.add_argument("--mass-tol", type=float, default=MASS_TOL, metavar="DA",
                    help=f"width of each m/z slice in Da (default {MASS_TOL}, "
                         "the value the published model was trained for)")
    ap.add_argument("--ms2-ppm", type=float, default=MS2_PPM, metavar="PPM",
                    help=f"m/z tolerance when a fragment spectrum is read from the raw "
                         f"files (default {MS2_PPM:g})")
    ap.add_argument("--ms2-rt-tol", type=float, default=MS2_RT_TOL, metavar="SECONDS",
                    help="DDA only: how far outside a peak's own RT window its fragment "
                         f"spectrum may lie (default {MS2_RT_TOL:g} s)")
    ap.add_argument("--dia-channels", choices=["function", "mslevel"], default=None,
                    help="DIA only: how MS1/MS2/lock mass are told apart; default: "
                         "function= native IDs when present (Waters), else MS level")
    ap.add_argument("--dia-functions", nargs=3, type=int, default=None,
                    metavar=("MS1", "MS2", "LOCKMASS"),
                    help="DIA only: the function numbers of the three channels "
                         "(default 1 2 3); pass 0 for LOCKMASS if there is none")
    ap.add_argument("--task-id", default="500")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--dry-run", action="store_true", help="write the config and stop")
    ap.add_argument("--demo", action="store_true",
                    help="download four public QC runs and analyse them; "
                         "ignores --input and every acquisition option")
    a = ap.parse_args()
    if a.demo:
        apply_demo_defaults(a)

    if not a.input or not a.output:
        raise SystemExit("--input and --output are required (or use --demo)")
    if not glob.glob(os.path.join(a.input, "*.mzML")):
        raise SystemExit(f"no .mzML files in {a.input}")
    if a.mass_tol <= 0:
        raise SystemExit(f"--mass-tol must be positive, got {a.mass_tol}")
    a.files = read_file_list(a.files)
    a.qc_sample = read_file_list(a.qc_sample)
    check_runs(a.input, a.files, "--files")
    check_runs(a.input, a.qc_sample, "--qc-sample", selected=a.files)
    name, task, cfg, why = build_config(a)
    os.makedirs(a.output, exist_ok=True)
    cfg_path = os.path.join(a.output, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    n = len(glob.glob(os.path.join(a.input, "*.mzML")))
    im = cfg["images"]
    print(f"dataset    : {name}  ({len(a.files) if a.files else n} of {n} mzML)", flush=True)
    print(f"acquisition: {im['acquisition']}{'' if a.acquisition else f'  (detected: {why})'}",
          flush=True)
    print(f"rt range   : {im['rt_range']} s{'' if a.rt_range else '  (detected)'}", flush=True)
    print(f"m/z range  : {im['mz_range']}", flush=True)
    print(f"m/z slices : {expected_eic_files(im)} images of {im['mass_tol']} Da", flush=True)
    print(f"config     : {cfg_path}", flush=True)
    if a.dry_run:
        return

    from pearl import patch_yucca_npy
    patch_yucca_npy.main(quiet=True)

    env = dict(os.environ)
    env["PEARL_CONFIG"] = os.path.abspath(cfg_path)
    env["PEARL_OUTPUT"] = os.path.abspath(a.output)
    env["CKPT"] = a.checkpoint
    rc = subprocess.call(["bash", os.path.join(ROOT, "run_pipeline.sh")], cwd=ROOT, env=env)
    if rc:
        sys.exit(rc)
    print(f"\npeak table : {cfg['peaks']['output']}")

if __name__ == "__main__":
    main()