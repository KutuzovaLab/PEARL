"""
    python -m src.taskconvert synthetic --src results/synthetic --task Task100_synth
    python -m src.taskconvert inference --src results/eic/images --task Task200_standNeg --prefix standNeg
"""

import argparse
import glob
import os
import re

import json

import numpy as np
from sklearn.model_selection import train_test_split
from yucca.pipeline.task_conversion.utils import generate_dataset_json

LABELS = {0: "no signal", 1: "signal", 2: "overlap"}

def _mark_npy(base):
    p = os.path.join(base, "dataset.json")
    d = json.load(open(p))
    d["image_extension"] = "npy"
    json.dump(d, open(p, "w"), indent=2)

def _task_dirs(task):
    raw = os.environ.get("YUCCA_RAW_DATA")
    if not raw:
        raise SystemExit("YUCCA_RAW_DATA is not set")
    base = os.path.join(raw, task)
    dirs = {d: os.path.join(base, d) for d in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return base, dirs

def convert_synthetic(src, task, label_version="v3", test_size=0.0, seed=42):
    raw = os.environ.get("YUCCA_RAW_DATA")
    if not raw:
        raise SystemExit("YUCCA_RAW_DATA is not set")

    base = os.path.join(raw, task)
    dirs = {d: os.path.join(base, d) for d in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    cases = []
    for sub in sorted(os.listdir(src)):
        img_dir = os.path.join(src, sub, "images")
        if not os.path.isdir(img_dir):
            continue
        for p in sorted(glob.glob(os.path.join(img_dir, "*.npy"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            cases.append((sub, stem, f"{sub}_{stem}"))

    if not cases:
        raise SystemExit(f"No images found under {src}/*/images")

    kept, dropped = [], 0
    for c in cases:
        sub, stem, _ = c
        if np.load(f"{src}/{sub}/labels/{label_version}/{stem}.npy").any():
            kept.append(c)
        else:
            dropped += 1
    cases = kept
    if dropped:
        print(f"dropped {dropped} all-background case(s) with no foreground label")
    if not cases:
        raise SystemExit("All cases were all-background; nothing to convert.")

    if test_size:
        tr, ts = train_test_split(range(len(cases)), test_size=test_size, random_state=seed)
    else:
        tr, ts = range(len(cases)), []

    def save_pair(i, img_out, lbl_out):
        sub, stem, cid = cases[i]
        img = np.load(f"{src}/{sub}/images/{stem}.npy").astype(np.float32)
        lbl = np.load(f"{src}/{sub}/labels/{label_version}/{stem}.npy").astype(np.uint8)
        np.save(f"{img_out}/{cid}_000.npy", img)
        np.save(f"{lbl_out}/{cid}.npy", lbl)

    for i in tr:
        save_pair(i, dirs["imagesTr"], dirs["labelsTr"])
    for i in ts:
        save_pair(i, dirs["imagesTs"], dirs["labelsTs"])

    generate_dataset_json(
        os.path.join(base, "dataset.json"),
        dirs["imagesTr"], dirs["imagesTs"],
        modalities=("T1",),   # what the original task conversion used; name is cosmetic
        labels=LABELS,
        dataset_name=task,
        license="internal",
        dataset_description=f"Synthetic LC-MS local smoke test (pooled, {label_version} labels)",
        dataset_reference="PEARL",
    )
    _mark_npy(base)
    print(f"converted {len(cases)} cases -> {len(list(tr))} train / {len(list(ts))} holdout in {base}")

def convert_inference(src, task, prefix=None, modality="T1", limit=None):
    raw = os.environ.get("YUCCA_RAW_DATA")
    if not raw:
        raise SystemExit("YUCCA_RAW_DATA is not set")
    if prefix is None:
        prefix = re.sub(r"^Task\d+_", "", task)  # strip the TaskXXX_ prefix

    base = os.path.join(raw, task)
    dirs = {d: os.path.join(base, d) for d in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "*.npy")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No .npy images found under {src}")

    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        cid = f"{prefix}_{stem}"
        img = np.load(p).astype(np.float32)
        np.save(f"{dirs['imagesTs']}/{cid}_000.npy", img)

    generate_dataset_json(
        os.path.join(base, "dataset.json"),
        dirs["imagesTs"], dirs["imagesTs"],
        modalities=(modality,),
        labels=LABELS,
        dataset_name=task,
        license="internal",
        dataset_description=f"Inference task: real EIC images ({len(files)} cases), no ground-truth labels",
        dataset_reference="PEARL",
    )
    _mark_npy(base)
    print(f"inference task ready: {len(files)} images -> {dirs['imagesTs']}")
    print(f"next: yucca_inference -t {task} -s <SOURCE_TASK> -d 2D -m UNet --no_wandb --disable_tta --no_eval")

def main():
    ap = argparse.ArgumentParser(description="Convert arrays into a Yucca task.")
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("synthetic", help="Pool synthetic images+labels into a training task.")
    s.add_argument("--src", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--label-version", default="v3")
    s.add_argument("--test-size", type=float, default=0.0,
                   help="Holdout into imagesTs. Default 0: yucca's own 5-fold over imagesTr "
                        "provides validation, and a holdout here is never read.")
    s.add_argument("--seed", type=int, default=42)

    i = sub.add_parser("inference", help="Convert EIC images into an inference task.")
    i.add_argument("--src", required=True)
    i.add_argument("--task", required=True)
    i.add_argument("--prefix", default=None)
    i.add_argument("--modality", default="T1")
    i.add_argument("--limit", type=int, default=None)

    a = ap.parse_args()
    if a.mode == "synthetic":
        convert_synthetic(a.src, a.task, a.label_version, a.test_size, a.seed)
    else:
        convert_inference(a.src, a.task, a.prefix, a.modality, a.limit)

if __name__ == "__main__":
    main()