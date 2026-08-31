import os
import glob
import argparse

import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from yucca.pipeline.task_conversion.utils import generate_dataset_json

def convert(src, task, label_version="v3", test_size=0.2, seed=42):
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

    tr, ts = train_test_split(range(len(cases)), test_size=test_size, random_state=seed)

    def save_pair(i, img_out, lbl_out):
        sub, stem, cid = cases[i]
        img = np.load(f"{src}/{sub}/images/{stem}.npy").astype(np.float32)
        lbl = np.load(f"{src}/{sub}/labels/{label_version}/{stem}.npy").astype(np.uint8)
        nib.save(nib.Nifti1Image(img, np.eye(4)), f"{img_out}/{cid}_000.nii.gz")
        nib.save(nib.Nifti1Image(lbl, np.eye(4)), f"{lbl_out}/{cid}.nii.gz")

    for i in tr:
        save_pair(i, dirs["imagesTr"], dirs["labelsTr"])
    for i in ts:
        save_pair(i, dirs["imagesTs"], dirs["labelsTs"])

    generate_dataset_json(
        os.path.join(base, "dataset.json"),
        dirs["imagesTr"], dirs["imagesTs"],
        modalities=("synthetic",),
        labels={0: "background", 1: "peak"},
        dataset_name=task,
        license="internal",
        dataset_description=f"Synthetic LC-MS local smoke test (pooled, {label_version} labels)",
        dataset_reference="PEARL",
    )
    print(f"converted {len(cases)} cases -> {len(list(tr))} train / {len(list(ts))} test in {base}")

def main():
    ap = argparse.ArgumentParser(description="Pool synthetic .npy into a Yucca raw_data Task (nii.gz).")
    ap.add_argument("--src", required=True, help="Synthetic data dir (with <subset>/images and <subset>/labels/<v>).")
    ap.add_argument("--task", required=True, help="Yucca task name, e.g. Task100_synthLocal.")
    ap.add_argument("--label-version", default="v3", help="Label version to use as ground truth (default v3).")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default 0.2).")
    ap.add_argument("--seed", type=int, default=42, help="Split random_state (default 42).")
    args = ap.parse_args()
    convert(args.src, args.task, args.label_version, args.test_size, args.seed)

if __name__ == "__main__":
    main()