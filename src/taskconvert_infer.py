"""
    python -m src.taskconvert_infer --src data/output/images --task Task200_inferLocal
"""

import os
import re
import glob
import argparse

import numpy as np
import nibabel as nib
from yucca.pipeline.task_conversion.utils import generate_dataset_json

LABEL_MAPS = {
    "binary": {0: "no signal", 1: "signal"},
    "3class": {0: "no signal", 1: "signal", 2: "overlap"},
}

def convert(src, task, prefix=None, modality="synthetic", label_map="binary", limit=None):
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
        nib.save(nib.Nifti1Image(img, np.eye(4)), f"{dirs['imagesTs']}/{cid}_000.nii.gz")

    generate_dataset_json(
        os.path.join(base, "dataset.json"),
        dirs["imagesTs"], dirs["imagesTs"],
        modalities=(modality,),
        labels=LABEL_MAPS[label_map],
        dataset_name=task,
        license="internal",
        dataset_description=f"Inference task: real EIC images ({len(files)} cases), no ground-truth labels",
        dataset_reference="PEARL",
    )
    print(f"inference task ready: {len(files)} images -> {dirs['imagesTs']}")
    print(f"next: yucca_inference -t {task} -s <SOURCE_TASK> -d 2D -m UNet --no_wandb --disable_tta --no_eval")


def main():
    ap = argparse.ArgumentParser(
        description="Convert real EIC .npy into a Yucca inference task (imagesTs nii.gz, no labels)."
    )
    ap.add_argument("--src", required=True, help="Directory of real EIC .npy images (mzml_to_npy output images/).")
    ap.add_argument("--task", required=True, help="Yucca task name, e.g. Task200_inferLocal.")
    ap.add_argument("--prefix", default=None,
                    help="Case-id prefix (default: task name without TaskXXX_). Keep underscore-free.")
    ap.add_argument("--modality", default="synthetic", help="Modality name for dataset.json (cosmetic).")
    ap.add_argument("--label-map", default="binary", choices=list(LABEL_MAPS),
                    help="labels field in dataset.json; match the trained model (default binary).")
    ap.add_argument("--limit", type=int, default=None, help="Only convert the first N images (for testing).")
    args = ap.parse_args()
    convert(args.src, args.task, args.prefix, args.modality, args.label_map, args.limit)


if __name__ == "__main__":
    main()
