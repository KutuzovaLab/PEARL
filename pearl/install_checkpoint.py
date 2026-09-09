import argparse, os, shutil

VDIR = ("{models}/{task}/UNet__2D/YuccaManager__YuccaPlanner/default/"
        "kfold_5_fold_0/version_0/checkpoints")

def main():
    ap = argparse.ArgumentParser(description="Install a checkpoint as a yucca source model.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="Task043_sim14L3")
    ap.add_argument("--name", default="best", help="checkpoint name yucca loads (default 'best')")
    a = ap.parse_args()
    models = os.environ.get("YUCCA_MODELS")
    if not models:
        raise SystemExit("YUCCA_MODELS is not set")
    dst = VDIR.format(models=models, task=a.task)
    os.makedirs(dst, exist_ok=True)

    hp = os.path.join(os.path.dirname(dst), "hparams.yaml")
    if os.path.exists(hp):
        os.remove(hp)
    out = os.path.join(dst, f"{a.name}.ckpt")
    shutil.copy2(a.ckpt, out)
    print(f"installed {a.ckpt} -> {out}")
    print(f"  use with: yucca_inference -t <TASK> -s {a.task} -d 2D -m UNet --version 0")

if __name__ == "__main__":
    main()
