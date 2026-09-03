import argparse
import shlex
import yaml
import numpy as np

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_config_arg(description="PEARL script"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default=None,
                        help="Top-level key to read when one YAML holds several stages.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.section:
        if args.section not in cfg:
            raise SystemExit(f"section '{args.section}' not in {args.config}; have: {sorted(cfg)}")
        return cfg[args.section]
    return cfg

PIPELINE_DEFAULTS = {
    "synthetic_config": "configs/synthetic.yaml",
    "synthetic_section": "base",
    "synthetic_task": "Task100_synth",
    "epochs": 600,
    "patch": "96 528",
    "batch": 44,
}

# How many .npy an images section should produce - DIA writes MS1+MS2, DDA MS1 only
def expected_eic_files(images):

    lo, hi = images["mz_range"]
    n_bins = len(np.arange(lo, hi, images["mass_tol"]))
    return n_bins * (1 if images.get("acquisition", "dia") == "dda" else 2)

def shell_exports(cfg):
    v = dict(PIPELINE_DEFAULTS, **{k: cfg[k] for k in PIPELINE_DEFAULTS if k in cfg})
    images = cfg["images"]
    lines = [
        f"TASK={shlex.quote(str(cfg['task']))}",
        f"SYNTH_CFG={shlex.quote(str(v['synthetic_config']))}",
        f"SYNTH_SEC={shlex.quote(str(v['synthetic_section']))}",
        f"SYNTH_TASK={shlex.quote(str(v['synthetic_task']))}",
        f"SYNTH_SRC={shlex.quote(str(load_config(v['synthetic_config'])[v['synthetic_section']]['out_dir']))}",
        f"EIC_DIR={shlex.quote(str(images['output_folder']))}",
        f"EIC_FILES={expected_eic_files(images)}",
        f"ACQUISITION={shlex.quote(str(images.get('acquisition', 'dia')))}",
        f'EPOCHS="${{EPOCHS:-{v["epochs"]}}}"',
        f'PATCH="${{PATCH:-{v["patch"]}}}"',
        f'BATCH="${{BATCH:-{v["batch"]}}}"',
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Emit pipeline parameters as shell assignments.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--shell", action="store_true", help="required; the only supported mode")
    a = ap.parse_args()
    if not a.shell:
        raise SystemExit("pass --shell")
    print(shell_exports(load_config(a.config)))