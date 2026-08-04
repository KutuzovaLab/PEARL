"""
Usage:  python mzml_to_npy.py --config configs/config_mzml_to_npy.yaml
"""

import os
from src.config import parse_config_arg
from src.mzml_to_npy import selecting_files, read_mzml_files, saving_results

def main(cfg):
    suspect_path = cfg["suspect_path"]
    mzml_path = cfg["mzml_path"]
    output_folder = cfg["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    rt_range = cfg["rt_range"]
    rt_bin_size = cfg["rt_bin_size"]
    mz_range = cfg["mz_range"]
    mass_tol = cfg["mass_tol"]
    length_spline = cfg.get("length_spline", 5)
    int_thres = cfg.get("int_thres", 1000)

    final_files = selecting_files(mzml_path, suspect_path)

    print('Reading raw files...', flush=True)
    ms1, ms2 = read_mzml_files(
        mzml_path, final_files, rt_bin_size, rt_range, mz_range, mass_tol,
        length_spline=length_spline,
    )

    saving_results(ms1, ms2, output_folder, mz_range, mass_tol, int_thres=int_thres)

if __name__ == "__main__":
    config = parse_config_arg(description="Convert mzML files to MS1/MS2 .npy images.")
    main(config)
