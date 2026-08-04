"""
Usage:  python sirius.py --config configs/config_sirius.yaml
"""

import os
import numpy as np
from src.config import parse_config_arg
from src.sirius import (
    get_suspects, write_sirius_ms_file, run_sirius, get_comp_spectra, get_ms1,
)

def main(cfg):
    suspect_path = cfg["suspect_path"]
    susp_name = cfg["susp_name"]
    cluster_path_base = cfg["cluster_path_base"]
    tol = cfg["tol"]
    mz_range = cfg["mz_range"]
    sirius_executable = cfg["sirius_executable"]
    output_directory = cfg["output_directory"]
    sample_ind = cfg.get("sample_ind", 15)
    msdial_int_thres = cfg.get("msdial_int_thres", 2)
    profile = cfg.get("profile", "qtof")

    # Getting suspects
    suspects = get_suspects(suspect_path)

    susp_mass = suspects.loc[suspects['name'] == susp_name, 'mz'].values[0]

    # Getting images in the cluster
    cluster_path = os.path.join(cluster_path_base, susp_name)

    # Mass slice info passed to SIRIUS
    mass_bins = np.arange(mz_range[0], mz_range[1], tol)

    # Grabbing MS2 spectra
    ms2_all = get_comp_spectra(
        suspects.loc[suspects['name'] == susp_name, 'MS2'].values[0],
        cluster_path, sample_ind, msdial_int_thres,
    )
    ms2_all.sort_values(by="mz", inplace=True)

    # Input to SIRIUS
    compound_name = susp_name
    parent_mass = np.mean([mass_bins[np.min(np.where(mass_bins >= susp_mass)) - 1],
                           mass_bins[np.min(np.where(mass_bins >= susp_mass))]])
    ion_mode = suspects.loc[suspects['name'] == susp_name, 'adduct'].values[0]
    ms1 = get_ms1(susp_mass, cluster_path, sample_ind)
    ms2 = ms2_all[ms2_all['method'] == 'Unet'][['mz', 'int']].values.tolist()

    # Make sure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    ms_file = os.path.join(output_directory, f'{susp_name}.ms')

    # Write the input file
    write_sirius_ms_file(ms_file, compound_name, parent_mass, ion_mode, ms1, ms2)

    # Run SIRIUS
    exit_code = run_sirius(sirius_executable, ms_file, output_directory, profile=profile)

    if exit_code == 0:
        print("SIRIUS analysis complete.")
    else:
        print("SIRIUS analysis failed.")


if __name__ == "__main__":
    config = parse_config_arg(description="Build a SIRIUS input file and run SIRIUS.")
    main(config)
