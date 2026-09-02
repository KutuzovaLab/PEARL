import os
import sys
import numpy as np

from src.common import MissingInput, parse_config_arg
from src.labels_io import make_rt_bins
from src.plots import calibration_curves, unet_vs_software, unet_vs_software_plain
from src.raw_spectra import load_qc_spectra
from src.reference_methods import load_reference
from src.metrics import (add_dda_metrics, add_library_metrics, add_msdial_cosine,
                         attach_unet_ms2, filter_library, load_clusters,
                         load_dda_fragments, match_compounds, pearson_per_compound)

ARRAY_COLUMNS = ["ms2_mz", "ms2_int"]

def _excel_safe(df):
    out = df.copy()
    for col in ARRAY_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: "" if v is None else " ".join(f"{x:g}" for x in np.atleast_1d(v))
            )
    return out

def _step(title):
    print(f"\n== {title}", flush=True)

def main(cfg):
    from matchms.logging_functions import set_matchms_logger_level
    set_matchms_logger_level("ERROR")

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    task_name = cfg.get("task_name", cfg["branch"])
    skipped = []          # (stage, reason) -> end-of-run summary
    deviations = []

    _step(f"1/8 reference method table ({cfg['branch']})")
    reference = load_reference(cfg)
    print("  " + reference.summary())
    deviations.extend(reference.deviations)
    for d in reference.deviations:
        print(f"  DEVIATION: {d}")

    _step("2/8 raw MS1/MS2 from QC file")
    df_ms1, df_ms2 = load_qc_spectra(reference.qc_mzml, mz_window=cfg.get("mz_window"))

    _step("3/8 matching compounds to U-Net single labels")
    rt_bins = make_rt_bins(reference.rt_range, reference.rt_bin_size)
    compound_info, comparison_table, not_matched = match_compounds(
        reference, cfg["single_labels_dir"], rt_bins, df_ms1
    )

    if compound_info.empty:
        print("\n  no compound matched any label - stopping before the spectral stages.")
        _write_tables(out_dir, compound_info, comparison_table, not_matched)
        _print_summary(reference, deviations,
                       skipped + [("spectral stages", "no compounds matched")], compound_info)
        return 0

    _step("4/8 U-Net MS2 spectra from MS1<->MS2 clusters")
    try:
        clusters = load_clusters(cfg["clusters_path"])
        compound_info = attach_unet_ms2(
            compound_info, clusters, cfg["single_labels_dir"], rt_bins, df_ms2
        )
    except MissingInput as exc:
        print(f"  SKIPPED: {exc}")
        skipped.append(("UNet MS2 extraction (cell 23)", exc.path))

    _step("5/8 cosine similarity vs MsDial")
    if "ms2_mz" in compound_info.columns:
        compound_info = add_msdial_cosine(compound_info, tol=cfg.get("cosine_tol", 0.5))
    else:
        print("  SKIPPED: no U-Net MS2 spectra available")
        skipped.append(("cosine vs MsDial (cell 34)", "depends on the skipped MS2 stage"))

    _step("6/8 MoNA reference-library metrics (MF / precision / recall / F1)")
    mona_lib = cfg.get("mona_library")
    if mona_lib and os.path.exists(str(mona_lib)) and "ms2_mz" in compound_info.columns:
        import pandas as pd
        lib = filter_library(pd.read_pickle(mona_lib),
                             instrument_pattern=cfg.get("library_instrument", "QTOF|Q-TOF"),
                             precursor_types=cfg.get("library_precursor_types"))
        compound_info = add_library_metrics(compound_info, lib,
                                            tol=cfg.get("library_tol", 0.1))
    else:
        reason = "no U-Net MS2 spectra" if "ms2_mz" not in compound_info.columns else str(mona_lib)
        print(f"  SKIPPED: {reason}")
        skipped.append(("MoNA library metrics", reason))

    _step("7/8 DDA agreement (precision / recall / F1)")
    try:
        dda_frag = load_dda_fragments(cfg.get("dda_path"))
        compound_info = add_dda_metrics(compound_info, dda_frag, tol=cfg.get("dda_tol", 0.1))
    except MissingInput as exc:
        print(f"  SKIPPED: {exc}")
        skipped.append(("DDA precision/recall/F1 (cell 35)", exc.path))

    sirius_cfg = cfg.get("sirius") or {}
    if sirius_cfg.get("enabled"):
        print("\n== SIRIUS formula scoring")
        try:
            from src.sirius import (LOGIN_HELP, NotLoggedIn, is_logged_in,
                                    resolve_sirius, score_compounds, sirius_login)

            sirius_path = resolve_sirius(sirius_cfg.get("sirius_path"))
            print(f"  using {sirius_path}")

            # SIRIUS 6's CLI refuses to compute anything without a login, formula
            # annotation included, and exits 0 while refusing. Establish the session
            # up front so a credential problem surfaces once, not per compound.
            login = sirius_cfg.get("login") or {}
            if not is_logged_in(sirius_path):
                if not login.get("enabled"):
                    raise MissingInput(
                        "SIRIUS login", "SIRIUS refuses all CLI computation when logged out",
                        "cell 27",
                    )
                print("  not logged in; authenticating from the environment")
                sirius_login(
                    sirius_path,
                    user_env=login.get("user_env", "SIRIUS_USER"),
                    password_env=login.get("password_env", "SIRIUS_PASSWORD"),
                )
            print("  login OK")

            for source in sirius_cfg.get("sources", ["unet", "msdial"]):
                if source == "unet" and "ms2_mz" not in compound_info.columns:
                    print("  skipping source 'unet': no U-Net MS2 spectra available")
                    continue
                print(f"  scoring source '{source}'")
                compound_info = score_compounds(compound_info, sirius_cfg, source=source)
        except MissingInput as exc:
            print(f"  SKIPPED: {exc}")
            if "login" in exc.path.lower() or "SIRIUS_" in exc.path:
                print(f"\n{LOGIN_HELP}\n")
            skipped.append(("SIRIUS formula scoring (cells 26-31)", exc.path))
        except NotLoggedIn as exc:
            print(f"  SKIPPED: {exc}")
            skipped.append(("SIRIUS formula scoring (cells 26-31)", "SIRIUS login required"))
    else:
        skipped.append((
            "SIRIUS formula scoring (cells 26-31)",
            "disabled in config (sirius.enabled = false)",
        ))

    _step("8/8 writing tables and figures")
    pearson = pearson_per_compound(comparison_table)
    valid = pearson["pearson_r"].dropna()
    if len(valid):
        print(f"  Pearson r vs the reference method: mean {valid.mean():.3f} over {len(valid)} "
              f"compound/label pairs ({(valid > 0.8).sum()} above 0.8)")
    pearson.to_excel(os.path.join(out_dir, "pearson_per_compound.xlsx"), index=False)
    _write_tables(out_dir, compound_info, comparison_table, not_matched)

    fig_dir = os.path.join(out_dir, "figures")
    for fn, name in (
        (calibration_curves(comparison_table, os.path.join(fig_dir, "calibration_curves.png")), "calibration_curves.png"),
        (unet_vs_software(comparison_table, os.path.join(fig_dir, "unet_vs_software.png"), task_name), "unet_vs_software.png"),
        (unet_vs_software_plain(comparison_table, os.path.join(fig_dir, "unet_vs_software_plain.png")), "unet_vs_software_plain.png"),
    ):
        print(f"  {'wrote' if fn else 'skipped (no data)'}: {name}")

    _print_summary(reference, deviations, skipped, compound_info)
    print(f"\nvalidation complete -> {out_dir}")
    return 0

def _write_tables(out_dir, compound_info, comparison_table, not_matched):
    _excel_safe(compound_info).to_excel(os.path.join(out_dir, "info_L3.xlsx"), index=False)
    comparison_table.to_excel(os.path.join(out_dir, "calibration_curve_L3.xlsx"), index=False)
    not_matched.to_excel(os.path.join(out_dir, "not_matched.xlsx"), index=False)
    compound_info.to_pickle(os.path.join(out_dir, "compound_info.pkl"))  # lossless, keeps arrays
    print(f"  info_L3.xlsx ({len(compound_info)} rows), "
          f"calibration_curve_L3.xlsx ({len(comparison_table)} rows), "
          f"not_matched.xlsx ({len(not_matched)} rows)")

def _print_summary(reference, deviations, skipped, compound_info):
    print("\n== summary")
    print(f"  {reference.name}: {len(compound_info)} matched compound/label pairs "
          f"from {len(reference.table)} compounds")
    print(f"  deviations from the notebook: {len(deviations)}")
    for d in deviations:
        print(f"    - {d}")
    print(f"  stages skipped for missing inputs: {len(skipped)}")
    for stage, reason in skipped:
        print(f"    - {stage}: {reason}")

if __name__ == "__main__":
    sys.exit(main(parse_config_arg(description="Validate U-Net peaks against a reference method.")))