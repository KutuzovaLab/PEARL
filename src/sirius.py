"""
SIRIUS molecular formula annotation (notebook cells 26-31).

Ported from the notebook, with three deliberate changes:

1. The password is read from the SIRIUS_PASSWORD environment variable rather than
   an interactive getpass prompt. A prompt cannot work in a batch job, and the
   credential must never be written into a config file.
2. Login state is checked up front and failures are reported clearly.

   SIRIUS 6.5.4's CLI REQUIRES a logged-in account for all computation, including
   plain formula annotation. The docs say formula annotation "is performed on your
   local computer and no license is needed" - that is about the webservice
   subscription, not about CLI access. Verified against 6.5.4:

       SEVERE - Login ERROR: Please Login to use the SIRIUS command line tool!

   Worse, SIRIUS EXITS 0 in that case, so the return code alone is not a success
   signal; we scan the output for the marker as well. Register with an
   INSTITUTIONAL email - the free academic licence is granted from the domain.
3. Compounds that cannot produce a valid .ms file (no MS2 peaks, no peaks below
   the precursor, everything filtered out, missing adduct) are counted and
   skipped instead of aborting the whole run.

CLI shape, unchanged from the notebook:

    sirius --input X.ms --project P.sirius formulas -p qtof [fingerprints] \
           summaries --format=TSV --top-k-summary=20 -o OUTDIR
"""

import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from src.common import MissingInput
from src.metrics import parse_msdial_ms2

# Locate the SIRIUS executable, or raise MissingInput.
def resolve_sirius(sirius_path):
    if not sirius_path:
        raise MissingInput("sirius", "SIRIUS formula annotation", "cells 26-31")
    found = shutil.which(sirius_path) or (sirius_path if os.path.exists(sirius_path) else None)
    if not found:
        raise MissingInput(sirius_path, "SIRIUS formula annotation", "cells 26-31")
    return found

def safe_project_name(text):
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text).strip())
    return re.sub(r"_+", "_", text).strip("_")

LOGIN_ERROR_MARKER = "Please Login to use the SIRIUS command line tool"

LOGIN_HELP = (
    "SIRIUS 6 requires a logged-in account for ALL CLI computation, including plain "
    "formula annotation.\n"
    "  1. Register with your INSTITUTIONAL email - the free academic licence is granted "
    "from the email domain, so a personal address will not qualify.\n"
    "  2. Verify the address from the confirmation mail.\n"
    "  3. export SIRIUS_USER='you@institution.edu' SIRIUS_PASSWORD='...'\n"
    "  4. Set sirius.login.enabled: true in the validation config."
)


def is_logged_in(sirius_path):
    result = subprocess.run(
        [sirius_path, "login", "--show"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    blob = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0 and "Not Logged in" not in blob and LOGIN_ERROR_MARKER not in blob

def sirius_login(sirius_path, username=None, user_env="SIRIUS_USER", password_env="SIRIUS_PASSWORD"):
    if username:
        os.environ.setdefault(user_env, username)
    if not os.environ.get(user_env):
        raise MissingInput(f"{user_env} (environment variable)", "SIRIUS login", "cell 27")
    if not os.environ.get(password_env):
        raise MissingInput(f"{password_env} (environment variable)", "SIRIUS login", "cell 27")

    result = subprocess.run(
        [sirius_path, "login", f"--user-env={user_env}", f"--password-env={password_env}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    blob = (result.stdout or "") + (result.stderr or "")
    # SIRIUS can report failure while still exiting 0, so check the text too.
    if result.returncode != 0 or LOGIN_ERROR_MARKER in blob or "ERROR" in blob:
        raise RuntimeError(
            f"SIRIUS login failed (exit {result.returncode}).\n\n{LOGIN_HELP}\n\n"
            f"SIRIUS said:\n{blob[:2000]}"
        )
    return result

class NotLoggedIn(Exception):
    """SIRIUS refused to compute because no account is logged in."""

class SkipCompound(Exception):
    """This compound cannot produce a usable .ms file."""

# Write one SIRIUS .ms file. Raises SkipCompound when there is nothing to write.
def write_sirius_ms_file(compound_name, parent_mass, precursor_int, ion_type,
                         ms2_mz, ms2_int, output_dir, min_ms2_int=None):
    compound_name = str(compound_name)
    if ion_type is None or (isinstance(ion_type, float) and np.isnan(ion_type)) or str(ion_type) in ("", "nan"):
        raise SkipCompound(f"{compound_name}: no adduct/ionization")
    try:
        parent_mass = float(parent_mass)
        precursor_int = float(precursor_int)
    except (TypeError, ValueError):
        raise SkipCompound(f"{compound_name}: precursor mass/intensity not numeric")

    ms2_mz = np.asarray(ms2_mz, dtype=float).ravel()
    ms2_int = np.asarray(ms2_int, dtype=float).ravel()
    if len(ms2_mz) == 0:
        raise SkipCompound(f"{compound_name}: no MS2 peaks")
    if len(ms2_mz) != len(ms2_int):
        raise ValueError(f"ms2_mz and ms2_int differ in length for {compound_name}")

    keep = ms2_mz < parent_mass          # drop fragments above the precursor
    ms2_mz, ms2_int = ms2_mz[keep], ms2_int[keep]
    if len(ms2_mz) == 0:
        raise SkipCompound(f"{compound_name}: no MS2 peaks below the precursor mass")

    order = np.argsort(ms2_mz)
    ms2_mz, ms2_int = ms2_mz[order], ms2_int[order]

    if min_ms2_int is not None:
        keep = ms2_int >= float(min_ms2_int)
        ms2_mz, ms2_int = ms2_mz[keep], ms2_int[keep]
        if len(ms2_mz) == 0:
            raise SkipCompound(f"{compound_name}: all MS2 peaks below min_ms2_int={min_ms2_int}")

    os.makedirs(output_dir, exist_ok=True)
    ms_file_path = os.path.join(output_dir, f"{safe_project_name(compound_name)}.ms")
    with open(ms_file_path, "w", encoding="utf-8") as f:
        f.write(f">compound {compound_name}\n")
        f.write(f">parentmass {parent_mass:.6f}\n")
        f.write(f">ionization {ion_type}\n\n")
        f.write(">ms1\n")
        f.write(f"{parent_mass:.6f} {precursor_int:.6f}\n\n")
        f.write(">ms2\n")
        for mz, intensity in zip(ms2_mz, ms2_int):
            f.write(f"{mz:.6f} {intensity:.6f}\n")
    return ms_file_path

def get_sirius_formula_matches(sirius_path, ms_file_path, project_path, output_dir,
                               profile="qtof", top_k=20, run_fingerprints=False,
                               timeout=None):
    project_dir = os.path.dirname(project_path)
    os.makedirs(project_dir, exist_ok=True)
    if os.path.exists(project_path):
        os.remove(project_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [sirius_path, "--input", ms_file_path, "--project", project_path,
           "formulas", "-p", profile]
    if run_fingerprints:
        cmd.append("fingerprints")
    cmd += ["summaries", "--format=TSV", f"--top-k-summary={top_k}", "-o", output_dir]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    blob = (result.stdout or "") + (result.stderr or "")

    # SIRIUS exits 0 when it refuses to run for lack of a login, so the exit code
    # alone is not a reliable success signal.
    if LOGIN_ERROR_MARKER in blob:
        raise NotLoggedIn(LOGIN_HELP)

    if result.returncode != 0:
        raise RuntimeError(
            f"SIRIUS failed (exit {result.returncode})\ncommand: {' '.join(cmd)}\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

    formula_file = os.path.join(output_dir, f"formula_identifications_top-{top_k}.tsv")
    if not os.path.exists(formula_file):
        available = os.listdir(output_dir) if os.path.isdir(output_dir) else []
        raise FileNotFoundError(
            f"{os.path.basename(formula_file)} not found in {output_dir}; found {available}\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
    return pd.read_csv(formula_file, sep="\t")

def _spectrum_for(compound_info, i, source):
    if source == "unet":
        return (np.asarray(compound_info["ms2_mz"].iloc[i], dtype=float),
                np.asarray(compound_info["ms2_int"].iloc[i], dtype=float))
    if source == "msdial":
        return parse_msdial_ms2(compound_info["MsDial MS2"].iloc[i])
    raise ValueError(f"unknown source '{source}'; expected 'unet' or 'msdial'")

# Add a Sirius score column
def score_compounds(compound_info, cfg, source="unet", verbose=True):
    sirius_path = resolve_sirius(cfg.get("sirius_path"))
    work_dir = cfg.get("work_dir", "results/sirius")
    profile = cfg.get("profile", "qtof")
    top_k = cfg.get("top_k", 20)
    min_ms2_int = cfg.get("min_ms2_int", 5000)
    run_fingerprints = cfg.get("run_fingerprints", False)
    timeout = cfg.get("timeout_seconds")

    label = {"unet": "UNet", "msdial": "MsDial"}[source]
    column = f"Sirius score ({label})"
    compound_info = compound_info.copy()
    compound_info[column] = np.nan

    n_scored = n_skipped = n_failed = n_no_formula = 0
    for i in range(len(compound_info)):
        compound = compound_info["Compound"].iloc[i]
        expected_formula = compound_info["formula"].iloc[i]
        name = safe_project_name(compound)
        base_out = os.path.join(work_dir, label, name)

        try:
            ms2_mz, ms2_int = _spectrum_for(compound_info, i, source)
            ms_file = write_sirius_ms_file(
                compound_name=compound,
                parent_mass=compound_info["precursor mass"].iloc[i],
                precursor_int=compound_info["precursor int"].iloc[i],
                ion_type=compound_info["adduct"].iloc[i],
                ms2_mz=ms2_mz, ms2_int=ms2_int,
                output_dir=base_out, min_ms2_int=min_ms2_int,
            )
        except SkipCompound as exc:
            n_skipped += 1
            if verbose:
                print(f"  skip: {exc}")
            continue

        try:
            matches = get_sirius_formula_matches(
                sirius_path=sirius_path,
                ms_file_path=ms_file,
                project_path=os.path.join(base_out, f"{name}_project.sirius"),
                output_dir=os.path.join(base_out, f"{name}_summary"),
                profile=profile, top_k=top_k,
                run_fingerprints=run_fingerprints, timeout=timeout,
            )
        except NotLoggedIn:
            raise
        except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            n_failed += 1
            if verbose:
                print(f"  FAILED: {compound}: {str(exc).splitlines()[0]}")
            continue

        hit = matches[matches["molecularFormula"] == expected_formula]
        if hit.empty:
            n_no_formula += 1
            continue
        compound_info.loc[i, column] = hit.iloc[0]["SiriusScore"]
        n_scored += 1

    if verbose:
        print(f"  SIRIUS ({label}): {n_scored} scored, {n_no_formula} without the expected "
              f"formula in the top {top_k}, {n_skipped} skipped, {n_failed} failed")
    return compound_info