import os
import re
import glob
import subprocess
import sys

import numpy as np
import yaml
import pytest

from pearl.common import (expected_eic_files, shell_exports, load_config, train_exports,
                          mz_decimals, slice_tag)
from pearl.constants import MASS_TOL, LENGTH_SPLINE, EROSION_WIDTH, K, MS2_PPM, MS2_RT_TOL

# mass_tol is deliberately NOT here: it is the one slice parameter a user may set.
RETIRED = ("int_thres", "drop_samples", "binary",
           "length_spline", "erosion_width", "reference_file")

def _run(args, tmp_path_factory):
    mzml = tmp_path_factory.mktemp("mzml")
    (mzml / "QC01.mzML").write_text("")
    out = tmp_path_factory.mktemp("out")
    rc = subprocess.call([sys.executable, "-m", "pearl.run", "--input", str(mzml),
                          "--output", str(out), "--dry-run"] + args, cwd=os.getcwd())
    assert rc == 0
    return load_config(str(out / "config.yaml"))

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    mzml = tmp_path_factory.mktemp("mzml")
    (mzml / "QC01.mzML").write_text("")
    out = tmp_path_factory.mktemp("out")
    rc = subprocess.call([sys.executable, "-m", "pearl.run", "--input", str(mzml),
                          "--output", str(out), "--acquisition", "dia",
                          "--mz-range", "50", "1000", "--rt-range", "100", "1100",
                          "--rt-bin-size", "0.73", "--qc-sample", "QC01.mzML",
                          "--dry-run"], cwd=os.getcwd())
    assert rc == 0
    return load_config(str(out / "config.yaml"))

def test_generated_config_has_required_keys(generated):
    assert "task" in generated
    for section in ("images", "labels", "matching", "peaks"):
        assert section in generated, f"missing {section}"
    im = generated["images"]
    for k in ("acquisition", "mzml_path", "mz_range", "rt_range", "rt_bin_size"):
        assert k in im, f"images missing {k}"
    assert im["acquisition"] in ("dia", "dda")

def test_qc_sample_is_a_list_without_extension(generated):
    assert generated["peaks"]["qc_sample"] == ["QC01"]

def test_generated_config_declares_no_retired_keys(generated):
    text = yaml.safe_dump(generated)
    for name in RETIRED:
        assert name not in text, f"generated config still declares {name}"

@pytest.mark.parametrize("name", RETIRED + ("sample_thres", "label_width", "k"))
def test_retired_names_are_never_read_from_config(name):
    read = re.compile(r"\w+(?:\.get\(|\[)[\"']" + re.escape(name) + r"[\"']")
    hits = []
    for path in glob.glob("pearl/*.py"):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if read.search(line):
                    hits.append(f"{path}:{i}: {line.strip()}")
    assert not hits, "still read from config:\n" + "\n".join(hits)

def test_constants_match_the_published_values():
    assert (MASS_TOL, LENGTH_SPLINE, EROSION_WIDTH, K) == (0.1, 5, 3, 30)
    assert (MS2_PPM, MS2_RT_TOL) == (20.0, 5.0)

def test_mass_tol_defaults_to_the_published_slice(generated):
    assert generated["images"]["mass_tol"] == MASS_TOL
    assert generated["peaks"]["ms2_ppm"] == MS2_PPM
    assert generated["peaks"]["ms2_rt_tol"] == MS2_RT_TOL

def test_mass_tol_reaches_the_config_and_the_file_count(tmp_path_factory):
    cfg = _run(["--acquisition", "dia", "--mz-range", "50", "1000",
                "--rt-range", "100", "1100", "--mass-tol", "0.05"], tmp_path_factory)
    assert cfg["images"]["mass_tol"] == 0.05
    assert expected_eic_files(cfg["images"]) == 2 * 19000
    assert "EIC_FILES=38000" in shell_exports(cfg)

def test_slice_names_are_unchanged_at_the_published_tolerance():
    assert mz_decimals(0.1) == 1
    for lo, hi in ((50, 1000), (50, 1200), (75, 751), (90, 1300)):
        edges = np.arange(lo, hi, 0.1)
        for mb in (edges, edges + 0.1):
            assert [slice_tag(v, 1) for v in mb] == [str(round(v, 1)).replace(".", "_")
                                                     for v in mb]

@pytest.mark.parametrize("tol,nd", [(0.1, 1), (0.05, 2), (0.01, 2), (0.001, 3), (0.5, 1)])
def test_mz_decimals_keeps_slice_names_distinct(tol, nd):
    assert mz_decimals(tol) == nd
    assert slice_tag(100.0, nd) != slice_tag(100.0 + tol, nd)

def test_expected_eic_files_dia_vs_dda():
    dia = {"mz_range": [50, 1000], "acquisition": "dia"}
    dda = {"mz_range": [50, 1000], "acquisition": "dda"}
    assert expected_eic_files(dia) == 2 * expected_eic_files(dda)
    assert expected_eic_files(dia) == 19000

def test_shell_exports_has_driver_vars(generated):
    out = shell_exports(generated)
    for key in ("TASK=", "SYNTH_TASK=", "EIC_DIR=", "EIC_FILES=", "ACQUISITION=dia"):
        assert key in out

def test_train_exports_needs_no_dataset_config():
    out = train_exports()
    for key in ("SYNTH_CFG=", "SYNTH_SEC=", "SYNTH_TASK=", "SYNTH_SRC=", "EPOCHS="):
        assert key in out
