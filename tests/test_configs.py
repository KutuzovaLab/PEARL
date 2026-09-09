import os
import re
import glob
import subprocess
import sys

import yaml
import pytest

from pearl.common import expected_eic_files, shell_exports, load_config, train_exports
from pearl.constants import MASS_TOL, LENGTH_SPLINE, EROSION_WIDTH, K

RETIRED = ("int_thres", "drop_samples", "binary", "mass_tol",
           "length_spline", "erosion_width", "reference_file")

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
    """They may exist as local variables or provenance stamps, but nothing may
    read them out of a config dict - that is what made them user-tunable."""
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
