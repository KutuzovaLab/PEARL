import glob
import yaml
import pytest
from src.common import expected_eic_files, shell_exports, load_config

CONFIGS = sorted(glob.glob("configs/*.yaml"))

@pytest.mark.parametrize("path", CONFIGS)
def test_config_parses(path):
    assert yaml.safe_load(open(path)) is not None

@pytest.mark.parametrize("name", ["wheat", "wastewater", "fluorine"])
def test_dataset_config_has_required_keys(name):
    c = load_config(f"configs/{name}.yaml")
    assert "task" in c
    for section in ("images", "labels", "matching"):
        assert section in c, f"{name} missing {section}"
    im = c["images"]
    for k in ("acquisition", "mzml_path", "mz_range", "rt_range", "rt_bin_size", "mass_tol"):
        assert k in im, f"{name}.images missing {k}"
    assert im["acquisition"] in ("dia", "dda")

def test_expected_eic_files_dia_vs_dda():
    dia = {"mz_range": [50, 1000], "mass_tol": 0.1, "acquisition": "dia"}
    dda = {"mz_range": [50, 1000], "mass_tol": 0.1, "acquisition": "dda"}
    assert expected_eic_files(dia) == 2 * expected_eic_files(dda)   # DIA writes MS1+MS2
    assert expected_eic_files(dia) == 19000

def test_shell_exports_has_driver_vars():
    out = shell_exports(load_config("configs/wheat.yaml"))
    for key in ("TASK=", "SYNTH_TASK=", "EIC_DIR=", "EIC_FILES=", "ACQUISITION=dia"):
        assert key in out

def test_wheat_reproduces_original_mz_range():
    assert load_config("configs/wheat.yaml")["images"]["mz_range"] == [50, 1000]
