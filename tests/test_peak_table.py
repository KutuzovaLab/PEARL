import csv

import numpy as np
import pyopenms as oms
import pytest

from pearl import peak_table

HEADER = ["MS1 m/z", "MS1 intensity", "mz min", "mz max", "RT", "RT min", "RT max", "MS2 spectrum"]
RT_RANGE, RT_BIN = [0, 10], 1.0

def write_mzml(path, spectra):
    exp = oms.MSExperiment()
    for rt, level, native_id, mz, inten, *precursor in spectra:
        sp = oms.MSSpectrum()
        sp.setRT(rt)
        sp.setMSLevel(level)
        sp.setNativeID(native_id)
        sp.set_peaks((np.array(mz, dtype=float), np.array(inten, dtype=float)))
        if precursor and precursor[0]:
            pr = oms.Precursor()
            pr.setMZ(float(precursor[0]))
            pr.setIsolationWindowLowerOffset(0.5)
            pr.setIsolationWindowUpperOffset(0.5)
            sp.setPrecursors([pr])
        exp.addSpectrum(sp)
    oms.MzMLFile().store(str(path), exp)

def blob(folder, name, rows):
    arr = np.zeros((2, 11), dtype=np.float32)
    for sample, values in rows.items():
        arr[sample, 3:6] = values 
    np.save(folder / name, arr)

def config(tmp_path, acquisition, qc_sample, **peaks):
    return {
        "images": {"acquisition": acquisition, "mzml_path": str(tmp_path / "mzml"),
                   "rt_range": RT_RANGE, "rt_bin_size": RT_BIN, "files": None},
        "labels": {"output_dir": str(tmp_path / "labels")},
        "matching": {"output_dir": str(tmp_path / "clusters")},
        "peaks": dict({"output": str(tmp_path / "peaks.csv"), "qc_sample": qc_sample}, **peaks),
    }

def read_rows(tmp_path):
    with open(tmp_path / "peaks.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER
    return rows[1:]

@pytest.fixture
def dia(tmp_path):
    mzml = tmp_path / "mzml"; mzml.mkdir()
    blobs = tmp_path / "labels" / "single_labels_overlap"; blobs.mkdir(parents=True)
    clusters = tmp_path / "clusters"; clusters.mkdir()
    write_mzml(mzml / "QC01.mzML", [
        (4.0, 1, "function=1 process=0 scan=1", [100.02, 100.05, 150.0], [500, 900, 10000]),
        (4.0, 2, "function=2 process=0 scan=2", [50.03, 60.0], [300, 5000]),
        (8.0, 1, "function=1 process=0 scan=3", [100.07], [99999]),
    ])
    write_mzml(mzml / "S02.mzML", [
        (4.0, 1, "function=1 process=0 scan=1", [100.06], [4321]),
        (4.0, 2, "function=2 process=0 scan=2", [50.04], [111]),
    ])
    blob(blobs, "ut_MS1_100_0_100_1_001.npy", {0: [10, 50, 20], 1: [5, 5, 5]})
    blob(blobs, "ut_MS1_200_0_200_1_001.npy", {0: [1, 2, 1]})        # no signal in any run -> dropped
    blob(blobs, "ut_MS2_50_0_50_1_001.npy", {0: [3, 9, 3]})
    with open(clusters / "final_matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ms1_name", "ms2_name"])
        w.writerow(["ut_MS1_100_0_100_1_001.npy", "ut_MS2_50_0_50_1_001.npy"])
    return tmp_path

def test_peak_table_dia_row_from_qc_sample(dia):
    peak_table.main(config(dia, "dia", ["QC01"]))
    assert read_rows(dia) == [["100.0500", "900.0", "100.0", "100.1", "4.00", "3.00", "5.00", "50.030;300.0"]]

def test_peak_table_auto_selects_run_with_most_signal(dia):
    blob(dia / "labels" / "single_labels_overlap", "ut_MS1_100_0_100_1_001.npy", {0: [1, 1, 1], 1: [10, 90, 10]})
    peak_table.main(config(dia, "dia", None))
    assert read_rows(dia) == [["100.0600", "4321.0", "100.0", "100.1", "4.00", "3.00", "5.00", "50.040;111.0"]]

def test_peak_table_rejects_unknown_qc_sample(dia):
    with pytest.raises(SystemExit):
        peak_table.main(config(dia, "dia", ["NOPE"]))

def test_peak_table_dia_fragment_comes_from_the_co_eluting_scan(tmp_path):
    mzml = tmp_path / "mzml"; mzml.mkdir()
    blobs = tmp_path / "labels" / "single_labels_overlap"; blobs.mkdir(parents=True)
    clusters = tmp_path / "clusters"; clusters.mkdir()
    write_mzml(mzml / "QC01.mzML", [
        (4.0, 1, "function=1 process=0 scan=1", [100.05], [900]),
        (4.0, 2, "function=2 process=0 scan=1", [50.09], [100]),
        (5.0, 2, "function=2 process=0 scan=2", [50.03], [300]),
        (5.0, 1, "function=1 process=0 scan=2", [100.05], [400]),
    ])
    blob(blobs, "ut_MS1_100_0_100_1_001.npy", {0: [10, 50, 20]})
    blob(blobs, "ut_MS2_50_0_50_1_001.npy", {0: [3, 9, 3]})
    with open(clusters / "final_matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ms1_name", "ms2_name"])
        w.writerow(["ut_MS1_100_0_100_1_001.npy", "ut_MS2_50_0_50_1_001.npy"])
    peak_table.main(config(tmp_path, "dia", ["QC01"]))
    assert read_rows(tmp_path)[0][-1] == "50.030;300.0"

def test_iter_dda_ms2_reads_every_fragment_scan_with_its_precursor(tmp_path):
    write_mzml(tmp_path / "QC.mzML", [
        (1.0, 1, "scan=1", [100.05], [900]),
        (1.2, 2, "scan=2", [50.03, 80.10], [300, 40], 100.0501),
        (1.4, 2, "scan=3", [60.02], [120]), 
        (1.6, 2, "scan=4", [70.01], [700], 268.1904),
    ])
    got = list(peak_table.iter_dda_ms2(str(tmp_path / "QC.mzML")))
    assert [r["scan"] for r in got] == [1, 3]
    assert [round(r["precursor_mz"], 4) for r in got] == [100.0501, 268.1904]
    assert [round(r["rt"], 2) for r in got] == [1.2, 1.6]
    assert [list(np.round(r["mz"], 2)) for r in got] == [[50.03, 80.10], [70.01]]
    assert [list(r["intensity"]) for r in got] == [[300, 40], [700]]

@pytest.fixture
def dda(tmp_path):
    mzml = tmp_path / "mzml"; mzml.mkdir()
    blobs = tmp_path / "labels" / "single_labels_overlap"; blobs.mkdir(parents=True)
    blob(blobs, "ut_MS1_100_0_100_1_001.npy", {0: [10, 50, 20]})
    write_mzml(mzml / "S02.mzML", [(4.0, 1, "scan=1", [100.06], [10])])
    return tmp_path

def dda_mzml(tmp_path, ms2):
    write_mzml(tmp_path / "mzml" / "QC.mzML",
               [(4.0, 1, "scan=1", [100.05], [900])] + ms2)

def test_peak_table_dda_reads_the_ms2_of_the_matching_precursor(dda):
    dda_mzml(dda, [(4.0, 2, "scan=2", [50.03, 80.1], [300, 40], 100.0501)])
    peak_table.main(config(dda, "dda", ["QC"]))
    assert read_rows(dda) == [["100.0500", "900.0", "100.0", "100.1", "4.00", "3.00", "5.00",
                               "50.030;300.0 80.100;40.0"]]

def test_peak_table_dda_ignores_a_precursor_outside_the_ppm_window(dda):
    dda_mzml(dda, [(4.0, 2, "scan=2", [50.03], [300], 100.06)])
    peak_table.main(config(dda, "dda", ["QC"]))
    assert read_rows(dda)[0][-1] == ""


def test_peak_table_dda_takes_the_scan_nearest_the_peak(dda):
    dda_mzml(dda, [
        (3.5, 2, "scan=2", [50.03], [300], 100.05),
        (5.0, 2, "scan=3", [77.07], [900], 100.05),
    ])
    peak_table.main(config(dda, "dda", ["QC"]))
    assert read_rows(dda)[0][-1] == "50.030;300.0"

def test_peak_table_dda_falls_back_to_the_nearest_ms2_outside_the_window(dda):
    dda_mzml(dda, [(8.0, 2, "scan=2", [50.03], [300], 100.05)])
    peak_table.main(config(dda, "dda", ["QC"]))
    assert read_rows(dda)[0][-1] == "50.030;300.0"

    peak_table.main(config(dda, "dda", ["QC"], ms2_rt_tol=1.0))
    assert read_rows(dda)[0][-1] == ""

def test_peak_table_dda_has_empty_ms2_column_without_a_precursor(dda):
    dda_mzml(dda, [(4.2, 2, "scan=2", [50.03], [300])])
    peak_table.main(config(dda, "dda", ["QC"]))
    assert read_rows(dda) == [["100.0500", "900.0", "100.0", "100.1", "4.00", "3.00", "5.00", ""]]

def test_peak_table_mz_columns_follow_the_mass_tolerance(dda):
    dda_mzml(dda, [])
    blobs = dda / "labels" / "single_labels_overlap"
    for p in blobs.glob("*.npy"):
        p.unlink()
    blob(blobs, "ut_MS1_100_05_100_10_001.npy", {0: [10, 50, 20]})
    cfg = config(dda, "dda", ["QC"])
    cfg["images"]["mass_tol"] = 0.05
    peak_table.main(cfg)
    assert read_rows(dda)[0][2:4] == ["100.05", "100.10"]