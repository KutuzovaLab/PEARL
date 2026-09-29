import csv

import numpy as np
import pyopenms as oms
import pytest

from pearl import peak_table

HEADER = ["MS1 m/z", "MS1 intensity", "mz min", "mz max", "RT", "RT min", "RT max", "MS2 spectrum"]
RT_RANGE, RT_BIN = [0, 10], 1.0          # rt_bins 0, 1, ..., 10


def write_mzml(path, spectra):
    exp = oms.MSExperiment()
    for rt, level, native_id, mz, inten in spectra:
        sp = oms.MSSpectrum()
        sp.setRT(rt)
        sp.setMSLevel(level)
        sp.setNativeID(native_id)
        sp.set_peaks((np.array(mz, dtype=float), np.array(inten, dtype=float)))
        exp.addSpectrum(sp)
    oms.MzMLFile().store(str(path), exp)


def blob(folder, name, rows):
    arr = np.zeros((2, 11), dtype=np.float32)
    for sample, values in rows.items():
        arr[sample, 3:6] = values                # RT window 3-5 s
    np.save(folder / name, arr)


def config(tmp_path, acquisition, qc_sample):
    return {
        "images": {"acquisition": acquisition, "mzml_path": str(tmp_path / "mzml"),
                   "rt_range": RT_RANGE, "rt_bin_size": RT_BIN, "files": None},
        "labels": {"output_dir": str(tmp_path / "labels")},
        "matching": {"output_dir": str(tmp_path / "clusters")},
        "peaks": {"output": str(tmp_path / "peaks.csv"), "qc_sample": qc_sample},
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
    # QC01: in the RT window the highest MS1 point in 100.0-100.1 is 100.05 @ 900; the brighter 100.07 @ 99999
    # lies outside it. MS2 (function=2) 50.03 @ 300 belongs to the fragment blob.
    write_mzml(mzml / "QC01.mzML", [
        (4.0, 1, "function=1 process=0 scan=1", [100.02, 100.05, 150.0], [500, 900, 10000]),
        (4.0, 2, "function=2 process=0 scan=2", [50.03, 60.0], [300, 5000]),
        (8.0, 1, "function=1 process=0 scan=3", [100.07], [99999]),
    ])
    # S02: the same peak at a different intensity, to tell which run was read
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


def test_peak_table_dda_has_empty_ms2_column(tmp_path):
    mzml = tmp_path / "mzml"; mzml.mkdir()
    blobs = tmp_path / "labels" / "single_labels_overlap"; blobs.mkdir(parents=True)
    write_mzml(mzml / "QC.mzML", [
        (4.0, 1, "scan=1", [100.05], [900]),
        (4.2, 2, "scan=2", [50.03], [300]),        # DDA MS2 is not imaged and never associated
    ])
    write_mzml(mzml / "S02.mzML", [(4.0, 1, "scan=1", [100.06], [10])])
    blob(blobs, "ut_MS1_100_0_100_1_001.npy", {0: [10, 50, 20]})
    peak_table.main(config(tmp_path, "dda", ["QC"]))
    assert read_rows(tmp_path) == [["100.0500", "900.0", "100.0", "100.1", "4.00", "3.00", "5.00", ""]]
