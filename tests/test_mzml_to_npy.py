import numpy as np
import pyopenms as oms
import pytest

from pearl.mzml_to_npy import mass_edges, read_mzml_files

RT_BIN = 0.68
RT_RANGE = [0, 10]
MZ_RANGE = [90, 94.15]

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

def slice_index(edges, lo):
    return int(np.argmin(np.abs(edges - lo)))

def test_mass_edges_dda_rounded_dia_arange():
    dda, dia = mass_edges(MZ_RANGE, 0.1, "dda"), mass_edges(MZ_RANGE, 0.1, "dia")
    assert len(dda) == len(dia) == 42
    assert dda[slice_index(dda, 93.9)] == 93.9
    assert dia[slice_index(dia, 93.9)] + 0.1 < 94.0

def test_dda_follows_the_original_dda_converter(tmp_path):
    write_mzml(tmp_path / "S1.mzML", [
        (1.0, 1, "scan=1", [94.0], [500]),
        (1.0, 2, "scan=2", [91.05], [7777]),
        (10.1, 1, "scan=3", [91.05], [300]),
        (10.3, 1, "scan=4", [91.05], [900]),
    ])
    ms1, ms2 = read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1, acquisition="dda")
    edges = mass_edges(MZ_RANGE, 0.1, "dda")
    assert ms2 is None
    assert ms1.shape == (42, 1, 15)
    col = int(1.0 // RT_BIN)
    assert ms1[slice_index(edges, 93.9), 0, col] == 500 
    assert ms1[slice_index(edges, 94.0), 0, col] == 500
    assert ms1[slice_index(edges, 91.0), 0, -1] == 300
    assert ms1[slice_index(edges, 91.0), 0].sum() == 300

def test_dia_follows_the_original_dia_converter(tmp_path):
    write_mzml(tmp_path / "S1.mzML", [
        (1.0, 1, "function=1 process=0 scan=1", [94.0], [500]),
        (1.0, 2, "function=2 process=0 scan=2", [91.05], [40]),
        (10.1, 1, "function=1 process=0 scan=3", [91.05], [300]),
    ])
    ms1, ms2 = read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1, acquisition="dia")
    edges = mass_edges(MZ_RANGE, 0.1, "dia")
    col = int(1.0 // RT_BIN)
    assert ms1[slice_index(edges, 93.9), 0, col] == 0 
    assert ms1[slice_index(edges, 94.0), 0, col] == 500
    assert ms1[slice_index(edges, 91.0), 0].sum() == 0
    assert ms2[slice_index(edges, 91.0), 0, col] == 40

def waters_like(lock_mz, lock_int):
    spectra = []
    for i in range(12):
        spectra.append((0.1 + i * 0.7, 1, f"function=1 process=0 scan={i+1}", [91.05], [100]))
        spectra.append((0.45 + i * 0.7, 2, f"function=2 process=0 scan={i+1}", [91.05], [50]))
    spectra.append((4.3, 1, "function=3 process=0 scan=1", [lock_mz], [lock_int]))
    return sorted(spectra, key=lambda s: s[0])

def test_dia_lock_mass_channel_is_interpolated_away(tmp_path):
    write_mzml(tmp_path / "S1.mzML", waters_like(91.05, 999999))
    ms1, ms2 = read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1,
                               acquisition="dia")
    row = slice_index(mass_edges(MZ_RANGE, 0.1, "dia"), 91.0)
    assert 999999 not in ms1[row, 0] and 999999 not in ms2[row, 0]
    bin_43 = int(4.3 // RT_BIN)
    assert ms1[row, 0, bin_43] == pytest.approx(100, rel=1e-5)
    assert ms2[row, 0, bin_43] == pytest.approx(50, rel=1e-5)

def test_dia_without_function_ids_splits_by_ms_level(tmp_path):
    write_mzml(tmp_path / "S1.mzML", [
        (1.0, 1, "controllerType=0 controllerNumber=1 scan=1", [91.05], [700]),
        (1.2, 2, "controllerType=0 controllerNumber=1 scan=2", [92.05], [40]),
    ])
    ms1, ms2 = read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1,
                               acquisition="dia")
    edges = mass_edges(MZ_RANGE, 0.1, "dia")
    assert ms1[slice_index(edges, 91.0), 0].sum() == 700
    assert ms1[slice_index(edges, 92.0), 0].sum() == 0
    assert ms2[slice_index(edges, 92.0), 0].sum() == 40
    assert ms2[slice_index(edges, 91.0), 0].sum() == 0

def test_dia_function_numbers_can_be_overridden(tmp_path):
    write_mzml(tmp_path / "S1.mzML", [
        (1.0, 1, "function=5 process=0 scan=1", [91.05], [700]),
        (1.2, 1, "function=6 process=0 scan=1", [92.05], [40]),
    ])
    ms1, ms2 = read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1,
                               acquisition="dia", dia_functions=[5, 6, 0])
    edges = mass_edges(MZ_RANGE, 0.1, "dia")
    assert ms1[slice_index(edges, 91.0), 0].sum() == 700
    assert ms2[slice_index(edges, 92.0), 0].sum() == 40

def test_dia_refuses_a_file_with_an_empty_channel(tmp_path):
    write_mzml(tmp_path / "S1.mzML", [
        (1.0, 1, "controllerType=0 controllerNumber=1 scan=1", [91.05], [700]),
        (1.2, 1, "controllerType=0 controllerNumber=1 scan=2", [92.05], [40]),
    ])
    with pytest.raises(SystemExit, match="MS2 channel is empty"):
        read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1,
                        acquisition="dia")