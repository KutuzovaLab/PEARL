import numpy as np
import pyopenms as oms

from pearl.mzml_to_npy import mass_edges, read_mzml_files

RT_BIN = 0.68
RT_RANGE = [0, 10]                  # rt bins 0 ... 10.2, last bin [9.52, 10.2)
MZ_RANGE = [90, 94.15]              # 42 slices; np.arange puts the 93.9 slice at 93.89999999999978


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
        (1.0, 1, "scan=1", [94.0], [500]),       # exactly on the 93.9 / 94.0 slice edge
        (1.0, 2, "scan=2", [91.05], [7777]),     # MS2 is not imaged for DDA
        (10.1, 1, "scan=3", [91.05], [300]),     # past rt_range[1], before the last bin edge
        (10.3, 1, "scan=4", [91.05], [900]),     # past the last bin edge
    ])
    ms1, ms2 = read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1, acquisition="dda")
    edges = mass_edges(MZ_RANGE, 0.1, "dda")
    assert ms2 is None
    assert ms1.shape == (42, 1, 15)
    col = int(1.0 // RT_BIN)
    assert ms1[slice_index(edges, 93.9), 0, col] == 500          # upper edge inclusive on the rounded edge
    assert ms1[slice_index(edges, 94.0), 0, col] == 500          # and the lower edge of the next slice
    assert ms1[slice_index(edges, 91.0), 0, -1] == 300           # the 10.1 s scan lands in the last bin
    assert ms1[slice_index(edges, 91.0), 0].sum() == 300         # the 10.3 s scan and the MS2 scan are dropped


def test_dia_follows_the_original_dia_converter(tmp_path):
    write_mzml(tmp_path / "S1.mzML", [
        (1.0, 1, "function=1 process=0 scan=1", [94.0], [500]),
        (1.0, 2, "function=2 process=0 scan=2", [91.05], [40]),
        (10.1, 1, "function=1 process=0 scan=3", [91.05], [300]),   # past rt_range[1]: dropped for DIA
    ])
    ms1, ms2 = read_mzml_files(str(tmp_path), ["S1"], RT_BIN, RT_RANGE, MZ_RANGE, 0.1, acquisition="dia")
    edges = mass_edges(MZ_RANGE, 0.1, "dia")
    col = int(1.0 // RT_BIN)
    assert ms1[slice_index(edges, 93.9), 0, col] == 0            # 94.0 is just above the drifted 93.9 slice
    assert ms1[slice_index(edges, 94.0), 0, col] == 500
    assert ms1[slice_index(edges, 91.0), 0].sum() == 0
    assert ms2[slice_index(edges, 91.0), 0, col] == 40
