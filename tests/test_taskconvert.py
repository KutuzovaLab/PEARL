import json
import numpy as np
from pearl import taskconvert

def test_inference_writes_npy_and_marks_extension(tmp_path, monkeypatch):
    src = tmp_path / "eic"; src.mkdir()
    for name in ("MS1_100_0_100_1", "MS2_200_0_200_1"):
        np.save(src / f"{name}.npy", np.ones((8, 40), dtype=np.float32))
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setenv("YUCCA_RAW_DATA", str(raw))

    taskconvert.convert_inference(str(src), "Task999_ut", prefix="ut")

    ts = raw / "Task999_ut" / "imagesTs"
    npys = list(ts.glob("*.npy"))
    assert len(npys) == 2
    assert all(p.name.endswith("_000.npy") for p in npys)
    dj = json.load(open(raw / "Task999_ut" / "dataset.json"))
    assert dj["image_extension"] == "npy"

def test_inference_skips_all_zero_slices(tmp_path, monkeypatch, capsys):
    src = tmp_path / "eic"; src.mkdir()
    np.save(src / "MS1_100_0_100_1.npy", np.zeros((8, 40), dtype=np.float32))
    one_pixel = np.zeros((8, 40), dtype=np.float32); one_pixel[3, 17] = 1e-3
    np.save(src / "MS1_100_1_100_2.npy", one_pixel)
    np.save(src / "MS1_100_2_100_3.npy", np.ones((8, 40), dtype=np.float32))
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setenv("YUCCA_RAW_DATA", str(raw))

    taskconvert.convert_inference(str(src), "Task999_ut", prefix="ut")

    ts = raw / "Task999_ut" / "imagesTs"
    assert sorted(p.name for p in ts.glob("*.npy")) == ["ut_MS1_100_1_100_2_000.npy", "ut_MS1_100_2_100_3_000.npy"]
    assert np.array_equal(np.load(ts / "ut_MS1_100_1_100_2_000.npy"), one_pixel)
    out = capsys.readouterr().out
    assert "skipped 1 all-zero slice(s)" in out
    assert "inference task ready: 2 images" in out

def test_synthetic_holds_out_20_percent(tmp_path, monkeypatch):
    src = tmp_path / "synthetic"
    for sub in ("resolved_low", "sparse_high_low"):
        (src / sub / "images").mkdir(parents=True)
        (src / sub / "labels" / "v3").mkdir(parents=True)
        for i in range(10):
            np.save(src / sub / "images" / f"{i:05d}.npy", np.ones((8, 40), dtype=np.float32))
            np.save(src / sub / "labels" / "v3" / f"{i:05d}.npy", np.ones((8, 40), dtype=np.uint8))
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setenv("YUCCA_RAW_DATA", str(raw))

    taskconvert.convert_synthetic(str(src), "Task998_ut")

    base = raw / "Task998_ut"
    train = sorted(p.name for p in (base / "imagesTr").glob("*.npy"))
    held = sorted(p.name for p in (base / "imagesTs").glob("*.npy"))
    assert (len(train), len(held)) == (16, 4)
    assert not set(train) & set(held)
    assert len(list((base / "labelsTr").glob("*.npy"))) == 16 and len(list((base / "labelsTs").glob("*.npy"))) == 4

    # same seed, same split
    raw2 = tmp_path / "raw2"; raw2.mkdir()
    monkeypatch.setenv("YUCCA_RAW_DATA", str(raw2))
    taskconvert.convert_synthetic(str(src), "Task998_ut")
    assert sorted(p.name for p in (raw2 / "Task998_ut" / "imagesTs").glob("*.npy")) == held

def test_taskconvert_labels_are_three_class():
    assert taskconvert.LABELS == {0: "no signal", 1: "signal", 2: "overlap"}

def test_inference_refuses_an_m_z_window_with_no_signal(tmp_path, monkeypatch):
    import pytest
    src = tmp_path / "eic"; src.mkdir()
    for i in range(3):
        np.save(src / f"MS1_100_{i}_100_{i+1}.npy", np.zeros((8, 40), dtype=np.float32))
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setenv("YUCCA_RAW_DATA", str(raw))

    with pytest.raises(SystemExit, match="no signal in this m/z range"):
        taskconvert.convert_inference(str(src), "Task999_ut", prefix="ut")