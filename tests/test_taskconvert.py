import json
import numpy as np
from src import taskconvert

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

def test_taskconvert_labels_are_three_class():
    assert taskconvert.LABELS == {0: "no signal", 1: "signal", 2: "overlap"}