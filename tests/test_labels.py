import numpy as np
import nibabel as nib
import pytest

from pearl import extracting_labels_erosion as labels

def write_case(tmp_path, raw, mask):
    pred = tmp_path / "pred"; pred.mkdir(exist_ok=True)
    rawd = tmp_path / "raw"; rawd.mkdir(exist_ok=True)
    np.save(rawd / "ut_MS1_100_0_100_1_000.npy", raw.astype(np.float32))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)),
             str(pred / "ut_MS1_100_0_100_1.nii.gz"))
    return {"predictions_dir": str(pred), "images_raw": str(rawd),
            "output_dir": str(tmp_path / "out"), "task_name": "ut"}

def test_labels_from_a_single_run_dataset(tmp_path):
    raw = np.zeros((1, 40), dtype=np.float32); raw[0, 10:20] = 500.0
    mask = np.zeros(40, dtype=np.uint8); mask[10:20] = 1      # 1-D, as yucca writes it
    labels.main(write_case(tmp_path, raw, mask))
    blobs = sorted((tmp_path / "out" / "single_labels_overlap").glob("*.npy"))
    assert len(blobs) == 1
    got = np.load(blobs[0])
    assert got.shape == raw.shape
    assert np.array_equal(got, raw)

def test_labels_reject_a_prediction_of_the_wrong_size(tmp_path):
    raw = np.zeros((2, 40), dtype=np.float32); raw[:, 5:9] = 1.0
    with pytest.raises(SystemExit, match="has shape"):
        labels.main(write_case(tmp_path, raw, np.zeros(39, dtype=np.uint8)))
