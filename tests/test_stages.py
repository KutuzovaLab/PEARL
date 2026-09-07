import numpy as np
from src.extracting_labels_erosion import apply_sn_w_filter
from src.matching_MS1_MS2 import image_projection_embedding
from src.mzml_to_npy import _fill_row_sums

def test_sn_w_filter_sample_thres():
    img = np.zeros((4, 6))
    img[:3, 2] = 1        # 3 samples -> passes sample_thres=3
    img[:1, 4] = 1        # 1 sample  -> fails sample_thres=3
    out = apply_sn_w_filter(img, sample_thres=3, label_width=1)
    assert out[:, 2].sum() == 3
    assert out[:, 4].sum() == 0

def test_sn_w_filter_label_width():
    img = np.zeros((5, 12))
    img[:, 1:3] = 1       # width 2 -> dropped
    img[:, 6:11] = 1      # width 5 -> kept
    out = apply_sn_w_filter(img, sample_thres=1, label_width=3)
    assert out[:, 1:3].sum() == 0
    assert out[:, 6:11].sum() == 25

def test_embedding_deterministic_and_normalized(tmp_path):
    arr = np.abs(np.random.RandomState(0).randn(8, 40)).astype(np.float32)
    p = tmp_path / "MS1_100_0_100_1_000.npy"; np.save(p, arr)
    v1 = image_projection_embedding(str(p))
    v2 = image_projection_embedding(str(p))
    assert np.array_equal(v1, v2)
    assert v1.shape[0] == arr.shape[0] + arr.shape[1]
    assert abs(v1.max() - 1.0) < 1e-12

def test_fill_row_sums_float64_accumulation():
    inten = np.full(1000, 0.1, dtype=np.float32)
    lo = np.array([0], dtype=np.int64)
    hi = np.array([1000], dtype=np.int64)
    out = np.zeros(1, dtype=np.float64)
    _fill_row_sums(lo, hi, inten, out)
    assert out[0] == float(np.float32(0.1)) * 1000