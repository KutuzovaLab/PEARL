import numpy as np
from pearl.matching_MS1_MS2 import image_projection_embedding
from pearl.mzml_to_npy import _fill_row_sums

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

def test_matching_skips_instead_of_killing_the_run(tmp_path, capsys):
    from pearl.matching_MS1_MS2 import main
    blobs = tmp_path / "single_labels_overlap"; blobs.mkdir()
    np.save(blobs / "ut_MS1_100_0_100_1_001.npy", np.ones((2, 8), dtype=np.float32))
    main({"input_dir": str(blobs), "output_dir": str(tmp_path / "clusters")})
    out = capsys.readouterr().out
    assert "no MS2 label images" in out
    assert not (tmp_path / "clusters" / "final_matches.csv").exists()
