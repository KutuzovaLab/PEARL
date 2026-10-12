import subprocess
import sys
import numpy as np
import pytest
import yaml
from pearl.simulating_data_paper import generate_dataset

def _gen_ours(out_dir, n=3):
    c = yaml.safe_load(open("configs/synthetic.yaml"))["main_equivalent"]
    c["out_dir"] = out_dir
    c["subsets"][0]["n_images"] = n
    generate_dataset(c)

def test_generator_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _gen_ours(str(a)); _gen_ours(str(b))
    for i in range(3):
        f = f"{70001+i:05d}.npy"
        assert np.array_equal(np.load(a/"original/images"/f), np.load(b/"original/images"/f))

def test_generator_byte_identical_to_main(tmp_path):
    try:
        src = subprocess.check_output(["git", "show", "main:simulating_data_paper.py"],
                                      text=True, stderr=subprocess.DEVNULL)
    except Exception:
        pytest.skip("main branch not available")
    main_dir = tmp_path / "main"
    (main_dir / "images").mkdir(parents=True)
    for v in ("v1", "v2", "v3"):
        (main_dir / "labels" / v).mkdir(parents=True)   # main still writes all three
    src = (src.replace("n_images = 10000", "n_images = 3")
              .replace("# Creating images", "random.seed(0); np.random.seed(0)\n# Creating images"))
    import re
    src = re.sub(r"^img_path = .*", f"img_path = r'{main_dir/'images'}'", src, flags=re.M)
    src = re.sub(r"^lbl_path = .*", f"lbl_path = r'{main_dir/'labels'}'", src, flags=re.M)
    script = tmp_path / "main_gen.py"; script.write_text(src)
    subprocess.check_call([sys.executable, str(script)], stderr=subprocess.DEVNULL)

    ours = tmp_path / "ours"; _gen_ours(str(ours))
    n = 0
    for i in range(3):
        f = f"{70001+i:05d}.npy"
        assert np.array_equal(np.load(main_dir/"images"/f), np.load(ours/"original/images"/f)); n += 1
        # v1 and v2 are gone; v3 - the only version PEARL ever used - must not have moved
        assert np.array_equal(np.load(main_dir/"labels/v3"/f),
                              np.load(ours/"original/labels/v3"/f)); n += 1
    assert n == 6


def test_generator_writes_only_v3_labels(tmp_path):
    _gen_ours(str(tmp_path))
    assert [p.name for p in sorted((tmp_path / "original" / "labels").iterdir())] == ["v3"]
