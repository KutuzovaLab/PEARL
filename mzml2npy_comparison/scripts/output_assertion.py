import os
import sys
import glob
import datetime
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "input", "gold_standard")

def compare_to_golden(out_dir, golden_dir):
    cmp_cnt = {"images": 0, "labels": 0}
    eq_cnt = {"images": 0, "labels": 0}
    maxd = 0.0
    mism = []
    for kind in ("images", "labels"):
        for gp in sorted(glob.glob(os.path.join(golden_dir, kind, "*.npy"))):
            name = os.path.basename(gp)
            op = os.path.join(out_dir, kind, name)
            if not os.path.exists(op):
                mism.append((f"{kind}/{name}", "missing in output"))
                continue
            g = np.load(gp)
            o = np.load(op)
            g = g[:o.shape[0]]
            cmp_cnt[kind] += 1
            if np.array_equal(o, g):
                eq_cnt[kind] += 1
            else:
                d = float(np.abs(o.astype("f8") - g.astype("f8")).max())
                maxd = max(maxd, d)
                if len(mism) < 20:
                    mism.append((f"{kind}/{name}", f"differ (max|Δ|={d:.3g})"))
    return {"images_compared": cmp_cnt["images"], "images_identical": eq_cnt["images"],
            "labels_compared": cmp_cnt["labels"], "labels_identical": eq_cnt["labels"],
            "max_abs_diff": maxd, "mismatches": mism}

def _passed(s):
    return (s["images_compared"] > 0
            and s["images_identical"] == s["images_compared"]
            and s["labels_identical"] == s["labels_compared"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--optimized-dir", default=os.path.join(ROOT, "output", "optimized"))
    ap.add_argument("--original-dir", default=os.path.join(ROOT, "output", "original"))
    ap.add_argument("--report", default=os.path.join(ROOT, "output", "output_assertion.txt"))
    a = ap.parse_args()

    stamp = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    lines = ["=" * 62, "OUTPUT ASSERTION — vs golden standard", "=" * 62,
             f"generated : {stamp}",
             f"golden    : {os.path.relpath(a.golden, ROOT)}", "-" * 62]

    all_ok = True
    checked_any = False
    for label, d in (("optimized", a.optimized_dir), ("original", a.original_dir)):
        if not os.path.isdir(os.path.join(d, "images")):
            lines.append(f"{label:9s}: (no output at {os.path.relpath(d, ROOT)} — skipped)")
            continue
        checked_any = True
        s = compare_to_golden(d, a.golden)
        ok = _passed(s)
        all_ok = all_ok and ok
        lines.append(f"{label:9s}: {'PASS' if ok else 'FAIL'}  "
                     f"images {s['images_identical']}/{s['images_compared']}  "
                     f"labels {s['labels_identical']}/{s['labels_compared']}  "
                     f"max|Δ|={s['max_abs_diff']:.3g}")
        for name, reason in s["mismatches"][:10]:
            lines.append(f"             - {name}: {reason}")
    if not checked_any:
        lines.append("no outputs found — run main.py or scripts/optimized.py first")
    lines.append("=" * 62)

    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {a.report}")
    sys.exit(0 if (checked_any and all_ok) else 1)

if __name__ == "__main__":
    main()
