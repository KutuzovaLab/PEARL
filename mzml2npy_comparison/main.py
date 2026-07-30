#!/usr/bin/env python

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import datetime
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import original
import optimized
import output_assertion as assertion

RT_RANGE = [100, 1100]
RT_BIN_SIZE = 0.73
MZ_RANGE = [50, 1200]
MASS_TOL = 0.1

INPUT = os.path.join(ROOT, "input", "mzml")
SUSPECT = os.path.join(ROOT, "input", "mzml", "Standards_NEG.xlsx")
GOLDEN = os.path.join(ROOT, "input", "gold_standard")
OPT_OUT = os.path.join(ROOT, "output", "optimized")
ORIG_OUT = os.path.join(ROOT, "output", "original")
BENCH_TXT = os.path.join(ROOT, "output", "time_benchmark.txt")
ASSERT_TXT = os.path.join(ROOT, "output", "output_assertion.txt")

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"wrote {path}", flush=True)

def _run(mod, out_dir, limit):
    t0 = time.time()
    files = mod.run(input_dir=INPUT, output_dir=out_dir, suspect_path=SUSPECT,
                    rt_range=RT_RANGE, rt_bin_size=RT_BIN_SIZE,
                    mz_range=MZ_RANGE, mass_tol=MASS_TOL, limit=limit)
    return len(files), time.time() - t0

def _assert_block(label, out_dir):
    s = assertion.compare_to_golden(out_dir, GOLDEN)
    ok = assertion._passed(s)
    line = (f"{label:9s}: {'PASS' if ok else 'FAIL'}  "
            f"images {s['images_identical']}/{s['images_compared']}  "
            f"labels {s['labels_identical']}/{s['labels_compared']}  max|Δ|={s['max_abs_diff']:.3g}")
    return ok, line, s["mismatches"]

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="first N selected files (0=all)")
    ap.add_argument("--skip-original", action="store_true",
                    help="skip the slow original run (compare only optimized)")
    a = ap.parse_args()

    stamp = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    n = a.limit if a.limit else len(optimized.selecting_files(INPUT, SUSPECT))
    print("=" * 64)
    print(f"golden benchmark: {n} file(s), m/z {MZ_RANGE}, golden = input/gold_standard")
    if not a.skip_original:
        print(f"heads-up: the original is ~2 h/file at [50,1200] -> ~{n*2}+ h for the original")
    print("=" * 64, flush=True)

    n_opt, t_opt = _run(optimized, OPT_OUT, a.limit)
    t_orig = None
    if not a.skip_original:
        _, t_orig = _run(original, ORIG_OUT, a.limit)

    bl = ["=" * 60, "TIME BENCHMARK — original vs optimized", "=" * 60,
          f"generated : {stamp}", f"files     : {n_opt}", f"m/z range : {MZ_RANGE}",
          f"rt range  : {RT_RANGE}  rt_bin_size: {RT_BIN_SIZE}  mass_tol: {MASS_TOL}", "-" * 60,
          f"optimized : {t_opt:9.1f} s   ({t_opt/max(n_opt,1):7.1f} s/file)"]
    if t_orig is not None:
        bl.append(f"original  : {t_orig:9.1f} s   ({t_orig/max(n_opt,1):7.1f} s/file)")
        bl.append(f"speedup   : {t_orig/max(t_opt,1e-9):9.1f}x")
    else:
        bl.append("original  : (skipped)")
    bl.append("=" * 60)
    _write(BENCH_TXT, "\n".join(bl))

    al = ["=" * 62, "OUTPUT ASSERTION — vs golden standard", "=" * 62,
          f"generated : {stamp}", f"golden    : input/gold_standard", "-" * 62]
    ok_opt, line_opt, mm_opt = _assert_block("optimized", OPT_OUT)
    al.append(line_opt)
    for nm, r in mm_opt[:8]:
        al.append(f"             - {nm}: {r}")
    overall = ok_opt
    if t_orig is not None:
        ok_orig, line_orig, mm_orig = _assert_block("original", ORIG_OUT)
        al.append(line_orig)
        for nm, r in mm_orig[:8]:
            al.append(f"             - {nm}: {r}")
        overall = overall and ok_orig
    al.append("-" * 62)
    al.append("RESULT: PASS" if overall else "RESULT: FAIL")
    al.append("=" * 62)
    _write(ASSERT_TXT, "\n".join(al))

    print("\nall done.", flush=True)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
