import os
import sys
import time
import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import original
import optimized

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _time(mod, out_dir, cfg):
    t0 = time.time()
    mod.run(output_dir=out_dir, **cfg)
    return time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1, help="number of files (0=all)")
    ap.add_argument("--mz-range", nargs=2, type=float, default=[50, 1200])
    ap.add_argument("--rt-range", nargs=2, type=float, default=[100, 1100])
    ap.add_argument("--rt-bin-size", type=float, default=0.73)
    ap.add_argument("--mass-tol", type=float, default=0.1)
    ap.add_argument("--report", default=os.path.join(ROOT, "output", "time_benchmark.txt"))
    a = ap.parse_args()

    cfg = dict(input_dir=os.path.join(ROOT, "input", "mzml"),
               rt_range=a.rt_range, rt_bin_size=a.rt_bin_size,
               mz_range=a.mz_range, mass_tol=a.mass_tol, limit=a.limit)
    n = a.limit if a.limit else len(
        optimized.selecting_files(cfg["input_dir"], optimized.default_suspect(cfg["input_dir"])))

    print(f"Benchmark: {n} file(s), m/z {a.mz_range} ...", flush=True)
    t_opt = _time(optimized, os.path.join(ROOT, "output", "optimized"), cfg)
    t_orig = _time(original, os.path.join(ROOT, "output", "original"), cfg)

    lines = [
        "=" * 60,
        "TIME BENCHMARK — original vs optimized",
        "=" * 60,
        f"generated   : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"files       : {n}",
        f"m/z range   : {a.mz_range}",
        f"rt range    : {a.rt_range}   rt_bin_size: {a.rt_bin_size}   mass_tol: {a.mass_tol}",
        "-" * 60,
        f"original    : {t_orig:8.1f} s   ({t_orig/n:6.1f} s/file)",
        f"optimized   : {t_opt:8.1f} s   ({t_opt/n:6.1f} s/file)",
        f"speedup     : {t_orig/max(t_opt, 1e-9):8.1f}x",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {a.report}")

if __name__ == "__main__":
    main()
