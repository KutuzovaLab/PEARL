import os
import json
import hashlib
import argparse
import urllib.request

STUDY = "MTBLS1129"
BASE = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{STUDY}"
HASHES_URL = f"{BASE}/HASHES/data_sha256.json"
DEFAULT_OUT = os.path.join("data", "mzml", "test")

FILES = [f"QC{i:02d}.mzML" for i in range(1, 9)]

def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def download(url, dest, expect=None):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if total:
                    pct = done / total * 100
                    print(f"\r    {os.path.basename(dest)} {pct:5.1f}% "
                          f"({done/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
    print()
    if expect:
        got = sha256(tmp)
        if got != expect:
            os.remove(tmp)
            raise SystemExit(f"checksum mismatch for {dest}\n  expected {expect}\n  got      {got}")
    os.replace(tmp, dest)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4,
                    help="how many QC files to fetch (default 4, ~330 MB)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"target dir (default {DEFAULT_OUT})")
    ap.add_argument("--force", action="store_true", help="re-download even if present and valid")
    ap.add_argument("--list", action="store_true", help="list available files and exit")
    a = ap.parse_args(argv)

    if a.list:
        print(f"{STUDY} QC replicates available: {', '.join(FILES)}")
        return

    if not 1 <= a.n <= len(FILES):
        raise SystemExit(f"--n must be between 1 and {len(FILES)}")

    wanted = FILES[:a.n]
    os.makedirs(a.out, exist_ok=True)

    print(f"MetaboLights {STUDY} -> {a.out}", flush=True)
    print("  fetching checksum manifest ...", flush=True)
    manifest = json.loads(_get(HASHES_URL))

    for name in wanted:
        dest = os.path.join(a.out, name)
        expect = manifest.get(f"FILES/{name}")
        if expect is None:
            raise SystemExit(f"{name} not in {STUDY} manifest -- study contents changed?")

        if os.path.exists(dest) and not a.force:
            if sha256(dest) == expect:
                print(f"  ok (cached)  {name}", flush=True)
                continue
            print(f"  re-fetching {name} (checksum mismatch on disk)", flush=True)

        print(f"  downloading  {name}", flush=True)
        download(f"{BASE}/FILES/{name}", dest, expect=expect)

    have = sorted(f for f in os.listdir(a.out) if f.endswith(".mzML"))
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in have)
    print(f"\n{len(have)} mzML in {a.out} ({total/1e9:.2f} GB), all checksums verified.")
    print("Now run:  ./run_pipeline.sh test", flush=True)

if __name__ == "__main__":
    main()
