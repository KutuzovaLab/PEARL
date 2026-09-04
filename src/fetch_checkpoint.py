import argparse, os, shutil, sys

def parse_hf(url):
    rest = url[len("hf://"):]
    parts = rest.split("/")
    if len(parts) < 3:
        raise SystemExit(f"expected hf://<owner>/<repo>/<file>, got {url!r}")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return repo_id, filename

def main():
    ap = argparse.ArgumentParser(description="Fetch a checkpoint from Hugging Face.")
    ap.add_argument("url", help="hf://owner/repo/path.ckpt")
    ap.add_argument("--out", default=None, help="destination (default checkpoints/<basename>)")
    a = ap.parse_args()
    if not a.url.startswith("hf://"):
        raise SystemExit("url must start with hf://")
    from huggingface_hub import hf_hub_download

    repo_id, filename = parse_hf(a.url)
    out = a.out or os.path.join("checkpoints", os.path.basename(filename))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        print(f"already present: {out}", file=sys.stderr)
        print(out)
        return
    print(f"downloading {repo_id}:{filename} ...", file=sys.stderr)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy2(cached, out)
    print(f"saved -> {out}", file=sys.stderr)
    print(out)

if __name__ == "__main__":
    main()