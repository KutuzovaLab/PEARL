import yucca.functional.utils.loading as loading

MARKER = "# PEARL: .npy support"
BRANCH = (
    '    ext = imagepath.split(os.extsep, 1)[1]\n'
    '    if ext == "npy":  ' + MARKER + '\n'
    '        return np.load(imagepath).astype(dtype)\n'
)
ANCHOR = "    ext = imagepath.split(os.extsep, 1)[1]\n"

def main():
    path = loading.__file__
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    if ANCHOR not in src:
        raise SystemExit(f"could not find the anchor line in {path}; yucca version changed?")
    open(path, "w").write(src.replace(ANCHOR, BRANCH, 1))
    print(f"patched .npy support into {path}")

if __name__ == "__main__":
    main()