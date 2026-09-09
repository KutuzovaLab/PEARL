import os

import yucca
import yucca.functional.utils.loading as loading

MANAGER = os.path.join(os.path.dirname(yucca.__file__),
                       "pipeline", "managers", "YuccaManager.py")
YUCCANET = os.path.join(os.path.dirname(yucca.__file__),
                        "modules", "networks", "networks", "YuccaNet.py")

NPY_MARKER = "# PEARL: .npy support"
NPY_ANCHOR = "    ext = imagepath.split(os.extsep, 1)[1]\n"
NPY_BRANCH = (
    '    ext = imagepath.split(os.extsep, 1)[1]\n'
    '    if ext == "npy":  ' + NPY_MARKER + '\n'
    '        return np.load(imagepath).astype(dtype)\n'
)

DEVICE_MARKER = "# PEARL: pinned device and precision"
DEVICE_ANCHOR = (
    '        self.accelerator = "cpu" if torch.backends.mps.is_available()'
    ' and self.model_dimensions == "3D" else "auto"\n'
)
DEVICE_BRANCH = DEVICE_ANCHOR + (
    "        import os  " + DEVICE_MARKER + "\n"
    '        self.accelerator = os.environ.get("PEARL_ACCELERATOR", "cpu")\n'
    '        self.precision = os.environ.get("PEARL_PRECISION", "32-true")\n'
)

BATCH_MARKER = "# PEARL: batched sliding window"
BATCH_ANCHOR = (
    "        for xs in x_steps:\n"
    "            for ys in y_steps:\n"
    "                # check if out of bounds\n"
    "                out = self.forward(data[:, :, xs : xs + px, ys : ys + py])\n"
    "                canvas[:, :, xs : xs + px, ys : ys + py] += out\n"
    "        return canvas"
)
BATCH_BRANCH = (
    "        import os  " + BATCH_MARKER + "\n"
    "        coords = [(xs, ys) for xs in x_steps for ys in y_steps]\n"
    '        chunk = int(os.environ.get("PEARL_WINDOW_BATCH", "8"))\n'
    "        for i in range(0, len(coords), chunk):\n"
    "            group = coords[i : i + chunk]\n"
    "            batch = torch.cat(\n"
    "                [data[:, :, xs : xs + px, ys : ys + py] for xs, ys in group], dim=0\n"
    "            )\n"
    "            outs = self.forward(batch)\n"
    "            for j, (xs, ys) in enumerate(group):\n"
    "                canvas[:, :, xs : xs + px, ys : ys + py] += outs[j : j + 1]\n"
    "        return canvas"
)

def _patch(path, marker, anchor, branch, what, quiet=False):
    src = open(path).read()
    if marker in src:
        if not quiet:
            print(f"already patched ({what}): {path}")
        return
    if anchor not in src:
        raise SystemExit(f"could not find the {what} anchor in {path}; yucca version changed?")
    open(path, "w").write(src.replace(anchor, branch, 1))
    print(f"patched {what} into {path}")

def main(quiet=False):
    _patch(loading.__file__, NPY_MARKER, NPY_ANCHOR, NPY_BRANCH, ".npy support", quiet)
    _patch(MANAGER, DEVICE_MARKER, DEVICE_ANCHOR, DEVICE_BRANCH, "device and precision", quiet)
    _patch(YUCCANET, BATCH_MARKER, BATCH_ANCHOR, BATCH_BRANCH, "batched sliding window", quiet)

if __name__ == "__main__":
    main()
