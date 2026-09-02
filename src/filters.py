import numpy as np

# --------------------------------------------------------------------------- labels
# Drop RT columns predicted in fewer than N samples.
def min_samples(mask, min_samples=1, **_):
    keep = mask.sum(axis=0) >= min_samples
    out = np.zeros_like(mask)
    out[:, keep] = mask[:, keep]
    return out

# Drop predicted RT runs narrower than `min_width` pixels. (`label_width` from preparing_for_clustering.py)
def min_width(mask, min_width=1, **_):
    col = mask.any(axis=0).astype(np.int8)
    out = np.zeros_like(mask)
    start = None
    for i, on in enumerate(np.append(col, 0)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_width:
                out[:, start:i] = mask[:, start:i]
            start = None
    return out

# --------------------------------------------------------------------------- links

# Keep only MS2->MS1 links closer than a Euclidean cutoff.
def max_distance(matches, max_distance=np.inf, **_):
    return matches[matches["distance"] <= max_distance].reset_index(drop=True)

# Keep only the k nearest MS1 neighbours per MS2 group.
def top_k(matches, top_k=None, **_):
    if not top_k:
        return matches
    return (matches.sort_values("distance")
                   .groupby("ms2_name", sort=False)
                   .head(top_k)
                   .reset_index(drop=True))

LABEL_FILTERS = {"min_samples": min_samples, "min_width": min_width}
LINK_FILTERS = {"max_distance": max_distance, "top_k": top_k}

def apply_filters(obj, specs, registry, verbose=True, what=""):
    if not specs:
        return obj
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("name", None)
        if name not in registry:
            raise ValueError(f"unknown {what} filter '{name}'; known: {sorted(registry)}")
        before = len(obj) if hasattr(obj, "__len__") else None
        obj = registry[name](obj, **spec)
        if verbose:
            after = len(obj) if hasattr(obj, "__len__") else None
            extra = f" ({before} -> {after})" if before is not None and after != before else ""
            print(f"  filter [{what}] {name}{spec and ' ' + str(spec) or ''}{extra}", flush=True)
    return obj

def apply_label_filters(mask, cfg, verbose=True):
    return apply_filters(mask, (cfg.get("filters") or {}).get("labels"),
                         LABEL_FILTERS, verbose=verbose, what="label")

def apply_link_filters(matches, cfg, verbose=True):
    return apply_filters(matches, (cfg.get("filters") or {}).get("links"),
                         LINK_FILTERS, verbose=verbose, what="link")
