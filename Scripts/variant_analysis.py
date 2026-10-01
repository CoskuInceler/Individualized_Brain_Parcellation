"""
BORDER SHIFTS AND ECTOPIC PARCELS
=================================
Separates the two reasons a parcel can come back in more than one piece.



When an individualised parcellation splits a parcel, the fragment is
either sitting just outside the main body, which means the border moved
relative to the group atlas, or it is somewhere else entirely, which is
the ectopic case: the same functional territory appearing as an island
in a different part of cortex.



The distinction matters because only the second is a genuinely
individual feature of topography rather than a small displacement, and
because a method can only detect it if nothing in its construction
forces parcels to stay in one piece.



    fragment within 30 mm of the main body   border shift
    fragment beyond 30 mm                    ectopic



Distance is geodesic along the surface, between the closest pair of
vertices. Fragments smaller than 10 vertices are counted separately,
since at that size the call is not reliable either way.



Output: Outputs/<variant>/Variants/<method>/{subject}.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components


import config
import utils

ECTOPIC_MM = 30.0  # beyond this, a fragment counts as ectopic
MIN_SIZE = 10  # smaller fragments are reported but not classified
SESSION = "ALL"


METHODS = {
    "Method_0_Schaefer": None,
    "Method_1_gMSHBM": ("Labels", False),
    "Method_2_AGP": ("Labels", True),
    "Method_3_SLIC_F": ("Labels", True),
    "Method_4_SLIC_C": ("Labels", True),
    "Method_5_Gordon": ("Labels_200", True),
}


def atlas_labels(subject_id):
    """Schaefer labels in valid-vertex space."""
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    vL = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    vR = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    aL, aR = utils.load_and_filter_atlas(config.SCHAEFER_200_FILE, vL, vR)
    return np.concatenate([aL, aR])


def load_labels(subject_id, method):
    """Parcel labels for one subject and method, in valid-vertex space."""
    if METHODS[method] is None:
        return atlas_labels(subject_id)

    folder, per_session = METHODS[method]
    lab_dir = config.OUTPUTS_DIR / method / folder
    name = (
        f"{subject_id}_{SESSION}_labels.npy"
        if per_session
        else f"{subject_id}_labels.npy"
    )
    return np.load(lab_dir / name)


def classify_hemisphere(labels, adj, geo):
    """
    Classify every fragment of every parcel in one hemisphere.



    Parameters
    ----------
    labels : (V,) parcel labels for this hemisphere, 0 = unassigned
    adj    : (V, V) sparse adjacency for this hemisphere
    geo    : (V, V) geodesic distances for this hemisphere



    Returns
    -------
    list of dicts, one per fragment beyond the main body
    """
    out = []

    for pid in np.unique(labels[labels > 0]):
        idx = np.flatnonzero(labels == pid)
        if idx.size == 0:
            continue

        n_parts, part = connected_components(adj[idx][:, idx], directed=False)
        if n_parts == 1:
            continue

        sizes = np.bincount(part)
        main = int(np.argmax(sizes))
        main_idx = idx[part == main]

        for p in range(n_parts):
            if p == main:
                continue

            frag_idx = idx[part == p]

            # closest approach between the fragment and the main body
            distance = float(geo[np.ix_(frag_idx, main_idx)].min())

            if frag_idx.size < MIN_SIZE:
                kind = "small"
            elif distance >= ECTOPIC_MM:
                kind = "ectopic"
            else:
                kind = "border"

            out.append(
                {
                    "parcel": int(pid),
                    "fragment_size": int(frag_idx.size),
                    "main_size": int(sizes[main]),
                    "distance_mm": round(distance, 2),
                    "kind": kind,
                }
            )

    return out


def analyse(subject_id, method):
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    valid_L = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    n_L = len(valid_L)

    labels = load_labels(subject_id, method)

    fragments = []
    for hemi, start, stop in (("L", 0, n_L), ("R", n_L, len(labels))):
        adj = load_npz(config.INPUTS_DIR / f"adjacency_{hemi}.npz").tocsr()
        geo = np.load(
            config.INPUTS_DIR / f"geodesic_{hemi}.npy", mmap_mode="r"
        )

        found = classify_hemisphere(labels[start:stop], adj, np.asarray(geo))
        for f in found:
            f["hemisphere"] = hemi
        fragments.extend(found)
        del geo

    return labels, fragments


def summarise(subject_id, method, labels, fragments):
    """One row per subject, counting each kind of fragment."""
    n_parcels = len(np.unique(labels[labels > 0]))
    kinds = [f["kind"] for f in fragments]

    ectopic = [f for f in fragments if f["kind"] == "ectopic"]
    border = [f for f in fragments if f["kind"] == "border"]

    parcels_with_ectopic = len({f["parcel"] for f in ectopic})

    return {
        "subject": subject_id,
        "method": method,
        "variant": config.VARIANT,
        "n_parcels": n_parcels,
        "n_fragments": len(fragments),
        "n_border": kinds.count("border"),
        "n_ectopic": kinds.count("ectopic"),
        "n_small": kinds.count("small"),
        "pct_parcels_with_ectopic": round(
            100.0 * parcels_with_ectopic / n_parcels, 2
        ),
        "mean_border_distance": (
            round(float(np.mean([f["distance_mm"] for f in border])), 2)
            if border
            else np.nan
        ),
        "mean_ectopic_distance": (
            round(float(np.mean([f["distance_mm"] for f in ectopic])), 2)
            if ectopic
            else np.nan
        ),
        "max_ectopic_distance": (
            round(max(f["distance_mm"] for f in ectopic), 2)
            if ectopic
            else np.nan
        ),
        "mean_ectopic_size": (
            round(float(np.mean([f["fragment_size"] for f in ectopic])), 1)
            if ectopic
            else np.nan
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Border and ectopic variants."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument(
        "--save-fragments",
        action="store_true",
        help="also write the per-fragment detail",
    )
    args = parser.parse_args()

    out_dir = config.OUTPUTS_DIR / "Variants" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    labels, fragments = analyse(args.subject, args.method)
    row = summarise(args.subject, args.method, labels, fragments)

    pd.DataFrame([row]).to_csv(out_dir / f"{args.subject}.csv", index=False)

    if args.save_fragments and fragments:
        detail = pd.DataFrame(fragments)
        detail.insert(0, "subject", args.subject)
        (out_dir / "Fragments").mkdir(exist_ok=True)
        detail.to_csv(
            out_dir / "Fragments" / f"{args.subject}.csv", index=False
        )

    print(f"Variant {config.VARIANT} | {args.method} | Subject {args.subject}")
    for k, v in row.items():
        if k not in ("subject", "method", "variant"):
            print(f"  {k:26s} {v}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
