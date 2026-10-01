"""
SLIC PARCELLATION (Levels 3 and 4)
==================================
Surface-based Simple Linear Iterative Clustering, following



    Wang, Hu & Wang (2016). Parcellating Whole Brain for Individuals by
    Simple Linear Iterative Clustering. ICONIP 2016, LNCS 9949, 131-139.



Two variants sit at different points of the individualisation gradient:



    SLIC-F  functional and spatial distance combined; the spatial term
            keeps parcels compact, so some group-level structure survives
    SLIC-C  functional distance only; nothing constrains parcels to be
            spatially contiguous



The combined distance for vertex i and cluster centre k is



    d = sqrt( d_func^2 / m^2 + d_spatial^2 / S^2 )



where S = sqrt(area / K) is the expected parcel spacing in millimetres and
m plays the same role for the functional term. Functional distance is
Euclidean distance between z-scored time series divided by sqrt(T), which
equals sqrt(2 (1 - r)) and so does not change with the number of
timepoints. m is calibrated per preprocessing variant by
slic_calibrate_m.py.



Outputs per subject and session:
    Method_3_SLIC_F/Labels/{subject}_{session}_labels.npy
    Method_4_SLIC_C/Labels/{subject}_{session}_labels.npy
    plus the corresponding parcel time series
"""

import argparse
import time
import numpy as np


import config
import utils

N_CLUSTERS = 100  # per hemisphere, 200 in total
MAX_ITER = 20
SEED = 42
SESSIONS = ["REST1", "REST2", "ALL"]


def zscore_rows(ts):
    """Z-score each vertex, with vertices along the first axis."""
    ts = np.asarray(ts, dtype=np.float64)
    out = ts - ts.mean(axis=1, keepdims=True)
    sd = out.std(axis=1, keepdims=True)
    sd[sd < np.finfo(np.float32).eps] = 1.0
    return out / sd


def load_m():
    """Compactness parameter for the current preprocessing variant."""
    path = config.INPUTS_DIR / f"slic_m_{config.VARIANT}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing; run slic_calibrate_m.py for this variant first"
        )
    return float(path.read_text().strip())


def load_areas():
    """Cortical surface area per hemisphere, in mm^2."""
    path = config.INPUTS_DIR / "surface_area.txt"
    areas = {}
    for line in path.read_text().split("\n"):
        if line.strip():
            hemi, val = line.split()
            areas[hemi] = float(val)
    return areas


def slic(ts, m, geo=None, area=None, engine="fast"):
    """
    Cluster the vertices of one hemisphere.



    Parameters
    ----------
    ts     : (V, T) time series, vertices along the first axis
    m      : compactness parameter
    geo    : (V, V) geodesic distances; omit for the functional-only variant
    area   : cortical area in mm^2, required alongside geo
    engine : "fast" expands the squared distance into a single matrix
             product; "direct" forms the difference cluster by cluster, as
             in the reference implementation. The two are algebraically the
             same and exist so the fast path can be checked.



    Returns
    -------
    (V,) labels in 0 .. K-1
    """
    # distances are accumulated in float64: in float32 the expansion
    # T - 2 v.c + ||c||^2 loses enough precision to flip occasional
    # assignments, which then compound over iterations
    z = zscore_rows(ts).astype(np.float64)
    n_vert, n_time = z.shape

    spatial = geo is not None
    if spatial:
        S = np.sqrt(area / N_CLUSTERS)

    rng = np.random.default_rng(SEED)
    centre_idx = rng.choice(n_vert, N_CLUSTERS, replace=False)
    centre_ts = z[centre_idx].copy()

    labels = np.full(n_vert, -1, dtype=np.int32)

    for it in range(MAX_ITER):
        if engine == "fast":
            # ||v - c||^2 = ||v||^2 - 2 v.c + ||c||^2, and ||v||^2 = T
            cross = z @ centre_ts.T
            d_func_sq = (
                n_time - 2.0 * cross + (centre_ts**2).sum(axis=1)[None, :]
            )
            np.maximum(d_func_sq, 0.0, out=d_func_sq)
        else:
            d_func_sq = np.empty((n_vert, N_CLUSTERS), dtype=np.float64)
            for k in range(N_CLUSTERS):
                diff = z - centre_ts[k]
                d_func_sq[:, k] = (diff**2).sum(axis=1)

        # scale so the value depends on correlation, not on T
        d_func_sq = d_func_sq / n_time

        if spatial:
            d_spatial = geo[:, centre_idx]
            total = d_func_sq / (m**2) + (d_spatial**2) / (S**2)
        else:
            total = d_func_sq / (m**2)

        new_labels = np.argmin(total, axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(N_CLUSTERS):
            members = np.flatnonzero(labels == k)
            if members.size == 0:
                continue
            centre_ts[k] = z[members].mean(axis=0)
            if spatial:
                sub = geo[np.ix_(members, members)]
                centre_idx[k] = members[np.argmin(sub.sum(axis=1))]

    return labels


def load_session(subject_id, session):
    """Cleaned time series for one session; ALL concatenates both."""
    cleaned = config.OUTPUTS_DIR / "Cleaned"
    if session == "ALL":
        return np.concatenate(
            [
                np.load(cleaned / f"{subject_id}_REST1.npy"),
                np.load(cleaned / f"{subject_id}_REST2.npy"),
            ],
            axis=0,
        )
    return np.load(cleaned / f"{subject_id}_{session}.npy")


def save_result(out_dir, subject_id, session, labels, ts, n_cortex):
    """Write labels and the matching parcel time series."""
    lab_dir = out_dir / "Labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    np.save(lab_dir / f"{subject_id}_{session}_labels.npy", labels)

    pids = np.unique(labels[labels > 0])
    parcels = np.zeros((ts.shape[0], len(pids)), dtype=np.float32)
    for i, pid in enumerate(pids):
        parcels[:, i] = ts[:, :n_cortex][:, labels == pid].mean(axis=1)
    np.save(
        out_dir
        / f"{subject_id}_{session}_{out_dir.name.split('_', 2)[-1]}.npy",
        parcels,
    )


def process_subject(subject_id, sessions, engine):
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))
    n_R = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT"))
    n_cortex = n_L + n_R

    m = load_m()
    areas = load_areas()
    print(f"  m = {m:.4f}")

    geo = {
        h: np.load(config.INPUTS_DIR / f"geodesic_{h}.npy", mmap_mode="r")
        for h in ("L", "R")
    }

    dir_f = config.OUTPUTS_DIR / "Method_3_SLIC_F"
    dir_c = config.OUTPUTS_DIR / "Method_4_SLIC_C"

    for session in sessions:
        t0 = time.time()
        ts = load_session(subject_id, session)

        out = {}
        for name, use_spatial, target in [
            ("SLIC_F", True, dir_f),
            ("SLIC_C", False, dir_c),
        ]:
            labels = np.zeros(n_cortex, dtype=np.int32)

            for hemi, start, stop in [("L", 0, n_L), ("R", n_L, n_L + n_R)]:
                block = ts[:, start:stop].T  # (V, T)
                g = np.asarray(geo[hemi]) if use_spatial else None
                lab = slic(
                    block, m, g, areas[hemi] if use_spatial else None, engine
                )
                # 1-100 for the left hemisphere, 101-200 for the right
                offset = 1 if hemi == "L" else N_CLUSTERS + 1
                labels[start:stop] = lab + offset

            save_result(target, subject_id, session, labels, ts, n_cortex)
            out[name] = len(np.unique(labels))

        print(
            f"  {session}: SLIC-F {out['SLIC_F']} parcels, "
            f"SLIC-C {out['SLIC_C']} parcels ({time.time()-t0:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser(description="SLIC parcellation.")
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--sessions", nargs="+", default=SESSIONS, choices=SESSIONS
    )
    parser.add_argument("--engine", default="fast", choices=["fast", "direct"])
    args = parser.parse_args()

    print(
        f"Variant {config.VARIANT} | Subject {args.subject} | engine {args.engine}"
    )
    t0 = time.time()
    process_subject(args.subject, args.sessions, args.engine)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
