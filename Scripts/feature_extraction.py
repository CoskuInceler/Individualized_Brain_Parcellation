"""
STAGE 2, STEP 1: FEATURE EXTRACTION
===================================
Per-subject features from one parcellation, for the brain-behaviour model.



Computed here, per subject and method:



    global_efficiency   raw, top10, r05
    avg_shortest_path   raw, top10, r05
    parcel_size_cv      one value
    parcel_size_mean    one value, for context
    n_parcels           one value, for context



Two further features are cross-subject and are added later by
aggregate_features.py: how close a subject's parcel-size profile is to the
group, and how close their connectivity matrix is. Both need every subject
before they can be computed, so this script saves the profiles they are
built from.



Graph variants, following the thesis:
    raw    weighted, edge weight = FC, negative correlations set to zero
    top10  binary, the strongest 10% of positive edges
    r05    binary, edges with FC > 0.5



Global efficiency and average shortest path do not depend on how parcels
are numbered, so they need no alignment. The parcel-size profile and the
connectivity matrix do: entry i has to mean the same region in every
subject. Those are put into atlas order by Hungarian matching against the
Schaefer atlas before being saved.



Average shortest path is undefined on a disconnected graph and is reported
as NaN rather than averaged over the pairs that happen to be connected,
which would be biased downwards since it is the distant pairs that go
missing. This mainly affects the r05 variant.



Usage:
    python3 feature_extraction.py --subject 100206 --method Method_2_AGP
"""

import argparse
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import shortest_path, connected_components


import config
import utils

SESSION = "ALL"
TOP_K_PERCENT = 10.0
R_THRESHOLD = 0.5


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


def load_timeseries(subject_id, n_cortex):
    """Cleaned cortical time series for the full session."""
    cleaned = config.OUTPUTS_DIR / "Cleaned"
    ts = np.concatenate(
        [
            np.load(cleaned / f"{subject_id}_REST1.npy"),
            np.load(cleaned / f"{subject_id}_REST2.npy"),
        ],
        axis=0,
    )
    return ts[:, :n_cortex]


def parcel_timeseries(ts, labels):
    """Mean time series of each parcel, in ascending label order."""
    pids = np.unique(labels[labels > 0])
    out = np.zeros((ts.shape[0], len(pids)), dtype=np.float32)
    for i, pid in enumerate(pids):
        out[:, i] = ts[:, labels == pid].mean(axis=1)
    return out, pids


def align_to_atlas(labels, atlas, pids):
    """
    Match each parcel to the atlas parcel it overlaps most.



    Parcel numbering is arbitrary in the data-driven methods, so a profile
    indexed by parcel number means nothing across subjects until the
    parcels are put in a common order. Hungarian matching on the overlap
    counts gives the one-to-one assignment that maximises total overlap.
    """
    atlas_ids = np.unique(atlas[atlas > 0])
    pos_p = {p: i for i, p in enumerate(pids)}
    pos_a = {a: j for j, a in enumerate(atlas_ids)}

    both = (labels > 0) & (atlas > 0)
    rows = np.array([pos_p[v] for v in labels[both]])
    cols = np.array([pos_a[v] for v in atlas[both]])

    overlap = np.bincount(
        rows * len(atlas_ids) + cols, minlength=len(pids) * len(atlas_ids)
    )
    overlap = overlap.reshape(len(pids), len(atlas_ids))

    r, c = linear_sum_assignment(-overlap)
    assigned = np.zeros(len(pids), dtype=np.int64)
    assigned[r] = atlas_ids[c]
    return assigned


def to_atlas_order(values, assigned, n_atlas=200):
    """Reindex a per-parcel vector into atlas order, NaN where unmatched."""
    out = np.full(n_atlas, np.nan)
    for value, atlas_id in zip(values, assigned):
        if atlas_id > 0:
            out[atlas_id - 1] = value
    return out


def fc_in_atlas_order(fc, assigned, n_atlas=200):
    """Permute an FC matrix into atlas order, NaN where a parcel is missing."""
    out = np.full((n_atlas, n_atlas), np.nan)
    idx = np.array([a - 1 for a in assigned])
    ok = idx >= 0
    sub = np.ix_(idx[ok], idx[ok])
    out[sub] = fc[np.ix_(np.flatnonzero(ok), np.flatnonzero(ok))]
    return out


def build_graphs(fc):
    """The three graph variants used in the thesis."""
    pos = np.where(np.isnan(fc), 0.0, fc)
    pos = np.where(pos < 0, 0.0, pos)
    np.fill_diagonal(pos, 0.0)

    iu = np.triu_indices_from(pos, k=1)
    vals = pos[iu]
    vals = vals[vals > 0]

    if vals.size:
        cut = np.percentile(vals, 100 - TOP_K_PERCENT)
        top10 = (pos >= cut).astype(np.float64)
    else:
        top10 = np.zeros_like(pos)
    np.fill_diagonal(top10, 0.0)

    r05 = (pos > R_THRESHOLD).astype(np.float64)
    np.fill_diagonal(r05, 0.0)

    return {"raw": pos, "top10": top10, "r05": r05}


def path_lengths(adj):
    """
    Shortest path lengths between all parcels.



    For a weighted graph a strong connection means a short distance, so
    edge weights are inverted before the search; for a binary graph the
    distance is the number of steps.
    """
    binary = np.all((adj == 0) | (adj == 1))

    if binary:
        return shortest_path(adj, method="D", directed=False, unweighted=True)

    inv = np.zeros_like(adj)
    nz = adj > 0
    inv[nz] = 1.0 / adj[nz]
    return shortest_path(inv, method="D", directed=False, unweighted=False)


def global_efficiency(dist):
    """Mean inverse shortest path length; unreachable pairs contribute zero."""
    iu = np.triu_indices(dist.shape[0], k=1)
    d = dist[iu]
    eff = np.zeros_like(d)
    finite = np.isfinite(d) & (d > 0)
    eff[finite] = 1.0 / d[finite]
    return float(eff.mean())


def avg_shortest_path(dist):
    """
    Mean shortest path length, NaN when the graph is disconnected.



    Averaging over only the connected pairs would report a shorter path
    than the graph actually supports, and systematically so.
    """
    iu = np.triu_indices(dist.shape[0], k=1)
    d = dist[iu]
    if not np.isfinite(d).all():
        return np.nan
    return float(d.mean())


def extract(subject_id, method):
    out_dir = config.OUTPUTS_DIR / "Features" / method
    (out_dir / "Profiles").mkdir(parents=True, exist_ok=True)

    atlas = atlas_labels(subject_id)
    labels = load_labels(subject_id, method)
    n_cortex = len(atlas)

    ts = load_timeseries(subject_id, n_cortex)
    parcels, pids = parcel_timeseries(ts[:, : len(labels)], labels)

    sizes = np.array([(labels == p).sum() for p in pids], dtype=float)
    fc = np.corrcoef(parcels.T)

    assigned = align_to_atlas(labels, atlas, pids)

    np.save(
        out_dir / "Profiles" / f"{subject_id}_sizes.npy",
        to_atlas_order(sizes, assigned),
    )

    fc_aligned = fc_in_atlas_order(fc, assigned)
    iu = np.triu_indices(fc_aligned.shape[0], k=1)
    np.save(out_dir / "Profiles" / f"{subject_id}_fc.npy", fc_aligned[iu])

    row = {
        "subject": subject_id,
        "method": method,
        "variant": config.VARIANT,
        "n_parcels": int(len(pids)),
        "parcel_size_mean": round(float(sizes.mean()), 3),
        "parcel_size_cv": round(float(sizes.std(ddof=1) / sizes.mean()), 5),
    }

    for name, adj in build_graphs(fc).items():
        dist = path_lengths(adj)
        row[f"global_efficiency_{name}"] = round(global_efficiency(dist), 6)
        n_comp, _ = connected_components(adj, directed=False)
        row[f"n_components_{name}"] = int(n_comp)
        aspl = avg_shortest_path(dist)
        row[f"avg_shortest_path_{name}"] = (
            round(aspl, 6) if np.isfinite(aspl) else np.nan
        )

    pd.DataFrame([row]).to_csv(out_dir / f"{subject_id}.csv", index=False)
    return row


def main():
    parser = argparse.ArgumentParser(description="Stage 2 feature extraction.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    args = parser.parse_args()

    t0 = time.time()
    row = extract(args.subject, args.method)

    print(f"Variant {config.VARIANT} | {args.method} | Subject {args.subject}")
    for k, v in row.items():
        if k not in ("subject", "method", "variant"):
            print(f"  {k:26s} {v}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
