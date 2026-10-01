"""
VALIDATION METRICS
==================
Metrics used to compare parcellations across methods.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components


def homogeneity(ts, labels):
    """
    Functional homogeneity: mean correlation between each vertex time series
    and the mean time series of its parcel.

    Parameters
    ----------
    ts     : (T, V) time series, columns aligned to `labels`
    labels : (V,) parcel labels, 0 = unassigned

    Returns
    -------
    mean_score : float, averaged over parcels
    per_parcel : dict {parcel_id: score}, NaN for parcels with < 2 vertices
    """
    per_parcel = {}
    scores = []
    for pid in np.unique(labels[labels > 0]):
        idx = np.flatnonzero(labels == pid)
        if len(idx) < 2:
            per_parcel[int(pid)] = np.nan
            continue
        v = ts[:, idx]
        m = v.mean(axis=1)
        vc = v - v.mean(axis=0)
        mc = m - m.mean()
        vn = np.sqrt((vc**2).sum(axis=0))
        mn = np.sqrt((mc**2).sum())
        vn[vn == 0] = 1.0
        if mn == 0:
            mn = 1.0
        score = float(np.mean(mc @ vc / (mn * vn)))
        per_parcel[int(pid)] = score
        scores.append(score)
    return (float(np.mean(scores)) if scores else np.nan), per_parcel


def contiguity(labels, adj):
    """
    Spatial contiguity of parcels within one hemisphere.
    A parcel is contiguous if its vertices form a single connected
    component in the surface adjacency graph.

    Parameters
    ----------
    labels : (V,) parcel labels for this hemisphere, 0 = unassigned
    adj    : (V, V) sparse adjacency matrix for this hemisphere

    Returns
    -------
    dict with parcel counts, the percentage that are contiguous, and
    the mean and maximum number of components per parcel
    """
    pids = np.unique(labels[labels > 0])
    n_components = []
    for pid in pids:
        idx = np.flatnonzero(labels == pid)
        if len(idx) == 0:
            continue
        sub = adj[idx][:, idx]
        n, _ = connected_components(sub, directed=False)
        n_components.append(n)
    n_components = np.array(n_components)
    n_total = len(n_components)
    n_contig = int((n_components == 1).sum())
    return {
        "n_parcels": n_total,
        "n_contiguous": n_contig,
        "n_fragmented": n_total - n_contig,
        "pct_contiguous": 100.0 * n_contig / n_total if n_total else np.nan,
        "mean_components": float(n_components.mean()) if n_total else np.nan,
        "max_components": int(n_components.max()) if n_total else 0,
    }


def parcel_timeseries(ts, labels):
    """
    Average vertex time series within each parcel.

    Parameters
    ----------
    ts     : (T, V) time series
    labels : (V,) parcel labels, 0 = unassigned

    Returns
    -------
    (T, n_parcels) array, parcels in ascending label order
    """
    pids = np.unique(labels[labels > 0])
    out = np.zeros((ts.shape[0], len(pids)), dtype=np.float32)
    for i, pid in enumerate(pids):
        out[:, i] = ts[:, labels == pid].mean(axis=1)
    return out


def fc_test_retest(ts1, ts2, labels):
    """
    Test-retest reliability of functional connectivity.
    Applies the same parcellation to two sessions, builds an FC matrix
    for each, and correlates their upper triangles.

    Parameters
    ----------
    ts1, ts2 : (T, V) time series from session 1 and session 2
    labels   : (V,) parcel labels, applied to both

    Returns
    -------
    float, Pearson r between the two FC patterns
    """
    fc1 = np.corrcoef(parcel_timeseries(ts1, labels).T)
    fc2 = np.corrcoef(parcel_timeseries(ts2, labels).T)
    iu = np.triu_indices_from(fc1, k=1)
    return float(np.corrcoef(fc1[iu], fc2[iu])[0, 1])


def dice_pairwise(labels1, labels2):
    """
    Dice overlap between two parcellations of the same vertices.
    Based on co-assignment of vertex pairs rather than parcel identity,
    so no correspondence between parcel labels is needed. This is what
    makes it usable for methods whose parcel numbering is arbitrary
    across subjects.
        Dice = 2 * (pairs together in both) / (pairs together in 1 +
                                               pairs together in 2)

    Parameters
    ----------
    labels1, labels2 : (V,) integer label arrays, 0 = unassigned

    Returns
    -------
    float in [0, 1]
    """
    valid = (labels1 > 0) & (labels2 > 0)
    l1 = labels1[valid]
    l2 = labels2[valid]
    if l1.size == 0:
        return 0.0

    def n_pairs(counts):
        c = counts.astype(np.int64)
        return float((c * (c - 1)).sum() / 2)

    # marginal counts
    sp1 = n_pairs(np.bincount(l1))
    sp2 = n_pairs(np.bincount(l2))
    if sp1 + sp2 == 0:
        return 0.0
    # joint counts, via a single flattened cross-tabulation
    k2 = l2.max() + 1
    joint = np.bincount(l1.astype(np.int64) * k2 + l2.astype(np.int64))
    agreement = n_pairs(joint)
    return float(2.0 * agreement / (sp1 + sp2))
