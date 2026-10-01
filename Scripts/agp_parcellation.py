"""
AGP PARCELLATION (Level 2 — atlas-guided region growing)
========================================================
Individualised parcellation that starts from the Schaefer atlas but lets
each parcel grow according to the subject's own functional data.
Each parcel begins at its seed vertex. At every step the unassigned vertex
with the highest mean correlation to the members of an adjacent parcel is
claimed by that parcel. Growth stops once as many vertices are assigned as
the atlas itself covers.
Reimplementation of Li et al. (2022), Computers in Biology and Medicine,
150, 106078.
Two engines compute the same quantity:
    fast    running sum per parcel, one dot product per candidate
    direct  full member-by-candidate correlation matrix, as in the original
They are algebraically identical; --engine direct exists to verify that.
Output per subject and session:
    Labels/{subject}_{session}_labels.npy   valid-vertex space
    {subject}_{session}_AGP.npy             parcel time series
"""

import argparse
import time
import numpy as np
from scipy.sparse import load_npz
import config
import utils

METHOD = "AGP"
SESSIONS = ["REST1", "REST2", "ALL"]


def zscore_columns(ts):
    """Z-score each vertex time series, so a dot product gives correlation."""
    out = ts - ts.mean(axis=0)
    sd = out.std(axis=0)
    sd[sd < np.finfo(np.float32).eps] = 1.0
    return out / sd


def neighbour_lists(adj):
    """Convert a sparse adjacency matrix into per-vertex index arrays."""
    adj = adj.tocsr()
    return [
        adj.indices[adj.indptr[i] : adj.indptr[i + 1]]
        for i in range(adj.shape[0])
    ]


def grow_regions(ts, seeds, neighbours, target, engine="fast"):
    """
    Grow parcels from their seeds until `target` vertices are assigned.

    Parameters
    ----------
    ts         : (T, V) time series for one hemisphere
    seeds      : (n_parcels, 2) array of parcel id and seed vertex
    neighbours : list of neighbour index arrays, one per vertex
    target     : number of vertices to assign in total
    engine     : "fast" or "direct"

    Returns
    -------
    (V,) int array of parcel labels, 0 where unassigned
    """
    import heapq

    z = zscore_columns(ts)
    n_time, n_vert = z.shape
    labels = np.zeros(n_vert, dtype=np.int32)
    members = {}
    running_sum = {}
    for pid, seed in seeds:
        pid, seed = int(pid), int(seed)
        labels[seed] = pid
        members[pid] = [seed]
        running_sum[pid] = z[:, seed].copy()
    assigned = len(members)

    def best_candidate(pid):
        """Highest-scoring unassigned neighbour of parcel `pid`."""
        cands = set()
        for m in members[pid]:
            for nb in neighbours[m]:
                if labels[nb] == 0:
                    cands.add(nb)
        if not cands:
            return None
        cands = np.fromiter(cands, dtype=np.int64, count=len(cands))
        if engine == "fast":
            scores = (running_sum[pid] @ z[:, cands]) / (
                n_time * len(members[pid])
            )
        else:
            pairwise = z[:, members[pid]].T @ z[:, cands] / n_time
            scores = pairwise.mean(axis=0)
        k = int(np.argmax(scores))
        return (-float(scores[k]), int(cands[k]), pid)

    heap = []
    for pid in members:
        c = best_candidate(pid)
        if c:
            heapq.heappush(heap, c)
    while heap and assigned < target:
        neg, vert, pid = heapq.heappop(heap)
        if labels[vert] != 0:
            c = best_candidate(pid)
            if c:
                heapq.heappush(heap, c)
            continue
        labels[vert] = pid
        members[pid].append(vert)
        running_sum[pid] += z[:, vert]
        assigned += 1
        c = best_candidate(pid)
        if c:
            heapq.heappush(heap, c)
    if assigned < target:
        print(
            f"    note: assigned {assigned} of {target}, some parcels ran out of neighbours"
        )
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


def process_subject(subject_id, sessions, engine):
    out_dir = config.OUTPUTS_DIR / f"Method_2_{METHOD}"
    lab_dir = out_dir / "Labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    valid_L = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    valid_R = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    n_L, n_R = len(valid_L), len(valid_R)
    atlas_L, atlas_R = utils.load_and_filter_atlas(
        config.SCHAEFER_200_FILE, valid_L, valid_R
    )
    seeds_L = np.load(config.INPUTS_DIR / "seeds_L.npy")
    seeds_R = np.load(config.INPUTS_DIR / "seeds_R.npy")
    nbr_L = neighbour_lists(load_npz(config.INPUTS_DIR / "adjacency_L.npz"))
    nbr_R = neighbour_lists(load_npz(config.INPUTS_DIR / "adjacency_R.npz"))
    for session in sessions:
        t0 = time.time()
        ts = load_session(subject_id, session)
        lab_L = grow_regions(
            ts[:, :n_L], seeds_L, nbr_L, int((atlas_L > 0).sum()), engine
        )
        lab_R = grow_regions(
            ts[:, n_L : n_L + n_R],
            seeds_R,
            nbr_R,
            int((atlas_R > 0).sum()),
            engine,
        )

        labels = np.concatenate([lab_L, lab_R])
        np.save(lab_dir / f"{subject_id}_{session}_labels.npy", labels)
        pids = np.unique(labels[labels > 0])
        parcels = np.zeros((ts.shape[0], len(pids)), dtype=np.float32)
        for i, pid in enumerate(pids):
            parcels[:, i] = ts[:, : len(labels)][:, labels == pid].mean(axis=1)
        np.save(out_dir / f"{subject_id}_{session}_{METHOD}.npy", parcels)
        print(
            f"  {session}: {len(pids)} parcels, "
            f"{(labels > 0).sum()} vertices ({time.time()-t0:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser(description="AGP parcellation.")
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
