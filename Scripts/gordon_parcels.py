"""
GORDON PARCELLATION, STEP 4: PARCEL CREATION
============================================
Turns a boundary map into discrete parcels.



Reimplementation of parcel_creator_cifti.m from
    https://github.com/MidnightScanClub/MSCcodebase
    Gordon et al. (2016), Cerebral Cortex, 26, 288-303.



Parcels grow outward from the low points of the boundary map and stop where
they meet. Neighbouring parcels separated only by a weak border are then
merged, small parcels are absorbed into their neighbours, and vertices
sitting on strong borders are dropped and treated as transition zones
rather than forced into a parcel.



The number of parcels is not fixed: it follows from the boundary map and
the merge threshold. This is the method's own behaviour and the reason a
second, resolution-matched version is produced separately.



Parameters follow the reference implementation, with the merge threshold
left as an argument since the original documents a useful range rather
than a single value.



Output: Labels/{subject}_{session}_labels.npy
"""

import argparse
import time
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components


import config
import utils

MINIMA_THRESH_PERC = 0.75  # minima above this percentile are discarded
EDGEVAL_THRESH_PERC = 0.75  # vertices above this percentile leave the parcels
MERGE_THRESH_PERC = 0.40  # parcels separated by weaker borders are merged
MIN_PARCEL_SIZE = 30  # smaller parcels are absorbed by a neighbour
MIN_ISOLATED_SIZE = 10  # smaller parcels with no neighbour are dropped
SEED = 42
SESSIONS = ["REST1", "REST2", "ALL"]


def neighbour_lists(adj_L, adj_R, n_L):
    """Neighbours of every cortical vertex, hemispheres kept separate."""
    out = []
    for adj, offset in ((adj_L, 0), (adj_R, n_L)):
        adj = adj.tocsr()
        for i in range(adj.shape[0]):
            out.append(adj.indices[adj.indptr[i] : adj.indptr[i + 1]] + offset)
    return out


def find_minima(edge, neighbours):
    """
    Local minima of the boundary map.



    A vertex is a minimum if no immediate neighbour is lower. Where several
    adjacent vertices tie, the reference implementation keeps the first and
    nudges the others up so that a flat basin yields a single seed.
    """
    work = edge.copy()
    minima = np.zeros(len(edge), dtype=bool)

    for i in range(len(edge)):
        nb = neighbours[i]
        if nb.size == 0:
            continue
        local = work[nb]
        if work[i] <= local.min():
            minima[i] = True
            tied = nb[local == work[i]]
            if tied.size:
                work[tied] += 1e-5
    return minima


def grow_parcels(edge, minima, neighbours, rng):
    """
    Flood the boundary map from its minima and label the resulting basins.



    Vertices claimed by two different basins become borders and stay
    unlabelled, which is what separates the parcels.
    """
    labels = np.zeros(len(edge), dtype=np.int32)
    border = np.zeros(len(edge), dtype=bool)

    seeds = np.flatnonzero(minima)
    labels[seeds] = np.argsort(np.argsort(edge[seeds])) + 1

    for level in np.unique(edge):
        pending = np.flatnonzero((edge < level) & (labels == 0) & ~border)
        if pending.size == 0:
            continue
        rng.shuffle(pending)

        for v in pending:
            nb_labels = labels[neighbours[v]]
            nb_labels = nb_labels[nb_labels > 0]
            if nb_labels.size == 0:
                continue
            if nb_labels.min() != nb_labels.max():
                border[v] = True
            else:
                labels[v] = nb_labels[0]

    return labels


def border_vertices(labels, neighbours):
    """
    For each pair of touching parcels, the unlabelled vertices between them.



    Returns a dict keyed by the pair, in ascending order.
    """
    pairs = {}
    for v in np.flatnonzero(labels == 0):
        nb = np.unique(labels[neighbours[v]])
        nb = nb[nb > 0]
        if nb.size < 2:
            continue
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                pairs.setdefault((int(nb[i]), int(nb[j])), []).append(v)
    return pairs


def close_merged_borders(labels, neighbours):
    """
    Absorb border vertices that no longer separate anything.



    After two parcels merge, the vertices that used to lie between them
    still carry no label. Left alone they keep the merged parcel in two
    spatially separate pieces, so any border vertex whose labelled
    neighbours now all agree joins them.
    """
    changed = True
    while changed:
        changed = False
        for v in np.flatnonzero(labels == 0):
            nb = labels[neighbours[v]]
            nb = nb[nb > 0]
            if nb.size and nb.min() == nb.max():
                labels[v] = nb[0]
                changed = True
    return labels


def merge_weak_borders(labels, edge, neighbours, threshold, target=None):
    """
    Merge parcel pairs across their weakest borders.



    The pair with the weakest median border goes first, and the process
    repeats. Without a target it stops once no border is weaker than the
    threshold, which is the method's own behaviour and leaves the number of
    parcels up to the data. With a target it instead continues until that
    many parcels remain, which is what the comparison against the other
    five methods needs, since those all produce 200.
    """
    merges = 0
    while True:
        pairs = border_vertices(labels, neighbours)
        if not pairs:
            break

        medians = {
            p: float(np.median(edge[v]))
            for p, v in pairs.items()
            if len(v) > 1
        }
        if not medians:
            break

        n_parcels = len(np.unique(labels[labels > 0]))

        if target is None:
            if min(medians.values()) >= threshold:
                break
        elif n_parcels <= target:
            break

        pair = min(medians, key=medians.get)
        labels[labels == pair[1]] = pair[0]
        merges += 1

    labels = close_merged_borders(labels, neighbours)
    return labels, merges


def merge_weak_borders_fast(labels, edge, neighbours, threshold, target=None):
    """
    Same merging as merge_weak_borders, with the border bookkeeping kept
    between iterations instead of rebuilt from scratch each time.



    When two parcels merge, only their own adjacencies change; every other
    pair keeps the border it had. Recomputing all of them each round is
    what makes the plain version too slow to run twice, which the parcel
    count target requires.
    """
    pairs = {
        k: list(v) for k, v in border_vertices(labels, neighbours).items()
    }
    medians = {
        p: float(np.median(edge[v])) for p, v in pairs.items() if len(v) > 1
    }

    alive = set(int(p) for p in np.unique(labels[labels > 0]))
    merges = 0

    while medians:
        if target is None:
            if min(medians.values()) >= threshold:
                break
        elif len(alive) <= target:
            break

        a, b = min(medians, key=medians.get)

        labels[labels == b] = a
        alive.discard(b)
        merges += 1

        # fold b's borders into a, dropping the one they shared
        moved = {}
        for (x, y), verts in list(pairs.items()):
            if b not in (x, y):
                continue
            other = y if x == b else x
            del pairs[(x, y)]
            medians.pop((x, y), None)
            if other == a:
                continue
            key = (min(a, other), max(a, other))
            moved.setdefault(key, []).extend(verts)

        for key, verts in moved.items():
            pairs[key] = sorted(set(pairs.get(key, [])) | set(verts))
            if len(pairs[key]) > 1:
                medians[key] = float(np.median(edge[pairs[key]]))
            else:
                medians.pop(key, None)

    labels = close_merged_borders(labels, neighbours)
    return labels, merges


def absorb_small_parcels(labels, edge, neighbours, min_size):
    """
    Merge parcels below the size limit into the neighbour they share the
    weakest border with.
    """
    pairs = border_vertices(labels, neighbours)
    sizes = {
        int(p): int((labels == p).sum()) for p in np.unique(labels) if p > 0
    }

    for parcel, size in sorted(sizes.items(), key=lambda kv: kv[1]):
        if size >= min_size or (labels == parcel).sum() == 0:
            continue

        options = {}
        for (a, b), verts in pairs.items():
            if a == parcel and (labels == b).any():
                options[b] = float(np.median(edge[verts]))
            elif b == parcel and (labels == a).any():
                options[a] = float(np.median(edge[verts]))

        if options:
            labels[labels == parcel] = min(options, key=options.get)

    return labels


def split_into_contiguous(labels, adj_L, adj_R, n_L):
    """
    Give every spatially separate piece of a parcel its own label.



    Merging can leave a parcel in two pieces, and dropping high-boundary
    vertices can split one as well, so this runs afterwards.
    """
    out = np.zeros(len(labels), dtype=np.int32)
    next_label = 1

    for adj, offset, size in (
        (adj_L, 0, adj_L.shape[0]),
        (adj_R, n_L, adj_R.shape[0]),
    ):
        block = labels[offset : offset + size]
        for parcel in np.unique(block[block > 0]):
            idx = np.flatnonzero(block == parcel)
            n_parts, part = connected_components(
                adj[idx][:, idx], directed=False
            )
            for p in range(n_parts):
                out[offset + idx[part == p]] = next_label
                next_label += 1

    return out


def drop_tiny_parcels(labels, min_size):
    """Remove parcels below the size limit outright."""
    for parcel in np.unique(labels[labels > 0]):
        if (labels == parcel).sum() < min_size:
            labels[labels == parcel] = 0
    return labels


def renumber(labels):
    """Relabel parcels as 1..n with no gaps."""
    out = np.zeros_like(labels)
    for new, old in enumerate(np.unique(labels[labels > 0]), start=1):
        out[labels == old] = new
    return out


def create_parcels(
    edge, neighbours, adj_L, adj_R, n_L, merge_perc, rng, target=None
):
    """Full parcel creation from a boundary map."""
    edge = edge - edge.min()

    minima_thresh = np.percentile(edge, MINIMA_THRESH_PERC * 100)
    edgeval_thresh = np.percentile(edge, EDGEVAL_THRESH_PERC * 100)
    merge_thresh = np.percentile(edge, merge_perc * 100)

    minima = find_minima(edge, neighbours)
    minima[edge > minima_thresh] = False

    labels = grow_parcels(edge, minima, neighbours, rng)
    n_initial = len(np.unique(labels[labels > 0]))

    labels, n_merges = merge_weak_borders_fast(
        labels, edge, neighbours, merge_thresh, target
    )
    labels = absorb_small_parcels(labels, edge, neighbours, MIN_PARCEL_SIZE)
    labels = close_merged_borders(labels, neighbours)

    # vertices on strong borders are transition zones, not parcel members
    labels[edge > edgeval_thresh] = 0

    labels = split_into_contiguous(labels, adj_L, adj_R, n_L)
    labels = drop_tiny_parcels(labels, MIN_ISOLATED_SIZE)

    # Dropping high-boundary vertices splits some parcels in two, so the
    # count drifts back up after the first merging pass. Where a fixed
    # number is required, merge again once the small fragments are gone.
    if target is not None:
        labels, extra = merge_weak_borders_fast(
            labels, edge, neighbours, merge_thresh, target
        )
        n_merges += extra
    labels = renumber(labels)

    return labels, n_initial, n_merges


def process_subject(subject_id, sessions, merge_perc, target):
    bnd_dir = config.OUTPUTS_DIR / "Method_5_Gordon" / "Boundary"
    suffix = "Labels" if target is None else f"Labels_{target}"
    out_dir = config.OUTPUTS_DIR / "Method_5_Gordon" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))

    adj_L = load_npz(config.INPUTS_DIR / "adjacency_L.npz").tocsr()
    adj_R = load_npz(config.INPUTS_DIR / "adjacency_R.npz").tocsr()
    neighbours = neighbour_lists(adj_L, adj_R, n_L)

    for session in sessions:
        src = bnd_dir / f"{subject_id}_{session}_boundary.npy"
        if not src.exists():
            print(f"  {session}: boundary map missing, skipped")
            continue

        t0 = time.time()
        rng = np.random.default_rng(SEED)
        labels, n_initial, n_merges = create_parcels(
            np.load(src),
            neighbours,
            adj_L,
            adj_R,
            n_L,
            merge_perc,
            rng,
            target,
        )

        np.save(out_dir / f"{subject_id}_{session}_labels.npy", labels)

        n_final = len(np.unique(labels[labels > 0]))
        print(
            f"  {session}: {n_initial} basins, {n_merges} merges, "
            f"{n_final} parcels, {(labels > 0).sum()} vertices assigned "
            f"({time.time()-t0:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser(description="Gordon parcel creation.")
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--sessions", nargs="+", default=SESSIONS, choices=SESSIONS
    )
    parser.add_argument(
        "--merge-perc",
        type=float,
        default=MERGE_THRESH_PERC,
        help="percentile of the boundary map used as the "
        "merge threshold (the original suggests .35-.50)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="merge until this many parcels remain; omit to "
        "let the boundary map decide",
    )
    args = parser.parse_args()

    print(
        f"Variant {config.VARIANT} | Subject {args.subject} | "
        f"merge {args.merge_perc}"
    )
    t0 = time.time()
    process_subject(args.subject, args.sessions, args.merge_perc, args.target)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
