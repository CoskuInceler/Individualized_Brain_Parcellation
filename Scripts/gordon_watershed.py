"""
GORDON PARCELLATION, STEP 3: EDGE DETECTION
===========================================
Runs watershed-by-flooding on every gradient map and records how often each
vertex ends up on a watershed line.



Reimplementation of metric_minima_all_cifti.m and
watershed_algorithm_all_par_cifti.m from
    https://github.com/MidnightScanClub/MSCcodebase
    Gordon et al. (2016), Cerebral Cortex, 26, 288-303.



The idea is the one from image segmentation: treat the gradient map as a
landscape, start water rising from every local minimum, and mark the lines
where two basins meet. Those lines are candidate areal borders. Running it
on all 594 gradient maps and averaging gives a boundary map, the value at
each vertex being the fraction of maps in which it came out as a border.



Parameters follow the reference implementation: minima are defined over a
3-step neighbourhood and flooding proceeds in 200 steps.



Output: {subject}_{session}_boundary.npy   (n_cortex,) float32 in [0, 1]
"""

import argparse
import time
import numpy as np
from scipy.sparse import load_npz


import config
import utils

NEIGH_DIST = 3  # steps used to define a local minimum
STEP_NUM = 200  # flooding steps
FRAC_MAX_H = 1.0  # flood up to this fraction of the maximum
SEED = 42
SESSIONS = ["REST1", "REST2", "ALL"]


def neighbour_lists(adj_L, adj_R, n_L):
    """
    Neighbours of every cortical vertex, in the combined index space.



    The two hemispheres stay separate: a vertex never has a neighbour on
    the other side, which is what the surface topology implies.
    """
    out = []
    for adj, offset in ((adj_L, 0), (adj_R, n_L)):
        adj = adj.tocsr()
        for i in range(adj.shape[0]):
            out.append(adj.indices[adj.indptr[i] : adj.indptr[i + 1]] + offset)
    return out


def expand_neighbourhood(neighbours, dist):
    """
    Vertices within `dist` steps of each vertex, excluding itself.



    Used to decide whether a vertex is a local minimum.
    """
    out = []
    for i in range(len(neighbours)):
        seen = {i}
        frontier = {i}
        for _ in range(dist):
            nxt = set()
            for v in frontier:
                nxt.update(neighbours[v].tolist())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        seen.discard(i)
        out.append(np.fromiter(seen, dtype=np.int64, count=len(seen)))
    return out


def local_minima(gradients, wide_neighbours):
    """
    Vertices lower than everything within the wide neighbourhood.



    Parameters
    ----------
    gradients       : (n_vertices, n_maps)
    wide_neighbours : neighbourhood index arrays from expand_neighbourhood



    Returns
    -------
    (n_vertices, n_maps) boolean
    """
    out = np.zeros(gradients.shape, dtype=bool)
    for i, nb in enumerate(wide_neighbours):
        if nb.size == 0:
            continue
        out[i] = (gradients[i] < gradients[nb]).all(axis=0)
    return out


def watershed_one_map(gradient, minima, neighbours, rng):
    """
    Flood one gradient map and return its watershed lines.



    Basins start at the local minima and grow as the water level rises in
    STEP_NUM steps. A vertex that would be claimed by two different basins
    becomes a watershed line and is never assigned.



    Returns
    -------
    (n_vertices,) boolean, True on watershed lines
    """
    n_vert = gradient.shape[0]
    labels = np.zeros(n_vert, dtype=np.int32)
    watershed = np.zeros(n_vert, dtype=bool)

    seeds = np.flatnonzero(minima)
    if seeds.size == 0:
        return watershed

    # basin numbering is shuffled, as in the reference implementation
    labels[seeds] = rng.permutation(seeds.size) + 1

    lo, hi = float(gradient.min()), float(gradient.max())
    levels = np.linspace(lo, lo + (hi - lo) * FRAC_MAX_H, STEP_NUM + 1)

    for level in levels:
        candidates = np.flatnonzero(
            (gradient < level) & (labels == 0) & ~watershed
        )
        if candidates.size == 0:
            continue
        rng.shuffle(candidates)

        for v in candidates:
            if labels[v] != 0 or watershed[v]:
                continue
            nb_labels = labels[neighbours[v]]
            nb_labels = nb_labels[nb_labels > 0]
            if nb_labels.size == 0:
                continue
            if nb_labels.min() != nb_labels.max():
                watershed[v] = True  # two basins meet here
            else:
                labels[v] = nb_labels[0]

    return watershed


def boundary_map(gradients, neighbours, wide_neighbours, seed=SEED):
    """
    Fraction of gradient maps in which each vertex is a watershed line.
    """
    minima = local_minima(gradients, wide_neighbours)
    n_maps = gradients.shape[1]

    counts = np.zeros(gradients.shape[0], dtype=np.float32)
    for m in range(n_maps):
        rng = np.random.default_rng(seed + m)
        counts += watershed_one_map(
            gradients[:, m], minima[:, m], neighbours, rng
        )

    return counts / n_maps


def process_subject(subject_id, sessions):
    grad_dir = config.OUTPUTS_DIR / "Method_5_Gordon" / "Gradients"
    out_dir = config.OUTPUTS_DIR / "Method_5_Gordon" / "Boundary"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))

    adj_L = load_npz(config.INPUTS_DIR / "adjacency_L.npz")
    adj_R = load_npz(config.INPUTS_DIR / "adjacency_R.npz")

    neighbours = neighbour_lists(adj_L, adj_R, n_L)
    wide = expand_neighbourhood(neighbours, NEIGH_DIST)

    for session in sessions:
        src = grad_dir / f"{subject_id}_{session}_gradients.npy"
        if not src.exists():
            print(f"  {session}: gradients missing, skipped")
            continue

        t0 = time.time()
        gradients = np.load(src)
        bmap = boundary_map(gradients, neighbours, wide)

        np.save(out_dir / f"{subject_id}_{session}_boundary.npy", bmap)
        print(
            f"  {session}: mean {bmap.mean():.4f}, max {bmap.max():.4f} "
            f"({time.time()-t0:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser(description="Gordon boundary maps.")
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--sessions", nargs="+", default=SESSIONS, choices=SESSIONS
    )
    args = parser.parse_args()

    print(f"Variant {config.VARIANT} | Subject {args.subject}")
    t0 = time.time()
    process_subject(args.subject, args.sessions)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
