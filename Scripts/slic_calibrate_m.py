"""
SLIC COMPACTNESS CALIBRATION
============================
Estimates the compactness parameter m for one preprocessing variant.



In SLIC the combined distance is



    d = sqrt( d_func^2 / m^2 + d_spatial^2 / S^2 )



so m is the scale that puts the functional term on the same footing as the
spatial term, just as S does for distance in millimetres. Wang et al. (2016)
set it near the median of all functional distances.



Functional distance here is the Euclidean distance between two z-scored
vertex time series, divided by sqrt(T):



    d_func = ||v_i - v_j|| / sqrt(T) = sqrt(2 (1 - r_ij))



Dividing by sqrt(T) makes the value depend only on the correlation between
the two vertices, not on how many timepoints went into it. Without it, a
parcellation from 2400 timepoints and one from 4800 would sit on different
scales and m would silently mean something different in each.



The median is estimated from a random sample of vertices in a random sample
of subjects, since it converges long before the full data is needed.



Output: Inputs/slic_m_{variant}.txt
"""

import argparse
import numpy as np


import config
import utils

N_SUBJECTS = 20
N_PAIRS = 50000
SEED = 0


def functional_distances(ts, n_pairs, rng, adj_L, adj_R, n_L):
    """
    Functional distances between neighbouring vertices.



    The earlier version of this sampled random vertex pairs, which turns
    out to describe the wrong distribution. SLIC decides where a vertex
    belongs by comparing it against nearby alternatives, never against
    the far side of cortex, so the spread that matters is the one among
    neighbours. Random pairs are almost all uncorrelated and their
    distances pile up near sqrt(2) with almost no variance, which leaves
    the functional term unable to distinguish anything once it is scaled
    by their median.



    This mirrors the original formulation, where the colour distance
    that sets m is the one between adjacent pixels.
    """
    z = ts - ts.mean(axis=0)
    sd = z.std(axis=0)
    sd[sd < np.finfo(np.float32).eps] = 1.0
    z = (z / sd).T
    n_time = z.shape[1]

    pairs = []
    for adj, offset in ((adj_L, 0), (adj_R, n_L)):
        rows, cols = adj.tocsr().nonzero()
        keep = rows < cols
        pairs.append(
            np.column_stack([rows[keep] + offset, cols[keep] + offset])
        )
    pairs = np.vstack(pairs)

    take = rng.choice(len(pairs), min(n_pairs, len(pairs)), replace=False)
    a, b = pairs[take, 0], pairs[take, 1]

    r = (z[a] * z[b]).sum(axis=1) / n_time
    return np.sqrt(np.maximum(2.0 * (1.0 - r), 0.0))


def main():
    parser = argparse.ArgumentParser(description="Calibrate SLIC compactness.")
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=N_PAIRS,
        help="neighbouring vertex pairs sampled per subject",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    from scipy.sparse import load_npz

    adj_L = load_npz(config.INPUTS_DIR / "adjacency_L.npz")
    adj_R = load_npz(config.INPUTS_DIR / "adjacency_R.npz")

    subjects = rng.choice(
        config.SUBJECT_IDS, size=args.n_subjects, replace=False
    )
    cleaned = config.OUTPUTS_DIR / "Cleaned"

    ref = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))
    n_R = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT"))
    n_cortex = n_L + n_R

    pooled = []
    for s in subjects:
        ts = np.concatenate(
            [
                np.load(cleaned / f"{s}_REST1.npy"),
                np.load(cleaned / f"{s}_REST2.npy"),
            ],
            axis=0,
        )[:, :n_cortex]
        d = functional_distances(ts, args.n_pairs, rng, adj_L, adj_R, n_L)
        pooled.append(d)
        print(f"  {s}: median {np.median(d):.4f}")

    pooled = np.concatenate(pooled)
    m = float(np.median(pooled))

    print()
    print(f"Variant   {config.VARIANT}")
    print(f"Subjects  {len(subjects)}")
    print(f"Pairs     {len(pooled):,}")
    print(f"Median    {m:.4f}")
    print(
        f"IQR       {np.percentile(pooled, 25):.4f} - {np.percentile(pooled, 75):.4f}"
    )

    out = config.INPUTS_DIR / f"slic_m_{config.VARIANT}.txt"
    out.write_text(f"{m:.6f}\n")
    print(f"\nsaved {out.name}")


if __name__ == "__main__":
    main()
