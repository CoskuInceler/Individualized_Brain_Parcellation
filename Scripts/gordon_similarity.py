"""
GORDON PARCELLATION, STEP 1: SIMILARITY MAPS
============================================
Builds the correlation-of-correlation maps that the gradient is taken on.



Reimplementation of the first half of surface_parcellation_singlesub.m from
    https://github.com/MidnightScanClub/MSCcodebase
    Gordon et al. (2016), Cerebral Cortex, 26, 288-303.



For every vertex the algorithm asks how similar its whole-cortex
connectivity map is to that of a set of reference vertices. Where that
similarity changes abruptly across the surface, an areal boundary is
likely. The similarity maps produced here are the input to the gradient
step; the gradient is what actually locates the boundaries.



Following the reference implementation, the gradient is run on a random
subsample of similarity maps rather than all of them. The original code
uses one map in every 100 and notes that this yields parcellations
correlating at r > .99 with the unsubsampled version.



The full dense connectome is 59,412 x 59,412, so it is never held in
memory: reference maps are built once and the remaining vertices are
streamed past them in blocks.



Output: {subject}_{session}_similarity.npy   (n_cortex, n_reference)
        float32, Fisher-transformed
"""

import argparse
import time
import numpy as np


import config
import utils

SUBSAMPLE = 100  # one reference map per 100 vertices, as in the original
BLOCK = 2000  # vertices per streaming block
SEED = 42
SESSIONS = ["REST1", "REST2", "ALL"]


def zscore_columns(ts):
    """Z-score each vertex so a dot product over time gives a correlation."""
    out = ts - ts.mean(axis=0)
    sd = out.std(axis=0)
    sd[sd < np.finfo(np.float32).eps] = 1.0
    return out / sd


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


def similarity_maps(z, ref_idx, block=BLOCK):
    """
    Correlation between each vertex's connectivity map and the reference maps.



    Parameters
    ----------
    z       : (T, V) z-scored time series
    ref_idx : indices of the reference vertices
    block   : how many vertices to process at once



    Returns
    -------
    (V, n_ref) float32, Fisher-transformed
    """
    n_time, n_vert = z.shape
    n_ref = len(ref_idx)

    # Reference connectivity maps, then z-scored down their length so the
    # correlation between two maps is again a dot product.
    ref_maps = (z[:, ref_idx].T @ z) / n_time  # (n_ref, V)
    np.nan_to_num(ref_maps, copy=False)
    ref_maps = ref_maps.T  # (V, n_ref)

    ref_c = ref_maps - ref_maps.mean(axis=0)
    ref_n = np.sqrt((ref_c**2).sum(axis=0))
    ref_n[ref_n < np.finfo(np.float32).eps] = 1.0
    ref_c = ref_c / ref_n

    out = np.zeros((n_vert, n_ref), dtype=np.float32)

    for start in range(0, n_vert, block):
        stop = min(start + block, n_vert)

        # connectivity maps for this block of vertices
        maps = (z[:, start:stop].T @ z) / n_time  # (block, V)
        np.nan_to_num(maps, copy=False)

        m = maps.T  # (V, block)
        m = m - m.mean(axis=0)
        n = np.sqrt((m**2).sum(axis=0))
        n[n < np.finfo(np.float32).eps] = 1.0
        m = m / n

        out[start:stop] = (m.T @ ref_c).astype(np.float32)

    # Fisher transform, as in the reference implementation
    np.clip(out, -0.9999999, 0.9999999, out=out)
    return np.arctanh(out).astype(np.float32)


def process_subject(subject_id, sessions):
    out_dir = config.OUTPUTS_DIR / "Method_5_Gordon" / "Similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))
    n_R = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT"))
    n_cortex = n_L + n_R

    n_ref = round(n_cortex / SUBSAMPLE)
    rng = np.random.default_rng(SEED)
    ref_idx = np.sort(rng.permutation(n_cortex)[:n_ref])
    np.save(out_dir / "reference_vertices.npy", ref_idx)

    for session in sessions:
        t0 = time.time()
        ts = load_session(subject_id, session)[:, :n_cortex]
        z = zscore_columns(ts.astype(np.float32))

        sim = similarity_maps(z, ref_idx)
        np.save(out_dir / f"{subject_id}_{session}_similarity.npy", sim)

        print(f"  {session}: {sim.shape}, {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Gordon similarity maps.")
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
