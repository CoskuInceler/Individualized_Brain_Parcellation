"""
GORDON PARCELLATION: FULL PIPELINE
==================================
Runs the four steps for one subject and keeps only what is worth keeping.



    similarity -> gradient -> boundary map -> parcels



The similarity and gradient arrays are about 135 MB per session and are
used by nothing except the step that follows them, so they are written to
a temporary directory and removed. Across the full sample they would come
to roughly 3 TB.



The boundary map is kept. It costs about five minutes to produce and seven
seconds to turn into parcels, so keeping it means the merge threshold or
the parcel target can be changed later without recomputing anything.



Two sets of labels are written:



    Labels      the number of parcels the boundary map implies
    Labels_200  merged down to 200, for comparison against the other
                methods, which all produce that many



Usage:
    python3 gordon_pipeline.py --subject 100206
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path


import numpy as np
from scipy.sparse import load_npz


import config
import utils
import gordon_similarity as g_sim
import gordon_gradient as g_grad
import gordon_watershed as g_wat
import gordon_parcels as g_par

SESSIONS = ["REST1", "REST2", "ALL"]
TARGET = 200


def process_session(subject_id, session, context, workdir):
    """Run all four steps for one session."""
    n_cortex = context["n_cortex"]
    out_root = config.OUTPUTS_DIR / "Method_5_Gordon"

    # 1. similarity maps
    t0 = time.time()
    ts = g_sim.load_session(subject_id, session)[:, :n_cortex]
    z = g_sim.zscore_columns(ts.astype(np.float32))
    similarity = g_sim.similarity_maps(z, context["ref_idx"])
    del ts, z
    t_sim = time.time() - t0

    # 2. gradient and smoothing, via Workbench
    t0 = time.time()
    gradients = g_grad.gradient_maps(
        similarity,
        context["brain_axis"],
        context["surf_L"],
        context["surf_R"],
        workdir,
    )
    del similarity
    t_grad = time.time() - t0

    # 3. watershed on every gradient map
    t0 = time.time()
    boundary = g_wat.boundary_map(
        gradients, context["neighbours"], context["wide"]
    )
    del gradients
    t_wat = time.time() - t0

    bnd_dir = out_root / "Boundary"
    bnd_dir.mkdir(parents=True, exist_ok=True)
    np.save(bnd_dir / f"{subject_id}_{session}_boundary.npy", boundary)

    # 4. parcels, both the method's own resolution and the matched one
    t0 = time.time()
    counts = {}
    for target, name in ((None, "Labels"), (TARGET, f"Labels_{TARGET}")):
        rng = np.random.default_rng(g_par.SEED)
        labels, _, _ = g_par.create_parcels(
            boundary.copy(),
            context["neighbours"],
            context["adj_L"],
            context["adj_R"],
            context["n_L"],
            g_par.MERGE_THRESH_PERC,
            rng,
            target,
        )

        lab_dir = out_root / name
        lab_dir.mkdir(parents=True, exist_ok=True)
        np.save(lab_dir / f"{subject_id}_{session}_labels.npy", labels)
        counts[name] = len(np.unique(labels[labels > 0]))
    t_par = time.time() - t0

    print(
        f"  {session}: {counts['Labels']} parcels natural, "
        f"{counts[f'Labels_{TARGET}']} matched "
        f"(sim {t_sim:.0f}s, grad {t_grad:.0f}s, "
        f"wat {t_wat:.0f}s, parc {t_par:.0f}s)"
    )


def build_context(subject_id):
    """Everything that does not change between sessions."""
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))
    n_R = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT"))
    n_cortex = n_L + n_R

    adj_L = load_npz(config.INPUTS_DIR / "adjacency_L.npz").tocsr()
    adj_R = load_npz(config.INPUTS_DIR / "adjacency_R.npz").tocsr()
    neighbours = g_wat.neighbour_lists(adj_L, adj_R, n_L)

    brain_axis, _ = g_grad.cortex_axis(subject_id)

    # the reference vertices are drawn once and reused, so that the three
    # sessions of a subject are compared against the same set
    rng = np.random.default_rng(g_sim.SEED)
    n_ref = round(n_cortex / g_sim.SUBSAMPLE)
    ref_idx = np.sort(rng.permutation(n_cortex)[:n_ref])

    return {
        "n_L": n_L,
        "n_cortex": n_cortex,
        "adj_L": adj_L,
        "adj_R": adj_R,
        "neighbours": neighbours,
        "wide": g_wat.expand_neighbourhood(neighbours, g_wat.NEIGH_DIST),
        "brain_axis": brain_axis,
        "ref_idx": ref_idx,
        "surf_L": config.get_surface_path(subject_id, "L"),
        "surf_R": config.get_surface_path(subject_id, "R"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gordon parcellation pipeline."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--sessions", nargs="+", default=SESSIONS, choices=SESSIONS
    )
    args = parser.parse_args()

    print(f"Variant {config.VARIANT} | Subject {args.subject}")
    t0 = time.time()

    context = build_context(args.subject)
    workdir = Path(tempfile.mkdtemp(prefix=f"gordon_{args.subject}_"))
    try:
        for session in args.sessions:
            process_session(args.subject, session, context, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
