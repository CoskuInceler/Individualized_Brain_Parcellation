"""
GORDON PARCELLATION, STEP 2: GRADIENT AND SMOOTHING
===================================================
Takes the spatial gradient of each similarity map along the cortical
surface, then smooths the result.



This is where boundaries are actually located: a large gradient means the
connectivity pattern changes sharply between neighbouring vertices, which
is what an areal border looks like.



The gradient and the smoothing both follow the cortical sheet, so they
depend on mesh geometry. Rather than reimplement that, this step calls
Connectome Workbench, which is what the reference implementation uses:



    wb_command -cifti-gradient
    wb_command -cifti-smoothing   (sigma 2.55, as in the original)



Surfaces are each subject's own MSMAll midthickness, matching the space
the functional data is in.



Output: {subject}_{session}_gradients.npy   (n_cortex, n_reference)
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel import cifti2


import config

SMOOTH_SIGMA = 2.55
SESSIONS = ["REST1", "REST2", "ALL"]


def cortex_axis(subject_id):
    """BrainModelAxis restricted to the two cortical surfaces."""
    img = nib.load(str(config.get_brain_path(subject_id, config.RUN_IDS[0])))
    axis = img.header.get_axis(1)
    n_cortex = 0
    for name, sl, _ in axis.iter_structures():
        if name in (
            "CIFTI_STRUCTURE_CORTEX_LEFT",
            "CIFTI_STRUCTURE_CORTEX_RIGHT",
        ):
            n_cortex = max(n_cortex, sl.stop)
    return axis[:n_cortex], n_cortex


def write_dtseries(data, brain_axis, path):
    """Write a (n_maps, n_vertices) array as a dtseries CIFTI."""
    series = cifti2.SeriesAxis(start=0, step=1, size=data.shape[0])
    img = cifti2.Cifti2Image(data, (series, brain_axis))
    img.to_filename(str(path))


def run(cmd):
    """Run a Workbench command and stop on failure."""
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} failed:\n{done.stderr}")


def gradient_maps(similarity, brain_axis, surf_L, surf_R, workdir):
    """
    Gradient of each similarity map, smoothed along the surface.



    Parameters
    ----------
    similarity : (n_vertices, n_maps) array
    brain_axis : cortical BrainModelAxis matching the first dimension
    surf_L, surf_R : paths to this subject's midthickness surfaces
    workdir    : directory for the intermediate CIFTI files



    Returns
    -------
    (n_vertices, n_maps) float32
    """
    raw = workdir / "similarity.dtseries.nii"
    grad = workdir / "gradient.dtseries.nii"
    smooth = workdir / "gradient_smooth.dtseries.nii"

    # CIFTI stores maps along the first axis
    write_dtseries(similarity.T.astype(np.float32), brain_axis, raw)

    run(
        [
            "wb_command",
            "-cifti-gradient",
            str(raw),
            "COLUMN",
            str(grad),
            "-left-surface",
            str(surf_L),
            "-right-surface",
            str(surf_R),
        ]
    )

    run(
        [
            "wb_command",
            "-cifti-smoothing",
            str(grad),
            str(SMOOTH_SIGMA),
            str(SMOOTH_SIGMA),
            "COLUMN",
            str(smooth),
            "-left-surface",
            str(surf_L),
            "-right-surface",
            str(surf_R),
        ]
    )

    out = nib.load(str(smooth)).get_fdata(dtype=np.float32).T
    return np.ascontiguousarray(out)


def process_subject(subject_id, sessions):
    sim_dir = config.OUTPUTS_DIR / "Method_5_Gordon" / "Similarity"
    out_dir = config.OUTPUTS_DIR / "Method_5_Gordon" / "Gradients"
    out_dir.mkdir(parents=True, exist_ok=True)

    brain_axis, n_cortex = cortex_axis(subject_id)
    surf_L = config.get_surface_path(subject_id, "L")
    surf_R = config.get_surface_path(subject_id, "R")

    for session in sessions:
        src = sim_dir / f"{subject_id}_{session}_similarity.npy"
        if not src.exists():
            print(f"  {session}: similarity missing, skipped")
            continue

        t0 = time.time()
        similarity = np.load(src)

        workdir = Path(tempfile.mkdtemp(prefix=f"gordon_{subject_id}_"))
        try:
            grads = gradient_maps(
                similarity, brain_axis, surf_L, surf_R, workdir
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        np.save(out_dir / f"{subject_id}_{session}_gradients.npy", grads)
        print(
            f"  {session}: {grads.shape}, "
            f"range {grads.min():.4f}-{grads.max():.4f} ({time.time()-t0:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser(description="Gordon gradient maps.")
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
