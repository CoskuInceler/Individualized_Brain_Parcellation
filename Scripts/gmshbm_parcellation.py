"""
gMSHBM PARCELLATION (Level 1 — group prior refined per individual)
==================================================================
Extracts individual parcellations from the Kong et al. (2022) group file
and applies them to the cleaned time series.
The parcellations themselves are fixed: Kong and colleagues estimated them
once, and we read them here. They differ between subjects but not between
preprocessing variants, so only the parcel time series depend on IP_VARIANT.
Label space
-----------
The .mat file stores labels on the full 32k mesh per hemisphere, medial
wall included as 0. The cleaned data holds only valid vertices, so labels
are filtered before they line up with data columns.
    LH labels: 1-100      RH labels: 101-200
Outputs (per subject)
---------------------
    Labels/{subj}_labels.npy              valid-vertex space, for Dice
    Labels/{subj}_LH.npy, _RH.npy         full mesh, for visualisation
    CIFTI/{subj}.dlabel.nii               full mesh, for viewing
    {subj}_REST1_gMSHBM.npy, _REST2_...   parcel time series
"""

import argparse
import time
import numpy as np
import h5py
import config
import utils

METHOD = "gMSHBM"
SESSIONS = ["REST1", "REST2"]
N_MESH = utils.N_VERTICES_PER_HEMI


def subject_index(subject_id):
    """Row of this subject in the group .mat file."""
    ids = [
        s.strip() for s in config.SUBJECT_FILE.read_text().split() if s.strip()
    ]
    if subject_id not in ids:
        raise ValueError(
            f"Subject {subject_id} not in {config.SUBJECT_FILE.name}"
        )
    return ids.index(subject_id)


def read_labels(subject_id):
    """
    Read one subject's labels from the group file.
    h5py returns MATLAB matrices transposed, so the stored (32492, 1029)
    arrays appear as (1029, 32492) and the subject is the first axis.
    """
    row = subject_index(subject_id)
    with h5py.File(str(config.GMSHBM_FILE), "r") as f:
        lh = f["lh_labels_all"][row, :].astype(int)
        rh = f["rh_labels_all"][row, :].astype(int)

    assert lh.shape[0] == N_MESH and rh.shape[0] == N_MESH
    return lh, rh


def process_subject(subject_id):
    out_dir = config.OUTPUTS_DIR / f"Method_1_{METHOD}"
    lab_dir = out_dir / "Labels"
    cii_dir = out_dir / "CIFTI"
    for d in (out_dir, lab_dir, cii_dir):
        d.mkdir(parents=True, exist_ok=True)

    lh_full, rh_full = read_labels(subject_id)
    print(
        f"  labels: LH 1-{lh_full.max()}, RH {rh_full[rh_full>0].min()}-{rh_full.max()}"
    )
    print(f"  medial wall: LH {(lh_full==0).sum()}, RH {(rh_full==0).sum()}")

    # full mesh, for visualisation
    np.save(lab_dir / f"{subject_id}_LH.npy", lh_full)
    np.save(lab_dir / f"{subject_id}_RH.npy", rh_full)

    utils.save_cifti(
        np.concatenate([lh_full, rh_full]).reshape(1, -1).astype(np.float32),
        config.SCHAEFER_200_FILE,
        cii_dir / f"{subject_id}.dlabel.nii",
    )

    # valid-vertex space, for metrics and Dice
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    valid_L = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    valid_R = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")

    labels = np.concatenate([lh_full[valid_L], rh_full[valid_R]])
    np.save(lab_dir / f"{subject_id}_labels.npy", labels)

    n_parcels = len(np.unique(labels[labels > 0]))
    print(f"  parcels present: {n_parcels}")

    # parcel time series, one file per session
    cleaned = config.OUTPUTS_DIR / "Cleaned"
    for sess in SESSIONS:
        src = cleaned / f"{subject_id}_{sess}.npy"
        if not src.exists():
            print(f"  {sess}: missing, skipped")
            continue

        ts = np.load(src)[:, : len(labels)]
        parcels = np.zeros((ts.shape[0], n_parcels), dtype=np.float32)
        for i, pid in enumerate(np.unique(labels[labels > 0])):
            parcels[:, i] = ts[:, labels == pid].mean(axis=1)

        np.save(out_dir / f"{subject_id}_{sess}_{METHOD}.npy", parcels)
        print(f"  {sess}: {parcels.shape}")


def main():
    parser = argparse.ArgumentParser(description="gMSHBM parcellation.")
    parser.add_argument("--subject", required=True, help="HCP subject ID")
    args = parser.parse_args()
    print(f"Variant {config.VARIANT} | Subject {args.subject}")
    t0 = time.time()
    process_subject(args.subject)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
