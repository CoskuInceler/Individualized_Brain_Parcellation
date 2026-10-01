"""
SCHAEFER PARCELLATION (Level 0 — group atlas)
=============================================
Applies the Schaefer 200-parcel atlas to the cleaned time series.
Atlas-to-data alignment
-----------------------
The atlas is defined on the full 32k mesh (64,984 vertices, medial wall
included). The cleaned .npy files come from CIFTI, which stores only valid
vertices, ordered as [valid_L | valid_R | subcortical]. The atlas must be
filtered to valid vertices before its indices line up with data columns.
Output: Outputs/Method_0_Schaefer/{subject}_{session}_Schaefer200.npy
        shape (timepoints, 200), float32
"""

import argparse
import time
import numpy as np
import config
import utils

SESSIONS = ["REST1", "REST2"]
N_PARCELS = 200


def parcellate(data, atlas):
    """
    Average vertex time series within each parcel.

    Parameters
    ----------
    data  : (T, V) array, full grayordinate data
    atlas : (n_cortex,) integer labels aligned to data columns

    Returns
    -------
    (T, n_parcels) float32 array, parcels in ascending label order
    """
    cortex = data[:, : len(atlas)]
    labels = np.unique(atlas[atlas > 0])
    out = np.zeros((cortex.shape[0], len(labels)), dtype=np.float32)
    for i, lab in enumerate(labels):
        out[:, i] = cortex[:, atlas == lab].mean(axis=1)
    return out


def build_atlas():
    """Load the Schaefer atlas and align it to the data column order."""
    ref = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    valid_R = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    atlas_L, atlas_R = utils.load_and_filter_atlas(
        config.SCHAEFER_200_FILE, valid_L, valid_R
    )
    return np.concatenate([atlas_L, atlas_R])


def process_subject(subject_id, atlas):
    """Parcellate both sessions of one subject."""
    in_dir = config.OUTPUTS_DIR / "Cleaned"
    out_dir = config.OUTPUTS_DIR / "Method_0_Schaefer"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sess in SESSIONS:
        src = in_dir / f"{subject_id}_{sess}.npy"
        if not src.exists():
            print(f"  {sess}: missing, skipped")
            continue
        t0 = time.time()
        ts = parcellate(np.load(src), atlas)
        np.save(out_dir / f"{subject_id}_{sess}_Schaefer200.npy", ts)
        print(f"  {sess}: {ts.shape} ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Schaefer parcellation.")
    parser.add_argument("--subject", required=True, help="HCP subject ID")
    args = parser.parse_args()
    print(f"Subject {args.subject}")
    t0 = time.time()
    process_subject(args.subject, build_atlas())
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
