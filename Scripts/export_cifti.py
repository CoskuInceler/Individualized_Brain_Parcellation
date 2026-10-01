"""
CIFTI EXPORT
============
Writes parcellations as CIFTI dlabel files on the full 32k mesh.



The .npy label files that drive the analysis live in valid-vertex space and
are only readable from this codebase. CIFTI is the standard surface format,
so exporting makes the parcellations viewable in Connectome Workbench and
usable by anyone working from the released data.



Labels are inflated back to the full mesh, with medial-wall vertices left
at 0, and the Schaefer atlas supplies the header.



Usage:
    python3 export_cifti.py --subject 100206 --method Method_2_AGP
    python3 export_cifti.py --subject 100206 --method Method_2_AGP --sessions ALL
"""

import argparse
import numpy as np


import config
import utils

N_MESH = utils.N_VERTICES_PER_HEMI
SESSIONS = ["REST1", "REST2", "ALL"]


def inflate(labels, valid_L, valid_R):
    """
    Put valid-vertex labels back on the full mesh.



    Returns a (1, 64984) array laid out as [left mesh | right mesh], which
    is what a dlabel file built on the Schaefer header expects.
    """
    n_L = len(valid_L)
    full_L = np.zeros(N_MESH, dtype=np.float32)
    full_R = np.zeros(N_MESH, dtype=np.float32)
    full_L[valid_L] = labels[:n_L]
    full_R[valid_R] = labels[n_L:]
    return np.concatenate([full_L, full_R]).reshape(1, -1)


def find_labels(method_dir, subject_id, sessions):
    """
    Label files for this subject, keyed by the name to write out.



    Methods fitted per session store one file per session; gMSHBM, whose
    parcellation does not depend on the data, stores a single file.
    """
    lab_dir = method_dir / "Labels"
    found = {}

    single = lab_dir / f"{subject_id}_labels.npy"
    if single.exists():
        found[subject_id] = single
        return found

    for s in sessions:
        path = lab_dir / f"{subject_id}_{s}_labels.npy"
        if path.exists():
            found[f"{subject_id}_{s}"] = path
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Export parcellations as CIFTI."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--method", required=True, help="method folder, e.g. Method_2_AGP"
    )
    parser.add_argument(
        "--sessions", nargs="+", default=SESSIONS, choices=SESSIONS
    )
    args = parser.parse_args()

    method_dir = config.OUTPUTS_DIR / args.method
    out_dir = method_dir / "CIFTI"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = config.get_brain_path(args.subject, config.RUN_IDS[0])
    valid_L = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    valid_R = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")

    files = find_labels(method_dir, args.subject, args.sessions)
    if not files:
        raise FileNotFoundError(
            f"no label files for {args.subject} in {method_dir / 'Labels'}"
        )

    for name, path in sorted(files.items()):
        labels = np.load(path)
        utils.save_cifti(
            inflate(labels, valid_L, valid_R),
            config.SCHAEFER_200_FILE,
            out_dir / f"{name}.dlabel.nii",
        )
        n = len(np.unique(labels[labels > 0]))
        print(f"  {name}: {n} parcels")


if __name__ == "__main__":
    main()
