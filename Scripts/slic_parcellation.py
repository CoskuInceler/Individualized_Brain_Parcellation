"""
SLIC PARCELLATION EXPORT (Methods 3 & 4)
==========================================
Exports SLIC parcellation results to all standard formats.

Takes the raw label arrays from slic_main.py, shifts labels for unique
IDs across the brain, inflates to full 32k mesh, and saves everything.

SLIC labels are 0-99 per hemisphere. For export:
    - Left:  1-100
    - Right: 101-200
    - Medial wall: 0

Output per subject per method (SLIC_F and SLIC_C):
    - {subj}_{method}_LH.npy / RH.npy             (full 32k labels)
    - {subj}_{method}_Parcellation.dscalar.nii     (CIFTI scalar)
    - {subj}_{method}_Parcellation.dlabel.nii      (CIFTI label)
    - {subj}_{method}_Timeseries.csv               (parcel-averaged timeseries)

Outputs go to: Outputs/Method_3_SLIC_F/ and Outputs/Method_4_SLIC_C/
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import config
from utils import get_valid_vertices, save_cifti

N_MESH = 32492


# =============================================================================
# LABEL SHIFTING
# =============================================================================

def shift_labels_for_cifti(raw_L, raw_R):
    """
    Shift SLIC labels to create unique IDs across the brain.
    Left: 0â†’1 ... 99â†’100, Right: 0â†’101 ... 99â†’200
    """
    shifted_L = raw_L + 1
    shifted_R = raw_R + 101
    return shifted_L, shifted_R


# =============================================================================
# CIFTI SAVING
# =============================================================================

def save_dlabel(data_matrix, template_path, output_path):
    """Save parcellation labels as .dlabel.nii using Schaefer template header."""
    print(f"   Saving dlabel: {os.path.basename(str(output_path))}")
    template_img = nib.load(template_path)
    new_img = nib.Cifti2Image(
        data_matrix,
        template_img.header,
        template_img.nifti_header
    )
    nib.save(new_img, output_path)


# =============================================================================
# TIMESERIES EXTRACTION
# =============================================================================

def extract_parcel_timeseries(subj_id, raw_L, raw_R, valid_L, valid_R):
    """
    Average dense timeseries within each SLIC parcel.
    L parcels (0-99) and R parcels (0-99) handled separately â†’ 200 columns.
    """
    ts_path = config.SHARED_DATA_DIR / "Aggregated" / f"{subj_id}_ALL_Dense.npy"

    if not ts_path.exists():
        print(f"   [Skip] Dense timeseries not found: {ts_path.name}")
        return None

    full_ts = np.load(ts_path)
    n_time = full_ts.shape[0]
    n_L = len(valid_L)
    n_R = len(valid_R)

    data_L = full_ts[:, :n_L]
    data_R = full_ts[:, n_L:n_L + n_R]

    unique_L = np.unique(raw_L[raw_L >= 0])
    unique_R = np.unique(raw_R[raw_R >= 0])

    ts_matrix = np.zeros((n_time, len(unique_L) + len(unique_R)))
    col_names = []

    for i, label in enumerate(unique_L):
        mask = (raw_L == label)
        if np.any(mask):
            ts_matrix[:, i] = np.mean(data_L[:, mask], axis=1)
        col_names.append(f"L_Parcel_{int(label)}")

    for i, label in enumerate(unique_R):
        mask = (raw_R == label)
        if np.any(mask):
            ts_matrix[:, len(unique_L) + i] = np.mean(data_R[:, mask], axis=1)
        col_names.append(f"R_Parcel_{int(label)}")

    return pd.DataFrame(ts_matrix, columns=col_names)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SLIC PARCELLATION EXPORT (Methods 3 & 4)")
    print("=" * 60)

    # Get valid vertex masks
    ref_path = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')

    template_path = config.SCHAEFER_200_FILE

    methods = [
        ('SLIC_F', config.METHOD_3_DIR),
        ('SLIC_C', config.METHOD_4_DIR),
    ]

    for method_name, method_dir in methods:
        os.makedirs(method_dir, exist_ok=True)

        print(f"\n{'â•' * 40}")
        print(f"Method: {method_name}")
        print(f"{'â•' * 40}")

        for subj in config.SUBJECT_IDS:
            print(f"\n[Subject] {subj}")

            label_L_path = method_dir / f"{subj}_Labels_L.npy"
            label_R_path = method_dir / f"{subj}_Labels_R.npy"

            if not (label_L_path.exists() and label_R_path.exists()):
                print(f"   [Skip] Label files missing â€” run slic_main first")
                continue

            # 1. Load raw labels (compressed valid-vertex space, 0-99)
            print("   [1/4] Loading labels...")
            raw_L = np.load(label_L_path)
            raw_R = np.load(label_R_path)
            print(f"   LH: {raw_L.shape}, RH: {raw_R.shape}")

            # 2. Shift + inflate to full 32k mesh
            print("   [2/4] Saving CIFTI files...")
            shifted_L, shifted_R = shift_labels_for_cifti(raw_L, raw_R)

            full_L = np.zeros(N_MESH, dtype=np.float32)
            full_R = np.zeros(N_MESH, dtype=np.float32)
            full_L[valid_L] = shifted_L
            full_R[valid_R] = shifted_R

            combined = np.concatenate([full_L, full_R]).reshape(1, -1)

            save_cifti(combined, template_path,
                       method_dir / f"{subj}_{method_name}_Parcellation.dscalar.nii")
            save_dlabel(combined, template_path,
                        method_dir / f"{subj}_{method_name}_Parcellation.dlabel.nii")

            # 3. Save full-mesh .npy (for visualization)
            print("   [3/4] Saving full-mesh .npy...")
            np.save(method_dir / f"{subj}_{method_name}_LH.npy", full_L)
            np.save(method_dir / f"{subj}_{method_name}_RH.npy", full_R)
            print(f"   Saved: {subj}_{method_name}_LH.npy, {subj}_{method_name}_RH.npy")

            # 4. Extract timeseries
            print("   [4/4] Extracting timeseries...")
            df = extract_parcel_timeseries(subj, raw_L, raw_R, valid_L, valid_R)
            if df is not None:
                csv_path = method_dir / f"{subj}_{method_name}_Timeseries.csv"
                df.to_csv(csv_path, index=False)
                print(f"   Saved: {csv_path.name}  ({df.shape})")

            print(f"   âœ“ {subj} complete")

    print("\n--- SLIC PARCELLATION EXPORT COMPLETE ---")