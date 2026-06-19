"""
AGP PARCELLATION EXPORT (Method 2)
====================================
Exports AGP region growing results to all standard formats.

Takes the raw label arrays from agp_step3_region_growing.py, inflates
them to the full 32k mesh, and saves in every format.

Output per subject:
    - {subj}_AGP_Labels_L.npy / R.npy             (raw compressed labels)
    - {subj}_AGP_Parcellation.dscalar.nii          (CIFTI scalar)
    - {subj}_AGP_Parcellation.dlabel.nii           (CIFTI label)
    - {subj}_AGP_Timeseries.csv                    (parcel-averaged timeseries)

All outputs go to: Outputs/Method_2_AGP/
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import config
from utils import get_valid_vertices, save_cifti

N_MESH = 32492


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
    """Average dense timeseries within each AGP parcel."""
    ts_path = config.SHARED_DATA_DIR / "Aggregated" / f"{subj_id}_ALL_Dense.npy"

    if not ts_path.exists():
        print(f"   [Skip] Dense timeseries not found: {ts_path.name}")
        return None

    full_ts = np.load(ts_path)
    print(f"   Dense timeseries: {full_ts.shape}")

    # AGP labels are in valid-vertex space (compressed ~29k)
    # Dense data columns: [LH valid | RH valid | subcortex]
    n_L = len(valid_L)
    n_R = len(valid_R)
    data_L = full_ts[:, :n_L]
    data_R = full_ts[:, n_L:n_L + n_R]

    unique_labels = np.unique(np.concatenate([raw_L, raw_R]))
    unique_labels = unique_labels[unique_labels > 0]

    n_time = full_ts.shape[0]
    ts_matrix = np.zeros((n_time, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask_L = (raw_L == label)
        mask_R = (raw_R == label)

        vals_L = data_L[:, mask_L] if np.any(mask_L) else np.empty((n_time, 0))
        vals_R = data_R[:, mask_R] if np.any(mask_R) else np.empty((n_time, 0))

        all_vals = np.hstack([vals_L, vals_R])
        ts_matrix[:, i] = np.mean(all_vals, axis=1)

    col_names = [f"Parcel_{int(l)}" for l in unique_labels]
    return pd.DataFrame(ts_matrix, columns=col_names)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    out_dir = config.METHOD_2_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("AGP PARCELLATION EXPORT (Method 2)")
    print("=" * 60)

    # Get valid vertex masks
    ref_path = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')

    template_path = config.SCHAEFER_200_FILE

    for subj in config.SUBJECT_IDS:
        print(f"\n{'─' * 40}")
        print(f"Subject: {subj}")
        print(f"{'─' * 40}")

        label_L_path = out_dir / f"{subj}_Labels_L.npy"
        label_R_path = out_dir / f"{subj}_Labels_R.npy"

        if not (label_L_path.exists() and label_R_path.exists()):
            print(f"   [Skip] Label files missing — run agp_step3 first")
            continue

        # 1. Load raw labels (compressed valid-vertex space)
        print("\n[1/4] Loading labels...")
        raw_L = np.load(label_L_path)
        raw_R = np.load(label_R_path)
        print(f"   LH: {raw_L.shape}, RH: {raw_R.shape}")

        # 2. Inflate to full 32k mesh
        print("\n[2/4] Saving CIFTI files...")
        full_L = np.zeros(N_MESH, dtype=np.float32)
        full_R = np.zeros(N_MESH, dtype=np.float32)
        full_L[valid_L] = raw_L
        full_R[valid_R] = raw_R

        combined = np.concatenate([full_L, full_R]).reshape(1, -1)

        save_cifti(combined, template_path,
                   out_dir / f"{subj}_AGP_Parcellation.dscalar.nii")
        save_dlabel(combined, template_path,
                    out_dir / f"{subj}_AGP_Parcellation.dlabel.nii")

        # 3. Save full-mesh .npy (for visualization scripts)
        print("\n[3/4] Saving full-mesh .npy...")
        np.save(out_dir / f"{subj}_AGP_LH.npy", full_L)
        np.save(out_dir / f"{subj}_AGP_RH.npy", full_R)
        print(f"   Saved: {subj}_AGP_LH.npy, {subj}_AGP_RH.npy")

        # 4. Extract timeseries
        print("\n[4/4] Extracting timeseries...")
        df = extract_parcel_timeseries(subj, raw_L, raw_R, valid_L, valid_R)
        if df is not None:
            csv_path = out_dir / f"{subj}_AGP_Timeseries.csv"
            df.to_csv(csv_path, index=False)
            print(f"   Saved: {csv_path.name}  ({df.shape})")

        print(f"\n   ✓ Subject {subj} complete")

    print("\n--- AGP PARCELLATION EXPORT COMPLETE ---")