"""
gMSHBM PARCELLATION (Method 1)
================================
Extracts individual gMS-HBM parcellations from the Kong et al. (2022)
group .mat file and exports in all formats.

This replaces both Extract.m (MATLAB) and gMSHBM_export.py by doing
everything in Python using h5py.

Subject indices are read from HCP_subject_list.txt in Group_Priors/,
so no hardcoding is needed.

Input:
    - HCP_1029sub_200Parcels_Kong2022_gMSHBM.mat (group parcellation)
    - HCP_subject_list.txt (subject IDs in order matching .mat columns)

Output per subject:
    - {subj}_gMSHBM_LH.npy / RH.npy              (labels per hemisphere, full 32k mesh)
    - {subj}_gMSHBM_Parcellation.dscalar.nii       (CIFTI scalar)
    - {subj}_gMSHBM_Parcellation.dlabel.nii        (CIFTI label — for visualization)
    - {subj}_gMSHBM_Timeseries.csv                 (parcel-averaged timeseries)

All outputs go to: Outputs/Method_1_MSHBM/

IMPORTANT — Label Space vs Data Space:
    The .mat file stores labels on the FULL 32k mesh (32,492 per hemisphere),
    including medial wall vertices (label 0). This is correct for:
        - .npy saving (visualization scripts plot on full mesh)
        - CIFTI saving (dlabel/dscalar use full 64,984 vertex structure)

    But the dense timeseries data from step1_clean_data.py stores only VALID
    vertices (medial wall excluded). The data columns are:
        [valid_L (~29,696) | valid_R (~29,716) | subcortical (~31,870)]

    So for timeseries extraction, we must FILTER the labels to valid vertices
    first, then use the correct column slicing.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import h5py
import config
from utils import get_valid_vertices, save_cifti

N_MESH = 32492


# =============================================================================
# SUBJECT INDEX LOOKUP
# =============================================================================

def load_subject_list(list_path):
    """
    Read HCP_subject_list.txt and return a dict mapping subject_id → column index.

    The .txt file has one subject ID per line, in the same order as the
    columns in the group .mat file. Line 1 = column 0 in Python.
    """
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Subject list not found: {list_path}")

    with open(list_path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    lookup = {subj: idx for idx, subj in enumerate(subjects)}
    print(f"   Loaded {len(lookup)} subjects from {os.path.basename(str(list_path))}")
    return lookup


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_labels_from_group(group_mat_path, subject_id, col_idx):
    """
    Extract one subject's parcellation from the group .mat file.

    The .mat file is MATLAB v7.3 (HDF5 format), read with h5py.
    h5py transposes MATLAB matrices, so:
        MATLAB shape: (32492, 1029)  →  h5py shape: (1029, 32492)
        Indexing: data[col_idx, :]  (subject on axis 0, vertices on axis 1)

    Contains:
        lh_labels_all: (1029, 32492) — left hemisphere labels
        rh_labels_all: (1029, 32492) — right hemisphere labels

    Labels are on the FULL 32k mesh:
        - LH: 0 (medial wall), 1-100 (parcels)
        - RH: 0 (medial wall), 101-200 (parcels)

    Args:
        group_mat_path: Path to the group .mat file
        subject_id: str, e.g., "100307"
        col_idx: int, subject index (row in h5py, column in MATLAB)

    Returns:
        lh_labels: (32492,) array, labels 0-100
        rh_labels: (32492,) array, labels 0-200
    """
    print(f"   Loading group .mat file (subject index {col_idx})...")

    with h5py.File(str(group_mat_path), 'r') as data:
        # h5py transposes: MATLAB (32492, 1029) → h5py (1029, 32492)
        # Subject on axis 0, vertices on axis 1
        lh_labels = data['lh_labels_all'][col_idx, :].flatten()
        rh_labels = data['rh_labels_all'][col_idx, :].flatten()

    assert lh_labels.shape[0] == N_MESH, f"LH wrong shape: {lh_labels.shape}"
    assert rh_labels.shape[0] == N_MESH, f"RH wrong shape: {rh_labels.shape}"

    print(f"   LH: {N_MESH} vertices, labels 0–{int(lh_labels.max())}, medial wall: {np.sum(lh_labels == 0)}")
    print(f"   RH: {N_MESH} vertices, labels 0–{int(rh_labels.max())}, medial wall: {np.sum(rh_labels == 0)}")

    return lh_labels.astype(int), rh_labels.astype(int)


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

def extract_parcel_timeseries(subject_id, lh_labels_full, rh_labels_full,
                              valid_L, valid_R):
    """
    Average dense timeseries within each gMSHBM parcel.

    IMPORTANT: The labels are on the full 32k mesh, but the dense data only
    contains valid vertices. We must filter labels to valid vertices first,
    then slice the data correctly.

    Args:
        subject_id:       e.g., "100307"
        lh_labels_full:   (32492,) full mesh labels for left hemisphere
        rh_labels_full:   (32492,) full mesh labels for right hemisphere
        valid_L:          Valid vertex indices for left hemisphere
        valid_R:          Valid vertex indices for right hemisphere

    Returns:
        DataFrame with parcel-averaged timeseries, or None if data missing.
    """
    ts_path = config.SHARED_DATA_DIR / "Aggregated" / f"{subject_id}_ALL_Dense.npy"

    if not ts_path.exists():
        print(f"   [Skip] Dense timeseries not found: {ts_path.name}")
        return None

    full_ts = np.load(ts_path)
    print(f"   Dense timeseries: {full_ts.shape}")

    # Filter labels from full 32k mesh to valid vertices only
    # After filtering, label index i matches data column i
    labels_L = lh_labels_full[valid_L]
    labels_R = rh_labels_full[valid_R]

    n_L = len(valid_L)
    n_R = len(valid_R)

    # Slice data correctly: [valid_L columns | valid_R columns | subcortical]
    data_L = full_ts[:, :n_L]
    data_R = full_ts[:, n_L:n_L + n_R]

    # Combine labels for finding unique parcel IDs
    all_labels = np.concatenate([labels_L, labels_R])
    unique_parcels = np.unique(all_labels)
    unique_parcels = unique_parcels[unique_parcels > 0]

    n_time = full_ts.shape[0]
    parcel_ts = np.zeros((n_time, len(unique_parcels)))

    for i, pid in enumerate(unique_parcels):
        # gMSHBM labels: LH uses 1-100, RH uses 101-200 (no overlap)
        # So each parcel ID only appears in one hemisphere
        mask_L = (labels_L == pid)
        mask_R = (labels_R == pid)

        vals = []
        if np.any(mask_L):
            vals.append(data_L[:, mask_L])
        if np.any(mask_R):
            vals.append(data_R[:, mask_R])

        if vals:
            parcel_ts[:, i] = np.mean(np.hstack(vals), axis=1)

    col_names = [f"Parcel_{int(p)}" for p in unique_parcels]
    return pd.DataFrame(parcel_ts, columns=col_names)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    out_dir = config.METHOD_1_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("gMSHBM PARCELLATION (Method 1)")
    print("=" * 60)

    # Group .mat file
    group_mat = config.GROUP_PRIORS_DIR / "HCP_1029sub_200Parcels_Kong2022_gMSHBM.mat"
    subject_list_file = config.GROUP_PRIORS_DIR / "HCP_subject_list.txt"

    if not group_mat.exists():
        print(f"ERROR: Group .mat file not found: {group_mat}")
        exit(1)

    # Load subject-to-index mapping from text file
    print("\n[Setup] Loading subject list...")
    subject_lookup = load_subject_list(subject_list_file)

    # Get valid vertex masks (needed for timeseries extraction)
    print("\n[Setup] Getting valid vertex masks...")
    ref_path = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')

    template_path = config.SCHAEFER_200_FILE

    for subj in config.SUBJECT_IDS:
        print(f"\n{'─' * 40}")
        print(f"Subject: {subj}")
        print(f"{'─' * 40}")

        if subj not in subject_lookup:
            print(f"   ✗ Subject {subj} not found in HCP_subject_list.txt — skipping")
            continue

        col_idx = subject_lookup[subj]

        try:
            # 1. Extract labels (full 32k mesh)
            print("\n[1/4] Extracting labels...")
            lh, rh = extract_labels_from_group(group_mat, subj, col_idx)

            # 2. Save .npy (full 32k mesh — used by visualization)
            print("\n[2/4] Saving .npy...")
            np.save(out_dir / f"{subj}_gMSHBM_LH.npy", lh)
            np.save(out_dir / f"{subj}_gMSHBM_RH.npy", rh)
            print(f"   Saved: {subj}_gMSHBM_LH.npy, {subj}_gMSHBM_RH.npy")

            # 3. Save CIFTI (full 64k = 32k+32k — matches template structure)
            print("\n[3/4] Saving CIFTI files...")
            combined = np.concatenate([lh, rh]).reshape(1, -1).astype(np.float32)

            save_cifti(combined, template_path,
                       out_dir / f"{subj}_gMSHBM_Parcellation.dscalar.nii")
            save_dlabel(combined, template_path,
                        out_dir / f"{subj}_gMSHBM_Parcellation.dlabel.nii")

            # 4. Extract timeseries (must filter labels to valid vertices)
            print("\n[4/4] Extracting timeseries...")
            df = extract_parcel_timeseries(subj, lh, rh, valid_L, valid_R)
            if df is not None:
                csv_path = out_dir / f"{subj}_gMSHBM_Timeseries.csv"
                df.to_csv(csv_path, index=False)
                print(f"   Saved: {csv_path.name}  ({df.shape})")

            print(f"\n   ✓ Subject {subj} complete")

        except Exception as e:
            print(f"\n   ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- gMSHBM PARCELLATION COMPLETE ---")