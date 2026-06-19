"""
SCHAEFER PARCELLATION (Method 0)
=================================
Applies the Schaefer 200-parcel atlas to cleaned data.
Saves parcellated timeseries for ALL combinations:

Per-run:     {subj}_{run}_Schaefer200.csv
Per-session: {subj}_REST1_Schaefer200.csv
             {subj}_REST2_Schaefer200.csv
All runs:    {subj}_ALL_Schaefer200.csv

All outputs go to: Outputs/Method_0_Schaefer/
Aggregated outputs go to: Outputs/Method_0_Schaefer/Aggregated/

IMPORTANT — Atlas-to-Data Alignment:
    The Schaefer atlas is stored on the FULL 32k mesh (64,984 vertices total),
    which includes the medial wall (label 0). But the cleaned .npy data from
    step1_clean_data.py comes from CIFTI dtseries files, which only store
    VALID vertices (medial wall excluded). The data columns are ordered:

        [valid_L (~29,696) | valid_R (~29,716) | subcortical (~31,870)]

    So we CANNOT directly index data columns with atlas entries — the atlas
    has 64,984 entries but the cortex only occupies ~59,412 data columns.
    We must filter the atlas to valid vertices first, then the indices match.
"""

import os
import numpy as np
import pandas as pd
import config
from utils import get_valid_vertices, load_and_filter_atlas


# =============================================================================
# PARCELLATION
# =============================================================================

def parcellate_data(data_matrix, atlas_labels, n_cortex):
    """
    Averages vertex-level time series within each Schaefer parcel.

    Args:
        data_matrix:  (Time x Vertices) — the full ~91k grayordinate data.
                      Columns are ordered: [valid_L | valid_R | subcortical].

        atlas_labels: (n_cortex,) integer labels, ALREADY FILTERED to valid
                      vertices only. Entry i corresponds to data column i.
                      Labels: 0 = unassigned (should not appear after filtering),
                      1-200 = Schaefer parcels.

        n_cortex:     Number of cortical columns in the data (n_valid_L + n_valid_R).
                      Used to slice out just the cortex from the full data.

    Returns:
        DataFrame with columns Parcel_1 ... Parcel_200, rows = timepoints.
    """
    # Slice out only the cortical columns — skip subcortical at the end
    cortex_data = data_matrix[:, :n_cortex]

    # Find all unique parcel IDs (skip 0 = background/medial wall)
    unique_labels = np.unique(atlas_labels)
    unique_labels = unique_labels[unique_labels > 0]

    n_timepoints = cortex_data.shape[0]
    n_parcels = len(unique_labels)

    parcellated_data = np.zeros((n_timepoints, n_parcels))

    for i, label_id in enumerate(unique_labels):
        # Find which columns (valid vertices) belong to this parcel
        parcel_indices = np.where(atlas_labels == label_id)[0]

        # Average the time series across all vertices in this parcel
        parcel_signal = cortex_data[:, parcel_indices]
        parcellated_data[:, i] = np.mean(parcel_signal, axis=1)

    col_names = [f"Parcel_{int(lbl)}" for lbl in unique_labels]
    return pd.DataFrame(parcellated_data, columns=col_names)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    out_dir = config.METHOD_0_DIR
    agg_dir = out_dir / "Aggregated"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(agg_dir, exist_ok=True)

    print("=" * 60)
    print("SCHAEFER PARCELLATION (Method 0)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load atlas and filter to valid vertices
    # ------------------------------------------------------------------
    # The atlas .dlabel.nii has 64,984 entries (full 32k×2 mesh).
    # The data .npy has ~59,412 cortical columns (valid vertices only).
    # We must filter the atlas to match the data column order.

    print("\n[Setup] Getting valid vertex masks...")
    ref_path = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')

    print("\n[Setup] Loading and filtering atlas...")
    atlas_L, atlas_R = load_and_filter_atlas(config.SCHAEFER_200_FILE, valid_L, valid_R)

    # Combine into a single array matching data column order:
    # data columns = [valid_L | valid_R | subcortical]
    # atlas_combined = [atlas_L | atlas_R]  (same length as cortical columns)
    atlas_combined = np.concatenate([atlas_L, atlas_R])
    n_cortex = len(atlas_combined)

    print(f"\n[Setup] Atlas aligned to data:")
    print(f"   Valid L: {len(valid_L)}, Valid R: {len(valid_R)}")
    print(f"   Combined cortex columns: {n_cortex}")
    print(f"   Unique labels: {len(np.unique(atlas_combined[atlas_combined > 0]))} parcels")

    shared_dir = config.SHARED_DATA_DIR
    shared_agg = shared_dir / "Aggregated"
    sessions = ["REST1", "REST2"]

    for subj in config.SUBJECT_IDS:
        print(f"\n[Subject] {subj}")

        # --------------------------------------------------------------
        # PHASE 1: Per-run parcellation
        # --------------------------------------------------------------
        for run in config.RUN_IDS:
            npy_path = shared_dir / f"{subj}_{run}_clean.npy"
            if not npy_path.exists():
                print(f"   [Skip] {run} — file missing")
                continue

            data = np.load(npy_path)
            df = parcellate_data(data, atlas_combined, n_cortex)

            out_path = out_dir / f"{subj}_{run}_Schaefer200.csv"
            df.to_csv(out_path, index=False)
            print(f"   [Saved] {out_path.name}  ({df.shape})")

        # --------------------------------------------------------------
        # PHASE 2: Per-session parcellation
        # --------------------------------------------------------------
        session_dfs = []

        for sess in sessions:
            sess_path = shared_agg / f"{subj}_{sess}_Dense.npy"
            if not sess_path.exists():
                print(f"   [Skip] {sess} — aggregated file missing")
                continue

            data = np.load(sess_path)
            df = parcellate_data(data, atlas_combined, n_cortex)

            out_path = agg_dir / f"{subj}_{sess}_Schaefer200.csv"
            df.to_csv(out_path, index=False)
            print(f"   [Saved] {out_path.name}  ({df.shape})")
            session_dfs.append(df)

        # --------------------------------------------------------------
        # PHASE 3: ALL runs parcellation
        # --------------------------------------------------------------
        all_path = shared_agg / f"{subj}_ALL_Dense.npy"
        if all_path.exists():
            data = np.load(all_path)
            df = parcellate_data(data, atlas_combined, n_cortex)

            out_path = agg_dir / f"{subj}_ALL_Schaefer200.csv"
            df.to_csv(out_path, index=False)
            print(f"   [Saved] {out_path.name}  ({df.shape})")

    print("\n--- SCHAEFER PARCELLATION COMPLETE ---")