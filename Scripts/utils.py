"""
SHARED UTILITIES FOR BRAIN PARCELLATION PIPELINE

This module centralizes functions used across multiple scripts to avoid
duplication and ensure consistency. All parcellation methods (M0–M4)
import shared logic from here.

Functions:
    get_valid_vertices()            - Extract valid vertex indices from CIFTI header
    load_pickle()                   - Load a pickled Python object
    load_and_filter_atlas()         - Load Schaefer atlas and filter to valid vertices
    load_subject_data()             - Load aggregated dense time series for one subject
    save_cifti()                    - Save a numpy array as a CIFTI file

    --- Validation metrics (shared across all methods) ---
    calculate_homogeneity()         - Functional homogeneity (mean, fast)
    calculate_homogeneity_detailed()- Functional homogeneity + per-parcel breakdown
    calculate_parcel_sizes()        - Parcel size statistics (mean, SD, CV, Gini, ...)
    calculate_contiguity_stats()    - Spatial contiguity statistics per hemisphere
    calculate_dice_adjacency()      - Dice overlap without requiring parcel correspondence
    parcellate_dense()              - Average dense timeseries within parcels
    calculate_fc_testretest()       - Test-retest reliability via FC matrix correlation
"""

import os
import numpy as np
import nibabel as nib
import pickle
import config


# =============================================================================
# 1. CIFTI VERTEX EXTRACTION
# =============================================================================

def get_valid_vertices(cifti_path, structure_name):
    """
    Extract valid (non-medial-wall) vertex indices from a CIFTI file header.

    Args:
        cifti_path (str or Path): Path to any CIFTI file (.dtseries.nii).
        structure_name (str): 'CIFTI_STRUCTURE_CORTEX_LEFT' or
                              'CIFTI_STRUCTURE_CORTEX_RIGHT'

    Returns:
        np.ndarray: Mesh vertex indices that contain data (~29k per hemisphere).
    """
    cifti_path = str(cifti_path)
    if not os.path.exists(cifti_path):
        raise FileNotFoundError(f"CIFTI file not found: {cifti_path}")

    img = nib.load(cifti_path)
    brain_axis = img.header.get_axis(1)

    for name, _, model in brain_axis.iter_structures():
        if name == structure_name:
            valid_indices = np.array(model.vertex)
            print(f"[Utils]   {structure_name}: {len(valid_indices)} valid vertices")
            return valid_indices

    raise ValueError(f"Structure '{structure_name}' not found in CIFTI file!")


# =============================================================================
# 2. PICKLE I/O
# =============================================================================

def load_pickle(path):
    """Load a pickled Python object from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# 3. ATLAS LOADING
# =============================================================================

def load_and_filter_atlas(atlas_path, valid_L, valid_R):
    """
    Load the Schaefer atlas and filter it to match valid vertex arrays.

    Returns:
        tuple: (atlas_L, atlas_R) — filtered integer label arrays in valid-vertex space.
    """
    img = nib.load(atlas_path)
    data = img.get_fdata().squeeze().astype(int)

    n_L_full = 32492
    atlas_L = data[:n_L_full][valid_L]
    atlas_R = data[n_L_full:][valid_R]

    print(f"[Utils]   Atlas: L={len(atlas_L)} vertices, R={len(atlas_R)} vertices")
    return atlas_L, atlas_R


# =============================================================================
# 4. TIME SERIES LOADING
# =============================================================================

def load_subject_data(subj_id):
    """
    Load the aggregated dense time series (ALL runs) for one subject.

    Returns:
        np.ndarray: Shape (Timepoints, ~91k vertices).
    """
    file_path = config.SHARED_DATA_DIR / "Aggregated" / f"{subj_id}_ALL_Dense.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    data = np.load(file_path)
    print(f"[Utils]   Loaded {subj_id}: {data.shape}")
    return data


# =============================================================================
# 5. FUNCTIONAL HOMOGENEITY (fast, mean only)
# =============================================================================

def calculate_homogeneity(ts_data, labels):
    """
    Calculate functional homogeneity of a parcellation.

    For each parcel, computes the average correlation of every vertex's
    time series with the parcel mean. Returns the grand mean across parcels.

    Args:
        ts_data (np.ndarray): (Timepoints, Vertices)
        labels (np.ndarray): Parcel assignments per vertex (0 = unassigned).

    Returns:
        float: Grand mean homogeneity score.
    """
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    scores = []

    for pid in unique_ids:
        idx = np.where(labels == pid)[0]
        if len(idx) < 2:
            continue
        vd = ts_data[:, idx]
        ms = np.mean(vd, axis=1)
        vc = vd - vd.mean(axis=0)
        mc = ms - ms.mean()
        vn = np.sqrt((vc ** 2).sum(axis=0))
        mn = float(np.sqrt((mc ** 2).sum()))
        vn[vn == 0] = 1.0
        if mn == 0:
            mn = 1.0
        corrs = np.dot(mc, vc) / (mn * vn)
        scores.append(float(np.mean(corrs)))

    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# 6. CIFTI EXPORT
# =============================================================================

def save_cifti(data_matrix, template_path, output_path):
    """Save a numpy matrix as a CIFTI file using a template header."""
    template_img = nib.load(template_path)
    new_img = nib.Cifti2Image(
        data_matrix, template_img.header, template_img.nifti_header
    )
    nib.save(new_img, output_path)
    print(f"[Utils]   Saved: {os.path.basename(str(output_path))}")


# =============================================================================
# 7. HOMOGENEITY — DETAILED (per-parcel breakdown)
# =============================================================================

def calculate_homogeneity_detailed(ts_data, labels):
    """
    Functional homogeneity with per-parcel breakdown.

    Args:
        ts_data: (T, V) timeseries
        labels:  (V,) parcel labels, 0 = unlabeled

    Returns:
        mean_score (float), per_parcel (dict {parcel_id: score})
    """
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    per_parcel = {}
    scores = []

    for pid in unique_ids:
        idx = np.where(labels == pid)[0]
        if len(idx) < 2:
            per_parcel[int(pid)] = float('nan')
            continue
        vd = ts_data[:, idx]
        ms = np.mean(vd, axis=1)
        vc = vd - vd.mean(axis=0)
        mc = ms - ms.mean()
        vn = np.sqrt((vc ** 2).sum(axis=0))
        mn = float(np.sqrt((mc ** 2).sum()))
        vn[vn == 0] = 1.0
        if mn == 0:
            mn = 1.0
        corrs = np.dot(mc, vc) / (mn * vn)
        s = float(np.mean(corrs))
        per_parcel[int(pid)] = s
        scores.append(s)

    mean_score = float(np.mean(scores)) if scores else 0.0
    return mean_score, per_parcel


# =============================================================================
# 8. PARCEL SIZE STATISTICS
# =============================================================================

def calculate_parcel_sizes(labels):
    """
    Compute parcel size statistics for a parcellation.

    Args:
        labels: 1D array, 0 = unlabeled/medial wall

    Returns:
        dict with keys: sizes (array), mean, std, cv, min, max, gini, n_parcels
    """
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return {'sizes': np.array([]), 'mean': 0.0, 'std': 0.0, 'cv': 0.0,
                'min': 0, 'max': 0, 'gini': 0.0, 'n_parcels': 0}

    sizes = np.array([np.sum(labels == pid) for pid in unique_ids], dtype=float)
    mean = float(np.mean(sizes))
    std = float(np.std(sizes))
    cv = float(std / mean) if mean > 0 else 0.0

    # Gini coefficient: 0 = perfectly equal, 1 = maximally unequal
    sorted_s = np.sort(sizes)
    n = len(sorted_s)
    total = sorted_s.sum()
    gini = float(
        (2.0 * np.dot(np.arange(1, n + 1), sorted_s) / (n * total)) - (n + 1) / n
    ) if total > 0 else 0.0

    return {
        'sizes': sizes,
        'mean': mean,
        'std': std,
        'cv': cv,
        'min': int(sizes.min()),
        'max': int(sizes.max()),
        'gini': gini,
        'n_parcels': int(len(sizes))
    }


# =============================================================================
# 9. SPATIAL CONTIGUITY STATISTICS
# =============================================================================

def calculate_contiguity_stats(labels, adj_list):
    """
    Compute spatial contiguity statistics for a parcellation hemisphere.

    Args:
        labels:   1D array of parcel assignments (valid-vertex space)
        adj_list: adjacency list (list of sets), or None to skip

    Returns:
        dict with keys: n_contiguous, n_fragmented, n_total,
                        pct_contiguous, mean_components
    """
    if adj_list is None:
        return {'n_contiguous': None, 'n_fragmented': None, 'n_total': None,
                'pct_contiguous': None, 'mean_components': None}

    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]

    n_contiguous = 0
    n_fragmented = 0
    component_counts = []

    for pid in unique_ids:
        remaining = set(np.where(labels == pid)[0].tolist())
        if not remaining:
            continue
        n_comp = 0
        while remaining:
            n_comp += 1
            stack = [next(iter(remaining))]
            while stack:
                v = stack.pop()
                if v not in remaining:
                    continue
                remaining.discard(v)
                for nb in adj_list[v]:
                    if nb in remaining:
                        stack.append(nb)
        component_counts.append(n_comp)
        if n_comp == 1:
            n_contiguous += 1
        else:
            n_fragmented += 1

    n_total = len(unique_ids)
    return {
        'n_contiguous': int(n_contiguous),
        'n_fragmented': int(n_fragmented),
        'n_total': int(n_total),
        'pct_contiguous': float(100.0 * n_contiguous / n_total) if n_total > 0 else 0.0,
        'mean_components': float(np.mean(component_counts)) if component_counts else 0.0
    }


# =============================================================================
# 10. DICE COEFFICIENT — ADJACENCY APPROACH (no parcel correspondence needed)
# =============================================================================

def calculate_dice_adjacency(labels1, labels2):
    """
    Compute Dice overlap between two parcellations.

    Uses the adjacency matrix approach so no parcel label correspondence is
    required. This is fair for comparing methods like SLIC whose parcels have
    no cross-subject IDs.

    Dice = 2 * |pairs in same parcel in BOTH P1 and P2|
               / (|same pairs in P1| + |same pairs in P2|)

    Args:
        labels1, labels2: 1D integer arrays, 0 = unlabeled

    Returns:
        float in [0, 1]
    """
    valid = (labels1 > 0) & (labels2 > 0)
    l1 = labels1[valid]
    l2 = labels2[valid]

    if len(l1) == 0:
        return 0.0

    def same_pairs(l):
        _, c = np.unique(l, return_counts=True)
        c = c.astype(np.int64)
        return float(np.sum(c * (c - 1)) / 2)

    sp1 = same_pairs(l1)
    sp2 = same_pairs(l2)

    if sp1 + sp2 == 0:
        return 0.0

    agreement = 0.0
    for pid in np.unique(l1):
        sub = l2[l1 == pid]
        _, c = np.unique(sub, return_counts=True)
        c = c.astype(np.int64)
        agreement += float(np.sum(c * (c - 1)) / 2)

    return float(2.0 * agreement / (sp1 + sp2))


# =============================================================================
# 11. PARCEL-AVERAGE DENSE TIMESERIES
# =============================================================================

def parcellate_dense(dense_data, labels_L, labels_R, n_L, n_R):
    """
    Average dense timeseries within parcels.

    Args:
        dense_data: (T, ~91k) full dense timeseries
        labels_L:   (n_L,) LH labels in valid-vertex space (>0)
        labels_R:   (n_R,) RH labels in valid-vertex space (>0, unique from LH)
        n_L, n_R:   number of valid vertices per hemisphere

    Returns:
        parcel_ts  : (T, n_parcels) float32
        parcel_ids : sorted array of parcel IDs
    """
    data_L = dense_data[:, :n_L]
    data_R = dense_data[:, n_L:n_L + n_R]
    combined_data = np.concatenate([data_L, data_R], axis=1)
    combined_labels = np.concatenate([labels_L, labels_R])

    unique_ids = np.unique(combined_labels)
    unique_ids = unique_ids[unique_ids > 0]

    T = dense_data.shape[0]
    parcel_ts = np.zeros((T, len(unique_ids)), dtype=np.float32)

    for i, pid in enumerate(unique_ids):
        mask = (combined_labels == pid)
        if np.any(mask):
            parcel_ts[:, i] = np.mean(combined_data[:, mask], axis=1)

    return parcel_ts, unique_ids


# =============================================================================
# 12. TEST-RETEST RELIABILITY
# =============================================================================

def calculate_fc_testretest(labels_L, labels_R, n_L, n_R, subj_id):
    """
    Compute test-retest reliability as Pearson r between REST1 and REST2 FC matrices.

    Applies the given parcellation labels to both REST1 and REST2 data,
    computes 200×200 FC matrices, then returns the correlation between
    their upper-triangular elements. This measures how stable the FC
    patterns are across scanning sessions.

    Args:
        labels_L, labels_R: parcellation labels in valid-vertex space (>0)
        n_L, n_R:           number of valid vertices per hemisphere
        subj_id:            subject ID string

    Returns:
        float (Pearson r), or None if REST data is unavailable
    """
    p1 = config.SHARED_DATA_DIR / "Aggregated" / f"{subj_id}_REST1_Dense.npy"
    p2 = config.SHARED_DATA_DIR / "Aggregated" / f"{subj_id}_REST2_Dense.npy"

    if not (p1.exists() and p2.exists()):
        return None

    d1 = np.load(p1)
    d2 = np.load(p2)

    ts1, _ = parcellate_dense(d1, labels_L, labels_R, n_L, n_R)
    ts2, _ = parcellate_dense(d2, labels_L, labels_R, n_L, n_R)

    fc1 = np.corrcoef(ts1.T)
    fc2 = np.corrcoef(ts2.T)

    idx = np.triu_indices_from(fc1, k=1)
    r = float(np.corrcoef(fc1[idx], fc2[idx])[0, 1])
    return r


# =============================================================================
# 13. INTER-SUBJECT DICE (group level)
# =============================================================================

def calculate_intersubject_dice(all_labels_list):
    """
    Compute mean pairwise Dice coefficient across all subject pairs.

    Args:
        all_labels_list: list of 1D label arrays (one per subject),
                         all in the same valid-vertex space

    Returns:
        mean_dice (float), std_dice (float), all_dice (list of floats)
    """
    n = len(all_labels_list)
    if n < 2:
        return float('nan'), float('nan'), []

    all_dice = []
    for i in range(n):
        for j in range(i + 1, n):
            d = calculate_dice_adjacency(all_labels_list[i], all_labels_list[j])
            all_dice.append(d)

    return float(np.mean(all_dice)), float(np.std(all_dice)), all_dice