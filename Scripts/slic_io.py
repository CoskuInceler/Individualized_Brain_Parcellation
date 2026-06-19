# slic_io.py
"""
Input/Output Functions for Surface-based SLIC Parcellation

This script handles loading data and saving results, adapted to work with
your thesis folder structure and standardized pipeline.

Key differences from supervisor's version:
------------------------------------------
1. Valid vertices: Extracted from CIFTI header (not separate mask files)
   - Consistent with your agp_step1_build_graphs.py approach
   - No extra files needed

2. Time series: Loaded from pre-cleaned .npy files
   - Your step2_clean_data.py already cleaned the data
   - Your step4_consolidate.py already aggregated runs
   - No need to re-process CIFTI files

3. Saving: Uses vertex index arrays (not boolean masks)
   - Matches how your pipeline tracks valid vertices
"""

import numpy as np
import nibabel as nib
import os
from utils import get_valid_vertices  # Re-exported for backward compatibility

# =============================================================================
# FUNCTION 2: LOAD TIME SERIES
# =============================================================================

def load_timeseries(npy_path, n_vertices_L, n_vertices_R, hemisphere):
    """
    Load pre-cleaned time series data for one hemisphere.
    
    Your pipeline (step2_clean_data.py + step4_consolidate.py) already:
    1. Removed motion artifacts and nuisance signals
    2. Applied bandpass filtering
    3. Concatenated runs (LR + RL) into sessions
    4. Saved as .npy files
    
    This function loads that data and extracts one hemisphere.
    
    Data organization:
    ------------------
    The .npy files contain all brain vertices in CIFTI order:
    [Left cortex | Right cortex | Subcortical structures]
    
    We extract just the cortical hemisphere we need.
    
    Args:
        npy_path (str or Path): Path to the aggregated .npy file.
                                e.g., "100307_REST1_Dense.npy"
        
        n_vertices_L (int): Number of valid LEFT hemisphere vertices.
                            Get this from get_valid_vertices().
        
        n_vertices_R (int): Number of valid RIGHT hemisphere vertices.
                            Get this from get_valid_vertices().
        
        hemisphere (str): Which hemisphere to extract: 'L' or 'R'
    
    Returns:
        ts_data (np.ndarray): Time series with shape (N_vertices, N_timepoints).
                              Already in the correct orientation for SLIC.
    
    Example:
        >>> ts_L = load_timeseries(
        ...     "100307_ALL_Dense.npy",
        ...     n_vertices_L=29696,
        ...     n_vertices_R=29716,
        ...     hemisphere='L'
        ... )
        >>> print(ts_L.shape)  # (29696, 4800)
    """
    # Convert to string in case Path object is passed
    npy_path = str(npy_path)
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Data file not found: {npy_path}")
    
    print(f"[IO] Loading time series: {os.path.basename(npy_path)}")
    
    # Load the full data array
    # Shape: (N_timepoints, N_all_vertices) where N_all_vertices â‰ˆ 91282
    full_data = np.load(npy_path)
    print(f"[IO]   Full data shape: {full_data.shape} (Timepoints x All_Vertices)")
    
    # Extract the requested hemisphere
    # CIFTI order: [Left cortex (0 to n_L) | Right cortex (n_L to n_L+n_R) | Subcortical]
    if hemisphere == 'L':
        ts_data = full_data[:, :n_vertices_L]
    elif hemisphere == 'R':
        ts_data = full_data[:, n_vertices_L : n_vertices_L + n_vertices_R]
    else:
        raise ValueError(f"Invalid hemisphere: '{hemisphere}'. Must be 'L' or 'R'.")
    
    print(f"[IO]   Extracted {hemisphere} hemisphere: {ts_data.shape}")
    
    # Transpose to (Vertices, Timepoints) as required by SLIC algorithm
    ts_data = ts_data.T
    print(f"[IO]   Transposed shape: {ts_data.shape} (Vertices x Timepoints)")
    
    return ts_data


# =============================================================================
# FUNCTION 3: NORMALIZE TIME SERIES
# =============================================================================

def normalize_timeseries(ts_data):
    """
    Z-score normalize time series data (per vertex).
    
    After normalization, each vertex's time series has:
    - Mean = 0
    - Standard deviation = 1
    
    Why normalize?
    --------------
    The SLIC algorithm computes functional distance as the L2 norm (sum of
    squared differences) between time series. Without normalization:
    - Vertices with larger signal amplitude would dominate
    - The compactness parameter (m) would be hard to interpret
    
    With normalization, the functional distance becomes equivalent to
    (a scaled version of) correlation distance.
    
    Args:
        ts_data (np.ndarray): Time series with shape (N_vertices, N_timepoints).
    
    Returns:
        ts_norm (np.ndarray): Normalized time series (same shape), as float32.
    
    Example:
        >>> ts_raw = load_timeseries(...)
        >>> ts_norm = normalize_timeseries(ts_raw)
        >>> print(np.mean(ts_norm[0, :]))  # ~0.0
        >>> print(np.std(ts_norm[0, :]))   # ~1.0
    """
    print(f"[IO] Normalizing time series...")
    print(f"[IO]   Input shape: {ts_data.shape}")
    print(f"[IO]   Input range: [{np.min(ts_data):.3f}, {np.max(ts_data):.3f}]")
    
    # Calculate mean and std for each vertex (along time axis)
    mean = np.mean(ts_data, axis=1, keepdims=True)
    std = np.std(ts_data, axis=1, keepdims=True)
    
    # Check for vertices with zero variance (constant signal)
    # This shouldn't happen in properly cleaned data, but let's be safe
    zero_var_mask = (std == 0).flatten()
    n_zero_var = np.sum(zero_var_mask)
    
    if n_zero_var > 0:
        print(f"[IO]   WARNING: {n_zero_var} vertices have zero variance!")
        print(f"[IO]   These will be set to 0 after normalization.")
        # Replace zero std with 1 to avoid division by zero
        std[std == 0] = 1.0
    
    # Apply z-score normalization
    ts_norm = ((ts_data - mean) / std).astype(np.float32)
    
    print(f"[IO]   Output range: [{np.min(ts_norm):.3f}, {np.max(ts_norm):.3f}]")
    
    return ts_norm


# =============================================================================
# FUNCTION 4: SAVE PARCELLATION
# =============================================================================

def save_parcellation(labels, valid_indices, output_path, n_mesh_vertices=32492):
    """
    Save parcellation labels as a GIFTI file (.label.gii).
    
    The labels array contains values only for valid vertices. This function
    "expands" the labels back to the full 32k mesh by:
    1. Creating a full array of zeros (medial wall = 0)
    2. Filling in the valid vertices with their parcel labels
    3. Saving as GIFTI with label intent
    
    Label numbering:
    ----------------
    - Input labels: 0 to K-1 (from SLIC algorithm)
    - Output labels: 1 to K (shifted by +1)
    - Medial wall: 0
    
    We shift by +1 so that parcel "0" becomes "1", keeping 0 reserved for
    medial wall / unlabeled vertices. This is standard convention.
    
    Args:
        labels (np.ndarray): Parcel assignments for valid vertices only.
                             Shape: (N_valid_vertices,), values 0 to K-1.
        
        valid_indices (np.ndarray): Mesh vertex indices for valid vertices.
                                    Shape: (N_valid_vertices,).
        
        output_path (str or Path): Where to save the .label.gii file.
        
        n_mesh_vertices (int): Total vertices in the mesh (32492 for HCP).
    
    Example:
        >>> save_parcellation(
        ...     labels=cluster_labels,        # Shape: (29696,), values 0-99
        ...     valid_indices=valid_L,        # Shape: (29696,)
        ...     output_path="100307_L_SLIC_F_K100.label.gii"
        ... )
    """
    # Convert to string in case Path object is passed
    output_path = str(output_path)
    
    print(f"[IO] Saving parcellation: {os.path.basename(output_path)}")
    
    # Validate inputs
    if len(labels) != len(valid_indices):
        raise ValueError(
            f"Length mismatch! labels has {len(labels)} elements, "
            f"but valid_indices has {len(valid_indices)} elements."
        )
    
    # Create full mesh array (all zeros = medial wall)
    full_labels = np.zeros(n_mesh_vertices, dtype=np.float32)
    
    # Fill in valid vertices with their labels (+1 to shift from 0-indexed)
    # So parcel 0 becomes 1, parcel 1 becomes 2, etc.
    full_labels[valid_indices] = labels.astype(np.float32) + 1
    
    # Report statistics
    n_parcels = len(np.unique(labels))
    print(f"[IO]   Valid vertices: {len(valid_indices)}")
    print(f"[IO]   Number of parcels: {n_parcels}")
    print(f"[IO]   Label range in file: {np.min(full_labels):.0f} to {np.max(full_labels):.0f}")
    
    # Create GIFTI data array with label intent
    darray = nib.gifti.GiftiDataArray(
        data=full_labels,
        intent='NIFTI_INTENT_LABEL'  # Tells viewers this is a parcellation
    )
    
    # Create GIFTI image and save
    img = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(img, output_path)
    
    print(f"[IO]   Saved successfully!")