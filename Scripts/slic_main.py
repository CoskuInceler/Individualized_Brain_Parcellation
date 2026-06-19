# slic_main.py
"""
Main Script for SLIC Parcellation (Methods 3 & 4)

This script orchestrates the full SLIC parcellation pipeline:
    1. Load valid vertex information from CIFTI header
    2. Compute geodesic distance matrices (cached for reuse)
    3. For each subject:
       a. Load pre-cleaned time series data
       b. Run SLIC-F (Method 3): Spatial + Functional
       c. Run SLIC-C (Method 4): Functional only
       d. Save results

Output Structure:
-----------------
    Outputs/Method_3_SLIC_F/
        ├── {SubjectID}_Labels_L.npy
        ├── {SubjectID}_Labels_R.npy
        └── Geodesic_L.npy, Geodesic_R.npy (cached)
    
    Outputs/Method_4_SLIC_C/
        ├── {SubjectID}_Labels_L.npy
        └── {SubjectID}_Labels_R.npy

Usage:
------
    Full run (all subjects, both methods):
        python slic_main.py
    
    Test run (one subject, one hemisphere, SLIC-F only):
        python slic_main.py --test

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import time
import numpy as np
import nibabel as nib

# Import configuration
import config

# Import SLIC modules
from slic_algorithm import SurfaceSLIC
from slic_geometry import compute_geodesic_distances
from slic_io import (
    get_valid_vertices,
    load_timeseries,
    normalize_timeseries,
    save_parcellation
)


# =============================================================================
# SLIC PARAMETERS
# =============================================================================
# These match the Wang et al. (2016) paper recommendations

N_CLUSTERS = 100        # Parcels per hemisphere (200 total brain, matching Schaefer)
COMPACTNESS_M = 95.0    # Balance between spatial and functional distance
                        # Paper recommends "around the median of all functional distances"
                        # m=40 was for raw volumetric data; m=95 is calibrated for
                        # z-scored HCP surface data (~4800 timepoints)
MAX_ITERATIONS = 20     # Maximum iterations (usually converges before this)
RANDOM_SEED = 42        # For reproducibility

# Data session to use
# Options: "REST1" (faster, ~2400 timepoints) or "ALL" (maximum data, ~4800 timepoints)
DATA_SESSION = "ALL"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_valid_surface_area(surf_path, valid_indices):
    """
    Compute total surface area (in mm²) for valid cortical vertices.
    
    Only counts triangles where ALL three vertices are valid (non-medial-wall).
    This gives S = sqrt(Area/K) in mm — matching geodesic distance units.
    
    Args:
        surf_path: Path to .surf.gii mesh file
        valid_indices: Array of valid vertex indices
    
    Returns:
        area: Surface area in mm²
    """
    mesh = nib.load(str(surf_path))
    vertices = mesh.darrays[0].data
    faces = mesh.darrays[1].data
    
    valid_set = set(valid_indices.tolist())
    
    total_area = 0.0
    for i in range(len(faces)):
        if faces[i, 0] in valid_set and faces[i, 1] in valid_set and faces[i, 2] in valid_set:
            v0 = vertices[faces[i, 0]]
            v1 = vertices[faces[i, 1]]
            v2 = vertices[faces[i, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            total_area += 0.5 * np.linalg.norm(cross)
    
    return total_area

def get_or_compute_geodesic(mesh_path, valid_indices, cache_path):
    """
    Get geodesic distance matrix, computing only if not cached.
    
    The geodesic matrix is computed on the FULL mesh (32k vertices), then
    masked to keep only valid vertices (~29k). This is correct because
    shortest paths might traverse medial wall vertices.
    
    Args:
        mesh_path: Path to .surf.gii mesh file
        valid_indices: Array of valid vertex indices
        cache_path: Path to save/load the MASKED geodesic matrix
    
    Returns:
        Masked geodesic matrix (n_valid x n_valid)
    """
    # Check if we have a cached MASKED matrix
    if os.path.exists(cache_path):
        print(f"[Geodesic] Loading cached matrix: {cache_path.name}")
        geo_masked = np.load(cache_path)
        print(f"[Geodesic]   Shape: {geo_masked.shape}")
        print(f"[Geodesic]   Range: [{np.min(geo_masked):.2f}, {np.max(geo_masked):.2f}] mm")
        return geo_masked
    
    # Need to compute from scratch
    print(f"[Geodesic] Computing geodesic distances (this takes ~5-10 minutes)...")
    
    # Compute full 32k x 32k matrix
    # We use a temporary cache for the full matrix
    full_cache_path = cache_path.parent / f"Geodesic_FULL_{cache_path.stem.split('_')[-1]}.npy"
    geo_full = compute_geodesic_distances(mesh_path, full_cache_path)
    
    # Mask to valid vertices only
    print(f"[Geodesic] Masking to valid vertices ({len(valid_indices)} x {len(valid_indices)})...")
    geo_masked = geo_full[np.ix_(valid_indices, valid_indices)]
    
    # Validate
    if np.any(np.isnan(geo_masked)) or np.any(np.isinf(geo_masked)):
        raise ValueError("Masked geodesic matrix contains invalid values!")
    
    # Save masked matrix for future use
    print(f"[Geodesic] Saving masked matrix: {cache_path.name}")
    np.save(cache_path, geo_masked.astype(np.float32))
    
    # Clean up full matrix from memory
    del geo_full
    
    print(f"[Geodesic]   Shape: {geo_masked.shape}")
    print(f"[Geodesic]   Range: [{np.min(geo_masked):.2f}, {np.max(geo_masked):.2f}] mm")
    
    return geo_masked.astype(np.float32)


def run_slic_for_subject(subject_id, hemisphere, ts_data, geo_matrix, mode, output_dir, valid_indices, surface_area=None):
    """
    Run SLIC parcellation for one subject/hemisphere/mode combination.
    
    Args:
        subject_id: e.g., "100307"
        hemisphere: 'L' or 'R'
        ts_data: Normalized time series (N_vertices x N_timepoints)
        geo_matrix: Geodesic distance matrix (only used for SLIC-F)
        mode: 'spatial_functional' (SLIC-F) or 'functional_only' (SLIC-C)
        output_dir: Where to save results
        valid_indices: Valid vertex indices (for saving GIFTI)
        surface_area: Cortical surface area in mm² (required for SLIC-F)
    
    Returns:
        labels: Cluster assignments (N_vertices,)
    """
    method_name = "SLIC-F" if mode == 'spatial_functional' else "SLIC-C"
    
    print(f"\n[{method_name}] Running for {subject_id} {hemisphere}...")
    print(f"[{method_name}]   Vertices: {ts_data.shape[0]}, Timepoints: {ts_data.shape[1]}")
    
    # Create SLIC instance
    slic = SurfaceSLIC(
        n_clusters=N_CLUSTERS,
        m=COMPACTNESS_M,
        max_iter=MAX_ITERATIONS,
        mode=mode,
        random_seed=RANDOM_SEED
    )
    
    # Run clustering
    start_time = time.time()
    
    if mode == 'spatial_functional':
        labels = slic.fit(ts_data, geo_matrix, surface_area=surface_area)
    else:
        labels = slic.fit(ts_data, geo_matrix=None)
    
    elapsed = time.time() - start_time
    
    # Report results
    n_unique = len(np.unique(labels))
    print(f"[{method_name}]   Completed in {elapsed:.1f} seconds")
    print(f"[{method_name}]   Parcels created: {n_unique} (target: {N_CLUSTERS})")
    
    if n_unique != N_CLUSTERS:
        print(f"[{method_name}]   WARNING: Expected {N_CLUSTERS} parcels!")
    
    # Save as .npy (matching AGP convention)
    npy_path = output_dir / f"{subject_id}_Labels_{hemisphere}.npy"
    np.save(npy_path, labels)
    print(f"[{method_name}]   Saved: {npy_path.name}")
    
    # Also save as .label.gii for visualization
    gii_path = output_dir / f"{subject_id}_Labels_{hemisphere}.label.gii"
    save_parcellation(labels, valid_indices, gii_path)
    
    return labels


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main execution: Process all subjects with both SLIC methods.
    """
    print("=" * 70)
    print("SLIC PARCELLATION - Methods 3 (SLIC-F) & 4 (SLIC-C)")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Clusters per hemisphere: {N_CLUSTERS}")
    print(f"  Compactness (m):         {COMPACTNESS_M}")
    print(f"  Max iterations:          {MAX_ITERATIONS}")
    print(f"  Random seed:             {RANDOM_SEED}")
    print(f"  Data session:            {DATA_SESSION}")
    print(f"\nSubjects: {config.SUBJECT_IDS}")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: GET VALID VERTICES
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DETERMINING VALID VERTICES")
    print("=" * 70)
    
    # Use first subject's CIFTI to get valid vertex indices
    # (All HCP subjects share the same cortical mask)
    ref_cifti = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    
    valid_L = get_valid_vertices(ref_cifti, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_cifti, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    
    n_valid_L = len(valid_L)
    n_valid_R = len(valid_R)
    
    print(f"\nValid vertices:")
    print(f"  Left hemisphere:  {n_valid_L}")
    print(f"  Right hemisphere: {n_valid_R}")
    
    # =========================================================================
    # STEP 2: COMPUTE/LOAD GEODESIC DISTANCES (for SLIC-F only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: GEODESIC DISTANCE MATRICES")
    print("=" * 70)
    
    # Cache geodesic matrices in Method_3 folder (SLIC-F uses them)
    geo_cache_L = config.METHOD_3_DIR / "Geodesic_L.npy"
    geo_cache_R = config.METHOD_3_DIR / "Geodesic_R.npy"
    
    print("\n--- Left Hemisphere ---")
    geo_L = get_or_compute_geodesic(config.MESH_FILE_L, valid_L, geo_cache_L)
    
    print("\n--- Right Hemisphere ---")
    geo_R = get_or_compute_geodesic(config.MESH_FILE_R, valid_R, geo_cache_R)
    
    # Store in dict for easy access
    geo_matrices = {'L': geo_L, 'R': geo_R}
    valid_indices = {'L': valid_L, 'R': valid_R}
    n_valid = {'L': n_valid_L, 'R': n_valid_R}
    
    # Compute surface areas for correct S parameter (in mm²)
    print("\n--- Computing Surface Areas ---")
    area_L = compute_valid_surface_area(config.MESH_FILE_L, valid_L)
    area_R = compute_valid_surface_area(config.MESH_FILE_R, valid_R)
    surface_areas = {'L': area_L, 'R': area_R}
    print(f"  Left:  {area_L:.0f} mm²  →  S = {np.sqrt(area_L / N_CLUSTERS):.2f} mm")
    print(f"  Right: {area_R:.0f} mm²  →  S = {np.sqrt(area_R / N_CLUSTERS):.2f} mm")
    
    # =========================================================================
    # STEP 3: PROCESS ALL SUBJECTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: PROCESSING SUBJECTS")
    print("=" * 70)
    
    # Track results
    results = []
    
    for subj_id in config.SUBJECT_IDS:
        print(f"\n{'=' * 70}")
        print(f"SUBJECT: {subj_id}")
        print(f"{'=' * 70}")
        
        # Load time series data
        data_path = config.SHARED_DATA_DIR / "Aggregated" / f"{subj_id}_{DATA_SESSION}_Dense.npy"
        
        if not data_path.exists():
            print(f"[ERROR] Data file not found: {data_path}")
            results.append((subj_id, "ALL", "FAILED - Data not found"))
            continue
        
        # Process each hemisphere
        for hemi in ['L', 'R']:
            print(f"\n--- {subj_id} | Hemisphere: {hemi} ---")
            
            # Load and normalize time series
            ts_raw = load_timeseries(
                data_path,
                n_valid_L,
                n_valid_R,
                hemi
            )
            ts_norm = normalize_timeseries(ts_raw)
            
            # -----------------------------------------------------------------
            # Run SLIC-F (Method 3): Spatial + Functional
            # -----------------------------------------------------------------
            try:
                run_slic_for_subject(
                    subject_id=subj_id,
                    hemisphere=hemi,
                    ts_data=ts_norm,
                    geo_matrix=geo_matrices[hemi],
                    mode='spatial_functional',
                    output_dir=config.METHOD_3_DIR,
                    valid_indices=valid_indices[hemi],
                    surface_area=surface_areas[hemi]
                )
                results.append((subj_id, hemi, "SLIC-F", "SUCCESS"))
            except Exception as e:
                print(f"[ERROR] SLIC-F failed: {e}")
                results.append((subj_id, hemi, "SLIC-F", f"FAILED - {e}"))
            
            # -----------------------------------------------------------------
            # Run SLIC-C (Method 4): Functional only
            # -----------------------------------------------------------------
            try:
                run_slic_for_subject(
                    subject_id=subj_id,
                    hemisphere=hemi,
                    ts_data=ts_norm,
                    geo_matrix=None,  # Not used for SLIC-C
                    mode='functional_only',
                    output_dir=config.METHOD_4_DIR,
                    valid_indices=valid_indices[hemi]
                )
                results.append((subj_id, hemi, "SLIC-C", "SUCCESS"))
            except Exception as e:
                print(f"[ERROR] SLIC-C failed: {e}")
                results.append((subj_id, hemi, "SLIC-C", f"FAILED - {e}"))
    
    # =========================================================================
    # STEP 4: SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_success = sum(1 for r in results if "SUCCESS" in r[-1])
    n_total = len(results)
    
    print(f"\nResults: {n_success}/{n_total} successful\n")
    
    for subj, hemi, method, status in results:
        symbol = "✓" if "SUCCESS" in status else "✗"
        print(f"  {symbol} {subj} | {hemi} | {method}: {status}")
    
    print(f"\nOutput locations:")
    print(f"  Method 3 (SLIC-F): {config.METHOD_3_DIR}")
    print(f"  Method 4 (SLIC-C): {config.METHOD_4_DIR}")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test():
    """
    Quick test: Run only Subject 100307, Left hemisphere, SLIC-F only.
    
    Use this to verify the pipeline works before running the full batch.
    """
    print("=" * 70)
    print("SLIC TEST MODE")
    print("=" * 70)
    print("Running: Subject 100307 | Left hemisphere | SLIC-F only")
    print("=" * 70)
    
    # Get valid vertices
    ref_cifti = config.get_brain_path("100307", config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_cifti, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_cifti, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    
    n_valid_L = len(valid_L)
    n_valid_R = len(valid_R)
    
    print(f"\nValid vertices: L={n_valid_L}, R={n_valid_R}")
    
    # Compute geodesic matrix
    print("\n--- Computing Geodesic Distances ---")
    geo_cache_L = config.METHOD_3_DIR / "Geodesic_L.npy"
    geo_L = get_or_compute_geodesic(config.MESH_FILE_L, valid_L, geo_cache_L)
    
    # Compute surface area
    print("\n--- Computing Surface Area ---")
    area_L = compute_valid_surface_area(config.MESH_FILE_L, valid_L)
    print(f"  Left: {area_L:.0f} mm²  →  S = {np.sqrt(area_L / N_CLUSTERS):.2f} mm")
    
    # Load time series
    print("\n--- Loading Time Series ---")
    data_path = config.SHARED_DATA_DIR / "Aggregated" / f"100307_{DATA_SESSION}_Dense.npy"
    ts_raw = load_timeseries(data_path, n_valid_L, n_valid_R, 'L')
    ts_norm = normalize_timeseries(ts_raw)
    
    # Run SLIC-F
    print("\n--- Running SLIC-F ---")
    labels = run_slic_for_subject(
        subject_id="100307",
        hemisphere='L',
        ts_data=ts_norm,
        geo_matrix=geo_L,
        mode='spatial_functional',
        output_dir=config.METHOD_3_DIR,
        valid_indices=valid_L,
        surface_area=area_L
    )
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {config.METHOD_3_DIR / '100307_Labels_L.npy'}")
    print(f"  {config.METHOD_3_DIR / '100307_Labels_L.label.gii'}")
    
    return labels


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test()
    else:
        main()