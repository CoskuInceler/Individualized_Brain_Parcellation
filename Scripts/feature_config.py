"""
FEATURE EXTRACTION — CONFIGURATION
====================================
Shared constants and paths for all feature extraction scripts.
Mirrors the structure of config.py from Stage 1 but adds
feature-specific output paths and parameters.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR  = BASE_DIR / "Inputs"
OUTPUTS_DIR = BASE_DIR / "Outputs"

# =============================================================================
# INPUT PATHS
# =============================================================================
ATLAS_DIR        = INPUTS_DIR / "Atlases" / "Schaefer2018"
SCHAEFER_FILE    = ATLAS_DIR  / "Schaefer2018_200Parcels_17Networks_order.dlabel.nii"
GROUP_PRIORS_DIR = INPUTS_DIR / "Group_Priors"
MESH_DIR         = INPUTS_DIR / "Standard_Mesh"
MESH_FILE_L      = MESH_DIR   / "L.inflated.32k_fs_LR.surf.gii"
MESH_FILE_R      = MESH_DIR   / "R.inflated.32k_fs_LR.surf.gii"
BEHAVIOR_DIR     = INPUTS_DIR / "Behavior"

# =============================================================================
# STAGE 1 OUTPUT PATHS (read-only inputs for feature extraction)
# =============================================================================
METHOD_0_DIR    = OUTPUTS_DIR / "Method_0_Schaefer"
METHOD_1_DIR    = OUTPUTS_DIR / "Method_1_MSHBM"
METHOD_2_DIR    = OUTPUTS_DIR / "Method_2_AGP"
METHOD_3_DIR    = OUTPUTS_DIR / "Method_3_SLIC_F"
METHOD_4_DIR    = OUTPUTS_DIR / "Method_4_SLIC_C"
SHARED_DATA_DIR = OUTPUTS_DIR / "Shared_Cleaned"

# =============================================================================
# FEATURE OUTPUT PATHS
# =============================================================================
FEATURES_DIR = OUTPUTS_DIR / "Features"

# One subdirectory per method
FEAT_M0 = FEATURES_DIR / "M0_Schaefer"
FEAT_M1 = FEATURES_DIR / "M1_MSHBM"
FEAT_M2 = FEATURES_DIR / "M2_AGP"
FEAT_M3 = FEATURES_DIR / "M3_SLIC_F"
FEAT_M4 = FEATURES_DIR / "M4_SLIC_C"

for d in [FEAT_M0, FEAT_M1, FEAT_M2, FEAT_M3, FEAT_M4]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# METHOD REGISTRY
# =============================================================================
# Each method entry defines where to find:
#   - timeseries CSV (for FC)
#   - full-mesh label .npy files (for alignment)
#   - col_format: how CSV columns are named
#   - needs_align: whether Hungarian alignment is required
#
# col_format:
#   "standard" → columns named "Parcel_1" ... "Parcel_200"
#                native_id = column number (1-200, globally unique)
#   "slic"     → columns named "L_Parcel_0"..."L_Parcel_99",
#                              "R_Parcel_0"..."R_Parcel_99"
#                native_id for LH = k+1 (1-100)
#                native_id for RH = k+101 (101-200)
#
# For Schaefer (M0): label IDs ARE atlas IDs — no alignment needed.
# For AGP (M2):      seeds come from Schaefer centroids so label IDs
#                    approximate atlas IDs, but we still run Hungarian
#                    to guarantee exact bijective correspondence.

METHODS = {
    "M0_Schaefer": {
        "feat_dir":     FEAT_M0,
        "method_dir":   METHOD_0_DIR,
        "needs_align":  False,
        "col_format":   "standard",
        "ts_pattern":   "Aggregated/{subj}_ALL_Schaefer200.csv",
        "lh_pattern":   None,   # atlas — no label files needed
        "rh_pattern":   None,
    },
    "M1_MSHBM": {
        "feat_dir":     FEAT_M1,
        "method_dir":   METHOD_1_DIR,
        "needs_align":  True,
        "col_format":   "standard",
        "ts_pattern":   "{subj}_gMSHBM_Timeseries.csv",
        "lh_pattern":   "{subj}_gMSHBM_LH.npy",
        "rh_pattern":   "{subj}_gMSHBM_RH.npy",
    },
    "M2_AGP": {
        "feat_dir":     FEAT_M2,
        "method_dir":   METHOD_2_DIR,
        "needs_align":  True,
        "col_format":   "standard",
        "ts_pattern":   "{subj}_AGP_Timeseries.csv",
        "lh_pattern":   "{subj}_AGP_LH.npy",
        "rh_pattern":   "{subj}_AGP_RH.npy",
    },
    "M3_SLIC_F": {
        "feat_dir":     FEAT_M3,
        "method_dir":   METHOD_3_DIR,
        "needs_align":  True,
        "col_format":   "slic",
        "ts_pattern":   "{subj}_SLIC_F_Timeseries.csv",
        "lh_pattern":   "{subj}_SLIC_F_LH.npy",
        "rh_pattern":   "{subj}_SLIC_F_RH.npy",
    },
    "M4_SLIC_C": {
        "feat_dir":     FEAT_M4,
        "method_dir":   METHOD_4_DIR,
        "needs_align":  True,
        "col_format":   "slic",
        "ts_pattern":   "{subj}_SLIC_C_Timeseries.csv",
        "lh_pattern":   "{subj}_SLIC_C_LH.npy",
        "rh_pattern":   "{subj}_SLIC_C_RH.npy",
    },
}

# =============================================================================
# GRAPH CONSTRUCTION PARAMETERS
# =============================================================================
# Three graph variants used for Features 1 & 2.
# All variants zero out negative FC edges before any thresholding.
#
# Variant definitions:
#   "raw"    : weighted graph, edge weight = FC value (negatives → 0)
#   "top10"  : binary graph, keep top 10% of positive edges
#   "r05"    : binary graph, keep edges where FC > 0.5

GRAPH_VARIANTS = ["raw", "top10", "r05"]
TOP_K_PERCENT  = 10     # for "top10" variant
R_THRESHOLD    = 0.5    # for "r05" variant

# =============================================================================
# PARCELLATION CONSTANTS
# =============================================================================
N_PARCELS  = 200
N_MESH     = 32492   # vertices per hemisphere on fs_LR32k mesh

# =============================================================================
# SUBJECT LIST
# =============================================================================
# On HPC this is overridden by --subject argument for array jobs.
# Default list here is for local testing.
SUBJECT_IDS = ["100307", "100408", "100610"]