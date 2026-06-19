import os
from pathlib import Path

# =============================================================================
# 1. PROJECT ROOT
# =============================================================================
# This finds the parent of the "Scripts" folder (i.e., "Analysis")
# Result: C:\THESIS_MAIN\Analysis
BASE_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# 2. INPUT PATHS (READ-ONLY)
# =============================================================================
INPUTS_DIR = BASE_DIR / "Inputs"
RAW_DATA_DIR = INPUTS_DIR / "Raw_Data"
ATLAS_DIR = INPUTS_DIR / "Atlases" / "Schaefer2018"
GROUP_PRIORS_DIR = INPUTS_DIR / "Group_Priors"

# The specific Atlas File
SCHAEFER_200_FILE = ATLAS_DIR / "Schaefer2018_200Parcels_17Networks_order.dlabel.nii"

MESH_DIR = INPUTS_DIR / "Standard_Mesh"
MESH_FILE_L = MESH_DIR / "L.inflated.32k_fs_LR.surf.gii"
MESH_FILE_R = MESH_DIR / "R.inflated.32k_fs_LR.surf.gii"

# =============================================================================
# 3. OUTPUT PATHS (WRITE)
# =============================================================================
OUTPUTS_DIR = BASE_DIR / "Outputs"

# Method 0: Schaefer Group Atlas
METHOD_0_DIR = OUTPUTS_DIR / "Method_0_Schaefer"

# Method 1: MS-HBM Parcellation
METHOD_1_DIR = OUTPUTS_DIR / "Method_1_MSHBM"

# Method 2: Atlas-Guided Parcellation
METHOD_2_DIR = OUTPUTS_DIR / "Method_2_AGP"

# Method 3: SLIC with Full spatial+functional constraint
METHOD_3_DIR = OUTPUTS_DIR / "Method_3_SLIC_F"

# Method 4: SLIC with Cut spatial constraint (functional only)
METHOD_4_DIR = OUTPUTS_DIR / "Method_4_SLIC_C"

# Shared cleaned data (used by all methods)
SHARED_DATA_DIR = OUTPUTS_DIR / "Shared_Cleaned"

# Create all output directories if they don't exist
os.makedirs(METHOD_0_DIR, exist_ok=True)
os.makedirs(METHOD_1_DIR, exist_ok=True)
os.makedirs(METHOD_2_DIR, exist_ok=True)
os.makedirs(METHOD_3_DIR, exist_ok=True)
os.makedirs(METHOD_4_DIR, exist_ok=True)
os.makedirs(SHARED_DATA_DIR, exist_ok=True)

# =============================================================================
# 4. EXPERIMENT CONSTANTS
# =============================================================================
SUBJECT_IDS = ["100307", "100408", "100610"]

# We list all 4 runs found in your tree.
# We can comment out REST2 later if you only want to process REST1.
RUN_IDS = [
    "rfMRI_REST1_LR",
    "rfMRI_REST1_RL",
    "rfMRI_REST2_LR",
    "rfMRI_REST2_RL"
]

# =============================================================================
# 5. PATH FINDER FUNCTIONS (The Bridge)
# =============================================================================
def get_brain_path(subject_id, run_id):
    """
    Returns the Path object for the .dtseries.nii file.
    Logic: Raw_Data / Subject / Direction / Filename
    """
    # Extract "LR" or "RL" from "rfMRI_REST1_LR"
    direction = run_id.split("_")[-1] 
    
    filename = f"{run_id}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    return RAW_DATA_DIR / subject_id / direction / filename

def get_confound_path(subject_id, run_id, confound_name="Movement_Regressors.txt"):
    """
    Returns the Path object for a specific confound text file.
    Logic: Raw_Data / Regressors / Subject / Run_Name / File
    """
    return RAW_DATA_DIR / "Regressors" / subject_id / run_id / confound_name

# =============================================================================
# END OF CONFIG
# =============================================================================