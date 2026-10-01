from pathlib import Path
import os

# Project root: the folder above the Scripts folder holding this file
BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = BASE_DIR / "Inputs"
SCRIPTS_DIR = BASE_DIR / "Scripts"
VARIANT = os.environ.get("IP_VARIANT", "none")
OUTPUTS_ROOT = BASE_DIR / "Outputs"
OUTPUTS_DIR = OUTPUTS_ROOT / VARIANT
LOGS_DIR = BASE_DIR / "Logs"
# --- Input files ---
ATLAS_DIR = INPUTS_DIR / "Atlases"
SCHAEFER_200_FILE = (
    ATLAS_DIR / "Schaefer2018_200Parcels_17Networks_order.dlabel.nii"
)
MESH_DIR = INPUTS_DIR / "Standard_Mesh"
MESH_FILE_L = MESH_DIR / "L.inflated.32k_fs_LR.surf.gii"
MESH_FILE_R = MESH_DIR / "R.inflated.32k_fs_LR.surf.gii"
GMSHBM_FILE = INPUTS_DIR / "HCP_1029sub_200Parcels_Kong2022_gMSHBM.mat"
SUBJECT_FILE = INPUTS_DIR / "HCP_subject_list.txt"
SUBJECT_FILE_FINAL = INPUTS_DIR / "subjects_final.txt"
# --- Participant list ---
SUBJECT_IDS = [
    s.strip()
    for s in SUBJECT_FILE_FINAL.read_text().splitlines()
    if s.strip() and not s.strip().startswith("#")
]
RUN_IDS = [
    "rfMRI_REST1_LR",
    "rfMRI_REST1_RL",
    "rfMRI_REST2_LR",
    "rfMRI_REST2_RL",
]
# --- HCP raw data (read-only, on the project share) ---
# Set IP_HCP_ROOT to point at your own copy of the HCP data.
HCP_ROOT = Path(os.environ.get("IP_HCP_ROOT", "/fs/s6k/project/hcpya25"))
HCP_REST = HCP_ROOT / "restingstate"


def get_brain_path(subject_id, run_id):
    """Path to the cleaned dtseries file of one run."""
    return (
        HCP_REST
        / subject_id
        / "MNINonLinear"
        / "Results"
        / run_id
        / f"{run_id}_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii"
    )


def get_confound_path(subject_id, run_id):
    """Path to the movement regressor file of one run."""
    return (
        HCP_REST
        / subject_id
        / "MNINonLinear"
        / "Results"
        / run_id
        / "Movement_Regressors.txt"
    )


# --- Subject-specific surfaces (MSMAll aligned, same space as the fMRI) ---
HCP_STRUCT = HCP_ROOT / "structural"


def get_surface_path(subject_id, hemi, kind="midthickness"):
    """
    Path to one subject's cortical surface.



    hemi : "L" or "R"
    kind : "midthickness", "inflated", "white", "pial", "sphere"



    Gradient and geodesic smoothing follow the cortical sheet, so the
    midthickness surface is the one that matters for those steps.
    """
    return (
        HCP_STRUCT
        / subject_id
        / "MNINonLinear"
        / "fsaverage_LR32k"
        / f"{subject_id}.{hemi}.{kind}_MSMAll.32k_fs_LR.surf.gii"
    )
