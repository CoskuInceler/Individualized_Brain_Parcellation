# Individualized Brain Parcellation

Code for a Master's thesis comparing five cortical parcellation methods along an
individualization gradient — Schaefer 200 (group atlas), gMS-HBM, Atlas-Guided
Parcellation (AGP), and surface SLIC in two variants (SLIC-F, SLIC-C) — and
evaluating how each supports the prediction of general intelligence (*g*) from
resting-state functional connectivity in Human Connectome Project (HCP) subjects.

## What's in this repository

This repository contains **code only** — every analysis script, in a single
`Scripts/` folder. No neuroimaging or behavioral data is included, by design (see
*Data availability* below). The scripts read their inputs from, and write their
outputs to, folders you create alongside `Scripts/` (described next).

## Folder layout

The repository root acts as the analysis root. The Python scripts locate inputs
and outputs **relative** to the `Scripts/` folder, so once you add the `Inputs/`
and `Outputs/` folders at the root, the paths resolve on their own:

```
Individualized_Brain_Parcellation/      <- repository root = analysis root
├── Scripts/                            <- included in this repo
├── Inputs/                             <- you create and populate this
│   ├── Raw_Data/                       HCP resting-state runs + Regressors/
│   ├── Atlases/Schaefer2018/           Schaefer 200 atlas
│   ├── Group_Priors/                   gMS-HBM group priors + subject list
│   ├── Standard_Mesh/                  fs_LR 32k inflated surfaces
│   └── Behavior/                       HCP behavioral data
└── Outputs/                            <- created automatically by the scripts
```

## Required inputs: where to get them and where to put them

| Input | Place in | Source / access |
|-------|----------|-----------------|
| `rfMRI_REST{1,2}_{LR,RL}_Atlas_MSMAll_hp2000_clean.dtseries.nii` (per subject) | `Inputs/Raw_Data/{subject}/{LR,RL}/` | HCP — requires a Data Use Agreement |
| `Movement_Regressors.txt`, `{run}_WM.txt`, `{run}_CSF.txt` | `Inputs/Raw_Data/Regressors/{subject}/{run}/` | HCP — requires a Data Use Agreement |
| `HCP_all_data.csv` (behavioral / cognitive scores) | `Inputs/Behavior/` | HCP — requires a Data Use Agreement |
| `Schaefer2018_200Parcels_17Networks_order.dlabel.nii` | `Inputs/Atlases/Schaefer2018/` | Public (Schaefer et al., 2018 / CBIG) |
| `HCP_1029sub_200Parcels_Kong2022_gMSHBM.mat`, `HCP_subject_list.txt` | `Inputs/Group_Priors/` | Public (Kong et al., 2022 / CBIG) |
| `L.inflated.32k_fs_LR.surf.gii`, `R.inflated.32k_fs_LR.surf.gii` | `Inputs/Standard_Mesh/` | Public (HCP fs_LR 32k standard mesh) |

## Run order

1. **Preprocessing** — `step1_clean_data.py` (confound regression, bandpass,
   detrend, z-score; consolidates runs into sessions and an ALL combination).
2. **Parcellation** (one method at a time):
   - Schaefer: `schaefer_parcellation.py`
   - gMS-HBM: `gMSHBM_parcellation.py`
   - AGP: `agp_step1_build_graphs.py` → `agp_step2_find_seeds.py` → `agp_step3_region_growing.py` → `agp_parcellation.py`
   - SLIC-F / SLIC-C: `slic_main.py` → `slic_parcellation.py`
     (`slic_geometry.py`, `slic_algorithm.py`, `slic_io.py` are imported helpers)
3. **Validation** — `schaefer_validation.py`, `gMSHBM_validation.py`,
   `agp_validation.py`, `slic_validation.py`
   (per-method `*_visualization.py` scripts render the parcellations).
4. **Cross-method comparison** — `stage_1_comparison.py`.
5. **Feature extraction** — `Feature_4_FC.py` (run first) →
   `Feature_F1F2_Graph_Metrics.py` → `Feature_3_Parcel_Size.py`
   (`feature_validation.py` checks outputs; `Feature_Report.py` makes figures).
6. **Prediction (Stage 2, *g*-only)** —
   `Stage_2_Unleaked_Step1.R` → `Stage_2_Unleaked_Step2.R` →
   `Stage_2_Unleaked_Step2_B.R` → `Stage_2_Unleaked_Step2_C.R` →
   `Stage_2_Unleaked_Step3_gOnly.R` → `Stage_2_Unleaked_Step3_B_gOnly.R`.
7. **Inferential statistics** — `Inferential_Stats.R`.

## Dependencies

- **Python**: numpy, pandas, scipy, nibabel, nilearn, matplotlib, seaborn, h5py
- **R**: tidyverse, lavaan, semPlot, afex (plus dplyr, tidyr)

## Setup notes

- The R scripts use a hard-coded project path near the top
  (`BASE <- "C:/Thesis_Main/Analysis"`). Edit that one line to point at your
  local copy of this repository.
- `config.py` and `feature_config.py` ship with a 3-subject test list in
  `SUBJECT_IDS`. Replace it with your full subject list before a complete run.

## Data availability

The HCP resting-state and behavioral data used in this project are governed by the
Human Connectome Project Data Use Agreement and cannot be redistributed here.
Access can be requested through the HCP / ConnectomeDB. The Schaefer atlas, the
gMS-HBM group priors, and the fs_LR standard-mesh surfaces are publicly available
from their respective sources listed above.
