# Individualized Brain Parcellation

Code for a study comparing six cortical parcellation methods along an
individualization gradient and evaluating what each of them supports in a
brain-behavior analysis of general intelligence (*g*) in the Human Connectome
Project Young Adult (HCP-YA) sample.

The six methods are:

| Method | Script prefix | Individualization |
| --- | --- | --- |
| Schaefer 200 (group atlas) | `schaefer_` | none, identical map for everyone |
| Gradient-infused multi-session hierarchical Bayesian model (gMSHBM) | `gmshbm_` | probabilistic group prior |
| Atlas-guided parcellation (AGP) | `agp_` | seeds taken from the group atlas |
| Surface SLIC, spatially constrained (SLIC-F) | `slic_` | group-level spatial term |
| Surface SLIC, unconstrained (SLIC-C) | `slic_` | none |
| Gradient-based parcellation | `gordon_` | none |

The gradient-based method follows Gordon et al. (2016), which is why its
scripts and its output folder carry the `gordon` name. The manuscript calls it
the gradient-based method throughout.

## What is in this repository

Code only. No neuroimaging or behavioral data is included: the HCP data are
governed by a data use agreement and cannot be redistributed (see *Data
availability*). All scripts live in a single `Scripts/` folder, because they
import one another (`config.py`, `utils.py`, `metrics.py`) and expect to be run
from that folder.

## Folder layout

The repository root doubles as the analysis root. Paths are resolved relative
to `Scripts/`, so once you add `Inputs/`, everything else falls into place:

```
Individualized_Brain_Parcellation/
├── Scripts/       included here
├── Inputs/        you create and populate this
└── Outputs/       created by the scripts, one subfolder per pipeline
```

## Required inputs

| File | Place in | Source |
| --- | --- | --- |
| `rfMRI_REST{1,2}_{LR,RL}_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii`, `Movement_Regressors.txt`, MSMAll midthickness surfaces | the HCP release itself, pointed at by `IP_HCP_ROOT` | HCP, data use agreement required |
| `HCP_all_data.csv` (behavioral and cognitive scores) | `Inputs/` | HCP, data use agreement required |
| `Schaefer2018_200Parcels_17Networks_order.dlabel.nii` | `Inputs/Atlases/` | public (Schaefer et al., 2018) |
| `HCP_1029sub_200Parcels_Kong2022_gMSHBM.mat`, `HCP_subject_list.txt` | `Inputs/` | public (Kong et al., 2021; CBIG) |
| `L.inflated.32k_fs_LR.surf.gii`, `R.inflated.32k_fs_LR.surf.gii` | `Inputs/Standard_Mesh/` | public (HCP fs_LR 32k standard mesh) |

`check_subjects.py` writes `Inputs/subjects_final.txt`, the list of
participants with all four resting-state runs. `build_adjacency.py` and
`agp_find_seeds.py` write the adjacency graphs and seed files that the
parcellation scripts read back from `Inputs/`.

## Environment variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `IP_VARIANT` | preprocessing pipeline, see below | `none` |
| `IP_HCP_ROOT` | root of the HCP release | the path used on our cluster |
| `IP_BASE` | analysis root, used by the R scripts | the folder above `Scripts/` |

The four preprocessing pipelines of the manuscript correspond to these variant
names, and each writes to its own folder under `Outputs/`:

| Manuscript | `IP_VARIANT` | Processing |
| --- | --- | --- |
| Pipeline A | `none` | minimally preprocessed data, no additional cleaning |
| Pipeline B | `thesis` | detrending, 12 motion parameters, 0.01–0.10 Hz bandpass |
| Pipeline C | `none_gsr` | pipeline A plus global signal regression |
| Pipeline D | `thesis_gsr` | pipeline B plus global signal regression |

## Run order

1. **Participants and surface graphs**
   `check_subjects.py` → `build_adjacency.py` → `slic_geodesic.py`
2. **Preprocessing**
   `step1_clean_data.py` (one participant at a time)
3. **Parcellation**, one method at a time
   `schaefer_parcellation.py`, `gmshbm_parcellation.py`,
   `agp_find_seeds.py` → `agp_parcellation.py`,
   `slic_calibrate_m.py` → `slic_parcellation.py`,
   `gordon_pipeline.py` (which runs `gordon_similarity.py`,
   `gordon_gradient.py`, `gordon_watershed.py` and `gordon_parcels.py` in turn)
4. **Validation**
   `*_validation.py` per method → `intersubject_dice.py` →
   `aggregate_validation.py`
5. **Fragment analysis**
   `variant_analysis.py` → `aggregate_variants.py`
6. **Features for the brain-behavior models**
   `feature_extraction.py` → `aggregate_features.py` → `build_kernels.py`
7. **Statistics**
   `stage1_stats.R` (method comparisons),
   `stage2_bifactor.R` → `stage2_sem.R` → `stage2_unique_variance.R` →
   `stage2_krr.R`
8. **Tables and figures**
   `make_tables.py`, then the plotting scripts below

Steps 2 to 6 run one participant at a time. We ran them as Slurm array jobs on
our university cluster, one array task per participant, with the preprocessing
variant passed in through `IP_VARIANT`. Those job files are not included here,
because they are tied to the queues, modules and paths of that one cluster;
every script in this repository can be called directly instead, for example
`python3 gordon_pipeline.py --subject 100206`. Repeat steps 2 to 7 with each
value of `IP_VARIANT` to reproduce the robustness analyses.

## Which script makes which figure

| Output | Script |
| --- | --- |
| Figure 1, parcellations of three participants | `make_figure1.py` |
| Figure 2, validation metrics | `plot_stage1.py` |
| Figure 3, fragment distances | `plot_fragment_distance.py` |
| Figure 4, graph metrics and parcel size | `plot_graph_metrics.py` |
| Figure 5, brain-behavior results | `plot_stage2.py` |
| Result tables (`table1_validation.csv` to `table4_variants.csv`, assembled into the manuscript tables) | `make_tables.py` |
| Supplementary fragment figure | `plot_variants.py` |

## Helper scripts

`export_cifti.py` writes parcellations as CIFTI dlabel files for viewing in
Connectome Workbench. `plot_parcellation.py` renders a single parcellation on
the inflated surface. `test_cleaning_effect.py` quantifies what each cleaning
step does to functional homogeneity in one participant.

## Dependencies

Python 3.9 with the packages in `requirements.txt`, and Connectome Workbench
1.5.0 for the spatial gradient and geodesic smoothing steps of the
gradient-based method.

R 4.3.1 with `lavaan` for the bifactor measurement model, the structural
equation models and the kernel ridge regression, and `afex`, `dplyr` and
`tidyr` for the method comparisons in `stage1_stats.R`.

## Data availability

The HCP-YA resting-state and behavioral data are governed by the Human
Connectome Project data use agreement and cannot be redistributed here. Access
can be requested through ConnectomeDB. The Schaefer atlas, the gMSHBM group
priors and the fs_LR standard-mesh surfaces are publicly available from the
sources listed above.

## License

MIT, see `LICENSE`.
