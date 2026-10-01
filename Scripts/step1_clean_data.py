"""
STEP 1 - Preprocessing
======================
The HCP resting-state data used here comes from the HCP-YA 2025 Release
and has already undergone:
  hp2000  : 2000 s high-pass filter
  clean   : sICA + FIX denoising
  rclean  : Reclean pipeline, improved sICA component classification
  tclean  : temporal ICA, removal of global structured artifacts
Three preprocessing variants are supported, selected with the
IP_VARIANT environment variable:
  none        no extra cleaning, only z-scoring          (default)
  thesis      the four steps used in the master's thesis
  thesis_gsr  thesis steps plus global signal regression
Usage:
    python3 step1_clean_data.py --subject 100206
    IP_VARIANT=thesis python3 step1_clean_data.py --subject 100206
"""

import argparse
import time
import numpy as np
import nibabel as nib
import config

# --- Cleaning options, per preprocessing variant ---
VARIANTS = {
    "none": dict(detrend=False, motion=False, bandpass=False, gsr=False),
    "thesis": dict(detrend=True, motion=True, bandpass=True, gsr=False),
    "thesis_gsr": dict(detrend=True, motion=True, bandpass=True, gsr=True),
    "none_gsr": dict(detrend=False, motion=False, bandpass=False, gsr=True),
}
if config.VARIANT not in VARIANTS:
    raise ValueError(
        f"Unknown variant '{config.VARIANT}'. "
        f"Expected one of: {', '.join(VARIANTS)}"
    )
OPTS = VARIANTS[config.VARIANT]
DO_DETREND = OPTS["detrend"]
DO_MOTION_REG = OPTS["motion"]
DO_BANDPASS = OPTS["bandpass"]
DO_GSR = OPTS["gsr"]
DO_ZSCORE = True  # always: required for SLIC/AGP distance computations
TR = 0.72
LOW_CUT = 0.01
HIGH_CUT = 0.10


def load_and_clean_run(subject_id, run_id):
    """
    Load one run's dtseries and apply the enabled cleaning steps.
    Motion parameters and the global signal, when enabled, enter a single
    regression together rather than being removed in separate passes.

    Returns
    -------
    data : (T, V) float32 array
        Cleaned time series, vertices in CIFTI order.
    """
    from nilearn.signal import clean as nl_clean

    img = nib.load(str(config.get_brain_path(subject_id, run_id)))
    data = img.get_fdata(dtype=np.float32)
    parts = []
    if DO_MOTION_REG:
        mov = np.loadtxt(config.get_confound_path(subject_id, run_id))
        parts.append(mov[:, :12])
    if DO_GSR:
        # Global signal: mean across all grayordinates at each timepoint
        parts.append(data.mean(axis=1, keepdims=True))
    if parts:
        confounds = np.hstack(parts)
        # Standardise confounds ourselves with the sample SD. nilearn's own
        # confound standardisation still uses the population SD, which would
        # be inconsistent with how the signal is standardised.
        confounds = confounds - confounds.mean(axis=0)
        sd = confounds.std(axis=0, ddof=1)
        sd[sd < np.finfo(np.float64).eps] = 1.0
        confounds = confounds / sd
    else:
        confounds = None
    data = nl_clean(
        data,
        detrend=DO_DETREND,
        standardize="zscore_sample" if DO_ZSCORE else False,
        confounds=confounds,
        standardize_confounds=False,
        low_pass=HIGH_CUT if DO_BANDPASS else None,
        high_pass=LOW_CUT if DO_BANDPASS else None,
        t_r=TR,
    )
    return data.astype(np.float32)


def process_subject(subject_id, save=True):
    """
    Clean all four runs of one subject and concatenate by session.
    Saves two files:
        {subject}_REST1.npy : runs 1-2 concatenated (2400 timepoints)
        {subject}_REST2.npy : runs 3-4 concatenated (2400 timepoints)
    The full 4800-timepoint series is obtained downstream by
    concatenating the two, so it is not stored separately.
    """
    out_dir = config.OUTPUTS_DIR / "Cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for run_id in config.RUN_IDS:
        t0 = time.time()
        runs[run_id] = load_and_clean_run(subject_id, run_id)
        print(f"  {run_id}: {runs[run_id].shape} ({time.time()-t0:.1f}s)")
    rest1 = np.concatenate(
        [runs[config.RUN_IDS[0]], runs[config.RUN_IDS[1]]], axis=0
    )
    rest2 = np.concatenate(
        [runs[config.RUN_IDS[2]], runs[config.RUN_IDS[3]]], axis=0
    )
    if save:
        np.save(out_dir / f"{subject_id}_REST1.npy", rest1)
        np.save(out_dir / f"{subject_id}_REST2.npy", rest2)
        print(f"  saved: {rest1.shape[0]} + {rest2.shape[0]} timepoints")
    return rest1, rest2


def main():
    parser = argparse.ArgumentParser(description="Preprocess one subject.")
    parser.add_argument("--subject", required=True, help="HCP subject ID")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="run without writing output files",
    )
    args = parser.parse_args()
    print(
        f"Variant: {config.VARIANT}  "
        f"(detrend={DO_DETREND}, motion={DO_MOTION_REG}, "
        f"bandpass={DO_BANDPASS}, gsr={DO_GSR})"
    )
    print(f"Subject {args.subject}")
    t0 = time.time()
    process_subject(args.subject, save=not args.no_save)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
