"""
EFFECT OF ADDITIONAL CLEANING ON HOMOGENEITY
============================================
The thesis applied detrending, motion regression and a 0.01-0.10 Hz
bandpass on top of the HCP files. The current pipeline applies none of
these. This script quantifies what each step does to functional
homogeneity, using the Schaefer atlas as a fixed reference parcellation.



One subject, REST1 only.
"""

import time
import numpy as np
import nibabel as nib
from nilearn.signal import clean


import config
import utils
import metrics

SUBJECT = config.SUBJECT_IDS[0]
TR = 0.72


SETTINGS = [
    ("none (current)", dict(detrend=False, motion=False, band=False)),
    ("detrend only", dict(detrend=True, motion=False, band=False)),
    ("bandpass only", dict(detrend=False, motion=False, band=True)),
    ("thesis (all three)", dict(detrend=True, motion=True, band=True)),
]


def load_runs(subject, runs):
    """Load raw dtseries and motion parameters for the given runs."""
    out = []
    for r in runs:
        img = nib.load(str(config.get_brain_path(subject, r)))
        data = img.get_fdata(dtype=np.float32)
        mov = np.loadtxt(config.get_confound_path(subject, r))[:, :12]
        out.append((data, mov))
    return out


def apply(data, mov, detrend, motion, band):
    return clean(
        data,
        detrend=detrend,
        standardize="zscore_sample",
        confounds=mov if motion else None,
        low_pass=0.10 if band else None,
        high_pass=0.01 if band else None,
        t_r=TR,
    )


def main():
    ref = config.get_brain_path(SUBJECT, config.RUN_IDS[0])
    vL = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    vR = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    aL, aR = utils.load_and_filter_atlas(config.SCHAEFER_200_FILE, vL, vR)
    atlas = np.concatenate([aL, aR])
    n_cortex = len(atlas)

    runs = load_runs(SUBJECT, config.RUN_IDS[:2])  # REST1 = runs 1-2

    print(f"\nSubject {SUBJECT}, REST1 (2400 timepoints), Schaefer 200\n")
    print(f"{'setting':<22} {'homogeneity':>12} {'mean FC':>10} {'time':>8}")
    print("-" * 56)

    for name, opt in SETTINGS:
        t0 = time.time()
        parts = [apply(d, m, **opt) for d, m in runs]
        ts = np.concatenate(parts, axis=0)[:, :n_cortex]

        h, _ = metrics.homogeneity(ts, atlas)

        # mean FC among parcels, for context
        pts = np.column_stack(
            [
                ts[:, atlas == p].mean(axis=1)
                for p in np.unique(atlas[atlas > 0])
            ]
        )
        fc = np.corrcoef(pts.T)
        mfc = float(fc[np.triu_indices(fc.shape[0], k=1)].mean())

        print(f"{name:<22} {h:>12.4f} {mfc:>10.4f} {time.time()-t0:>7.1f}s")


if __name__ == "__main__":
    main()
