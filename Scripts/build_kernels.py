"""
STAGE 2: SIMILARITY KERNELS
===========================
Builds the subject-by-subject similarity matrices that kernel ridge
regression uses as kernels.



Two per method, both correlations between subjects:
    sizes   over the 200 parcel sizes
    fc      over the 19,900 connectivity edges



Both profiles were already put into atlas order by feature_extraction.py,
so entry i means the same region in every subject.



Schaefer's size kernel is degenerate, since every subject carries the
same atlas and therefore the same parcel sizes. It is written anyway and
skipped downstream.



Output: Features/<method>/kernel_sizes.csv, kernel_fc.csv
"""

import argparse
import numpy as np
import pandas as pd


import config

METHODS = [
    "Method_0_Schaefer",
    "Method_1_gMSHBM",
    "Method_2_AGP",
    "Method_3_SLIC_F",
    "Method_4_SLIC_C",
    "Method_5_Gordon",
]


def build(method):
    feat_dir = config.OUTPUTS_DIR / "Features" / method
    profile_dir = feat_dir / "Profiles"

    subjects = [
        s
        for s in config.SUBJECT_IDS
        if (profile_dir / f"{s}_sizes.npy").exists()
    ]

    for suffix in ("sizes", "fc"):
        profiles = np.vstack(
            [np.load(profile_dir / f"{s}_{suffix}.npy") for s in subjects]
        )

        usable = ~np.isnan(profiles).any(axis=0)
        data = profiles[:, usable]

        if data.shape[1] < 2 or np.allclose(data.std(axis=0), 0):
            kernel = np.full((len(subjects), len(subjects)), np.nan)
            note = "degenerate"
        else:
            kernel = np.corrcoef(data)
            off = kernel[np.triu_indices_from(kernel, k=1)]
            note = f"mean off-diagonal {off.mean():.4f}"

        pd.DataFrame(kernel, index=subjects, columns=subjects).to_csv(
            feat_dir / f"kernel_{suffix}.csv"
        )

        print(f"  {method:20s} {suffix:6s} {kernel.shape}  {note}")


def main():
    parser = argparse.ArgumentParser(description="Build similarity kernels.")
    parser.add_argument("--method", default=None)
    args = parser.parse_args()

    print(f"Variant {config.VARIANT}")
    for m in ([args.method] if args.method else METHODS):
        build(m)


if __name__ == "__main__":
    main()
