"""
STAGE 2, STEP 2: FEATURE AGGREGATION
====================================
Combines the per-subject feature files of one method and adds the two
features that only exist once every subject is available.



    parcel_size_similarity   mean correlation between a subject's
                             parcel-size profile and every other
                             subject's
    fc_similarity            mean correlation between a subject's
                             connectivity matrix and every other
                             subject's



Both are the mean off-diagonal of the subject-by-subject correlation
matrix, following the thesis. A subject who sits close to everyone else
scores high; one whose topography is unusual scores low.



Usage:
    python3 aggregate_features.py --method Method_2_AGP
"""

import argparse
import numpy as np
import pandas as pd


import config


def load_profiles(profile_dir, subjects, suffix):
    """Stack per-subject profiles, keeping track of which subjects loaded."""
    kept, rows = [], []
    for s in subjects:
        path = profile_dir / f"{s}_{suffix}.npy"
        if path.exists():
            rows.append(np.load(path))
            kept.append(s)
    if not rows:
        return [], None
    return kept, np.vstack(rows)


def mean_similarity(profiles):
    """
    Mean correlation of each subject with every other subject.



    Columns missing for any subject are dropped first, so that every
    subject is scored on the same set of parcels.
    """
    usable = ~np.isnan(profiles).any(axis=0)
    if usable.sum() < 3:
        return np.full(profiles.shape[0], np.nan), int(usable.sum())

    corr = np.corrcoef(profiles[:, usable])
    n = corr.shape[0]
    off_diagonal_sum = corr.sum(axis=1) - np.diag(corr)
    return off_diagonal_sum / (n - 1), int(usable.sum())


def main():
    parser = argparse.ArgumentParser(description="Aggregate stage 2 features.")
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    feat_dir = config.OUTPUTS_DIR / "Features" / args.method
    files = sorted(
        f
        for f in feat_dir.glob("*.csv")
        if not f.name.startswith(("features_", "kernel_"))
    )
    if not files:
        raise FileNotFoundError(f"no per-subject features in {feat_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("subject").reset_index(drop=True)
    subjects = df["subject"].astype(str).tolist()
    print(f"Combined {len(df)} subjects")

    profile_dir = feat_dir / "Profiles"
    for suffix, column in (
        ("sizes", "parcel_size_similarity"),
        ("fc", "fc_similarity"),
    ):
        kept, profiles = load_profiles(profile_dir, subjects, suffix)
        if profiles is None:
            print(f"  {column}: no profiles found")
            continue
        values, n_used = mean_similarity(profiles)
        df[column] = df["subject"].astype(str).map(dict(zip(kept, values)))
        print(f"  {column}: {len(kept)} subjects, {n_used} columns used")

    # reversed, so that a higher value always means a better-connected
    # network, whichever measure it comes from
    for variant in ("raw", "top10", "r05"):
        col = f"avg_shortest_path_{variant}"
        if col in df.columns:
            df[f"aspl_reversed_{variant}"] = -df[col]
        col = f"n_components_{variant}"
        if col in df.columns:
            df[f"components_reversed_{variant}"] = -df[col]

    out = feat_dir / "features_all_subjects.csv"
    df.to_csv(out, index=False)

    num = df.select_dtypes(include=[np.number]).drop(
        columns=["subject"], errors="ignore"
    )
    summary = pd.DataFrame(
        {
            "feature": num.columns,
            "n": num.notna().sum().values,
            "mean": num.mean().values,
            "sd": num.std().values,
            "min": num.min().values,
            "max": num.max().values,
        }
    ).round(5)
    summary.to_csv(feat_dir / "features_summary.csv", index=False)

    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
