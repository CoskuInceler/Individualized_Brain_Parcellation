"""
AGGREGATE VALIDATION RESULTS
============================
Combines the per-subject validation files of one method into a single
table and computes summary statistics.



If inter-subject Dice was run for this method, those per-subject values
are merged in as well, so the four validation criteria of stage one end up
in one place: homogeneity, test-retest reliability, spatial contiguity and
inter-subject Dice.



Usage:
    python3 aggregate_validation.py --method Method_2_AGP
"""

import argparse
import numpy as np
import pandas as pd


import config


def combine(directory):
    """Read every per-subject CSV in a directory into one DataFrame."""
    files = sorted(directory.glob("*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values("subject").reset_index(drop=True)


def summarise(df):
    """Mean, SD and range for each numeric column."""
    num = df.select_dtypes(include=[np.number]).drop(
        columns=["subject"], errors="ignore"
    )
    out = pd.DataFrame(
        {
            "metric": num.columns,
            "n": num.notna().sum().values,
            "mean": num.mean().values,
            "sd": num.std().values,
            "min": num.min().values,
            "max": num.max().values,
        }
    )
    return out.round(5)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate validation results."
    )
    parser.add_argument(
        "--method", required=True, help="method folder, e.g. Method_2_AGP"
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="validation subfolder suffix, e.g. 200 for " "Validation_200",
    )
    args = parser.parse_args()

    method_dir = config.OUTPUTS_DIR / args.method

    suffix = "" if args.labels is None else f"_{args.labels}"
    df = combine(method_dir / f"Validation{suffix}")
    if df is None:
        raise FileNotFoundError(
            f"no validation files in {method_dir / ('Validation' + suffix)}"
        )
    print(f"Combined {len(df)} subjects")

    dice = combine(method_dir / f"Dice{suffix}")
    if dice is not None:
        keep = ["subject", "dice_mean", "dice_sd", "dice_min", "dice_max"]
        dice = dice[[c for c in keep if c in dice.columns]]
        dice = dice.rename(
            columns={
                "dice_mean": "intersubject_dice",
                "dice_sd": "intersubject_dice_sd",
                "dice_min": "intersubject_dice_min",
                "dice_max": "intersubject_dice_max",
            }
        )
        df = df.merge(dice, on="subject", how="left")
        print(
            f"Merged inter-subject Dice for {dice['subject'].nunique()} subjects"
        )
    else:
        print("No inter-subject Dice found for this method")

    df.to_csv(method_dir / f"validation_all_subjects{suffix}.csv", index=False)

    summary = summarise(df)
    summary.to_csv(method_dir / f"validation_summary{suffix}.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
