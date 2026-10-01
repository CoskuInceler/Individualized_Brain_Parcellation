"""
AGGREGATE VARIANT ANALYSIS
==========================
Combines the per-subject variant counts of one method.



The summary answers the question the analysis was built for: how often
does each method find a parcel fragment far from its main body, and how
far away. A method that forces parcels to stay in one piece reports
zero, which is itself the finding.



Usage:
    python3 aggregate_variants.py --method Method_4_SLIC_C
"""

import argparse
import numpy as np
import pandas as pd


import config


def main():
    parser = argparse.ArgumentParser(description="Aggregate variant analysis.")
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    var_dir = config.OUTPUTS_DIR / "Variants" / args.method
    files = sorted(
        f for f in var_dir.glob("*.csv") if not f.name.startswith("variants_")
    )
    if not files:
        raise FileNotFoundError(f"no per-subject files in {var_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("subject").reset_index(drop=True)
    df.to_csv(var_dir / "variants_all_subjects.csv", index=False)

    print(f"{args.method}: {len(df)} subjects")

    num = df.select_dtypes(include=[np.number]).drop(
        columns=["subject"], errors="ignore"
    )
    summary = pd.DataFrame(
        {
            "measure": num.columns,
            "n": num.notna().sum().values,
            "mean": num.mean().values,
            "sd": num.std().values,
            "min": num.min().values,
            "max": num.max().values,
        }
    ).round(4)
    summary.to_csv(var_dir / "variants_summary.csv", index=False)

    # the subjects that show no fragments at all are worth naming, since
    # for some methods that is every subject
    none_at_all = int((df["n_fragments"] == 0).sum())
    print(f"  subjects with no fragments: {none_at_all} of {len(df)}")

    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
