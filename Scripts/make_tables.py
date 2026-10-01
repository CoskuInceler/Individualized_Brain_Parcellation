"""
RESULT TABLES
=============
Turns the per-method output files into the tables the manuscript needs.



The pipeline writes one file per method per preprocessing variant, which
is the right shape for computing but the wrong shape for reading. This
assembles them into method-by-measure tables, with the mean and standard
deviation across subjects in each cell.



One set of tables is written per preprocessing variant.



Output: Outputs/<variant>/Tables/
    table1_validation.csv     stage 1 metrics
    table2_sem.csv            standardised paths from the SEM
    table3_prediction.csv     KRR accuracy and unique variance
    table4_variants.csv       border shifts and ectopic fragments
"""

import argparse
import numpy as np
import pandas as pd


import config

METHODS = [
    ("Method_0_Schaefer", "Schaefer"),
    ("Method_1_gMSHBM", "gMSHBM"),
    ("Method_2_AGP", "AGP"),
    ("Method_3_SLIC_F", "SLIC-F"),
    ("Method_4_SLIC_C", "SLIC-C"),
    ("Method_5_Gordon", "Gordon"),
]


# Gordon is read from the 200-parcel label set, so every method is
# summarised at the same resolution
VALIDATION_FILE = {
    "Method_5_Gordon": "validation_all_subjects_200.csv",
}


def mean_sd(series, digits=3):
    """Format a column as mean followed by standard deviation."""
    s = series.dropna()
    if s.empty:
        return ""
    return f"{s.mean():.{digits}f} ({s.std():.{digits}f})"


def table_validation(out_dir):
    """Stage 1: the four validation criteria, per method."""
    rows = []
    for folder, label in METHODS:
        name = VALIDATION_FILE.get(folder, "validation_all_subjects.csv")
        path = config.OUTPUTS_DIR / folder / name
        if not path.exists():
            continue

        df = pd.read_csv(path)

        # Schaefer is one atlas shared by every subject, so its
        # inter-subject Dice is 1 by construction and never computed
        dice = (
            "1.000 (0.000)"
            if "intersubject_dice" not in df.columns
            else mean_sd(df["intersubject_dice"])
        )

        rows.append(
            {
                "Method": label,
                "Parcels": f"{df['n_parcels'].mean():.0f}",
                "Homogeneity": mean_sd(df["homogeneity_ALL"]),
                "Test-retest": mean_sd(df["fc_test_retest"]),
                "Contiguity (%)": mean_sd(df["pct_contiguous"], 2),
                "Inter-subject Dice": dice,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table1_validation.csv", index=False)
    return df


def table_variants(out_dir):
    """Border shifts and ectopic fragments, per method."""
    rows = []
    for folder, label in METHODS:
        path = (
            config.OUTPUTS_DIR
            / "Variants"
            / folder
            / "variants_all_subjects.csv"
        )
        if not path.exists():
            continue

        df = pd.read_csv(path)
        rows.append(
            {
                "Method": label,
                "Fragments": mean_sd(df["n_fragments"], 1),
                "Border shifts": mean_sd(df["n_border"], 1),
                "Ectopic": mean_sd(df["n_ectopic"], 1),
                "Parcels with ectopic (%)": mean_sd(
                    df["pct_parcels_with_ectopic"], 2
                ),
                "Ectopic distance (mm)": mean_sd(
                    df["mean_ectopic_distance"], 1
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table4_variants.csv", index=False)
    return df


def table_sem(out_dir):
    """
    Standardised paths from the SEM, one row per method and predictor.



    Only the raw graph variant is tabulated. The two binary variants
    answer the same question at a different threshold and would treble
    the table without adding to it; they remain in sem_paths.csv.
    """
    path = config.OUTPUTS_DIR / "Stage2" / "sem_paths.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = df[df["model"].str.startswith("raw")]

    def cell(row):
        stars = (
            "***"
            if row["p_fdr"] < 0.001
            else (
                "**"
                if row["p_fdr"] < 0.01
                else "*" if row["p_fdr"] < 0.05 else ""
            )
        )
        return f"{row['beta']:.3f}{stars}"

    df["entry"] = df.apply(cell, axis=1)

    wide = df.pivot_table(
        index=["method", "model"],
        columns="predictor",
        values="entry",
        aggfunc="first",
    )
    wide = wide.reset_index()
    wide.to_csv(out_dir / "table2_sem.csv", index=False)
    return wide


def table_prediction(out_dir):
    """KRR accuracy alongside the unique variance from the SEM."""
    stage2 = config.OUTPUTS_DIR / "Stage2"

    krr_path = stage2 / "krr_results.csv"
    uniq_path = stage2 / "sem_unique_variance.csv"
    if not krr_path.exists():
        return None

    krr = pd.read_csv(krr_path)

    def cell(row):
        stars = (
            "***"
            if row["p_fdr"] < 0.001
            else (
                "**"
                if row["p_fdr"] < 0.01
                else "*" if row["p_fdr"] < 0.05 else ""
            )
        )
        return f"{row['r']:.3f}{stars}"

    krr["entry"] = krr.apply(cell, axis=1)
    wide = krr.pivot_table(
        index="method", columns="kernel", values="entry", aggfunc="first"
    )
    wide.columns = [f"KRR r ({c})" for c in wide.columns]
    wide = wide.reset_index()

    if uniq_path.exists():
        uniq = pd.read_csv(uniq_path)
        r2 = uniq.groupby("method")["r2_full"].first().reset_index()
        r2.columns = ["method", "SEM R2"]
        r2["SEM R2"] = r2["SEM R2"].round(4)

        cv = uniq[uniq["predictor"] == "parcel_size_cv"]
        cv = cv[["method", "unique_variance"]]
        cv.columns = ["method", "Unique R2 (parcel size)"]

        wide = wide.merge(r2, on="method", how="left")
        wide = wide.merge(cv, on="method", how="left")

    wide.to_csv(out_dir / "table3_prediction.csv", index=False)
    return wide


def main():
    parser = argparse.ArgumentParser(description="Build result tables.")
    args = parser.parse_args()

    out_dir = config.OUTPUTS_DIR / "Tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Variant {config.VARIANT}\n")

    for name, builder in (
        ("Table 1: validation", table_validation),
        ("Table 2: SEM paths", table_sem),
        ("Table 3: prediction", table_prediction),
        ("Table 4: variants", table_variants),
    ):
        df = builder(out_dir)
        print(strrep := "=" * 60)
        print(name)
        if df is None:
            print("  inputs not available yet")
        else:
            print(df.to_string(index=False))
        print()

    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
