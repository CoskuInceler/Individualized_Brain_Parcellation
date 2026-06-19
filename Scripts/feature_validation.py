"""
FEATURE EXTRACTION VALIDATION
================================
Validates all outputs from the feature extraction pipeline.

Checks F3, F4, and F1/F2 outputs for all 5 methods and all subjects.

PER-SUBJECT CHECKS:
    F3: parcel_sizes.csv
        - Exactly 200 rows
        - atlas_parcel_id covers 1-200, all unique
        - All n_vertices > 0

    F4: fc_matrix.npy + fc_vector.npy
        - FC matrix: shape (200,200), symmetric, diagonal=1, no NaN
        - FC vector: length 19900, no NaN

    F1/F2: graph_metrics.csv
        - All 6 rows present (2 metrics × 3 variants)
        - Global efficiency in [0, 1]
        - Average shortest path > 0

CROSS-SUBJECT CHECKS:
    F4 similarity matrix:
        - Shape (n_subj × n_subj)
        - Symmetric
        - Diagonal = 1

    F3 master table:
        - Shape (n_subj × 200)
        - No missing values

    F1/F2 master table:
        - Shape (n_subj × 6)
        - No missing values for GE columns
        - ASPL NaN allowed (disconnected graph)

Usage:
    python feature_validation.py                    # validate all
    python feature_validation.py --feature F3       # validate F3 only
    python feature_validation.py --feature F4       # validate F4 only
    python feature_validation.py --feature F1F2     # validate F1/F2 only
    python feature_validation.py --subjects 100307 133928   # subset of subjects
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import feature_config as cfg

# Feature output directories
F3_DIR  = cfg.FEATURES_DIR / "F3_Parcel_Sizes"
F4_DIR  = cfg.FEATURES_DIR / "F4_FC"
F12_DIR = cfg.FEATURES_DIR / "F1F2_Graph"

# Counters
n_pass = 0
n_fail = 0
n_warn = 0


def ok(msg):
    global n_pass; n_pass += 1
    print(f"  ✓ {msg}")


def fail(msg):
    global n_fail; n_fail += 1
    print(f"  ✗ FAIL: {msg}")


def warn(msg):
    global n_warn; n_warn += 1
    print(f"  ⚠ WARN: {msg}")


# =============================================================================
# F3 VALIDATION
# =============================================================================

def validate_F3_subject(subj, method_name):
    path = F3_DIR / method_name / f"{subj}_parcel_sizes.csv"
    label = f"F3/{method_name}/{subj}"

    if not path.exists():
        fail(f"{label}: file missing"); return

    df = pd.read_csv(path)

    # Row count
    if len(df) != 200:
        fail(f"{label}: {len(df)} rows ≠ 200")
    else:
        ok(f"{label}: 200 parcels")

    # Atlas IDs
    if df["atlas_parcel_id"].nunique() != 200:
        fail(f"{label}: atlas_parcel_id not unique")
    elif df["atlas_parcel_id"].min() < 1 or df["atlas_parcel_id"].max() > 200:
        fail(f"{label}: atlas_parcel_id outside 1-200")
    else:
        ok(f"{label}: atlas IDs 1-200, all unique")

    # Vertex counts
    if (df["n_vertices"] <= 0).any():
        fail(f"{label}: parcel(s) with n_vertices ≤ 0")
    else:
        ok(f"{label}: all parcels have positive vertex count")


def validate_F3_master(method_name, expected_n):
    path = F3_DIR / method_name / "all_subjects_parcel_sizes.csv"
    label = f"F3/{method_name}/master"

    if not path.exists():
        fail(f"{label}: all_subjects_parcel_sizes.csv missing"); return

    df = pd.read_csv(path)

    # Shape
    if df.shape[1] != 201:  # subject_id + 200 parcel columns
        fail(f"{label}: {df.shape[1]} columns ≠ 201")
    else:
        ok(f"{label}: 201 columns (subject_id + 200 parcels)")

    if len(df) != expected_n:
        warn(f"{label}: {len(df)} subjects, expected {expected_n}")
    else:
        ok(f"{label}: {len(df)} subjects")

    # No missing values
    parcel_cols = [c for c in df.columns if c.startswith("parcel_")]
    n_nan = df[parcel_cols].isna().sum().sum()
    if n_nan > 0:
        fail(f"{label}: {n_nan} missing values in parcel columns")
    else:
        ok(f"{label}: no missing values")


# =============================================================================
# F4 VALIDATION
# =============================================================================

def validate_F4_subject(subj, method_name):
    fc_path  = F4_DIR / method_name / f"{subj}_fc_matrix.npy"
    vec_path = F4_DIR / method_name / f"{subj}_fc_vector.npy"
    label    = f"F4/{method_name}/{subj}"

    # FC matrix
    if not fc_path.exists():
        fail(f"{label}: fc_matrix.npy missing"); return

    fc = np.load(fc_path)

    if fc.shape != (200, 200):
        fail(f"{label}: FC shape {fc.shape} ≠ (200,200)"); return
    else:
        ok(f"{label}: FC shape (200,200)")

    if np.isnan(fc).any():
        fail(f"{label}: FC contains NaN")

    if not np.allclose(fc, fc.T, atol=1e-5):
        fail(f"{label}: FC not symmetric")
    else:
        ok(f"{label}: FC symmetric")

    diag_err = float(np.max(np.abs(np.diag(fc) - 1.0)))
    if diag_err > 1e-3:
        fail(f"{label}: FC diagonal ≠ 1 (err={diag_err:.2e})")
    else:
        ok(f"{label}: FC diagonal=1 (err={diag_err:.2e})")

    # FC vector
    if not vec_path.exists():
        fail(f"{label}: fc_vector.npy missing"); return

    vec = np.load(vec_path)
    if len(vec) != 19900:
        fail(f"{label}: FC vector length {len(vec)} ≠ 19900")
    elif np.isnan(vec).any():
        fail(f"{label}: FC vector contains NaN")
    else:
        ok(f"{label}: FC vector length=19900, no NaN")

    # Cross-check: vector should match lower triangle of matrix
    expected = fc[np.tril_indices_from(fc, k=-1)]
    if not np.allclose(vec, expected, atol=1e-6):
        fail(f"{label}: FC vector does not match lower triangle of FC matrix")
    else:
        ok(f"{label}: FC vector consistent with matrix lower triangle")


def validate_F4_similarity(method_name, expected_n):
    sim_path  = F4_DIR / method_name / "fc_similarity_matrix.npy"
    subj_path = F4_DIR / method_name / "fc_similarity_subjects.txt"
    label     = f"F4/{method_name}/similarity"

    if not sim_path.exists():
        fail(f"{label}: fc_similarity_matrix.npy missing"); return
    if not subj_path.exists():
        fail(f"{label}: fc_similarity_subjects.txt missing"); return

    sim = np.load(sim_path)
    with open(subj_path) as f:
        subjects = [l.strip() for l in f if l.strip()]

    n = len(subjects)

    if sim.shape != (n, n):
        fail(f"{label}: shape {sim.shape} ≠ ({n},{n})")
    else:
        ok(f"{label}: shape ({n}×{n})")

    if n != expected_n:
        warn(f"{label}: {n} subjects, expected {expected_n}")

    if np.isnan(sim).any():
        fail(f"{label}: contains NaN")

    if not np.allclose(sim, sim.T, atol=1e-5):
        fail(f"{label}: not symmetric")
    else:
        ok(f"{label}: symmetric")

    diag_err = float(np.max(np.abs(np.diag(sim) - 1.0)))
    if diag_err > 1e-3:
        fail(f"{label}: diagonal ≠ 1 (err={diag_err:.2e})")
    else:
        ok(f"{label}: diagonal=1 (err={diag_err:.2e})")

    off_mean = float(sim[np.triu_indices_from(sim, k=1)].mean())
    ok(f"{label}: off-diagonal mean={off_mean:.4f}")


# =============================================================================
# F1/F2 VALIDATION
# =============================================================================

def validate_F12_subject(subj, method_name):
    path  = F12_DIR / method_name / f"{subj}_graph_metrics.csv"
    label = f"F1F2/{method_name}/{subj}"

    if not path.exists():
        fail(f"{label}: file missing"); return

    df = pd.read_csv(path)

    # Check all 6 rows present
    expected_rows = {
        ("global_efficiency", "raw"),
        ("global_efficiency", "top10"),
        ("global_efficiency", "r05"),
        ("avg_shortest_path", "raw"),
        ("avg_shortest_path", "top10"),
        ("avg_shortest_path", "r05"),
    }
    actual_rows = set(zip(df["metric"], df["variant"]))
    missing = expected_rows - actual_rows
    if missing:
        fail(f"{label}: missing rows {missing}")
    else:
        ok(f"{label}: all 6 metric/variant combinations present")

    # Global efficiency range check
    ge_rows = df[df["metric"] == "global_efficiency"]
    if (ge_rows["value"].dropna() < 0).any() or \
       (ge_rows["value"].dropna() > 1).any():
        fail(f"{label}: global_efficiency outside [0,1]")
    else:
        ok(f"{label}: global_efficiency in valid range")

    # ASPL > 0 where not NaN
    aspl_rows = df[df["metric"] == "avg_shortest_path"]
    valid_aspl = aspl_rows["value"].dropna()
    if len(valid_aspl) == 0:
        warn(f"{label}: all ASPL values are NaN (fully disconnected graphs?)")
    elif (valid_aspl <= 0).any():
        fail(f"{label}: avg_shortest_path ≤ 0")
    else:
        ok(f"{label}: avg_shortest_path > 0 for all non-NaN values")

    # Warn if any ASPL is NaN (disconnected graph)
    n_nan_aspl = aspl_rows["value"].isna().sum()
    if n_nan_aspl > 0:
        warn(f"{label}: {n_nan_aspl}/3 ASPL variants are NaN "
             f"(disconnected graph for those variants)")


def validate_F12_master(method_name, expected_n):
    path  = F12_DIR / method_name / "all_subjects_graph_metrics.csv"
    label = f"F1F2/{method_name}/master"

    if not path.exists():
        fail(f"{label}: all_subjects_graph_metrics.csv missing"); return

    df = pd.read_csv(path)

    expected_cols = [
        "subject_id",
        "global_efficiency_raw", "global_efficiency_top10", "global_efficiency_r05",
        "avg_shortest_path_raw", "avg_shortest_path_top10", "avg_shortest_path_r05",
    ]

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        fail(f"{label}: missing columns {missing_cols}")
    else:
        ok(f"{label}: all 7 columns present")

    if len(df) != expected_n:
        warn(f"{label}: {len(df)} subjects, expected {expected_n}")
    else:
        ok(f"{label}: {len(df)} subjects")

    # GE columns should have no NaN
    ge_cols = [c for c in df.columns if c.startswith("global_efficiency")]
    n_nan_ge = df[ge_cols].isna().sum().sum()
    if n_nan_ge > 0:
        fail(f"{label}: {n_nan_ge} NaN values in global_efficiency columns")
    else:
        ok(f"{label}: no NaN in global_efficiency columns")

    # ASPL NaN allowed — just report counts
    aspl_cols = [c for c in df.columns if c.startswith("avg_shortest_path")]
    for col in aspl_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            warn(f"{label}: {n_nan} NaN values in {col} (disconnected graphs)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate feature extraction outputs")
    parser.add_argument("--feature",  type=str, default=None,
                        choices=["F3", "F4", "F1F2"],
                        help="Validate one feature only")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject IDs to check (default: first 3)")
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else cfg.SUBJECT_IDS[:3]
    features = [args.feature] if args.feature else ["F3", "F4", "F1F2"]

    # Expected subject counts per method
    expected_n = {m: 95 for m in cfg.METHODS}

    print("=" * 60)
    print("FEATURE EXTRACTION VALIDATION")
    print("=" * 60)
    print(f"Checking subjects: {subjects}")
    print(f"Checking features: {features}")
    print(f"Checking methods:  {list(cfg.METHODS.keys())}")

    for method_name in cfg.METHODS:
        n = expected_n[method_name]

        if "F3" in features:
            print(f"\n{'━'*60}")
            print(f"  F3 | {method_name}")
            print(f"{'━'*60}")
            for subj in subjects:
                validate_F3_subject(subj, method_name)
            validate_F3_master(method_name, n)

        if "F4" in features:
            print(f"\n{'━'*60}")
            print(f"  F4 | {method_name}")
            print(f"{'━'*60}")
            for subj in subjects:
                validate_F4_subject(subj, method_name)
            validate_F4_similarity(method_name, n)

        if "F1F2" in features:
            print(f"\n{'━'*60}")
            print(f"  F1/F2 | {method_name}")
            print(f"{'━'*60}")
            for subj in subjects:
                validate_F12_subject(subj, method_name)
            validate_F12_master(method_name, n)

    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass} passed  |  {n_warn} warnings  |  {n_fail} failed")
    print(f"{'='*60}")

    if n_fail > 0:
        print("\n✗ VALIDATION FAILED")
        raise SystemExit(1)
    elif n_warn > 0:
        print("\n⚠ VALIDATION PASSED WITH WARNINGS")
    else:
        print("\n✓ FULL VALIDATION PASSED")


if __name__ == "__main__":
    main()