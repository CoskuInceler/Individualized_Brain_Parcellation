"""
FEATURE F3 — PARCEL SIZE
==========================
Computes vertex count per parcel for all 5 parcellation methods.

For each method, for each subject:
    - Loads native label files (full 32k mesh .npy)
    - Runs Hungarian alignment → maps native parcel IDs to atlas IDs 1-200
    - Counts vertices per parcel in native geometry
    - Reports under atlas-aligned IDs so row i = same brain region across methods

Output: 200 values per subject per method.
All 200 atlas parcel IDs always present (no dropped parcels).

WHY ATLAS-ALIGNMENT MATTERS HERE:
    Without alignment, "Parcel 1" means something different in each method.
    After Hungarian alignment, parcel ID 1 refers to the same Schaefer
    region across all 5 methods, enabling direct cross-method comparison
    of e.g. "is the motor cortex parcel larger in SLIC-C vs Schaefer?"

OUTPUTS:
    Per subject per method:
        Outputs/Features/F3_Parcel_Sizes/{METHOD}/{subj}_parcel_sizes.csv
            columns: atlas_parcel_id (1-200), n_vertices
    
    After --aggregate:
        Outputs/Features/F3_Parcel_Sizes/{METHOD}/all_subjects_parcel_sizes.csv
            columns: subject_id, parcel_1, parcel_2, ..., parcel_200
            shape:   n_subjects × 200

Usage:
    # Single subject (HPC array job):
    python feat_F3_parcel_sizes.py --subject 100307

    # All subjects locally:
    python feat_F3_parcel_sizes.py --all

    # After all subjects done — build master table per method:
    python feat_F3_parcel_sizes.py --aggregate
"""

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

import feature_config as cfg
import feature_utils  as utils

# Output directory for this feature
F3_DIR = cfg.FEATURES_DIR / "F3_Parcel_Sizes"
for method_name in cfg.METHODS:
    (F3_DIR / method_name).mkdir(parents=True, exist_ok=True)


# =============================================================================
# ALIGNMENT HELPER
# =============================================================================

def get_mapping(method_name, method_info, subj, native_lh, native_rh,
                atlas_lh, atlas_rh):
    """
    Return native→atlas mapping dict for a subject.

    For M0 (Schaefer): identity mapping, no alignment needed.
    For M1-M4: Hungarian bijective alignment.

    Returns:
        mapping: dict {native_global_id: atlas_id (1-200)}
    """
    if not method_info["needs_align"]:
        # Schaefer: label IDs are already atlas IDs
        return {i: i for i in range(1, 201)}

    print(f"      [Align] Hungarian alignment for {method_name}/{subj}...")
    mapping = utils.align_to_atlas(native_lh, native_rh, atlas_lh, atlas_rh)

    # Quick QC
    n_zero = sum(
        1 for nat_id, atl_id in mapping.items()
        if _overlap_count(native_lh, native_rh, nat_id, atlas_lh, atlas_rh, atl_id) == 0
    )
    if n_zero > 0:
        print(f"      [Align] ⚠ {n_zero} zero-overlap forced assignments")

    return mapping


def _overlap_count(native_lh, native_rh, nat_id, atlas_lh, atlas_rh, atl_id):
    """Quick overlap check for QC — not used in computation."""
    rh_unique = np.unique(native_rh[native_rh > 0])
    rh_per_hemi = bool(len(rh_unique) > 0 and rh_unique.max() <= 100)

    if nat_id <= 100:
        # LH
        atl_lh_id = atl_id if atl_id <= 100 else atl_id
        return int(np.sum((native_lh == nat_id) & (atlas_lh == atl_id)))
    else:
        # RH
        rh_native_val = nat_id - 100 if rh_per_hemi else nat_id
        return int(np.sum((native_rh == rh_native_val) & (atlas_rh == atl_id)))


# =============================================================================
# CORE: PARCEL SIZE EXTRACTION FOR ONE SUBJECT
# =============================================================================

def extract_subject(subj, valid_L, valid_R, atlas_lh, atlas_rh):
    """
    Extract parcel sizes for all 5 methods for one subject.

    Saves one CSV per method:
        {F3_DIR}/{method_name}/{subj}_parcel_sizes.csv

    Returns:
        dict {method_name: pd.DataFrame} — sizes per method
    """
    results = {}

    for method_name, method_info in cfg.METHODS.items():
        out_path = F3_DIR / method_name / f"{subj}_parcel_sizes.csv"

        print(f"\n    [{method_name}]")

        # --- Load native labels ---
        native_lh, native_rh = utils.load_native_labels(
            method_info, subj, valid_L, valid_R)

        if native_lh is None:
            print(f"      [SKIP] Label files missing for {subj}")
            continue

        # --- Get alignment mapping ---
        mapping = get_mapping(method_name, method_info, subj,
                              native_lh, native_rh, atlas_lh, atlas_rh)

        # --- Compute parcel sizes (vertex counts, native geometry) ---
        sizes = utils.compute_parcel_sizes_aligned(
            native_lh, native_rh, mapping)

        # Verify all 200 parcels present
        assert len(sizes) == 200, \
            f"Expected 200 parcels, got {len(sizes)} for {method_name}/{subj}"

        # Build DataFrame: one row per parcel, sorted by atlas ID
        rows = [{"atlas_parcel_id": pid, "n_vertices": nv}
                for pid, nv in sorted(sizes.items())]
        df = pd.DataFrame(rows)

        df.to_csv(out_path, index=False)

        total_v = df["n_vertices"].sum()
        mean_v  = df["n_vertices"].mean()
        print(f"      Saved: {out_path.name}  "
              f"(200 parcels, mean={mean_v:.0f} vertices, "
              f"total={total_v} vertices)")

        results[method_name] = df

    return results


# =============================================================================
# AGGREGATE: BUILD MASTER TABLE PER METHOD
# =============================================================================

def aggregate():
    """
    After all subjects are processed, build one master CSV per method.

    Master CSV shape: (n_subjects, 200)
    Columns: subject_id, parcel_1, parcel_2, ..., parcel_200
    """
    print("\n" + "=" * 60)
    print("AGGREGATING F3 PARCEL SIZES")
    print("=" * 60)

    for method_name in cfg.METHODS:
        method_dir = F3_DIR / method_name
        print(f"\n  {method_name}...")

        rows = []
        subjects_found = []

        for subj in cfg.SUBJECT_IDS:
            csv_path = method_dir / f"{subj}_parcel_sizes.csv"
            if not csv_path.exists():
                print(f"    [WARN] Missing: {csv_path.name}")
                continue

            df = pd.read_csv(csv_path)
            if len(df) != 200:
                print(f"    [WARN] {subj}: {len(df)} parcels ≠ 200, skipping")
                continue

            # Build one row: subject_id + 200 parcel size values
            # Sort by atlas_parcel_id to guarantee consistent column order
            df_sorted = df.sort_values("atlas_parcel_id")
            row = {"subject_id": subj}
            for _, r in df_sorted.iterrows():
                row[f"parcel_{int(r['atlas_parcel_id'])}"] = int(r["n_vertices"])
            rows.append(row)
            subjects_found.append(subj)

        if not rows:
            print(f"    [WARN] No subjects found for {method_name}")
            continue

        master = pd.DataFrame(rows)
        # Ensure column order: subject_id, parcel_1, ..., parcel_200
        parcel_cols = [f"parcel_{i}" for i in range(1, 201)]
        master = master[["subject_id"] + parcel_cols]

        out_path = method_dir / "all_subjects_parcel_sizes.csv"
        master.to_csv(out_path, index=False)

        print(f"    Saved: {out_path.name}  "
              f"({len(subjects_found)} subjects × 200 parcels)")

        # Quick sanity check
        size_matrix = master[parcel_cols].values
        print(f"    Size matrix: shape={size_matrix.shape}, "
              f"mean={size_matrix.mean():.0f}, "
              f"min={size_matrix.min()}, "
              f"max={size_matrix.max()}")

    print("\n  Aggregation complete.")
    print(f"  Master tables saved to: {F3_DIR}/{{method}}/all_subjects_parcel_sizes.csv")


# =============================================================================
# MAIN
# =============================================================================

def process_subject(subj):
    print(f"\n{'═' * 60}")
    print(f"  F3 PARCEL SIZES | Subject: {subj}")
    print(f"{'═' * 60}")

    # Load valid vertices and atlas labels once (shared across methods)
    print("\n  [Setup] Loading valid vertices and atlas...")
    valid_L, valid_R = utils.get_valid_vertices()
    atlas_lh, atlas_rh = utils.load_atlas_labels(valid_L, valid_R)
    print(f"  Valid vertices: LH={len(valid_L)}, RH={len(valid_R)}")

    # Extract for all methods
    results = extract_subject(subj, valid_L, valid_R, atlas_lh, atlas_rh)

    print(f"\n  ✓ Subject {subj} complete — "
          f"{len(results)}/5 methods processed")


def main():
    parser = argparse.ArgumentParser(
        description="F3: Parcel size extraction (all 5 methods)")
    parser.add_argument("--subject",   type=str, default=None,
                        help="Single subject ID (for HPC array job)")
    parser.add_argument("--all",       action="store_true",
                        help="Process all subjects sequentially (local use)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Build master tables after all subjects done")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  FEATURE F3: PARCEL SIZES")
    print(f"{'═' * 60}")
    print(f"  Output: {F3_DIR}")
    print(f"  Methods: {list(cfg.METHODS.keys())}")

    if args.aggregate:
        aggregate()

    elif args.subject:
        process_subject(args.subject)

    elif args.all:
        for subj in cfg.SUBJECT_IDS:
            try:
                process_subject(subj)
            except Exception as e:
                print(f"\n  [ERROR] {subj}: {e}")
                import traceback
                traceback.print_exc()
        aggregate()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()