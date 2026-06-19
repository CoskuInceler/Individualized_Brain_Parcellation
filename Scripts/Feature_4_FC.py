"""
FEATURE F4 — FUNCTIONAL CONNECTIVITY
=======================================
Computes functional connectivity matrices for all 5 parcellation methods.

For each method, for each subject:
    F4a: FC Matrix (200×200)
        - Pearson correlation between all parcel timeseries pairs
        - Computed in native column order then permuted to atlas order
          via Hungarian alignment (same alignment as F3)
        - Saved as .npy

    F4b: FC Vector (19,900 values)
        - Lower triangle of FC matrix (excluding diagonal)
        - 200×199/2 = 19,900 values
        - Saved as .npy

After all subjects are processed (--similarity):
    F4c: FC Similarity Matrix (95×95)
        - Entry [i,j] = Pearson r between subject i and subject j's
          FC vectors (19,900-dimensional)
        - One matrix per method
        - Saved as .npy + subject order .txt

WHY ATLAS ALIGNMENT FOR FC:
    The FC matrix entry [i,j] must refer to the same pair of brain
    regions across all subjects. Without alignment, position [1,2]
    in gMSHBM refers to a different pair of regions than in Schaefer.
    After Hungarian alignment and permutation, [i,j] consistently
    refers to atlas parcels i and j across all methods and subjects.

OUTPUTS:
    Per subject per method:
        Outputs/Features/F4_FC/{METHOD}/{subj}_fc_matrix.npy     (200×200)
        Outputs/Features/F4_FC/{METHOD}/{subj}_fc_vector.npy     (19900,)

    Cross-subject per method (after --similarity):
        Outputs/Features/F4_FC/{METHOD}/fc_similarity_matrix.npy (95×95)
        Outputs/Features/F4_FC/{METHOD}/fc_similarity_subjects.txt

Usage:
    # Single subject (HPC array job):
    python feat_F4_fc.py --subject 100307

    # All subjects locally:
    python feat_F4_fc.py --all

    # After all subjects done — build similarity matrices:
    python feat_F4_fc.py --similarity
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import feature_config as cfg
import feature_utils  as utils

# Output directory for this feature
F4_DIR = cfg.FEATURES_DIR / "F4_FC"
for method_name in cfg.METHODS:
    (F4_DIR / method_name).mkdir(parents=True, exist_ok=True)


# =============================================================================
# ALIGNMENT (same logic as F3 — kept independent so scripts don't couple)
# =============================================================================

def get_mapping(method_name, method_info, subj,
                native_lh, native_rh, atlas_lh, atlas_rh):
    """
    Return native→atlas mapping for FC permutation.
    Identity mapping for M0, Hungarian alignment for M1-M4.
    """
    if not method_info["needs_align"]:
        return {i: i for i in range(1, 201)}

    print(f"      [Align] Hungarian alignment for {method_name}/{subj}...")
    mapping = utils.align_to_atlas(native_lh, native_rh, atlas_lh, atlas_rh)
    return mapping


# =============================================================================
# CORE: FC EXTRACTION FOR ONE SUBJECT
# =============================================================================

def extract_subject(subj, valid_L, valid_R, atlas_lh, atlas_rh):
    """
    Compute FC matrix and FC vector for all 5 methods for one subject.

    Saves per method:
        {F4_DIR}/{method_name}/{subj}_fc_matrix.npy   (200×200)
        {F4_DIR}/{method_name}/{subj}_fc_vector.npy   (19900,)

    Returns:
        dict {method_name: fc_vector} for use in similarity matrix
    """
    results = {}

    for method_name, method_info in cfg.METHODS.items():
        fc_path  = F4_DIR / method_name / f"{subj}_fc_matrix.npy"
        vec_path = F4_DIR / method_name / f"{subj}_fc_vector.npy"

        print(f"\n    [{method_name}]")

        # --- Load timeseries CSV ---
        ts_path = method_info["method_dir"] / \
                  method_info["ts_pattern"].format(subj=subj)

        if not ts_path.exists():
            print(f"      [SKIP] Timeseries not found: {ts_path.name}")
            continue

        ts_df = pd.read_csv(ts_path)
        print(f"      Timeseries: {ts_df.shape[0]} timepoints × "
              f"{ts_df.shape[1]} parcels")

        # --- Load native labels for alignment ---
        native_lh, native_rh = utils.load_native_labels(
            method_info, subj, valid_L, valid_R)

        if native_lh is None and method_info["needs_align"]:
            print(f"      [SKIP] Label files missing for {subj}")
            continue

        # --- Get alignment mapping ---
        mapping = get_mapping(method_name, method_info, subj,
                              native_lh, native_rh, atlas_lh, atlas_rh)

        # --- Compute FC in native column order ---
        fc_native = utils.compute_fc_matrix(ts_df)

        # --- Permute to atlas order ---
        fc = utils.permute_fc_to_atlas_order(
            fc_native,
            list(ts_df.columns),
            method_info["col_format"],
            mapping
        )

        # Validate permutation
        diag_err = float(np.max(np.abs(np.diag(fc) - 1.0)))
        sym_err  = float(np.max(np.abs(fc - fc.T)))
        if diag_err > 1e-5 or sym_err > 1e-5:
            print(f"      [WARN] FC validation: diag_err={diag_err:.2e}, "
                  f"sym_err={sym_err:.2e}")
        else:
            print(f"      FC validated: symmetric, diagonal=1 ✓")

        # --- Save FC matrix ---
        np.save(fc_path, fc)
        upper = fc[np.triu_indices_from(fc, k=1)]
        print(f"      Saved: {fc_path.name}  "
              f"(200×200, mean|r|={np.abs(upper).mean():.3f})")

        # --- FC vector (lower triangle) ---
        fc_vec = utils.fc_lower_triangle(fc)
        np.save(vec_path, fc_vec)
        print(f"      Saved: {vec_path.name}  ({len(fc_vec)} values)")

        results[method_name] = fc_vec

    return results


# =============================================================================
# SIMILARITY MATRIX (cross-subject, run after all subjects done)
# =============================================================================

def compute_similarity_matrices():
    """
    Build one 95×95 FC similarity matrix per method.
    Loads saved FC vectors for all subjects and computes pairwise
    Pearson correlation between their 19,900-dimensional FC vectors.
    """
    print("\n" + "=" * 60)
    print("COMPUTING FC SIMILARITY MATRICES (F4c)")
    print("=" * 60)

    for method_name in cfg.METHODS:
        method_dir = F4_DIR / method_name
        print(f"\n  {method_name}...")

        fc_vectors = {}
        missing    = []

        for subj in cfg.SUBJECT_IDS:
            vec_path = method_dir / f"{subj}_fc_vector.npy"
            if vec_path.exists():
                fc_vectors[subj] = np.load(vec_path)
            else:
                missing.append(subj)

        if missing:
            print(f"    [WARN] Missing FC vectors for {len(missing)} subjects: "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}")

        if len(fc_vectors) < 2:
            print(f"    [SKIP] Need at least 2 subjects.")
            continue

        # Validate vector lengths
        lengths = set(len(v) for v in fc_vectors.values())
        if lengths != {19900}:
            print(f"    [WARN] Unexpected vector lengths: {lengths}")

        # Compute similarity matrix
        sim, subj_order = utils.fc_similarity_matrix(fc_vectors)

        # Save
        sim_path  = method_dir / "fc_similarity_matrix.npy"
        subj_path = method_dir / "fc_similarity_subjects.txt"

        np.save(sim_path, sim)
        with open(subj_path, "w") as f:
            f.write("\n".join(subj_order))

        # Report
        diag_mean = float(np.diag(sim).mean())
        off_mean  = float(sim[np.triu_indices_from(sim, k=1)].mean())
        print(f"    Saved: fc_similarity_matrix.npy  "
              f"({sim.shape[0]}×{sim.shape[1]})")
        print(f"    Diagonal mean={diag_mean:.4f} (should be 1.0), "
              f"off-diagonal mean={off_mean:.4f}")

    print("\n  Similarity matrices complete.")
    print(f"  Saved to: {F4_DIR}/{{method}}/fc_similarity_matrix.npy")


# =============================================================================
# MAIN
# =============================================================================

def process_subject(subj):
    print(f"\n{'═' * 60}")
    print(f"  F4 FC MATRICES | Subject: {subj}")
    print(f"{'═' * 60}")

    # Load valid vertices and atlas labels once (shared across methods)
    print("\n  [Setup] Loading valid vertices and atlas...")
    valid_L, valid_R = utils.get_valid_vertices()
    atlas_lh, atlas_rh = utils.load_atlas_labels(valid_L, valid_R)

    # Extract for all methods
    results = extract_subject(subj, valid_L, valid_R, atlas_lh, atlas_rh)

    print(f"\n  ✓ Subject {subj} complete — "
          f"{len(results)}/5 methods processed")


def main():
    parser = argparse.ArgumentParser(
        description="F4: FC matrix extraction (all 5 methods)")
    parser.add_argument("--subject",    type=str, default=None,
                        help="Single subject ID (for HPC array job)")
    parser.add_argument("--all",        action="store_true",
                        help="Process all subjects sequentially (local use)")
    parser.add_argument("--similarity", action="store_true",
                        help="Build similarity matrices (run after --all)")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  FEATURE F4: FUNCTIONAL CONNECTIVITY")
    print(f"{'═' * 60}")
    print(f"  Output: {F4_DIR}")
    print(f"  Methods: {list(cfg.METHODS.keys())}")

    if args.similarity:
        compute_similarity_matrices()

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
        compute_similarity_matrices()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()