"""
FEATURES F1 & F2 — GRAPH METRICS
===================================
Computes global efficiency (F1) and average shortest path length (F2)
for all 5 parcellation methods across 3 graph variants.

Both features require building a graph from the FC matrix, so they
are computed together in one script to avoid redundant computation.

GRAPH VARIANTS:
    raw   : weighted graph, edge weight = FC value (negatives → 0)
    top10 : binary graph, top 10% of positive FC edges kept
    r05   : binary graph, edges where FC > 0.5 kept

For weighted graphs (raw):
    - Edge distance = 1 / FC value (strength → proximity)
    - Dijkstra shortest paths
    - Negative FC → excluded (set to 0)

For binary graphs (top10, r05):
    - BFS shortest paths
    - Distance = hop count

IMPORTANT — FC MATRICES REUSED FROM F4:
    This script loads the atlas-aligned FC matrices saved by
    feat_F4_fc.py rather than recomputing them. F4 must be
    run before F1/F2.

OUTPUT: 3 values per metric × 3 variants × 5 methods × 95 subjects

OUTPUTS:
    Per subject per method:
        Outputs/Features/F1F2_Graph/{METHOD}/{subj}_graph_metrics.csv
            columns: metric, variant, value
            rows: 6 (2 metrics × 3 variants)

    After --aggregate:
        Outputs/Features/F1F2_Graph/{METHOD}/all_subjects_graph_metrics.csv
            columns: subject_id,
                     global_efficiency_raw, global_efficiency_top10, global_efficiency_r05,
                     avg_shortest_path_raw, avg_shortest_path_top10, avg_shortest_path_r05

Usage:
    # Single subject (HPC array job):
    python feat_F1F2_graph_metrics.py --subject 100307

    # All subjects locally:
    python feat_F1F2_graph_metrics.py --all

    # After all subjects done — build master table:
    python feat_F1F2_graph_metrics.py --aggregate
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import feature_config as cfg
import feature_utils  as utils

# Output directory for this feature
F12_DIR = cfg.FEATURES_DIR / "F1F2_Graph"
for method_name in cfg.METHODS:
    (F12_DIR / method_name).mkdir(parents=True, exist_ok=True)

# FC matrices are loaded from F4 outputs
F4_DIR = cfg.FEATURES_DIR / "F4_FC"


# =============================================================================
# CORE: GRAPH METRICS FOR ONE SUBJECT
# =============================================================================

def extract_subject(subj):
    """
    Compute graph metrics for all 5 methods for one subject.

    Loads atlas-aligned FC matrices from F4 outputs.
    Builds 3 graph variants per method.
    Computes global efficiency and average shortest path length.

    Saves one CSV per method:
        {F12_DIR}/{method_name}/{subj}_graph_metrics.csv

    Returns:
        dict {method_name: dict {metric_variant: value}}
    """
    results = {}

    for method_name in cfg.METHODS:
        out_path = F12_DIR / method_name / f"{subj}_graph_metrics.csv"

        print(f"\n    [{method_name}]")

        # --- Load FC matrix from F4 outputs ---
        fc_path = F4_DIR / method_name / f"{subj}_fc_matrix.npy"

        if not fc_path.exists():
            print(f"      [SKIP] FC matrix not found: {fc_path.name}")
            print(f"      Run feat_F4_fc.py first.")
            continue

        fc = np.load(fc_path)

        if fc.shape != (200, 200):
            print(f"      [SKIP] Unexpected FC shape: {fc.shape}")
            continue

        # --- Build all 3 graph variants ---
        graphs = utils.build_all_graphs(fc)

        # --- Compute F1 + F2 for each variant ---
        rows = []
        method_results = {}

        for variant, adj in graphs.items():
            n_edges = int((adj > 0).sum()) // 2
            print(f"      [{variant}] {n_edges} edges", end="")

            # F1: Global Efficiency
            ge = utils.global_efficiency(adj)

            # F2: Average Shortest Path Length
            aspl = utils.average_shortest_path_length(adj)

            print(f"  |  GE={ge:.6f}  "
                  + (f"ASPL={aspl:.6f}" if not np.isnan(aspl) else "ASPL=NaN"))

            rows.append({
                "metric":  "global_efficiency",
                "variant": variant,
                "value":   round(ge, 8)
            })
            rows.append({
                "metric":  "avg_shortest_path",
                "variant": variant,
                "value":   round(aspl, 8) if not np.isnan(aspl) else np.nan
            })

            method_results[f"global_efficiency_{variant}"] = round(ge, 8)
            method_results[f"avg_shortest_path_{variant}"] = (
                round(aspl, 8) if not np.isnan(aspl) else np.nan)

        # Save per-subject CSV
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"      Saved: {out_path.name}")

        results[method_name] = method_results

    return results


# =============================================================================
# AGGREGATE: BUILD MASTER TABLE PER METHOD
# =============================================================================

def aggregate():
    """
    After all subjects are processed, build one master CSV per method.

    Master CSV shape: (n_subjects, 7)
    Columns: subject_id,
             global_efficiency_raw, global_efficiency_top10, global_efficiency_r05,
             avg_shortest_path_raw, avg_shortest_path_top10, avg_shortest_path_r05
    """
    print("\n" + "=" * 60)
    print("AGGREGATING F1/F2 GRAPH METRICS")
    print("=" * 60)

    value_cols = [
        "global_efficiency_raw",
        "global_efficiency_top10",
        "global_efficiency_r05",
        "avg_shortest_path_raw",
        "avg_shortest_path_top10",
        "avg_shortest_path_r05",
    ]

    for method_name in cfg.METHODS:
        method_dir = F12_DIR / method_name
        print(f"\n  {method_name}...")

        rows = []
        subjects_found = []

        for subj in cfg.SUBJECT_IDS:
            csv_path = method_dir / f"{subj}_graph_metrics.csv"
            if not csv_path.exists():
                print(f"    [WARN] Missing: {csv_path.name}")
                continue

            df = pd.read_csv(csv_path)

            # Build one row from the per-subject CSV
            row = {"subject_id": subj}
            for _, r in df.iterrows():
                col_name = f"{r['metric']}_{r['variant']}"
                row[col_name] = r["value"]

            # Ensure all expected columns present
            for col in value_cols:
                if col not in row:
                    row[col] = np.nan

            rows.append(row)
            subjects_found.append(subj)

        if not rows:
            print(f"    [WARN] No subjects found for {method_name}")
            continue

        master = pd.DataFrame(rows)[["subject_id"] + value_cols]
        out_path = method_dir / "all_subjects_graph_metrics.csv"
        master.to_csv(out_path, index=False)

        print(f"    Saved: {out_path.name}  "
              f"({len(subjects_found)} subjects × 6 metrics)")

        # Quick summary
        for col in value_cols:
            vals = master[col].dropna()
            if len(vals) > 0:
                print(f"    {col:<35}: "
                      f"mean={vals.mean():.6f}, "
                      f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    print("\n  Aggregation complete.")
    print(f"  Master tables saved to: {F12_DIR}/{{method}}/all_subjects_graph_metrics.csv")


# =============================================================================
# MAIN
# =============================================================================

def process_subject(subj):
    print(f"\n{'═' * 60}")
    print(f"  F1/F2 GRAPH METRICS | Subject: {subj}")
    print(f"{'═' * 60}")

    results = extract_subject(subj)

    print(f"\n  ✓ Subject {subj} complete — "
          f"{len(results)}/5 methods processed")


def main():
    parser = argparse.ArgumentParser(
        description="F1/F2: Graph metrics (all 5 methods, 3 variants)")
    parser.add_argument("--subject",   type=str, default=None,
                        help="Single subject ID (for HPC array job)")
    parser.add_argument("--all",       action="store_true",
                        help="Process all subjects sequentially (local use)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Build master tables after all subjects done")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  FEATURES F1 & F2: GRAPH METRICS")
    print(f"{'═' * 60}")
    print(f"  Output:   {F12_DIR}")
    print(f"  Methods:  {list(cfg.METHODS.keys())}")
    print(f"  Variants: {cfg.GRAPH_VARIANTS}")
    print(f"  Requires: F4 FC matrices in {F4_DIR}")

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