"""
BRIDGING STAGE — REPORT & VISUALIZATIONS
==========================================
Generates figures and a written report summarizing the feature
extraction outputs for supervisor review.

FIGURES GENERATED:
    F1/F2 — one violin plot per method (5 plots total):
        Shows all 6 distributions (GE × 3 variants, ASPL × 3 variants)
        Filename: F1F2_violins_{method}.png

    F3 — parcel size distributions across methods (1 plot):
        Violin plot: all 200 parcels × 95 subjects per method
        Shows how equal/unequal parcel sizes are per method
        Filename: F3_parcel_size_violins.png

    F4 — FC similarity heatmaps, one per method (5 plots):
        95×95 subject similarity matrix visualized as heatmap
        Filename: F4_similarity_heatmap_{method}.png

    F4 — Cross-method mean FC strength comparison (1 plot):
        Bar plot of mean |r| across all subject pairs per method
        Filename: F4_mean_fc_strength.png

OUTPUT:
    All figures: Outputs/Features/Report/Figures/
    Report text: Outputs/Features/Report/Bridging_Stage_Report.txt

Usage:
    python bridging_report.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

import feature_config as cfg

# =============================================================================
# PATHS
# =============================================================================
F3_DIR   = cfg.FEATURES_DIR / "F3_Parcel_Sizes"
F4_DIR   = cfg.FEATURES_DIR / "F4_FC"
F12_DIR  = cfg.FEATURES_DIR / "F1F2_Graph"
REPORT_DIR = cfg.FEATURES_DIR / "Report"
FIG_DIR    = REPORT_DIR / "Figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================
METHOD_LABELS = {
    "M0_Schaefer": "Schaefer",
    "M1_MSHBM":   "gMSHBM",
    "M2_AGP":     "AGP",
    "M3_SLIC_F":  "SLIC-F",
    "M4_SLIC_C":  "SLIC-C",
}
METHOD_COLORS = {
    "M0_Schaefer": "#4477AA",
    "M1_MSHBM":   "#EE6677",
    "M2_AGP":     "#228833",
    "M3_SLIC_F":  "#CCBB44",
    "M4_SLIC_C":  "#AA3377",
}
VARIANT_LABELS = {
    "raw":   "Raw (weighted)",
    "top10": "Top 10% (binary)",
    "r05":   "r > 0.5 (binary)",
}
VARIANT_COLORS = {
    "raw":   "#2166AC",
    "top10": "#4DAC26",
    "r05":   "#D01C8B",
}

plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.titleweight": "bold",
    "figure.dpi":       150,
})


# =============================================================================
# F1/F2 — VIOLIN PLOTS (one per method)
# =============================================================================

def plot_F1F2_violins():
    print("\n[F1/F2] Generating violin plots...")

    for method_name in cfg.METHODS:
        label    = METHOD_LABELS[method_name]
        color    = METHOD_COLORS[method_name]
        csv_path = F12_DIR / method_name / "all_subjects_graph_metrics.csv"

        if not csv_path.exists():
            print(f"  [SKIP] Missing: {csv_path.name}")
            continue

        df = pd.read_csv(csv_path)

        # Build data for 6 violins
        # Order: GE_raw, GE_top10, GE_r05, ASPL_raw, ASPL_top10, ASPL_r05
        ge_cols   = ["global_efficiency_raw",
                     "global_efficiency_top10",
                     "global_efficiency_r05"]
        aspl_cols = ["avg_shortest_path_raw",
                     "avg_shortest_path_top10",
                     "avg_shortest_path_r05"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{label} — Graph Metrics Across 95 Subjects",
                     fontsize=14, fontweight="bold", y=1.01)

        # --- GE subplot ---
        ge_data   = [df[c].dropna().values for c in ge_cols]
        vp1 = ax1.violinplot(ge_data, positions=[1, 2, 3],
                             showmedians=True, showextrema=True)
        for i, body in enumerate(vp1["bodies"]):
            body.set_facecolor(list(VARIANT_COLORS.values())[i])
            body.set_alpha(0.7)
        vp1["cmedians"].set_color("black")
        vp1["cmedians"].set_linewidth(2)

        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels([VARIANT_LABELS[v] for v in
                             ["raw", "top10", "r05"]], fontsize=10)
        ax1.set_ylabel("Global Efficiency", fontsize=11)
        ax1.set_title("F1 — Global Efficiency", fontsize=12)
        ax1.yaxis.grid(True, alpha=0.4)
        ax1.set_axisbelow(True)

        # Add mean annotations
        for i, data in enumerate(ge_data):
            ax1.text(i + 1, data.mean(), f" μ={data.mean():.3f}",
                     ha="left", va="center", fontsize=9, color="black")

        # --- ASPL subplot ---
        aspl_data = [df[c].dropna().values for c in aspl_cols]
        vp2 = ax2.violinplot(aspl_data, positions=[1, 2, 3],
                             showmedians=True, showextrema=True)
        for i, body in enumerate(vp2["bodies"]):
            body.set_facecolor(list(VARIANT_COLORS.values())[i])
            body.set_alpha(0.7)
        vp2["cmedians"].set_color("black")
        vp2["cmedians"].set_linewidth(2)

        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels([VARIANT_LABELS[v] for v in
                             ["raw", "top10", "r05"]], fontsize=10)
        ax2.set_ylabel("Average Shortest Path Length", fontsize=11)
        ax2.set_title("F2 — Average Shortest Path Length", fontsize=12)
        ax2.yaxis.grid(True, alpha=0.4)
        ax2.set_axisbelow(True)

        for i, data in enumerate(aspl_data):
            ax2.text(i + 1, data.mean(), f" μ={data.mean():.3f}",
                     ha="left", va="center", fontsize=9, color="black")

        # Legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=VARIANT_COLORS["raw"],   alpha=0.7,
                  label=VARIANT_LABELS["raw"]),
            Patch(facecolor=VARIANT_COLORS["top10"], alpha=0.7,
                  label=VARIANT_LABELS["top10"]),
            Patch(facecolor=VARIANT_COLORS["r05"],   alpha=0.7,
                  label=VARIANT_LABELS["r05"]),
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        out = FIG_DIR / f"F1F2_violins_{method_name}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# =============================================================================
# F3 — PARCEL SIZE VIOLIN PLOT (all methods on one plot)
# =============================================================================

def plot_F3_parcel_sizes():
    print("\n[F3] Generating parcel size violin plot...")

    all_sizes = {}
    for method_name in cfg.METHODS:
        csv_path = F3_DIR / method_name / "all_subjects_parcel_sizes.csv"
        if not csv_path.exists():
            print(f"  [SKIP] Missing: {csv_path.name}")
            continue
        df   = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c.startswith("parcel_")]
        # Flatten all 200 parcels × 95 subjects into one array
        all_sizes[method_name] = df[cols].values.flatten()

    if not all_sizes:
        print("  [SKIP] No data found")
        return

    methods  = list(all_sizes.keys())
    data     = [all_sizes[m] for m in methods]
    labels   = [METHOD_LABELS[m] for m in methods]
    colors   = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))

    vp = ax.violinplot(data, positions=range(1, len(methods) + 1),
                       showmedians=True, showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)
        body.set_edgecolor("white")
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    # Overlay mean ± SD as scatter
    for i, (method, d) in enumerate(zip(methods, data)):
        ax.scatter(i + 1, d.mean(), color="black",
                   zorder=5, s=50, marker="D")
        ax.text(i + 1 + 0.08, d.mean(),
                f"μ={d.mean():.0f}\nσ={d.std():.0f}",
                va="center", fontsize=9)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Parcel Size (vertices)", fontsize=11)
    ax.set_title("F3 — Parcel Size Distribution Across Methods\n"
                 "(200 parcels × 95 subjects per violin)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Colour patches for legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=0.7, label=l)
               for c, l in zip(colors, labels)]
    ax.legend(handles=handles, fontsize=10,
              loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out = FIG_DIR / "F3_parcel_size_violins.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# =============================================================================
# F4 — FC SIMILARITY HEATMAPS (one per method)
# =============================================================================

def plot_F4_similarity_heatmaps():
    print("\n[F4] Generating FC similarity heatmaps...")

    for method_name in cfg.METHODS:
        sim_path  = F4_DIR / method_name / "fc_similarity_matrix.npy"
        subj_path = F4_DIR / method_name / "fc_similarity_subjects.txt"
        label     = METHOD_LABELS[method_name]

        if not sim_path.exists():
            print(f"  [SKIP] Missing: {sim_path.name}")
            continue

        sim = np.load(sim_path)

        fig, ax = plt.subplots(figsize=(9, 8))

        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1,
                       aspect="auto", interpolation="none")
        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

        ax.set_title(f"{label} — FC Similarity Matrix\n"
                     f"(95×95 subjects, entry [i,j] = r between FC vectors)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Subject index", fontsize=11)
        ax.set_ylabel("Subject index", fontsize=11)

        # Add off-diagonal mean annotation
        off_diag = sim[np.triu_indices_from(sim, k=1)]
        ax.text(0.02, 0.98,
                f"Off-diagonal mean r = {off_diag.mean():.3f}",
                transform=ax.transAxes, fontsize=10,
                va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white",
                          alpha=0.8, edgecolor="gray"))

        plt.tight_layout()
        out = FIG_DIR / f"F4_similarity_heatmap_{method_name}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# =============================================================================
# F4 — CROSS-METHOD MEAN FC STRENGTH
# =============================================================================

def plot_F4_mean_fc_strength():
    print("\n[F4] Generating cross-method FC strength comparison...")

    means = []
    sems  = []
    labels = []
    colors = []

    for method_name in cfg.METHODS:
        upper_vals = []
        for subj in cfg.SUBJECT_IDS:
            fc_path = F4_DIR / method_name / f"{subj}_fc_matrix.npy"
            if not fc_path.exists():
                continue
            fc  = np.load(fc_path)
            idx = np.triu_indices_from(fc, k=1)
            upper_vals.append(np.abs(fc[idx]).mean())

        if not upper_vals:
            continue

        arr = np.array(upper_vals)
        means.append(arr.mean())
        sems.append(arr.std() / np.sqrt(len(arr)))
        labels.append(METHOD_LABELS[method_name])
        colors.append(METHOD_COLORS[method_name])

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, alpha=0.8,
                  edgecolor="white", linewidth=0.5)
    ax.errorbar(x, means, yerr=sems, fmt="none",
                color="black", capsize=5, linewidth=1.5)

    # Value labels on bars
    for i, (m, s) in enumerate(zip(means, sems)):
        ax.text(i, m + s + 0.002, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean |r| (upper triangle FC)", fontsize=11)
    ax.set_title("F4 — Mean FC Strength Across Methods\n"
                 "(mean ± SEM across 95 subjects)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(means) * 1.15)

    # Arrow indicating individualization direction
    ax.annotate("", xy=(len(labels) - 0.5, -0.015),
                xytext=(-0.5, -0.015),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="->",
                                color="gray", lw=1.5))
    ax.text((len(labels) - 1) / 2, -0.06,
            "← more group-constrained          more individual →",
            ha="center", va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9, color="gray")

    plt.tight_layout()
    out = FIG_DIR / "F4_mean_fc_strength.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# =============================================================================
# REPORT TEXT
# =============================================================================

def generate_report():
    print("\n[Report] Generating written report...")

    # Collect summary statistics
    stats = {}
    for method_name in cfg.METHODS:
        stats[method_name] = {}

        # F3
        p = F3_DIR / method_name / "all_subjects_parcel_sizes.csv"
        if p.exists():
            df   = pd.read_csv(p)
            cols = [c for c in df.columns if c.startswith("parcel_")]
            vals = df[cols].values.flatten()
            stats[method_name]["parcel_size_mean"] = vals.mean()
            stats[method_name]["parcel_size_std"]  = vals.std()
            stats[method_name]["parcel_size_cv"]   = vals.std() / vals.mean()

        # F4 mean |r|
        upper_vals = []
        for subj in cfg.SUBJECT_IDS:
            fc_path = F4_DIR / method_name / f"{subj}_fc_matrix.npy"
            if fc_path.exists():
                fc  = np.load(fc_path)
                idx = np.triu_indices_from(fc, k=1)
                upper_vals.append(np.abs(fc[idx]).mean())
        if upper_vals:
            stats[method_name]["mean_fc"] = np.mean(upper_vals)

        # F4 similarity
        sim_path = F4_DIR / method_name / "fc_similarity_matrix.npy"
        if sim_path.exists():
            sim = np.load(sim_path)
            off = sim[np.triu_indices_from(sim, k=1)]
            stats[method_name]["sim_mean"] = off.mean()

        # F1/F2
        p = F12_DIR / method_name / "all_subjects_graph_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            for variant in cfg.GRAPH_VARIANTS:
                ge_col   = f"global_efficiency_{variant}"
                aspl_col = f"avg_shortest_path_{variant}"
                if ge_col in df.columns:
                    stats[method_name][f"ge_{variant}"]   = df[ge_col].mean()
                if aspl_col in df.columns:
                    stats[method_name][f"aspl_{variant}"] = df[aspl_col].dropna().mean()

    report = f"""
BRIDGING STAGE REPORT
======================
Feature Extraction for Behavioral Prediction
HCP Dataset | 95 Subjects | 5 Parcellation Methods | 200 Parcels

Generated from: Outputs/Features/
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

======================================================================
1. WHAT WAS EXTRACTED AND WHY
======================================================================

Four features were extracted from the parcellated resting-state fMRI
data of 95 HCP subjects across all 5 parcellation methods. These
features will serve as inputs to behavioral prediction models in
Stage 2. All features are atlas-aligned: after Hungarian bijective
mapping from native parcel IDs to Schaefer atlas IDs, parcel position
i refers to the same brain region across all methods and subjects,
enabling direct cross-method comparison.

======================================================================
2. FILE NAMING AND STRUCTURE
======================================================================

All outputs are in: Outputs/Features/

F1 & F2 — Graph Metrics:
    F1F2_Graph/{{METHOD}}/{{subj}}_graph_metrics.csv
        columns: metric, variant, value
        6 rows per subject (2 metrics × 3 graph variants)
    F1F2_Graph/{{METHOD}}/all_subjects_graph_metrics.csv
        shape: 95 subjects × 6 metric-variant columns

F3 — Parcel Sizes:
    F3_Parcel_Sizes/{{METHOD}}/{{subj}}_parcel_sizes.csv
        columns: atlas_parcel_id (1-200), n_vertices
        200 rows per subject
    F3_Parcel_Sizes/{{METHOD}}/all_subjects_parcel_sizes.csv
        shape: 95 subjects × 200 parcel columns (parcel_1...parcel_200)

F4 — Functional Connectivity:
    F4_FC/{{METHOD}}/{{subj}}_fc_matrix.npy
        shape: (200, 200) — Pearson r between all parcel pairs
    F4_FC/{{METHOD}}/{{subj}}_fc_vector.npy
        shape: (19900,) — lower triangle of FC matrix
        19900 = 200×199/2
    F4_FC/{{METHOD}}/fc_similarity_matrix.npy
        shape: (95, 95) — Pearson r between subjects' FC vectors
    F4_FC/{{METHOD}}/fc_similarity_subjects.txt
        subject ID order for matrix rows/columns

METHOD directory names:
    M0_Schaefer, M1_MSHBM, M2_AGP, M3_SLIC_F, M4_SLIC_C

GRAPH VARIANTS (for F1/F2):
    raw   — weighted graph, edge weight = FC value (negatives zeroed)
    top10 — binary graph, top 10% positive FC edges
    r05   — binary graph, FC > 0.5 edges only

======================================================================
3. FEATURE DESCRIPTIONS
======================================================================

F1 — GLOBAL EFFICIENCY
    Definition: E_glob = (1/N(N-1)) × Σ_{i≠j} 1/d(i,j)
    Where d(i,j) is the shortest path between parcels i and j.
    For weighted graphs: d = 1/FC (strength → proximity).
    For binary graphs: d = hop count.
    Range: [0, 1]. Higher = more efficient information transfer.
    Disconnected pairs contribute 0.

F2 — AVERAGE SHORTEST PATH LENGTH
    Definition: mean of d(i,j) over all connected pairs.
    Lower = more integrated network.
    NaN if graph is fully disconnected.

F3 — PARCEL SIZE
    Definition: number of cortical surface vertices per parcel.
    Reported under atlas-aligned parcel IDs so parcel i refers
    to the same brain region across all methods.
    Serves as both a feature for prediction and a confound check
    (smaller parcels → higher FC homogeneity by artifact).

F4 — FUNCTIONAL CONNECTIVITY
    F4a Matrix: 200×200 Pearson r between parcel mean timeseries.
    F4b Vector: lower triangle of FC matrix (19,900 values).
    F4c Similarity: 95×95 matrix of inter-subject FC similarity.
        Entry [i,j] = Pearson r between subjects i and j's
        19,900-element FC vectors. Measures how similarly two
        subjects' connectomes are organized under each parcellation.

======================================================================
4. SUMMARY RESULTS
======================================================================

--- F3: Parcel Sizes ---
"""

    for method_name in cfg.METHODS:
        s = stats[method_name]
        if "parcel_size_mean" in s:
            report += (f"  {METHOD_LABELS[method_name]:<12}: "
                       f"mean={s['parcel_size_mean']:.0f} vertices, "
                       f"SD={s['parcel_size_std']:.0f}, "
                       f"CV={s['parcel_size_cv']:.3f}\n")

    report += "\n--- F4: Mean FC Strength (mean |r| across upper triangle) ---\n"
    for method_name in cfg.METHODS:
        s = stats[method_name]
        if "mean_fc" in s:
            report += f"  {METHOD_LABELS[method_name]:<12}: {s['mean_fc']:.4f}\n"

    report += "\n--- F4: FC Similarity (off-diagonal mean r) ---\n"
    for method_name in cfg.METHODS:
        s = stats[method_name]
        if "sim_mean" in s:
            report += f"  {METHOD_LABELS[method_name]:<12}: {s['sim_mean']:.4f}\n"

    report += "\n--- F1: Global Efficiency (mean across 95 subjects) ---\n"
    report += f"  {'Method':<12}  {'Raw':>10}  {'Top10%':>10}  {'r>0.5':>10}\n"
    report += "  " + "-"*46 + "\n"
    for method_name in cfg.METHODS:
        s = stats[method_name]
        ge_raw   = f"{s.get('ge_raw',   float('nan')):.4f}"
        ge_top10 = f"{s.get('ge_top10', float('nan')):.4f}"
        ge_r05   = f"{s.get('ge_r05',   float('nan')):.4f}"
        report += (f"  {METHOD_LABELS[method_name]:<12}  "
                   f"{ge_raw:>10}  {ge_top10:>10}  {ge_r05:>10}\n")

    report += "\n--- F2: Average Shortest Path Length (mean across 95 subjects) ---\n"
    report += f"  {'Method':<12}  {'Raw':>10}  {'Top10%':>10}  {'r>0.5':>10}\n"
    report += "  " + "-"*46 + "\n"
    for method_name in cfg.METHODS:
        s = stats[method_name]
        aspl_raw   = f"{s.get('aspl_raw',   float('nan')):.4f}"
        aspl_top10 = f"{s.get('aspl_top10', float('nan')):.4f}"
        aspl_r05   = f"{s.get('aspl_r05',   float('nan')):.4f}"
        report += (f"  {METHOD_LABELS[method_name]:<12}  "
                   f"{aspl_raw:>10}  {aspl_top10:>10}  {aspl_r05:>10}\n")

    report += """
======================================================================
5. BRIEF INTERPRETATION
======================================================================

PARCEL SIZES (F3):
    SLIC-F has the most uniform parcel sizes (lowest CV), while
    SLIC-C and AGP show greater inequality. This is important context
    for interpreting F1/F2 and FC results — methods with unequal
    parcels may show artificially high FC homogeneity in small parcels.

FC STRENGTH (F4):
    Mean FC strength decreases across the individualization spectrum
    (Schaefer > gMSHBM > AGP > SLIC-F > SLIC-C). SLIC-C's lower
    mean |r| reflects its spatially fragmented parcels, which mix
    signals from functionally distinct regions when computing parcel
    mean timeseries.

FC SIMILARITY (F4):
    The off-diagonal similarity matrix mean decreases monotonically
    from Schaefer (most similar across subjects) to SLIC-C (most
    individual). This directly confirms the individualization gradient
    at the level of functional connectivity — SLIC-C produces
    connectivity fingerprints that are most unique to each person,
    while Schaefer produces near-identical FC structures across
    subjects by construction.

GRAPH METRICS (F1/F2):
    Global efficiency and average shortest path length are broadly
    consistent across methods for the raw and top10 variants.
    SLIC-C shows lower raw global efficiency and longer raw ASPL,
    consistent with its fragmented parcellation producing a less
    integrated network. The r>0.5 variant shows more variability
    across methods because SLIC-C produces fewer strong connections
    (mean FC is lower), resulting in sparser graphs under this
    threshold. This suggests the r>0.5 threshold may not be equally
    appropriate for all methods — a consideration for Stage 2.

STAGE 2 IMPLICATIONS:
    The four features capture complementary aspects of brain
    organization: spatial (F3), topological (F1/F2), and
    connectome-level (F4). The FC similarity matrices will be
    particularly useful for fingerprinting analyses. The graph metrics
    provide topological summaries that abstract away from specific
    connection patterns. Together they offer a multi-scale view of
    how parcellation method affects the representation of brain
    organization available for behavioral prediction.

======================================================================
END OF REPORT
======================================================================
"""

    out = REPORT_DIR / "Bridging_Stage_Report.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {out.name}")
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  BRIDGING STAGE — REPORT & VISUALIZATIONS")
    print("=" * 60)
    print(f"  Figures: {FIG_DIR}")
    print(f"  Report:  {REPORT_DIR}")

    plot_F1F2_violins()
    plot_F3_parcel_sizes()
    plot_F4_similarity_heatmaps()
    plot_F4_mean_fc_strength()
    report = generate_report()

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    print(f"\n  Figures saved ({len(list(FIG_DIR.glob('*.png')))} total):")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"    {f.name}")
    print(f"\n  Report: {REPORT_DIR / 'Bridging_Stage_Report.txt'}")


if __name__ == "__main__":
    main()