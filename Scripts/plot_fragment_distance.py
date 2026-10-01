"""
FRAGMENT DISTANCE FIGURE
========================
Distribution of how far parcel fragments sit from the body they belong to.



The border/ectopic distinction rests on a threshold that is not settled.
Dworetsky et al. (2024) used 3.5 mm at network level and reported that
roughly a third of ectopic variants fell beyond 30 mm from their network
boundary; at areal level the appropriate value is likely lower, and it
depends on data quality and on spatial smoothing.



Rather than classify fragments against one cut-off, this shows the
distribution (A) and how the share called "ectopic" changes with the
threshold (B), so the reader can see where any threshold would fall.



Output: Outputs/<variant>/Figures/fragment_distance.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import config

# the folder names stay as they are on disk; only the displayed label
# of Method 5 is "Gradient-based", as in the manuscript
METHODS = [
    ("Method_1_gMSHBM", "gMSHBM", "#EE6677"),
    ("Method_2_AGP", "AGP", "#228833"),
    ("Method_3_SLIC_F", "SLIC-F", "#CCBB44"),
    ("Method_4_SLIC_C", "SLIC-C", "#AA3377"),
    ("Method_5_Gordon", "Gradient-based", "#66CCEE"),
]


THRESHOLDS = [3.5, 10, 15, 30]


# font sizes, matched to Figure 2
TICK_SIZE = 12
LABEL_SIZE = 14
TITLE_SIZE = 14
LEGEND_SIZE = 12
LETTER_SIZE = 18


def load_fragments(folder):
    d = config.OUTPUTS_DIR / "Variants" / folder / "Fragments"
    files = sorted(d.glob("*.csv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def style_axis(ax, letter):
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.12,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=LETTER_SIZE,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def main():
    argparse.ArgumentParser(
        description="Fragment distance figure."
    ).parse_args()
    out_dir = config.OUTPUTS_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    no_fragments = []
    for folder, label, colour in METHODS:
        df = load_fragments(folder)
        if df is None or df.empty:
            # a method with fully contiguous parcels has nothing to plot,
            # but it is still reported below as 0 %
            no_fragments.append(label)
            print(f"  {label}: 0 fragments")
            continue
        data[label] = (df, colour)
        print(
            f"  {label}: {len(df)} fragments from "
            f"{df['subject'].nunique()} participants"
        )

    if not data:
        print("no fragment files found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # A: where the fragments sit
    ax = axes[0]
    for label, (df, colour) in data.items():
        d = df["distance_mm"].dropna().values
        if d.size < 10:
            continue
        ax.hist(
            d,
            bins=np.linspace(0, 120, 61),
            density=True,
            histtype="step",
            linewidth=2.0,
            color=colour,
            label=label,
        )
    for t in THRESHOLDS:
        ax.axvline(t, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Distance (mm)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Density (log scale)", fontsize=LABEL_SIZE)
    ax.set_yscale("log")
    ax.set_title("Distance from the main body", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    style_axis(ax, "A")

    # B: how the share called ectopic depends on the threshold
    ax = axes[1]
    grid = np.linspace(1, 60, 120)
    for label, (df, colour) in data.items():
        d = df["distance_mm"].dropna().values
        if d.size < 10:
            continue
        ax.plot(
            grid,
            [(d >= t).mean() * 100 for t in grid],
            color=colour,
            linewidth=2.2,
            label=label,
        )
    for t in THRESHOLDS:
        ax.axvline(t, color="grey", linestyle=":", linewidth=1.0)
        ax.text(
            t,
            102,
            f"{t:g}",
            fontsize=11,
            ha="center",
            color="grey",
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )
    ax.set_xlabel("Threshold applied (mm)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Classified as ectopic (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 108)
    ax.set_title("Dependence on the threshold", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    style_axis(ax, "B")

    fig.tight_layout(w_pad=3)
    path = out_dir / "fragment_distance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  {path.name}")

    print("\nProportion beyond each candidate threshold (%):")
    print(f"{'method':15s} " + " ".join(f"{t:>8.1f}" for t in THRESHOLDS))
    for label, (df, _) in data.items():
        d = df["distance_mm"].dropna().values
        row = " ".join(f"{(d >= t).mean()*100:8.1f}" for t in THRESHOLDS)
        print(f"{label:15s} {row}")
    for label in no_fragments:
        row = " ".join(f"{0.0:8.1f}" for _ in THRESHOLDS)
        print(f"{label:15s} {row}   (no fragments)")


if __name__ == "__main__":
    main()
