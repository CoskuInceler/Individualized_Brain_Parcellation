"""
VARIANT ANALYSIS FIGURE
=======================
Shows what each method can and cannot report about fragmented parcels.



Two panels. The first counts fragments per participant, split into
border shifts and ectopic fragments, on a log scale because the methods
differ by orders of magnitude. The second shows how far the ectopic
fragments sit from the parcel they belong to, which is what separates a
displaced boundary from a genuinely relocated piece of cortex.



Methods that produce no fragments at all are shown as empty bars rather
than omitted, since their absence is the finding.



Output: Outputs/<variant>/Figures/stage1_variants.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import config

METHODS = [
    ("Method_0_Schaefer", "Schaefer"),
    ("Method_1_gMSHBM", "gMSHBM"),
    ("Method_2_AGP", "AGP"),
    ("Method_3_SLIC_F", "SLIC-F"),
    ("Method_4_SLIC_C", "SLIC-C"),
    ("Method_5_Gordon", "Gordon"),
]


ECTOPIC_MM = 30.0


def load():
    frames = []
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
        df["label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def counts_panel(ax, df):
    labels = [m[1] for m in METHODS]
    x = np.arange(len(labels))
    width = 0.38

    border = [df.loc[df["label"] == l, "n_border"].mean() for l in labels]
    ectopic = [df.loc[df["label"] == l, "n_ectopic"].mean() for l in labels]

    # a log axis cannot show zero, so empty categories are drawn at the
    # floor and left visibly empty
    floor = 0.5
    b = ax.bar(
        x - width / 2,
        [max(v, floor) for v in border],
        width,
        label="Border shifts",
        color="#4477AA",
        alpha=0.85,
    )
    e = ax.bar(
        x + width / 2,
        [max(v, floor) for v in ectopic],
        width,
        label="Ectopic",
        color="#EE6677",
        alpha=0.85,
    )

    for bars, values in ((b, border), (e, ectopic)):
        for bar, value in zip(bars, values):
            if value < 0.05:
                bar.set_facecolor("white")
                bar.set_edgecolor("grey")
                bar.set_linewidth(0.8)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    floor * 1.15,
                    "0",
                    ha="center",
                    fontsize=7,
                    color="grey",
                )

    ax.set_yscale("log")
    ax.set_ylim(floor, max(max(border), max(ectopic)) * 3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fragments per participant (log scale)", fontsize=10)
    ax.set_title("How many fragments each method reports", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def distance_panel(ax, df):
    labels = [m[1] for m in METHODS]
    values, positions, colors = [], [], []

    for i, l in enumerate(labels):
        d = df.loc[df["label"] == l, "mean_ectopic_distance"].dropna()
        if d.size:
            values.append(d.values)
            positions.append(i)

    if values:
        parts = ax.violinplot(
            values, positions=positions, widths=0.6, showmedians=True
        )
        for body in parts["bodies"]:
            body.set_facecolor("#EE6677")
            body.set_alpha(0.6)

    ax.axhline(ECTOPIC_MM, color="black", linestyle="--", linewidth=0.8)
    ax.text(
        len(labels) - 0.5,
        ECTOPIC_MM + 2,
        "ectopic threshold",
        fontsize=7,
        ha="right",
        color="black",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Distance to parent parcel (mm)", fontsize=10)
    ax.set_title("How far the ectopic fragments sit", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)


def main():
    argparse.ArgumentParser(description="Variant figure.").parse_args()

    out_dir = config.OUTPUTS_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts_panel(axes[0], df)
    distance_panel(axes[1], df)
    fig.tight_layout()

    path = out_dir / "stage1_variants.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Variant {config.VARIANT}\n  {path.name}")


if __name__ == "__main__":
    main()
