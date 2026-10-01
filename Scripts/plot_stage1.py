"""
STAGE 1 FIGURES
===============
Raincloud plots of the validation metrics across methods.



Each panel shows one metric: a half violin for the shape of the
distribution, a narrow box for the quartiles, and the individual
subjects as points. With 947 subjects the points are drawn small and
faint, since at that density they are there to show the spread rather
than to be read one by one.



Colours follow Paul Tol's qualitative palette, which stays legible in
greyscale and for the common forms of colour blindness.



Output: Outputs/<variant>/Figures/
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
    ("Method_0_Schaefer", "Schaefer", "#4477AA"),
    ("Method_1_gMSHBM", "gMSHBM", "#EE6677"),
    ("Method_2_AGP", "AGP", "#228833"),
    ("Method_3_SLIC_F", "SLIC-F", "#CCBB44"),
    ("Method_4_SLIC_C", "SLIC-C", "#AA3377"),
    ("Method_5_Gordon", "Gradient-based", "#66CCEE"),
]


# the gradient-based method is read at 200 parcels so every method is
# shown at the same resolution
VALIDATION_FILE = {"Method_5_Gordon": "validation_all_subjects_200.csv"}


METRICS = [
    ("homogeneity_ALL", "Functional homogeneity"),
    ("fc_test_retest", "Test-retest reliability (r)"),
    ("pct_contiguous", "Spatial contiguity (%)"),
    ("intersubject_dice", "Inter-individual Dice"),
]


PANEL_LETTERS = ["A", "B", "C", "D"]


# font sizes
TICK_SIZE = 12
LABEL_SIZE = 14
LETTER_SIZE = 18


def load_all():
    """One long table with every method's per-subject metrics."""
    frames = []
    for folder, label, _ in METHODS:
        name = VALIDATION_FILE.get(folder, "validation_all_subjects.csv")
        path = config.OUTPUTS_DIR / folder / name
        if not path.exists():
            continue

        df = pd.read_csv(path)

        # Schaefer shares one atlas across subjects, so its inter-subject
        # Dice is 1 by construction and is never computed
        if "intersubject_dice" not in df.columns:
            df["intersubject_dice"] = 1.0

        df["method"] = label
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def half_violin(ax, values, position, color, width=0.35):
    """
    Density curve drawn on one side of the position only.



    The other side is left free for the points, which is what makes a
    raincloud readable: the distribution and the raw data sit next to
    each other rather than on top of one another.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, values[0]):
        # a constant metric has no density to draw; mark the value
        ax.plot(
            [position, position + width],
            [values[0], values[0]],
            color=color,
            linewidth=2,
        )
        return

    from scipy.stats import gaussian_kde

    kde = gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), 200)
    density = kde(grid)
    density = density / density.max() * width

    ax.fill_betweenx(
        grid,
        position,
        position + density,
        facecolor=color,
        alpha=0.6,
        linewidth=0,
    )
    ax.plot(position + density, grid, color=color, linewidth=1)


def raincloud_panel(ax, long_df, metric, label, rng, letter=None):
    positions = np.arange(len(METHODS))

    for i, (_, method, color) in enumerate(METHODS):
        values = (
            long_df.loc[long_df["method"] == method, metric].dropna().values
        )
        if values.size == 0:
            continue

        half_violin(ax, values, i + 0.08, color)

        ax.boxplot(
            values,
            positions=[i],
            widths=0.12,
            vert=True,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )

        jitter = rng.uniform(-0.09, 0.09, values.size)
        ax.scatter(
            np.full(values.size, i - 0.20) + jitter,
            values,
            s=2,
            color=color,
            alpha=0.12,
            linewidths=0,
            zorder=1,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [m[1] for m in METHODS], rotation=25, ha="right", fontsize=TICK_SIZE
    )
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylabel(label, fontsize=LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    # panel letter in the top-left corner, outside the plotting area
    if letter is not None:
        ax.text(
            -0.14,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=LETTER_SIZE,
            fontweight="bold",
            ha="right",
            va="bottom",
        )


def main():
    parser = argparse.ArgumentParser(description="Stage 1 figures.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = config.OUTPUTS_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_all()
    rng = np.random.default_rng(args.seed)

    print(f"Variant {config.VARIANT}")
    print(
        f"Subjects per method: "
        f"{long_df.groupby('method').size().to_dict()}\n"
    )

    # one panel per metric, as a single figure (no overall title; the
    # caption in the manuscript does that job)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (metric, label), letter in zip(
        axes.ravel(), METRICS, PANEL_LETTERS
    ):
        raincloud_panel(ax, long_df, metric, label, rng, letter=letter)

    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    path = out_dir / "stage1_validation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.name}")

    # and each panel on its own, for slides or supplements
    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        raincloud_panel(ax, long_df, metric, label, rng)
        fig.tight_layout()
        path = out_dir / f"stage1_{metric}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
