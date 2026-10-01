"""
GRAPH METRICS FIGURE
====================
Graph-level properties of the parcellations.



    A  global efficiency of the weighted graph, by method
    B  coefficient of variation of parcel size, by method
    C  within-participant relationship between the two, across the six
       methods: one correlation per participant, computed over the six
       (CV, global efficiency) pairs that the methods produce for that
       participant



Panels A and B use the same raincloud style as the Stage 1 figure: a
half violin for the shape of the distribution, a narrow box for the
quartiles, and the individual participants as faint points.



Output: Outputs/<variant>/Figures/graph_metrics.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr


import config

# folder names stay as they are on disk; only the displayed label of
# Method 5 is "Gradient-based", as in the manuscript
METHODS = [
    ("Method_0_Schaefer", "Schaefer", "#4477AA"),
    ("Method_1_gMSHBM", "gMSHBM", "#EE6677"),
    ("Method_2_AGP", "AGP", "#228833"),
    ("Method_3_SLIC_F", "SLIC-F", "#CCBB44"),
    ("Method_4_SLIC_C", "SLIC-C", "#AA3377"),
    ("Method_5_Gordon", "Gradient-based", "#66CCEE"),
]


# font sizes, matched to the other figures
TICK_SIZE = 12
LABEL_SIZE = 14
LETTER_SIZE = 18


def load_features():
    """One row per participant and method."""
    frames = []
    for folder, label, _ in METHODS:
        files = sorted(
            (config.OUTPUTS_DIR / "Features" / folder).glob("*.csv")
        )
        if not files:
            continue
        parts = []
        for f in files:
            one = pd.read_csv(f)
            # the folder also holds similarity matrices and summary
            # tables; the per-participant feature files are the ones
            # with a single row and the expected columns
            ok = len(one) == 1
            ok = ok and "global_efficiency_raw" in one.columns
            ok = ok and "Unnamed: 0" not in one.columns
            if ok:
                parts.append(one)
        df = pd.concat(parts, ignore_index=True)
        df["label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def half_violin(ax, values, position, color, width=0.35):
    """Density curve drawn on one side of the position only."""
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


def add_letter(ax, letter, x=-0.14):
    ax.text(
        x,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=LETTER_SIZE,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def raincloud_panel(ax, table, column, label, rng, letter):
    positions = np.arange(len(METHODS))

    for i, (_, method, color) in enumerate(METHODS):
        values = table.loc[table["label"] == method, column].dropna().values
        if values.size == 0:
            continue

        half_violin(ax, values, i + 0.08, color)

        if np.ptp(values) > 0:
            ax.boxplot(
                values,
                positions=[i],
                widths=0.12,
                vert=True,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor="white", edgecolor="black", linewidth=0.8
                ),
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
    add_letter(ax, letter)


def correlation_panel(ax, table, letter):
    """One correlation per participant, computed across the six methods."""
    ge = table.pivot_table(
        index="subject", columns="label", values="global_efficiency_raw"
    )
    cv = table.pivot_table(
        index="subject", columns="label", values="parcel_size_cv"
    )
    order = [m[1] for m in METHODS if m[1] in ge.columns]
    ge, cv = ge[order].dropna(), cv[order].dropna()
    common = ge.index.intersection(cv.index)

    values = np.array([pearsonr(cv.loc[s], ge.loc[s])[0] for s in common])
    print(
        f"  within-participant r: mean {values.mean():.3f}, "
        f"SD {values.std():.3f}, negative in "
        f"{(values < 0).sum()} of {values.size}"
    )

    ax.hist(values, bins=40, color="#4477AA", alpha=0.8, linewidth=0)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(
        values.mean(),
        color="#AA3377",
        linestyle="--",
        linewidth=1.6,
        label=f"mean r = {values.mean():.2f}",
    )
    ax.set_xlabel(
        "Correlation between parcel size CV and global efficiency\n"
        "across the six methods, within participant",
        fontsize=LABEL_SIZE,
    )
    ax.set_ylabel("Participants", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    add_letter(ax, letter, x=-0.07)


def main():
    parser = argparse.ArgumentParser(description="Graph metrics figure.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = config.OUTPUTS_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_features()
    rng = np.random.default_rng(args.seed)

    print(f"Variant {config.VARIANT}")
    print(
        f"Participants per method: "
        f"{table.groupby('label').size().to_dict()}\n"
    )

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1, 0.85], hspace=0.45, wspace=0.3
    )

    raincloud_panel(
        fig.add_subplot(grid[0, 0]),
        table,
        "global_efficiency_raw",
        "Global efficiency",
        rng,
        "A",
    )
    raincloud_panel(
        fig.add_subplot(grid[0, 1]),
        table,
        "parcel_size_cv",
        "Parcel size CV",
        rng,
        "B",
    )
    correlation_panel(fig.add_subplot(grid[1, :]), table, "C")

    path = out_dir / "graph_metrics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  {path.name}")


if __name__ == "__main__":
    main()
