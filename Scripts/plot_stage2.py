"""
STAGE 2 FIGURES
===============
Three views of the brain-behaviour results.



    SEM paths        standardized regression coefficients, method by
                     predictor, with the ones surviving correction marked
    Unique variance  how much of g each predictor accounts for on its own
    KRR accuracy     how well g can be predicted from each kernel



The combined figure places the heatmap on its own row and the two bar
panels below it, so that nothing has to be shrunk to fit the page width.



Output: Outputs/<variant>/Figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import config

# "Gordon" is the name used in the data files; only the displayed label
# is "Gradient-based", as in the manuscript
METHOD_ORDER = ["Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C", "Gordon"]
METHOD_LABEL = {
    "SLIC_F": "SLIC-F",
    "SLIC_C": "SLIC-C",
    "Gordon": "Gradient-based",
}


METHOD_COLORS = {
    "Schaefer": "#4477AA",
    "gMSHBM": "#EE6677",
    "AGP": "#228833",
    "SLIC_F": "#CCBB44",
    "SLIC_C": "#AA3377",
    "Gordon": "#66CCEE",
}


PREDICTOR_LABEL = {
    "global_efficiency_raw": "Global efficiency",
    "parcel_size_cv": "Parcel size CV",
    "parcel_size_similarity": "Parcel size similarity",
    "fc_similarity": "FC similarity",
}


# font sizes, matched to the other figures
TICK_SIZE = 12
LABEL_SIZE = 14
TITLE_SIZE = 13
LEGEND_SIZE = 11
CELL_SIZE = 11
LETTER_SIZE = 18


def label(name):
    return METHOD_LABEL.get(name, name)


def load(filename):
    path = config.OUTPUTS_DIR / "Stage2" / filename
    return pd.read_csv(path) if path.exists() else None


def add_letter(ax, letter, x=-0.14):
    """Bold panel letter in the top-left corner, outside the axes."""
    if letter is None:
        return
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


def sem_heatmap(ax, letter=None):
    """
    Standardized paths for the raw graph variant, method by predictor.



    Cells that survived correction are annotated with an asterisk, so
    the reader can separate a path that is merely large from one the
    data actually supports.
    """
    df = load("sem_paths.csv")
    if df is None:
        return False

    # the cv and meansim models each contribute one parcel-size term,
    # so both are pulled in and the rest taken from either
    df = df[df["model"].str.startswith("raw")]

    predictors = list(PREDICTOR_LABEL)
    matrix = np.full((len(METHOD_ORDER), len(predictors)), np.nan)
    stars = np.empty(matrix.shape, dtype=object)
    stars[:] = ""

    for i, method in enumerate(METHOD_ORDER):
        for j, pred in enumerate(predictors):
            rows = df[(df["method"] == method) & (df["predictor"] == pred)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            matrix[i, j] = row["beta"]
            if row["p_fdr"] < 0.05:
                stars[i, j] = "*"

    limit = np.nanmax(np.abs(matrix))
    im = ax.imshow(
        matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    color="grey",
                    fontsize=CELL_SIZE + 1,
                )
            else:
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}{stars[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=CELL_SIZE,
                    color=(
                        "black" if abs(matrix[i, j]) < limit * 0.6 else "white"
                    ),
                )

    ax.set_xticks(range(len(predictors)))
    ax.set_xticklabels(
        [PREDICTOR_LABEL[p] for p in predictors],
        rotation=25,
        ha="right",
        fontsize=TICK_SIZE,
    )
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([label(m) for m in METHOD_ORDER], fontsize=TICK_SIZE)
    ax.set_title(
        "Standardized paths to g\n(* survives FDR correction)",
        fontsize=TITLE_SIZE,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    # the method names on the left are long, so the letter sits further out
    add_letter(ax, letter, x=-0.30)
    return True


def unique_variance_bars(ax, letter=None):
    """Unique contribution of each predictor, stacked by method."""
    df = load("sem_unique_variance.csv")
    if df is None:
        return False

    predictors = [p for p in PREDICTOR_LABEL if p in df["predictor"].unique()]
    width = 0.8 / len(predictors)
    x = np.arange(len(METHOD_ORDER))

    shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(predictors)))

    for k, pred in enumerate(predictors):
        values = []
        for method in METHOD_ORDER:
            row = df[(df["method"] == method) & (df["predictor"] == pred)]
            values.append(
                row["unique_variance"].iloc[0] if not row.empty else np.nan
            )
        ax.bar(
            x + k * width - 0.4 + width / 2,
            values,
            width,
            label=PREDICTOR_LABEL[pred],
            color=shades[k],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [label(m) for m in METHOD_ORDER],
        rotation=25,
        ha="right",
        fontsize=TICK_SIZE,
    )
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylabel("Unique variance in g", fontsize=LABEL_SIZE)
    ax.set_title("Variance only that predictor explains", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    add_letter(ax, letter)
    return True


def krr_bars(ax, letter=None):
    """Prediction accuracy per method, one bar per kernel."""
    df = load("krr_results.csv")
    if df is None:
        return False

    kernels = ["fc", "sizes"]
    kernel_label = {"fc": "Connectivity", "sizes": "Parcel size"}
    width = 0.38
    x = np.arange(len(METHOD_ORDER))

    for k, kern in enumerate(kernels):
        values, marks = [], []
        for method in METHOD_ORDER:
            row = df[(df["method"] == method) & (df["kernel"] == kern)]
            if row.empty:
                values.append(np.nan)
                marks.append("")
            else:
                values.append(row["r"].iloc[0])
                marks.append("*" if row["p_fdr"].iloc[0] < 0.05 else "")

        offset = (k - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=kernel_label[kern],
            color=["#4477AA", "#EE6677"][k],
            alpha=0.85,
        )

        for bar, mark, value in zip(bars, marks, values):
            if mark and np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    mark,
                    ha="center",
                    fontsize=14,
                )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label(m) for m in METHOD_ORDER],
        rotation=25,
        ha="right",
        fontsize=TICK_SIZE,
    )
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylabel("Prediction accuracy (r)", fontsize=LABEL_SIZE)
    ax.set_title(
        "Predicting g by kernel ridge regression\n"
        "(* survives FDR correction)",
        fontsize=TITLE_SIZE,
    )
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    add_letter(ax, letter)
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 2 figures.")
    parser.parse_args()

    out_dir = config.OUTPUTS_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Variant {config.VARIANT}")

    # each panel on its own, without a letter, for slides or supplements
    panels = [
        ("stage2_sem_paths", sem_heatmap, (7.5, 5.5)),
        ("stage2_unique_variance", unique_variance_bars, (8, 5.5)),
        ("stage2_krr", krr_bars, (8, 5.5)),
    ]

    for name, builder, size in panels:
        fig, ax = plt.subplots(figsize=size)
        ok = builder(ax)
        if not ok:
            plt.close(fig)
            print(f"  {name}: inputs missing")
            continue
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{name}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        print(f"  {name}.png")

    # the three together: heatmap on its own row, bars below, so the
    # figure is closer to square and survives being scaled to the page
    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(
        2, 4, height_ratios=[1, 0.9], hspace=0.55, wspace=0.9
    )

    sem_heatmap(fig.add_subplot(grid[0, 1:3]), letter="A")
    unique_variance_bars(fig.add_subplot(grid[1, 0:2]), letter="B")
    krr_bars(fig.add_subplot(grid[1, 2:4]), letter="C")

    fig.savefig(
        out_dir / "stage2_combined.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("  stage2_combined.png")


if __name__ == "__main__":
    main()
