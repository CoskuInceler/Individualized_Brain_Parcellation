"""
FIGURE 1: PARCELLATION METHODS
Schaefer gets one centred row, since the atlas is identical for everyone.
The five personalized methods get three participants each, with the
participant headers placed between the Schaefer row and the rest.
Left hemisphere, lateral and medial views. White margins around each
rendered surface are cropped so the brains fill the panels.
Reads the PNGs written by plot_parcellation.py.
Output: Outputs/<variant>/Figures/figure1_methods.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import config

SUBJECTS = ["105115", "133928", "198451"]
HEMI = "L"
VIEWS = ["lateral", "medial"]
SCHAEFER = ("Method_0_Schaefer", "Schaefer\n(group atlas)")
PERSONALIZED = [
    ("Method_1_gMSHBM", "gMSHBM"),
    ("Method_2_AGP", "AGP"),
    ("Method_3_SLIC_F", "SLIC-F"),
    ("Method_4_SLIC_C", "SLIC-C"),
    ("Method_5_Gordon", "Gradient-based"),
]
N_COLS = len(SUBJECTS) * len(VIEWS)


def panel_path(method, subject, view):
    return (
        config.OUTPUTS_DIR
        / method
        / "Figures"
        / f"{subject}_{HEMI}_{view}.png"
    )


def crop_white(img, tol=0.98, pad=4):
    rgb = img[..., :3].astype(float)
    if rgb.max() > 1:
        rgb = rgb / 255.0
    mask = (rgb < tol).any(axis=2)
    if img.ndim == 3 and img.shape[2] == 4:
        mask &= img[..., 3] > 0
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0:
        return img
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def show(ax, method, subject, view):
    ax.axis("off")
    path = panel_path(method, subject, view)
    if path.exists():
        ax.imshow(crop_white(mpimg.imread(str(path))))
        return 0
    print(f"  missing: {path}")
    return 1


def row_label(ax, text):
    ax.text(
        -0.06,
        0.5,
        text,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="right",
        va="center",
    )


def main():
    out_dir = config.OUTPUTS_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 1 + len(PERSONALIZED)
    fig = plt.figure(figsize=(N_COLS * 2.6, n_rows * 2.0 + 0.9))
    # row 0: Schaefer, row 1: header spacer, rows 2+: personalized methods
    gs = GridSpec(
        n_rows + 1,
        N_COLS,
        figure=fig,
        height_ratios=[1, 0.32] + [1] * len(PERSONALIZED),
        left=0.15,
        right=0.995,
        top=0.96,
        bottom=0.005,
        wspace=0.02,
        hspace=0.04,
    )
    missing = 0

    # Schaefer, centred in the two middle columns
    mid = N_COLS // 2 - 1
    for j, view in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, mid + j])
        missing += show(ax, SCHAEFER[0], SUBJECTS[0], view)
        ax.set_title(view.capitalize(), fontsize=13)
        if j == 0:
            first_schaefer = ax
    # the row label sits at the left edge like the others
    label_ax = fig.add_subplot(gs[0, 0])
    label_ax.axis("off")
    row_label(label_ax, SCHAEFER[1])

    # personalized methods
    grid_axes = []
    for r, (method, label) in enumerate(PERSONALIZED):
        row_axes = []
        col = 0
        for subject in SUBJECTS:
            for view in VIEWS:
                ax = fig.add_subplot(gs[r + 2, col])
                missing += show(ax, method, subject, view)
                row_axes.append(ax)
                col += 1
        row_label(row_axes[0], label)
        grid_axes.append(row_axes)

    # participant and view headers above the personalized block
    top = grid_axes[0][0].get_position().y1
    for k in range(len(SUBJECTS)):
        a = grid_axes[0][2 * k].get_position()
        b = grid_axes[0][2 * k + 1].get_position()
        fig.text(
            (a.x0 + b.x1) / 2,
            top + 0.028,
            f"Participant {k + 1}",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        for j, view in enumerate(VIEWS):
            pos = grid_axes[0][2 * k + j].get_position()
            fig.text(
                (pos.x0 + pos.x1) / 2,
                top + 0.006,
                view.capitalize(),
                fontsize=13,
                ha="center",
                va="bottom",
            )

    path = out_dir / "figure1_methods.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Variant {config.VARIANT}")
    print(f"  {path}")
    print(f"  missing panels: {missing}")


if __name__ == "__main__":
    main()
