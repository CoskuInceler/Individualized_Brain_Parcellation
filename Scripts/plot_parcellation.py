"""
PARCELLATION FIGURES
====================
Renders a parcellation on the inflated cortical surface.



Follows the visualisation used in the thesis: pastel colours spaced by
the golden ratio so neighbouring parcels stay distinguishable, shading
from surface depth to convey the folding, and black contours along
parcel borders.



Labels live in valid-vertex space in this pipeline, so they are first
inflated back onto the full mesh, with the medial wall left as NaN so
it renders as background rather than as a parcel.



Usage:
    python3 plot_parcellation.py --subject 100206 --method Method_2_AGP
    python3 plot_parcellation.py --subject 100206 --method all
"""

import argparse
import colorsys
import random


import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting, surface


import config
import utils

N_MESH = utils.N_VERTICES_PER_HEMI
SESSION = "ALL"


# where each method keeps its labels, and whether they vary by session
METHODS = {
    "Method_0_Schaefer": None,
    "Method_1_gMSHBM": ("Labels", False),
    "Method_2_AGP": ("Labels", True),
    "Method_3_SLIC_F": ("Labels", True),
    "Method_4_SLIC_C": ("Labels", True),
    "Method_5_Gordon": ("Labels_200", True),
}


def pastel_colormap(n_colors=256, seed=42):
    """
    Colours spaced around the hue circle by the golden ratio.



    Stepping by the golden ratio keeps successive colours far apart, so
    parcels that happen to be adjacent rarely end up in similar shades.
    """
    random.seed(seed)
    golden = 0.618033988749895
    h = random.random()

    colors = []
    for i in range(n_colors):
        h = (h + golden) % 1.0
        s = 0.45 + (i % 3) * 0.05
        v = 0.90 + (i % 2) * 0.05
        colors.append(colorsys.hsv_to_rgb(h, s, v))

    return ListedColormap(colors)


def surface_depth(surf_path):
    """
    Depth proxy for shading, taken from the vertical coordinate.



    The inflated surface has no sulcal depth left to read, so this
    stands in for it: enough variation to give the render some relief.
    """
    coords, _ = surface.load_surf_mesh(str(surf_path))
    z = coords[:, 2]
    return 1.0 - (z - z.min()) / (z.max() - z.min())


def atlas_labels(subject_id):
    """Schaefer labels in valid-vertex space."""
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    vL = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    vR = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    aL, aR = utils.load_and_filter_atlas(config.SCHAEFER_200_FILE, vL, vR)
    return np.concatenate([aL, aR])


def load_labels(subject_id, method):
    """Parcel labels for one subject and method, in valid-vertex space."""
    if METHODS[method] is None:
        return atlas_labels(subject_id)

    folder, per_session = METHODS[method]
    lab_dir = config.OUTPUTS_DIR / method / folder
    name = (
        f"{subject_id}_{SESSION}_labels.npy"
        if per_session
        else f"{subject_id}_labels.npy"
    )
    return np.load(lab_dir / name)


def inflate(labels, valid_L, valid_R):
    """
    Put valid-vertex labels back on the full mesh.



    Unassigned vertices become NaN so that nilearn draws the background
    surface there instead of colouring them as a parcel. That matters
    for the gradient method, which leaves the strongest borders out of
    the parcels on purpose.
    """
    n_L = len(valid_L)
    full_L = np.full(N_MESH, np.nan)
    full_R = np.full(N_MESH, np.nan)

    lab_L = labels[:n_L].astype(float)
    lab_R = labels[n_L:].astype(float)
    lab_L[lab_L == 0] = np.nan
    lab_R[lab_R == 0] = np.nan

    full_L[valid_L] = lab_L
    full_R[valid_R] = lab_R
    return full_L, full_R


def render(surf_path, data, hemi, view, depth, cmap, out_path):
    """One hemisphere, one view."""
    fig = plt.figure(figsize=(12, 10), facecolor="white")

    plotting.plot_surf_roi(
        str(surf_path),
        data,
        hemi=hemi,
        view=view,
        cmap=cmap,
        colorbar=False,
        bg_map=depth,
        bg_on_data=True,
        alpha=0.85,
        figure=fig,
    )

    levels = np.unique(data[~np.isnan(data)])
    if len(levels) > 1:
        try:
            plotting.plot_surf_contours(
                str(surf_path),
                data,
                levels=list(levels),
                figure=fig,
                hemi=hemi,
                view=view,
                colors=["black"] * len(levels),
                linewidths=1.2,
                alpha=0.8,
            )
        except Exception:
            # contours can fail on parcels reduced to a handful of
            # vertices; the filled render is still usable without them
            pass

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def plot_method(subject_id, method, cmap, context):
    out_dir = config.OUTPUTS_DIR / method / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(subject_id, method)
    data_L, data_R = inflate(labels, context["valid_L"], context["valid_R"])

    n_parcels = len(np.unique(labels[labels > 0]))
    print(f"  {method}: {n_parcels} parcels")

    for hemi, data, surf, depth, tag in (
        ("left", data_L, config.MESH_FILE_L, context["depth_L"], "L"),
        ("right", data_R, config.MESH_FILE_R, context["depth_R"], "R"),
    ):
        for view in ("lateral", "medial"):
            out = out_dir / f"{subject_id}_{tag}_{view}.png"
            render(surf, data, hemi, view, depth, cmap, out)
            print(f"    {out.name}")


def main():
    parser = argparse.ArgumentParser(description="Render parcellations.")
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--method", required=True, help="one method folder, or 'all'"
    )
    args = parser.parse_args()

    ref = config.get_brain_path(args.subject, config.RUN_IDS[0])
    context = {
        "valid_L": utils.get_valid_vertices(
            ref, "CIFTI_STRUCTURE_CORTEX_LEFT"
        ),
        "valid_R": utils.get_valid_vertices(
            ref, "CIFTI_STRUCTURE_CORTEX_RIGHT"
        ),
        "depth_L": surface_depth(config.MESH_FILE_L),
        "depth_R": surface_depth(config.MESH_FILE_R),
    }

    cmap = pastel_colormap()
    methods = sorted(METHODS) if args.method == "all" else [args.method]

    print(f"Variant {config.VARIANT} | Subject {args.subject}")
    for m in methods:
        plot_method(args.subject, m, cmap, context)


if __name__ == "__main__":
    main()
