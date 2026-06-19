"""
SLIC VISUALIZATION (Methods 3 & 4)
=====================================
Renders SLIC parcellations on inflated cortical surface.

Handles both SLIC-F (Method 3) and SLIC-C (Method 4).

Features:
    - 3D shading from sulcal depth
    - Black contour lines between parcels
    - Enhanced pastel colors (reproducible with fixed seed)
    - Both hemispheres, lateral + medial views
    - Per subject, per method
    - 300 DPI

Figures go to: Outputs/Method_3_SLIC_F/Figures/
               Outputs/Method_4_SLIC_C/Figures/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting, surface
import colorsys
import random
import config

N_MESH = 32492


# =============================================================================
# COLORMAP
# =============================================================================

def create_pastel_colormap(n_colors=256, seed=42):
    """Golden-ratio pastel colormap with reproducible colors."""
    random.seed(seed)
    colors = []
    golden_ratio = 0.618033988749895
    h = random.random()

    for i in range(n_colors):
        h += golden_ratio
        h = h % 1.0
        s = 0.45 + (i % 3) * 0.05
        v = 0.90 + (i % 2) * 0.05
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))

    return ListedColormap(colors)


# =============================================================================
# SULCAL DEPTH (3D shading)
# =============================================================================

def compute_sulcal_depth(surf_path):
    """Compute depth from surface coordinates for 3D shading effect."""
    try:
        coords, faces = surface.load_surf_mesh(surf_path)
        z = coords[:, 2]
        depth = (z - z.min()) / (z.max() - z.min())
        depth = 1 - depth
        return depth
    except Exception as e:
        print(f"   Warning: Could not compute depth ({e})")
        return None


# =============================================================================
# PLOTTING
# =============================================================================

def plot_parcellation(surf, data, hemi, view, depth, cmap, out_path):
    """Plot one view of the parcellation with 3D shading and contours."""
    fig = plt.figure(figsize=(12, 10), facecolor='white')

    plotting.plot_surf_roi(
        surf, data,
        hemi=hemi, view=view,
        cmap=cmap, colorbar=False,
        bg_map=depth, bg_on_data=True,
        alpha=0.85, figure=fig
    )

    unique_labels = np.unique(data[~np.isnan(data)])
    if len(unique_labels) > 1:
        try:
            plotting.plot_surf_contours(
                surf, data,
                levels=unique_labels,
                figure=fig, hemi=hemi, view=view,
                colors=['black'] * len(unique_labels),
                linewidths=1.2, alpha=0.8
            )
        except:
            pass

    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SLIC VISUALIZATION (Methods 3 & 4)")
    print("=" * 60)

    surf_L = str(config.MESH_FILE_L)
    surf_R = str(config.MESH_FILE_R)

    print("\n[Computing sulcal depth...]")
    depth_L = compute_sulcal_depth(surf_L)
    depth_R = compute_sulcal_depth(surf_R)

    cmap = create_pastel_colormap()

    methods = [
        ('SLIC_F', config.METHOD_3_DIR),
        ('SLIC_C', config.METHOD_4_DIR),
    ]

    for method_name, method_dir in methods:
        fig_dir = method_dir / "Figures"
        os.makedirs(fig_dir, exist_ok=True)

        print(f"\n{'â•' * 40}")
        print(f"Method: {method_name}")
        print(f"{'â•' * 40}")

        for subj in config.SUBJECT_IDS:
            print(f"\n[Subject] {subj}")

            # Load full-mesh labels (saved by slic_parcellation.py)
            lh_path = method_dir / f"{subj}_{method_name}_LH.npy"
            rh_path = method_dir / f"{subj}_{method_name}_RH.npy"

            if not (lh_path.exists() and rh_path.exists()):
                print(f"   [Skip] Full-mesh label files missing â€” run slic_parcellation first")
                continue

            data_L = np.load(lh_path).astype(float)
            data_R = np.load(rh_path).astype(float)

            # Mask medial wall with NaN
            data_L[data_L == 0] = np.nan
            data_R[data_R == 0] = np.nan

            for hemi, data, surf, depth, name in [
                ('left',  data_L, surf_L, depth_L, 'L'),
                ('right', data_R, surf_R, depth_R, 'R'),
            ]:
                for view in ['lateral', 'medial']:
                    out = fig_dir / f"{subj}_{method_name}_{name}_{view}.png"
                    plot_parcellation(surf, data, hemi, view, depth, cmap, out)
                    print(f"   [Saved] {subj}_{method_name}_{name}_{view}.png")

    print("\n--- SLIC VISUALIZATION COMPLETE ---")