"""
GEODESIC DISTANCE MATRICES
==========================
All-pairs shortest-path distances along the cortical surface, used as the
spatial term in SLIC-F.



Distances are computed on the full 32k mesh and then restricted to valid
vertices. Computing first and masking second matters: a shortest path
between two cortical vertices may legitimately run through medial-wall
vertices, and removing them beforehand would lengthen such paths.



The mesh is the fs_LR 32k template, identical across subjects, so the
matrices depend on neither subject nor preprocessing variant and are
computed once.



Note on the surface used: distances come from the inflated surface, as does
the surface area that sets SLIC's spatial scale S. The two are therefore on
the same metric, which is what the ratio d_spatial / S requires.



Output: Inputs/geodesic_L.npy, Inputs/geodesic_R.npy
        float32, shape (n_valid, n_valid), about 3.5 GB each
"""

import numpy as np
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


import config
import utils


def surface_graph(mesh_path):
    """Sparse graph of the mesh, edges weighted by their Euclidean length."""
    mesh = nib.load(str(mesh_path))
    coords = mesh.darrays[0].data
    faces = mesh.darrays[1].data

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.concatenate([lengths, lengths])

    n = coords.shape[0]
    print(
        f"  mesh: {n} vertices, {len(edges)} edges, "
        f"edge length {lengths.min():.2f}-{lengths.max():.2f} mm"
    )

    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def cortical_area(mesh_path, valid):
    """
    Surface area covered by valid vertices, in mm^2.



    Only triangles whose three vertices are all valid are counted, so the
    medial wall does not inflate the area that sets S.
    """
    mesh = nib.load(str(mesh_path))
    coords = mesh.darrays[0].data
    faces = mesh.darrays[1].data

    keep = np.zeros(coords.shape[0], dtype=bool)
    keep[valid] = True
    faces = faces[keep[faces].all(axis=1)]

    a = coords[faces[:, 1]] - coords[faces[:, 0]]
    b = coords[faces[:, 2]] - coords[faces[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def main():
    ref = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])

    areas = {}
    for hemi, struct, mesh_file in [
        ("L", "CIFTI_STRUCTURE_CORTEX_LEFT", config.MESH_FILE_L),
        ("R", "CIFTI_STRUCTURE_CORTEX_RIGHT", config.MESH_FILE_R),
    ]:
        print(f"{hemi} hemisphere")
        valid = utils.get_valid_vertices(ref, struct)

        graph = surface_graph(mesh_file)
        print(
            "  running dijkstra over the full mesh, this takes a few minutes"
        )
        full = dijkstra(graph, directed=False, return_predecessors=False)

        if not np.isfinite(full).all():
            raise ValueError(f"{hemi}: mesh has disconnected components")

        geo = full[np.ix_(valid, valid)].astype(np.float32)
        del full

        out = config.INPUTS_DIR / f"geodesic_{hemi}.npy"
        np.save(out, geo)
        print(f"  {geo.shape}, {geo.max():.1f} mm across, saved {out.name}")

        areas[hemi] = cortical_area(mesh_file, valid)
        S = np.sqrt(areas[hemi] / 100)
        print(
            f"  area {areas[hemi]:.0f} mm2 -> S = {S:.2f} mm for 100 parcels\n"
        )

    out = config.INPUTS_DIR / "surface_area.txt"
    out.write_text(f"L {areas['L']:.4f}\nR {areas['R']:.4f}\n")
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
