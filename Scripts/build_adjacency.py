"""
SURFACE ADJACENCY GRAPHS
========================
Builds a vertex adjacency graph for each hemisphere from the fs_LR 32k mesh.
Two vertices are neighbours if they share an edge of a mesh triangle.
Medial-wall vertices are dropped and the remaining indices are remapped to
data-column space, so node i of the graph is column i of the hemisphere's
time series.
The mesh is identical for all subjects (fs_LR 32k), so this runs once.
Output: Inputs/adjacency_L.npz, Inputs/adjacency_R.npz
        scipy sparse matrices, symmetric, binary
"""

import numpy as np
import nibabel as nib
from scipy.sparse import coo_matrix, save_npz
import config
import utils

N_MESH = utils.N_VERTICES_PER_HEMI


def load_triangles(surf_path):
    """Read the triangle list from a GIFTI surface file."""
    img = nib.load(str(surf_path))
    for arr in img.darrays:
        if (
            arr.data.ndim == 2
            and arr.data.shape[1] == 3
            and arr.data.dtype.kind in "iu"
        ):
            return arr.data
    raise ValueError(f"No triangle array found in {surf_path}")


def build_adjacency(triangles, valid_indices):
    """
    Build a sparse adjacency matrix in data-column space.

    Parameters
    ----------
    triangles : (n_tri, 3) mesh triangle vertex indices
    valid_indices : mesh vertex indices present in the data

    Returns
    -------
    (n_valid, n_valid) symmetric binary sparse matrix
    """
    mesh_to_data = np.full(N_MESH, -1, dtype=np.int64)
    mesh_to_data[valid_indices] = np.arange(len(valid_indices))
    tri = mesh_to_data[triangles]
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = edges[(edges[:, 0] >= 0) & (edges[:, 1] >= 0)]
    both = np.vstack([edges, edges[:, ::-1]])
    n = len(valid_indices)
    adj = coo_matrix(
        (np.ones(len(both), dtype=np.uint8), (both[:, 0], both[:, 1])),
        shape=(n, n),
    ).tocsr()
    adj.data[:] = 1
    return adj


def main():
    ref = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    for hemi, struct, mesh_file in [
        ("L", "CIFTI_STRUCTURE_CORTEX_LEFT", config.MESH_FILE_L),
        ("R", "CIFTI_STRUCTURE_CORTEX_RIGHT", config.MESH_FILE_R),
    ]:
        valid = utils.get_valid_vertices(ref, struct)
        tris = load_triangles(mesh_file)
        adj = build_adjacency(tris, valid)
        degrees = np.asarray(adj.sum(axis=1)).ravel()
        out = config.INPUTS_DIR / f"adjacency_{hemi}.npz"
        save_npz(out, adj)
        print(f"{hemi}: {adj.shape[0]} vertices, {adj.nnz // 2} edges")
        print(
            f"   degree  min={degrees.min()}  mean={degrees.mean():.2f}  max={degrees.max()}"
        )
        print(f"   isolated vertices: {(degrees == 0).sum()}")
        print(f"   saved {out.name}")


if __name__ == "__main__":
    main()
