"""
AGP SEED VERTICES
=================
Finds the seed vertex for each Schaefer parcel, used to initialise region
growing in the atlas-guided parcellation.
A seed is the most interior point of its parcel: the vertex furthest, in
geodesic distance along the mesh, from any parcel boundary. Medial-wall
neighbours count as boundaries, so seeds are also pushed away from the
cortical edge.
Reimplementation of CenterBackFM_ly.m from
    https://github.com/ly6ustc/Atlas-guided-parcellation
    Li et al. (2022), Computers in Biology and Medicine, 150, 106078.
Seeds depend only on the atlas and the mesh, not on subject data or on the
preprocessing variant, so this runs once.
Output: Inputs/seeds_L.npy, Inputs/seeds_R.npy
        shape (n_parcels, 2): parcel id, data-space vertex index
"""

import heapq
import numpy as np
import nibabel as nib
import config
import utils

N_MESH = utils.N_VERTICES_PER_HEMI


def load_mesh(path):
    """Vertex coordinates and triangles from a GIFTI surface."""
    img = nib.load(str(path))
    return img.darrays[0].data, img.darrays[1].data


def mesh_adjacency(triangles, n=N_MESH):
    """Neighbour sets on the full mesh, medial wall included."""
    adj = [set() for _ in range(n)]
    for a, b, c in triangles:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def edge_lengths(adj, coords):
    """Euclidean length of every mesh edge."""
    dist = [dict() for _ in range(len(adj))]
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if j not in dist[i]:
                d = float(np.linalg.norm(coords[i] - coords[j]))
                dist[i][j] = d
                dist[j][i] = d
    return dist


def boundary_vertices(labels, adj):
    """Parcel members touching a different label, medial wall included."""
    out = set()
    for i, pid in enumerate(labels):
        if pid == 0:
            continue
        for j in adj[i]:
            if labels[j] != pid:
                out.add(i)
                break
    return out


def geodesic_depth(labels, adj, dist, boundary):
    """
    Geodesic distance from every parcel vertex to the nearest boundary.
    Fast marching outward from all boundary vertices at once. Medial-wall
    vertices are excluded. Unreached vertices come back as -1.
    """
    n = len(labels)
    FAR, OPEN, DEAD, USELESS = 0, 1, 2, -2
    state = np.zeros(n, dtype=np.int32)
    depth = np.full(n, np.inf)
    state[labels == 0] = USELESS
    queue = []
    for v in boundary:
        depth[v] = 0.0
        state[v] = OPEN
        heapq.heappush(queue, (0.0, v))
    while queue:
        d, cur = heapq.heappop(queue)
        if state[cur] == DEAD or d > depth[cur]:
            continue
        state[cur] = DEAD
        for nb in adj[cur]:
            if state[nb] == FAR:
                state[nb] = OPEN
        for nb in adj[cur]:
            if state[nb] != OPEN:
                continue
            improved = False
            for j in adj[nb]:
                if state[j] > 0:
                    cand = depth[j] + dist[nb][j]
                    if cand < depth[nb]:
                        depth[nb] = cand
                        improved = True
            if improved:
                heapq.heappush(queue, (depth[nb], nb))
    depth[np.isinf(depth)] = -1.0
    return depth


def select_seeds(labels, depth, coords):
    """
    Deepest vertex of each parcel.
    On ties, the vertex closest to the parcel's Euclidean centroid is used,
    following the original implementation.
    """
    seeds = {}
    for pid in np.unique(labels[labels > 0]):
        members = np.flatnonzero(labels == pid)
        d = depth[members]
        winners = members[d == d.max()]
        if len(winners) == 1:
            seeds[int(pid)] = int(winners[0])
        else:
            centroid = coords[members].mean(axis=0)
            offsets = np.linalg.norm(coords[members] - centroid, axis=1)
            seeds[int(pid)] = int(members[np.argmin(offsets)])
    return seeds


def to_data_space(seeds, valid):
    """Convert mesh vertex indices to data-column indices."""
    lookup = np.full(N_MESH, -1, dtype=np.int64)
    lookup[valid] = np.arange(len(valid))
    out = {}
    for pid, mesh_idx in seeds.items():
        col = lookup[mesh_idx]
        if col < 0:
            print(
                f"  warning: seed for parcel {pid} falls outside the data mask"
            )
        else:
            out[pid] = int(col)
    return out


def process_hemisphere(atlas_full, mesh_file, valid, name):
    coords, tris = load_mesh(mesh_file)
    adj = mesh_adjacency(tris)
    dist = edge_lengths(adj, coords)
    bnd = boundary_vertices(atlas_full, adj)
    print(
        f"  {name}: {len(bnd)} boundary of {(atlas_full > 0).sum()} parcel vertices"
    )
    depth = geodesic_depth(atlas_full, adj, dist, bnd)
    print(f"  {name}: max geodesic depth {depth.max():.2f} mm")
    seeds = select_seeds(atlas_full, depth, coords)
    seeds = to_data_space(seeds, valid)
    print(f"  {name}: {len(seeds)} seeds")
    return seeds


def main():
    ref = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    valid_R = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    # atlas on the full mesh, medial wall as 0
    raw = (
        nib.load(str(config.SCHAEFER_200_FILE))
        .get_fdata()
        .squeeze()
        .astype(int)
    )
    atlas_L_full = raw[:N_MESH]
    atlas_R_full = raw[N_MESH:]
    for hemi, atlas_full, mesh_file, valid in [
        ("L", atlas_L_full, config.MESH_FILE_L, valid_L),
        ("R", atlas_R_full, config.MESH_FILE_R, valid_R),
    ]:
        seeds = process_hemisphere(atlas_full, mesh_file, valid, hemi)
        arr = np.array(sorted(seeds.items()), dtype=np.int64)
        out = config.INPUTS_DIR / f"seeds_{hemi}.npy"
        np.save(out, arr)
        print(f"  saved {out.name}\n")


if __name__ == "__main__":
    main()
