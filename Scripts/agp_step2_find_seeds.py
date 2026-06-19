"""
AGP STEP 2: FIND PARCEL SEEDS
===============================
Finds the seed vertex for each atlas parcel to initialize region growing.

ACTIVE METHOD (GitHub repo): Geodesic Interior Center
    Faithful reimplementation of CenterBackFM_ly.m from the authors' repo.
    Uses fast marching from parcel boundaries inward to find the vertex
    maximally far from any boundary (the most interior point).

    1. Find boundary vertices: parcel members whose mesh neighbors belong to
       a different parcel OR medial wall (label 0). This pushes seeds AWAY
       from both inter-parcel boundaries and the medial wall edge.

    2. Fast marching from ALL boundary vertices simultaneously, using actual
       mesh edge (Euclidean) distances. Each vertex gets a "time" = geodesic
       distance to the nearest boundary.

    3. Seed = vertex within each parcel with MAXIMUM geodesic distance from
       any boundary (the most interior point).

    4. Tie-breaking fallback: if multiple vertices share the max distance
       (e.g., all vertices are boundary vertices in a tiny parcel), pick the
       vertex closest to the Euclidean centroid of ALL parcel members.

    CRITICAL: This operates on the FULL 32k mesh (not valid-vertex space),
    because medial wall vertices must act as boundaries. Seeds are then
    converted to data-space indices for downstream region growing.

PAPER METHOD (commented out below): Euclidean Centroid
    The paper describes seeds as "geometric centers" without specifying the
    algorithm. A naive reading suggests a simple Euclidean centroid: compute
    the mean (X,Y,Z) of all parcel member vertices and pick the vertex
    closest to that centroid. This is simpler and faster but can place seeds
    near parcel boundaries or the medial wall edge, unlike the repo version.

    To switch to the paper method:
        1. Uncomment the PAPER METHOD section below
        2. Comment out the REPO METHOD sections (or use the alternate __main__)
        3. The paper method uses load_and_filter_atlas from utils instead of
           load_atlas_full_mesh, since it operates in data space (~29k vertices)

Reference:
    CenterBackFM_ly.m from https://github.com/ly6ustc/Atlas-guided-parcellation
    Li et al. (2022), Computers in Biology and Medicine, 150, 106078.
"""

import os
import numpy as np
import nibabel as nib
import pickle
import heapq
import config
from utils import get_valid_vertices
# from utils import load_and_filter_atlas  # Uncomment for paper method

N_MESH = 32492


# =============================================================================
# 1. MESH LOADING
# =============================================================================

def load_mesh(mesh_path):
    """
    Load GIFTI surface file: 3D coordinates and triangle topology.

    Returns:
        coords:    (N_MESH, 3) vertex positions
        triangles: (N_triangles, 3) triangle vertex indices
    """
    print(f"  [Mesh] Loading: {mesh_path.name}")
    img = nib.load(mesh_path)
    coords = img.darrays[0].data       # (32492, 3)
    triangles = img.darrays[1].data     # (N_tri, 3)
    return coords, triangles


def build_full_mesh_adjacency(triangles, n_vertices=N_MESH):
    """
    Build adjacency list on the FULL 32k mesh (including medial wall).

    This matches the structure of atlas.l_neib from atlas.mat in the
    original MATLAB code: every vertex knows its mesh neighbors.

    Returns:
        adj: list of sets, adj[i] = set of neighbor vertex indices
    """
    adj = [set() for _ in range(n_vertices)]
    for v1, v2, v3 in triangles:
        adj[v1].add(v2); adj[v1].add(v3)
        adj[v2].add(v1); adj[v2].add(v3)
        adj[v3].add(v1); adj[v3].add(v2)
    return adj


# =============================================================================
# 2. ATLAS LOADING (FULL 32k MESH)
# =============================================================================

def load_atlas_full_mesh(atlas_path, valid_L, valid_R):
    """
    Load atlas labels expanded to full 32k mesh per hemisphere.
    Medial wall / unassigned vertices = 0.

    This matches the output of cifti_struct_dense_extract_surface_data()
    in the MATLAB code, which returns a (32492, 1) vector with zeros for
    vertices without data.

    Args:
        atlas_path: Path to .dlabel.nii atlas file
        valid_L:    Array of valid mesh vertex indices for left hemisphere
        valid_R:    Array of valid mesh vertex indices for right hemisphere

    Returns:
        atlas_L: (32492,) int array — parcel labels on full L mesh
        atlas_R: (32492,) int array — parcel labels on full R mesh
    """
    print(f"  [Atlas] Loading full-mesh: {os.path.basename(str(atlas_path))}")
    img = nib.load(atlas_path)
    raw = img.get_fdata().squeeze()

    n_L_full = 32492

    # Map CIFTI entries to full mesh positions
    atlas_L = raw[:n_L_full].astype(np.int32)
    atlas_R = raw[n_L_full:].astype(np.int32)

    n_L_parcels = len(np.unique(atlas_L[atlas_L > 0]))
    n_R_parcels = len(np.unique(atlas_R[atlas_R > 0]))
    print(f"         L: {np.sum(atlas_L > 0)} labeled vertices, {n_L_parcels} parcels")
    print(f"         R: {np.sum(atlas_R > 0)} labeled vertices, {n_R_parcels} parcels")

    return atlas_L, atlas_R


# =============================================================================
# 3. COMPUTE MESH EDGE DISTANCES
# =============================================================================

def compute_edge_distances(adj, coords):
    """
    Compute Euclidean distance for every mesh edge.

    Matches CenterBackFM_ly.m lines 18-25:
        nei_dist(i,j) = sqrt(sum((location_ver(i,:) - location_ver(neighbor(i,j),:)).^2))

    Returns:
        edge_dist: list of dicts, edge_dist[i][j] = distance from vertex i to neighbor j
    """
    n = len(adj)
    edge_dist = [dict() for _ in range(n)]
    for i in range(n):
        ci = coords[i]
        for j in adj[i]:
            if j not in edge_dist[i]:
                d = float(np.sqrt(np.sum((ci - coords[j]) ** 2)))
                edge_dist[i][j] = d
                edge_dist[j][i] = d
    return edge_dist


# =============================================================================
# 4. FIND BOUNDARY (EDGE) VERTICES
# =============================================================================

def find_boundary_vertices(atlas_labels, adj):
    """
    Identify parcel boundary vertices on the full 32k mesh.

    A parcel member is a boundary vertex if ANY of its mesh neighbors has
    a DIFFERENT label (including label 0 = medial wall).

    Matches CenterBackFM_ly.m lines 28-41:
        if atlas_parcel(cur(k)) ~= parcels(num)
            ver_label(ind(j)) = 1;  break;

    This is critical: medial wall neighbors (label 0) count as "different",
    so parcels touching medial wall have boundary vertices along that edge
    too. This pushes seeds away from the medial wall.

    Returns:
        boundary: set of mesh vertex indices that are boundary vertices
    """
    boundary = set()
    for i in range(len(atlas_labels)):
        pid = atlas_labels[i]
        if pid == 0:
            continue  # Skip medial wall / background
        for j in adj[i]:
            if atlas_labels[j] != pid:
                boundary.add(i)
                break  # One different neighbor is enough
    return boundary


# =============================================================================
# 5. FAST MARCHING FROM BOUNDARIES
# =============================================================================

def fast_march_from_boundaries(boundary_verts, atlas_labels, adj, edge_dist):
    """
    Fast marching from all boundary vertices inward.

    Faithful reimplementation of CenterBackFM_ly.m lines 43-90.

    Algorithm:
        - Boundary vertices start with time=0, state=Open
        - Medial wall (label 0) vertices: state=Useless (excluded)
        - All other parcel interior vertices: state=Far
        - Each iteration: pop the Open vertex with minimum time, mark Dead
        - For each of its Open neighbors: recheck ALL of that neighbor's
          Dead/Open neighbors for a shorter path (matches MATLAB behavior)
        - Continue until no Far vertices remain

    Returns:
        v_time: (N_MESH,) array where v_time[i] = geodesic distance from
                vertex i to the nearest boundary. Higher = more interior.
                -1 for unreached vertices.
    """
    n = len(atlas_labels)

    # States: matches MATLAB line 56 comment
    FAR     =  0
    OPEN    =  1
    DEAD    =  2
    USELESS = -2

    stat = np.zeros(n, dtype=np.int32)
    v_time = np.full(n, 1e10, dtype=np.float64)

    # Medial wall = Useless (MATLAB line 58)
    # Must be set BEFORE boundary vertices, in case boundary overlaps
    stat[atlas_labels == 0] = USELESS

    # Boundary vertices = Open with time 0 (MATLAB lines 45, 57)
    pq = []
    for v in boundary_verts:
        v_time[v] = 0.0
        stat[v] = OPEN
        heapq.heappush(pq, (0.0, v))

    # Main loop (MATLAB lines 62-88)
    # MATLAB condition: while sum(stat==0)>0  (while any Far vertices exist)
    # With priority queue: equivalent to processing until queue is empty
    # (all reachable vertices will be enqueued when their neighbors are processed)
    while pq:
        cur_time, cur = heapq.heappop(pq)

        # Skip stale entries (vertex already finalized or outdated time)
        if stat[cur] == DEAD:
            continue
        if cur_time > v_time[cur]:
            continue

        # Mark current vertex as Dead (MATLAB line 66)
        stat[cur] = DEAD

        # Phase 1: Convert Far neighbors to Open (MATLAB lines 67-71)
        for nei in adj[cur]:
            if stat[nei] == FAR:
                stat[nei] = OPEN

        # Phase 2: For each Open neighbor, recheck ALL its neighbors
        # for a shorter path (MATLAB lines 72-86)
        #
        # This matches the MATLAB behavior exactly:
        #   for i=1:max_neighbor(cur)
        #       nei = neighbor(cur,i);
        #       if stat(nei)==1  % Open
        #           for j=1:max_neighbor(nei)
        #               if stat(neighbor(nei,j))>0  % Open or Dead
        #                   c_time = v_time(neighbor(nei,j)) + dist(nei,j);
        #                   if c_time < v_time(nei): v_time(nei) = c_time
        for nei in adj[cur]:
            if stat[nei] == OPEN:
                updated = False
                for j in adj[nei]:
                    if stat[j] > 0:  # Open or Dead
                        c_time = v_time[j] + edge_dist[nei][j]
                        if c_time < v_time[nei]:
                            v_time[nei] = c_time
                            updated = True
                if updated:
                    heapq.heappush(pq, (v_time[nei], nei))

    # Mark unreached vertices (MATLAB lines 89-90)
    v_time[v_time >= 1e10] = -1.0

    return v_time


# =============================================================================
# 6. SELECT SEED VERTICES
# =============================================================================

def find_geodesic_seeds(atlas_labels, v_time, coords):
    """
    Pick the seed vertex for each parcel.

    Seed = vertex with MAXIMUM geodesic distance from any boundary
    (= most interior point of the parcel).

    Tie-breaking (MATLAB lines 97-107):
        When multiple vertices share max distance, pick the vertex
        closest to the Euclidean centroid of ALL parcel members.
        Note: the MATLAB code picks from ALL members (indd), not just
        the tied candidates (tmp). We replicate this exactly.

    Returns:
        seeds_mesh: dict {parcel_id: mesh_vertex_index}
    """
    parcels = np.unique(atlas_labels)
    parcels = parcels[parcels > 0]

    seeds_mesh = {}

    for pid in parcels:
        # All members of this parcel (MATLAB: indd = find(atlas_parcel==parcels(i)))
        members = np.where(atlas_labels == pid)[0]
        times = v_time[members]
        max_time = np.max(times)

        # Candidates with max time (MATLAB: tmp = indd(find(temp2==max(temp2))))
        candidates = members[times == max_time]

        if len(candidates) == 1:
            # Single winner — no tie-breaking needed (MATLAB line 106)
            seeds_mesh[int(pid)] = int(candidates[0])
        else:
            # Tie-breaking: closest to centroid of ALL members
            # (MATLAB lines 98-103 — note: operates on indd, not tmp)
            centroid = np.mean(coords[members], axis=0)
            dists = np.sqrt(np.sum((coords[members] - centroid) ** 2, axis=1))
            best_local = np.argmin(dists)  # First index on ties (matches MATLAB)
            seeds_mesh[int(pid)] = int(members[best_local])

    return seeds_mesh


# =============================================================================
# 7. CONVERT SEEDS: MESH SPACE → DATA SPACE
# =============================================================================

def convert_seeds_to_data_space(seeds_mesh, valid_vertices):
    """
    Convert seed indices from full 32k mesh to valid-vertex (data) space.

    This matches the implicit conversion in MATLAB's region growing:
        seeds = seeds(mask)   % line 7 of region_growing_ly_r1_T.m
    where mask = (medialwall == 0), filtering 32k → ~29k valid vertices.

    Args:
        seeds_mesh:      {parcel_id: mesh_vertex_index}
        valid_vertices:  array of mesh indices that are valid (non-medial-wall)

    Returns:
        seeds_data: {parcel_id: data_space_index}
    """
    # Build reverse lookup: mesh_index → data_index
    mesh_to_data = np.full(N_MESH, -1, dtype=np.int32)
    for data_idx, mesh_idx in enumerate(valid_vertices):
        mesh_to_data[mesh_idx] = data_idx

    seeds_data = {}
    for pid, mesh_idx in seeds_mesh.items():
        data_idx = mesh_to_data[mesh_idx]
        if data_idx == -1:
            print(f"  WARNING: Seed for parcel {pid} at mesh vertex {mesh_idx} "
                  f"is not a valid vertex! Skipping.")
        else:
            seeds_data[pid] = int(data_idx)

    return seeds_data


# =============================================================================
# 8. TOP-LEVEL: PROCESS ONE HEMISPHERE
# =============================================================================

def find_seeds_hemisphere(atlas_labels_full, coords, triangles, valid_vertices, hemi_name):
    """
    Complete seed-finding pipeline for one hemisphere.

    Operates on the full 32k mesh, then converts to data space.

    Args:
        atlas_labels_full: (32492,) parcel labels on full mesh (0 = medial wall)
        coords:            (32492, 3) vertex coordinates
        triangles:         (N_tri, 3) mesh triangles
        valid_vertices:    array of valid mesh vertex indices
        hemi_name:         "Left" or "Right" (for logging)

    Returns:
        seeds_data: dict {parcel_id: data_space_index}
    """
    print(f"\n  --- {hemi_name} Hemisphere ---")

    # A. Build full-mesh adjacency
    print(f"  [A] Building full-mesh adjacency...")
    adj = build_full_mesh_adjacency(triangles)
    n_edges = sum(len(s) for s in adj) // 2
    print(f"       {len(adj)} vertices, {n_edges} edges")

    # B. Compute mesh edge distances
    print(f"  [B] Computing edge distances...")
    edge_dist = compute_edge_distances(adj, coords)

    # C. Find boundary vertices
    print(f"  [C] Finding boundary vertices...")
    boundary = find_boundary_vertices(atlas_labels_full, adj)
    n_parcel_verts = np.sum(atlas_labels_full > 0)
    print(f"       {len(boundary)} boundary vertices "
          f"(of {n_parcel_verts} parcel vertices)")

    # D. Fast marching from boundaries
    print(f"  [D] Running fast marching...")
    v_time = fast_march_from_boundaries(boundary, atlas_labels_full, adj, edge_dist)
    reached = np.sum(v_time >= 0)
    max_time = np.max(v_time)
    print(f"       Reached {reached} vertices, max geodesic depth = {max_time:.4f}")

    # E. Select seeds on full mesh
    print(f"  [E] Selecting geodesic center seeds...")
    seeds_mesh = find_geodesic_seeds(atlas_labels_full, v_time, coords)
    print(f"       Found {len(seeds_mesh)} seeds")

    # F. Convert to data space
    print(f"  [F] Converting to data-space indices...")
    seeds_data = convert_seeds_to_data_space(seeds_mesh, valid_vertices)
    print(f"       {len(seeds_data)} seeds in data space")

    # G. Diagnostics: how many seeds are boundary vs interior
    n_boundary_seeds = sum(1 for mid in seeds_mesh.values() if mid in boundary)
    n_interior_seeds = len(seeds_mesh) - n_boundary_seeds
    print(f"       Interior seeds: {n_interior_seeds}, "
          f"Boundary seeds (fallback): {n_boundary_seeds}")

    return seeds_data


# =============================================================================
# PAPER METHOD: EUCLIDEAN CENTROID (COMMENTED OUT)
# =============================================================================
# The paper describes seeds as "geometric centers" of each parcel. The simplest
# interpretation is: compute the Euclidean centroid (mean X, Y, Z) of all
# parcel member vertices, then pick the closest actual vertex as the seed.
#
# This operates in data space (~29k valid vertices), not the full 32k mesh.
# Simpler and much faster than the repo's fast marching, but seeds may land
# near parcel boundaries or the medial wall edge.
#
# To activate: uncomment this section and the PAPER METHOD __main__ block below,
# then comment out the REPO METHOD __main__ block.
# =============================================================================

# def load_mesh_coords(mesh_path):
#     """
#     Reads a GIFTI file and extracts the 3D Coordinates (X, Y, Z) only.
#     Input: .surf.gii file
#     Output: Numpy array (N_Vertices, 3)
#     """
#     print(f"[Mesh] Loading Coordinates: {mesh_path.name}")
#     img = nib.load(mesh_path)
#     coords = img.darrays[0].data
#     return coords
#
#
# def calculate_euclidean_centroid_seeds(atlas_labels, coordinates):
#     """
#     Finds the 'Geometric Center' for every parcel via Euclidean centroid.
#
#     For each parcel:
#         1. Collect all member vertices' (X, Y, Z) coordinates
#         2. Compute the centroid (mean X, mean Y, mean Z)
#         3. Find the member vertex closest to the centroid
#
#     Inputs:
#         atlas_labels: Array of parcel IDs (Size: ~29k, data space)
#         coordinates:  Array of (X,Y,Z) positions (Size: ~29k, data space)
#
#     Returns:
#         seeds: Dictionary {Parcel_ID: Vertex_Index}
#                (Vertex_Index is in Data Space: 0 to ~29k)
#     """
#     seeds = {}
#     unique_ids = np.unique(atlas_labels)
#     unique_ids = unique_ids[unique_ids > 0]  # Ignore 0 (Background)
#
#     print(f" -> Calculating seeds for {len(unique_ids)} parcels...")
#
#     for pid in unique_ids:
#         # 1. Find all vertices belonging to this Parcel (data-space indices)
#         indices = np.where(atlas_labels == pid)[0]
#
#         # 2. Get their 3D Coordinates
#         parcel_coords = coordinates[indices]
#
#         # 3. Calculate Center of Mass (Mean X, Mean Y, Mean Z)
#         center_of_mass = np.mean(parcel_coords, axis=0)
#
#         # 4. Find the vertex closest to this center
#         distances = np.sqrt(np.sum((parcel_coords - center_of_mass)**2, axis=1))
#         best_local_idx = np.argmin(distances)
#
#         # 5. Convert back to data-space index
#         best_global_idx = indices[best_local_idx]
#
#         seeds[pid] = best_global_idx
#
#     return seeds


# =============================================================================
# 9. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # REPO METHOD __main__: GEODESIC CENTER (ACTIVE)
    # =========================================================================

    print("=" * 60)
    print("STEP 2: FIND SEEDS (Geodesic Center — CenterBackFM_ly)")
    print("=" * 60)

    # 1. Get valid vertex indices (for atlas loading and final conversion)
    subj = config.SUBJECT_IDS[0]
    run = config.RUN_IDS[0]
    data_path = config.get_brain_path(subj, run)

    print("\n[1/4] Determining valid vertices...")
    valid_L = get_valid_vertices(data_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(data_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    print(f"       Left:  {len(valid_L)} valid vertices")
    print(f"       Right: {len(valid_R)} valid vertices")

    # 2. Load atlas on FULL 32k mesh (medial wall = 0)
    print("\n[2/4] Loading atlas on full 32k mesh...")
    atlas_L_full, atlas_R_full = load_atlas_full_mesh(
        config.SCHAEFER_200_FILE, valid_L, valid_R
    )

    # 3. Load mesh geometry
    print("\n[3/4] Loading mesh geometry...")
    coords_L, tris_L = load_mesh(config.MESH_FILE_L)
    coords_R, tris_R = load_mesh(config.MESH_FILE_R)

    # 4. Find seeds for each hemisphere
    print("\n[4/4] Finding geodesic center seeds...")

    seeds_L = find_seeds_hemisphere(
        atlas_L_full, coords_L, tris_L, valid_L, "Left"
    )
    seeds_R = find_seeds_hemisphere(
        atlas_R_full, coords_R, tris_R, valid_R, "Right"
    )

    # 5. Save
    os.makedirs(config.METHOD_2_DIR, exist_ok=True)

    out_L = config.METHOD_2_DIR / "Seeds_Geometric_L.pkl"
    out_R = config.METHOD_2_DIR / "Seeds_Geometric_R.pkl"

    with open(out_L, 'wb') as f:
        pickle.dump(seeds_L, f)
    with open(out_R, 'wb') as f:
        pickle.dump(seeds_R, f)

    print(f"\n{'=' * 60}")
    print(f"SUCCESS — Seeds saved:")
    print(f"  {out_L}")
    print(f"  {out_R}")
    print(f"  Left:  {len(seeds_L)} parcels")
    print(f"  Right: {len(seeds_R)} parcels")
    print(f"{'=' * 60}")

    # =========================================================================
    # PAPER METHOD __main__ (UNCOMMENT TO USE INSTEAD OF ABOVE)
    # =========================================================================
    # Also uncomment: load_mesh_coords, calculate_euclidean_centroid_seeds,
    # and the "from utils import load_and_filter_atlas" import at the top.
    # Then comment out the REPO METHOD __main__ block above.
    # =========================================================================

    # print("--- STARTING STEP 2: DEFINING SEEDS (Euclidean Centroid) ---")
    #
    # # 1. SETUP
    # subj = config.SUBJECT_IDS[0]
    # run = config.RUN_IDS[0]
    # data_path = config.get_brain_path(subj, run)
    #
    # # 2. GET VALID VERTICES
    # print("\n[1] Determining Valid Vertices...")
    # valid_L = get_valid_vertices(data_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    # valid_R = get_valid_vertices(data_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    #
    # # 3. PREPARE ATLAS (Filter to match data space)
    # print("\n[2] Preparing Atlas...")
    # atlas_L, atlas_R = load_and_filter_atlas(config.SCHAEFER_200_FILE, valid_L, valid_R)
    #
    # # 4. PREPARE COORDINATES (Filter to match data space)
    # print("\n[3] Preparing Coordinates...")
    # coords_L_raw = load_mesh_coords(config.MESH_FILE_L)
    # coords_R_raw = load_mesh_coords(config.MESH_FILE_R)
    #
    # # Filter to keep only the ~29k valid vertices
    # coords_L = coords_L_raw[valid_L]
    # coords_R = coords_R_raw[valid_R]
    #
    # # 5. CALCULATE SEEDS
    # print("\n[4] Finding Euclidean Centroid Seeds (Left)...")
    # seeds_L = calculate_euclidean_centroid_seeds(atlas_L, coords_L)
    #
    # print("\n[5] Finding Euclidean Centroid Seeds (Right)...")
    # seeds_R = calculate_euclidean_centroid_seeds(atlas_R, coords_R)
    #
    # # 6. SAVE
    # os.makedirs(config.METHOD_2_DIR, exist_ok=True)
    #
    # out_path_L = config.METHOD_2_DIR / "Seeds_Geometric_L.pkl"
    # out_path_R = config.METHOD_2_DIR / "Seeds_Geometric_R.pkl"
    #
    # with open(out_path_L, 'wb') as f: pickle.dump(seeds_L, f)
    # with open(out_path_R, 'wb') as f: pickle.dump(seeds_R, f)
    #
    # print(f"\n[SUCCESS] Seeds saved to:\n  {out_path_L}\n  {out_path_R}")