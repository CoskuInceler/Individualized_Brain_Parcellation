# slic_geometry.py
"""
Geodesic Distance Computation for Surface-based SLIC

This script computes geodesic distances between all pairs of vertices on a
cortical surface mesh. Geodesic distance is the shortest path ALONG the surface,
not through 3D space.

Why Geodesic Instead of Euclidean?
----------------------------------
Imagine two vertices on opposite banks of a deep sulcus (brain fold):
- Euclidean distance: Short (straight line through the air/tissue)
- Geodesic distance: Long (must travel around the fold along the cortex)

For brain parcellation, geodesic is correct because:
1. Brain regions are defined by cortical organization, not 3D proximity
2. Functional signals propagate along the cortical sheet
3. Two nearby points in 3D might be in completely different functional areas

Algorithm:
----------
1. Load the surface mesh (vertices + triangles)
2. Convert triangles to edges (graph representation)
3. Weight each edge by its Euclidean length
4. Run Dijkstra's algorithm to find all shortest paths
5. Cache the result (computation is slow, ~5-10 min per hemisphere)

Note on Mesh Size:
------------------
The HCP mesh has 32,492 vertices per hemisphere, but only ~29,000 are valid
cortex (the rest are medial wall). We compute distances on the FULL mesh,
then the main script extracts only the valid vertices. This is correct because
shortest paths might travel through medial wall vertices even if we don't
parcellate them.
"""

import numpy as np
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import os


def compute_geodesic_distances(gifti_path, cache_path=None):
    """
    Compute all-pairs geodesic distance matrix for a cortical surface mesh.
    
    This function either:
    - Loads a pre-computed matrix from cache (fast), OR
    - Computes the matrix from scratch using Dijkstra's algorithm (slow)
    
    Args:
        gifti_path (str or Path): Path to the surface mesh file (.surf.gii).
                                  This should be your HCP template mesh, e.g.,
                                  "L.inflated.32k_fs_LR.surf.gii"
        
        cache_path (str or Path): Path to save/load the cached matrix (.npy).
                                  If None, no caching is performed.
                                  Recommended: Always use caching!
    
    Returns:
        d_geo (np.ndarray): Geodesic distance matrix of shape (N_vertices, N_vertices).
                            Entry [i,j] = geodesic distance from vertex i to vertex j.
                            Units are millimeters (same as the mesh coordinates).
    
    Example:
        >>> d_geo = compute_geodesic_distances(
        ...     "L.inflated.32k_fs_LR.surf.gii",
        ...     "geodesic_cache_L.npy"
        ... )
        >>> print(d_geo.shape)  # (32492, 32492)
        >>> print(d_geo[0, 100])  # Distance from vertex 0 to vertex 100
    """
    
    # Convert to string in case Path objects are passed
    gifti_path = str(gifti_path)
    if cache_path:
        cache_path = str(cache_path)
    
    # =================================================================
    # STEP 1: CHECK CACHE
    # =================================================================
    # Computing geodesics is expensive (~5-10 minutes, ~4GB RAM for 32k mesh).
    # If we've done it before, just load the saved result.
    
    if cache_path and os.path.exists(cache_path):
        print(f"[Geometry] Loading cached geodesic matrix: {os.path.basename(cache_path)}")
        d_geo = np.load(cache_path)
        
        # Validate the cached data isn't corrupted
        if np.any(np.isnan(d_geo)) or np.any(np.isinf(d_geo)):
            print(f"[Geometry] WARNING: Cached matrix contains invalid values, recomputing...")
        else:
            print(f"[Geometry]   Shape: {d_geo.shape}")
            print(f"[Geometry]   Distance range: [{np.min(d_geo):.3f}, {np.max(d_geo):.3f}] mm")
            return d_geo

    # =================================================================
    # STEP 2: LOAD THE SURFACE MESH
    # =================================================================
    # A GIFTI surface file contains:
    #   - darrays[0]: Vertex coordinates (N x 3 matrix of x,y,z positions)
    #   - darrays[1]: Face definitions (M x 3 matrix of vertex indices forming triangles)
    
    print(f"[Geometry] Computing geodesic matrix from: {os.path.basename(gifti_path)}")
    print(f"[Geometry] (This will take several minutes...)")
    
    try:
        mesh = nib.load(gifti_path)
        vertices = mesh.darrays[0].data  # Shape: (N_vertices, 3)
        faces = mesh.darrays[1].data     # Shape: (N_faces, 3)
    except Exception as e:
        raise ValueError(f"Failed to load GIFTI mesh file: {e}")

    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    print(f"[Geometry]   Mesh loaded: {n_vertices} vertices, {n_faces} faces")

    # =================================================================
    # STEP 3: CONVERT MESH TO GRAPH
    # =================================================================
    # A triangle mesh defines connectivity: if two vertices share a triangle edge,
    # they are connected. We extract all unique edges from the triangles.
    #
    # Each triangle ABC has three edges: A-B, B-C, C-A
    # We collect all edges, remove duplicates, and calculate their lengths.
    
    # Extract edges from each triangle
    # faces[:, [0, 1]] gives the first and second vertex of each triangle (edge A-B)
    edges = np.vstack([
        faces[:, [0, 1]],  # Edge: vertex 0 to vertex 1 of each triangle
        faces[:, [1, 2]],  # Edge: vertex 1 to vertex 2 of each triangle
        faces[:, [2, 0]]   # Edge: vertex 2 to vertex 0 of each triangle
    ])
    
    # Sort each edge so (5, 3) becomes (3, 5) - this helps identify duplicates
    edges.sort(axis=1)
    
    # Remove duplicate edges (each internal edge is shared by two triangles)
    edges = np.unique(edges, axis=0)
    
    n_edges = len(edges)
    print(f"[Geometry]   Extracted {n_edges} unique edges")

    # =================================================================
    # STEP 4: CALCULATE EDGE WEIGHTS
    # =================================================================
    # The weight of each edge is its Euclidean length (distance between endpoints).
    # This is correct because geodesic distance = sum of edge lengths along path.
    
    # Get the 3D coordinates of each edge's endpoints
    v1 = vertices[edges[:, 0]]  # First vertex of each edge (N_edges x 3)
    v2 = vertices[edges[:, 1]]  # Second vertex of each edge (N_edges x 3)
    
    # Calculate Euclidean distance for each edge
    edge_lengths = np.linalg.norm(v1 - v2, axis=1)
    
    print(f"[Geometry]   Edge lengths: min={np.min(edge_lengths):.3f}, "
          f"max={np.max(edge_lengths):.3f}, mean={np.mean(edge_lengths):.3f} mm")

    # =================================================================
    # STEP 5: BUILD SPARSE GRAPH MATRIX
    # =================================================================
    # For Dijkstra's algorithm, we need an adjacency matrix where:
    #   - graph[i, j] = weight of edge from i to j (or 0 if no edge)
    #
    # We use a sparse matrix because most entries are 0 (each vertex connects
    # to only ~6 neighbors on average, not all 32,000 vertices).
    #
    # The graph is undirected, so we add both directions: (i,j) and (j,i)
    
    # Prepare indices and values for sparse matrix construction
    rows = np.concatenate([edges[:, 0], edges[:, 1]])  # From vertices
    cols = np.concatenate([edges[:, 1], edges[:, 0]])  # To vertices
    data = np.concatenate([edge_lengths, edge_lengths])  # Edge weights (both directions)
    
    # Create sparse matrix in CSR format (efficient for Dijkstra)
    graph_sparse = csr_matrix(
        (data, (rows, cols)), 
        shape=(n_vertices, n_vertices)
    )
    
    print(f"[Geometry]   Sparse graph constructed: {graph_sparse.nnz} non-zero entries")

    # =================================================================
    # STEP 6: RUN DIJKSTRA'S ALGORITHM
    # =================================================================
    # Dijkstra finds the shortest path from every vertex to every other vertex.
    # This is the expensive step - O(N^2 log N) time complexity.
    #
    # Result: d_geo[i, j] = length of shortest path from vertex i to vertex j
    #                     = geodesic distance between i and j
    
    print("[Geometry]   Running Dijkstra's algorithm (this takes several minutes)...")
    
    d_geo = dijkstra(
        graph_sparse, 
        directed=False,           # Undirected graph (can traverse edges both ways)
        return_predecessors=False  # We only need distances, not the actual paths
    )
    
    # Convert to float32 to save memory (float64 is overkill for distances)
    d_geo = d_geo.astype(np.float32)
    
    # =================================================================
    # STEP 7: VALIDATE RESULT
    # =================================================================
    # Check for problems that would indicate a bug or disconnected mesh
    
    if np.any(np.isnan(d_geo)):
        raise ValueError(
            "Geodesic matrix contains NaN values! "
            "This should not happen with a valid mesh."
        )
    
    if np.any(np.isinf(d_geo)):
        raise ValueError(
            "Geodesic matrix contains Inf values! "
            "This indicates disconnected components in the mesh - "
            "some vertices cannot reach others."
        )
    
    print(f"[Geometry]   Computation complete!")
    print(f"[Geometry]   Distance range: [{np.min(d_geo):.3f}, {np.max(d_geo):.3f}] mm")

    # =================================================================
    # STEP 8: SAVE TO CACHE
    # =================================================================
    # Save the result so we don't have to recompute next time
    
    if cache_path:
        print(f"[Geometry]   Saving to cache: {os.path.basename(cache_path)}")
        np.save(cache_path, d_geo)

    return d_geo