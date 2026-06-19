"""
FEATURE EXTRACTION — SHARED UTILITIES
=======================================
Shared functions used by all five per-method feature extraction scripts.

Contents:
    1. Hungarian alignment          - bijective native → atlas label mapping
    2. Label loading                - load full-mesh .npy label files
    3. Column parsing               - extract native parcel ID from CSV column name
    4. FC computation               - Pearson correlation matrix from timeseries
    5. Graph construction           - build adjacency from FC (3 variants)
    6. Graph metrics                - global efficiency, average shortest path
    7. Parcel size                  - vertex count per parcel (atlas-aligned)
    8. FC vectorization             - lower triangle extraction (19,900 values)
"""

import numpy as np
import pandas as pd
import nibabel as nib
from collections import deque
import feature_config as cfg


# =============================================================================
# 1. HUNGARIAN ALIGNMENT
# =============================================================================

def _hungarian_maximize(profit_matrix):
    """
    Solve maximum-weight bipartite assignment (pure NumPy, no SciPy).
    Converts to minimization by subtracting from max, then runs
    shortest-augmenting-path O(n³) algorithm.

    Args:
        profit_matrix: (n, n) non-negative NumPy array.

    Returns:
        row_ind: (n,) array
        col_ind: (n,) array — optimal assignment maximizing total profit.
    """
    n = profit_matrix.shape[0]
    cost = float(profit_matrix.max()) - profit_matrix.astype(np.float64)
    INF = float('inf')

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0   = 0
        dist = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        prev = np.zeros(n + 1, dtype=int)

        while True:
            used[j0] = True
            i0    = p[j0]
            delta = INF
            j1    = -1

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < dist[j]:
                        dist[j] = cur
                        prev[j] = j0
                    if dist[j] < delta:
                        delta = dist[j]
                        j1    = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j]    -= delta
                else:
                    dist[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0 != 0:
            p[j0] = p[prev[j0]]
            j0    = prev[j0]

    col_ind = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        if p[j] > 0:
            col_ind[p[j] - 1] = j - 1

    return np.arange(n), col_ind


def build_overlap_matrix(native_labels, atlas_labels, n_native=100, n_atlas=100):
    """
    Build vertex-overlap matrix between native and atlas parcels.

    Entry [i, j] = number of vertices where native parcel (i+1)
    overlaps with atlas parcel (j+1).

    Both label arrays must be in valid-vertex space and cover the
    same hemisphere (labels 1..n_native and 1..n_atlas respectively).

    Args:
        native_labels: (n_vertices,) integer array, 0 = medial wall
        atlas_labels:  (n_vertices,) integer array, 0 = medial wall
        n_native:      number of native parcels (default 100 per hemi)
        n_atlas:       number of atlas parcels  (default 100 per hemi)

    Returns:
        overlap: (n_native, n_atlas) int array
    """
    overlap = np.zeros((n_native, n_atlas), dtype=np.int32)
    valid = (native_labels > 0) & (atlas_labels > 0)
    nat = native_labels[valid]
    atl = atlas_labels[valid]

    for ni in range(1, n_native + 1):
        mask = (nat == ni)
        if not mask.any():
            continue
        atl_sub = atl[mask]
        for ai in range(1, n_atlas + 1):
            overlap[ni - 1, ai - 1] = int(np.sum(atl_sub == ai))

    return overlap


def align_to_atlas(native_lh, native_rh, atlas_lh, atlas_rh):
    """
    Build bijective native → atlas parcel mapping using Hungarian algorithm.

    Processes LH (parcels 1-100) and RH (parcels 101-200) independently,
    then combines into a single mapping covering 1-200.

    Args:
        native_lh: (n_valid_L,) native parcel labels, 0=medial wall, values 1-100
        native_rh: (n_valid_R,) native parcel labels, 0=medial wall, values 1-100 or 101-200
        atlas_lh:  (n_valid_L,) Schaefer atlas labels, values 1-100
        atlas_rh:  (n_valid_R,) Schaefer atlas labels, values 101-200

    Returns:
        mapping: dict {native_global_id: atlas_id}
                 native_global_id is 1-100 for LH, 101-200 for RH
                 (SLIC per-hemi 1-100 convention is handled by the caller)
    """
    # --- LH: native 1-100, atlas 1-100 ---
    overlap_L = build_overlap_matrix(native_lh, atlas_lh,
                                     n_native=100, n_atlas=100)
    _, col_L = _hungarian_maximize(overlap_L)
    # col_L[i] = atlas 0-indexed position for native parcel i+1
    mapping = {}
    for native_0idx, atlas_0idx in enumerate(col_L):
        mapping[native_0idx + 1] = atlas_0idx + 1       # 1-indexed

    # --- RH ---
    # Detect whether native RH uses per-hemisphere 1-100 (SLIC) or
    # global 101-200 (gMSHBM, AGP).
    rh_unique = np.unique(native_rh[native_rh > 0])
    rh_per_hemi = bool(rh_unique.max() <= 100)

    # Atlas RH always uses 101-200; shift to 1-100 for overlap matrix
    atlas_rh_shifted = atlas_rh.copy()
    atlas_rh_shifted[atlas_rh_shifted > 0] -= 100

    if rh_per_hemi:
        # SLIC: native RH uses 1-100
        overlap_R = build_overlap_matrix(native_rh, atlas_rh_shifted,
                                         n_native=100, n_atlas=100)
        _, col_R = _hungarian_maximize(overlap_R)
        for native_0idx, atlas_0idx in enumerate(col_R):
            # Store with +100 offset so keys are globally unique 101-200
            mapping[native_0idx + 101] = atlas_0idx + 101
    else:
        # gMSHBM/AGP: native RH uses 101-200; shift to 1-100 for matrix
        native_rh_shifted = native_rh.copy()
        native_rh_shifted[native_rh_shifted > 0] -= 100
        overlap_R = build_overlap_matrix(native_rh_shifted, atlas_rh_shifted,
                                         n_native=100, n_atlas=100)
        _, col_R = _hungarian_maximize(overlap_R)
        for native_0idx, atlas_0idx in enumerate(col_R):
            mapping[native_0idx + 101] = atlas_0idx + 101

    return mapping


# =============================================================================
# 2. LABEL LOADING
# =============================================================================

def load_atlas_labels(valid_L, valid_R):
    """
    Load Schaefer atlas and filter to valid-vertex space.

    Returns:
        atlas_lh: (n_valid_L,) labels 1-100
        atlas_rh: (n_valid_R,) labels 101-200
    """
    img  = nib.load(cfg.SCHAEFER_FILE)
    data = img.get_fdata().squeeze().astype(int)
    atlas_lh = data[:cfg.N_MESH][valid_L]
    atlas_rh = data[cfg.N_MESH:][valid_R]
    return atlas_lh, atlas_rh


def load_native_labels(method_info, subj, valid_L, valid_R):
    """
    Load native parcel labels for a subject, filtered to valid-vertex space.

    For Schaefer (needs_align=False), returns atlas labels directly
    (identity mapping, no alignment needed).

    For all other methods, loads the full-mesh .npy files and filters.

    Returns:
        native_lh: (n_valid_L,) labels
        native_rh: (n_valid_R,) labels
        or (None, None) if files are missing.
    """
    if not method_info["needs_align"]:
        # M0: atlas IS the parcellation
        return load_atlas_labels(valid_L, valid_R)

    mdir    = method_info["method_dir"]
    lh_path = mdir / method_info["lh_pattern"].format(subj=subj)
    rh_path = mdir / method_info["rh_pattern"].format(subj=subj)

    if not (lh_path.exists() and rh_path.exists()):
        print(f"    [WARN] Label files missing for {subj}: {lh_path.name}")
        return None, None

    lh_full = np.load(lh_path).astype(int)
    rh_full = np.load(rh_path).astype(int)

    return lh_full[valid_L], rh_full[valid_R]


def get_valid_vertices():
    """
    Extract valid (non-medial-wall) vertex indices from Schaefer atlas header.

    Returns:
        valid_L: (n_valid_L,) indices into the 32k LH mesh
        valid_R: (n_valid_R,) indices into the 32k RH mesh
    """
    img        = nib.load(cfg.SCHAEFER_FILE)
    brain_axis = img.header.get_axis(1)

    valid_L = valid_R = None
    for name, _, model in brain_axis.iter_structures():
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            valid_L = np.array(model.vertex)
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            valid_R = np.array(model.vertex)

    if valid_L is None or valid_R is None:
        raise ValueError("Could not extract valid vertex indices from atlas.")

    return valid_L, valid_R


# =============================================================================
# 3. COLUMN PARSING
# =============================================================================

def parse_col_to_native_id(col_name, col_format):
    """
    Extract globally unique native parcel ID from a timeseries CSV column name.

    Standard format  ("Parcel_N"):
        "Parcel_5"    → 5    (already globally unique 1-200)

    SLIC format  ("L_Parcel_K" / "R_Parcel_K"):
        "L_Parcel_5"  → 6    (k+1,   range 1-100)
        "R_Parcel_5"  → 106  (k+101, range 101-200)

    Returns:
        int native ID, or None if unparseable.
    """
    if col_format == "standard":
        parts = col_name.split("_")
        if len(parts) == 2 and parts[0] == "Parcel":
            try:
                return int(parts[1])
            except ValueError:
                return None

    elif col_format == "slic":
        parts = col_name.split("_")
        if len(parts) == 3 and parts[1] == "Parcel":
            try:
                k = int(parts[2])
            except ValueError:
                return None
            if parts[0] == "L":
                return k + 1        # 0-99 → 1-100
            elif parts[0] == "R":
                return k + 101      # 0-99 → 101-200

    return None


# =============================================================================
# 4. FC COMPUTATION
# =============================================================================

def compute_fc_matrix(ts_df):
    """
    Compute 200×200 Pearson correlation FC matrix from parcel timeseries.

    Args:
        ts_df: DataFrame (Timepoints × 200 parcels)

    Returns:
        fc: (200, 200) float64 array, NaNs replaced with 0.
    """
    fc = ts_df.corr().to_numpy(copy=True)
    np.nan_to_num(fc, copy=False, nan=0.0)
    return fc


def permute_fc_to_atlas_order(fc_native, ts_columns, col_format, mapping):
    """
    Permute FC matrix from native column order to atlas parcel order (1..200).

    This is a LOSSLESS row/column permutation — no values are changed,
    only their positions. After permutation, position [i,j] corresponds
    to atlas parcels (i+1, j+1).

    Args:
        fc_native:   (200, 200) FC in native column order
        ts_columns:  list of CSV column names (length 200)
        col_format:  "standard" or "slic"
        mapping:     dict {native_global_id: atlas_id}

    Returns:
        fc_atlas: (200, 200) FC in atlas order

    Raises:
        ValueError if any column cannot be mapped (always fatal —
        an incomplete mapping means silent corruption otherwise).
    """
    n = len(ts_columns)
    # perm[i] = 0-indexed atlas position for native column i
    perm = np.full(n, -1, dtype=int)

    for i, col in enumerate(ts_columns):
        native_id = parse_col_to_native_id(col, col_format)
        if native_id is None:
            raise ValueError(f"Cannot parse column '{col}' with format '{col_format}'")
        atlas_id = mapping.get(native_id)
        if atlas_id is None:
            raise ValueError(
                f"Native ID {native_id} (from column '{col}') not in mapping. "
                f"Mapping covers: {sorted(mapping.keys())[:10]}..."
            )
        perm[i] = atlas_id - 1   # 0-indexed

    # Sanity: every slot must be filled and bijective
    if (perm < 0).any():
        raise ValueError(f"{(perm < 0).sum()} columns could not be mapped.")
    if len(np.unique(perm)) != n:
        raise ValueError("Duplicate atlas positions — mapping is not bijective.")

    # Apply permutation: inv_perm[atlas_pos] = native_col
    inv_perm = np.argsort(perm)
    fc_atlas = fc_native[np.ix_(inv_perm, inv_perm)]
    return fc_atlas


def fc_lower_triangle(fc):
    """
    Extract lower triangle of FC matrix (excluding diagonal).

    For a 200×200 matrix: 200×199/2 = 19,900 values.

    Args:
        fc: (200, 200) array

    Returns:
        vec: (19900,) float64 array
    """
    idx = np.tril_indices_from(fc, k=-1)
    return fc[idx].astype(np.float64)


# =============================================================================
# 5. GRAPH CONSTRUCTION
# =============================================================================

def _zero_negatives(fc):
    """Return copy of FC with all negative values set to 0."""
    fc_pos = fc.copy()
    fc_pos[fc_pos < 0] = 0.0
    np.fill_diagonal(fc_pos, 0.0)   # no self-loops
    return fc_pos


def build_graph_raw(fc):
    """
    Weighted graph: edge weight = FC value.
    Negative FC → 0. Diagonal → 0.

    Returns:
        adj: (200, 200) float64 weighted adjacency matrix
    """
    return _zero_negatives(fc)


def build_graph_top10(fc):
    """
    Binary graph: keep top 10% of positive FC edges.
    Negative FC → excluded. Diagonal → excluded.

    Returns:
        adj: (200, 200) binary adjacency matrix (0/1)
    """
    fc_pos = _zero_negatives(fc)
    # Get all upper-triangle positive values
    idx_upper = np.triu_indices_from(fc_pos, k=1)
    vals = fc_pos[idx_upper]
    pos_vals = vals[vals > 0]

    if len(pos_vals) == 0:
        return np.zeros_like(fc_pos)

    threshold = np.percentile(pos_vals,
                              100 - cfg.TOP_K_PERCENT)   # e.g. 90th percentile
    adj = (fc_pos >= threshold).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def build_graph_r05(fc):
    """
    Binary graph: keep edges where FC > 0.5.
    Negative FC → excluded. Diagonal → excluded.

    Returns:
        adj: (200, 200) binary adjacency matrix (0/1)
    """
    fc_pos = _zero_negatives(fc)
    adj = (fc_pos > cfg.R_THRESHOLD).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def build_all_graphs(fc):
    """
    Build all three graph variants from an FC matrix.

    Returns:
        dict with keys "raw", "top10", "r05"
    """
    return {
        "raw":   build_graph_raw(fc),
        "top10": build_graph_top10(fc),
        "r05":   build_graph_r05(fc),
    }


# =============================================================================
# 6. GRAPH METRICS
# =============================================================================

def global_efficiency(adj):
    """
    Compute global efficiency of a graph.

    Global efficiency = mean of inverse shortest path lengths across
    all pairs of nodes. For disconnected pairs the path length is
    infinite, contributing 0 to the mean.

    E_glob = (1 / N(N-1)) * sum_{i≠j} 1/d(i,j)

    For weighted graphs, edge weight is treated as connection strength
    (higher = closer), so effective distance = 1/weight. This is the
    standard definition from Latora & Marchiori (2001).

    For binary graphs, distance = hop count (BFS).

    Args:
        adj: (N, N) adjacency matrix (weighted or binary, non-negative,
             diagonal = 0)

    Returns:
        float: global efficiency in [0, 1]
    """
    N   = adj.shape[0]
    total = 0.0
    is_weighted = not np.all((adj == 0) | (adj == 1))

    for i in range(N):
        if is_weighted:
            inv_lengths = _dijkstra_efficiency(adj, i)
        else:
            inv_lengths = _bfs_efficiency(adj, i)
        total += inv_lengths.sum()

    return total / (N * (N - 1))


def average_shortest_path_length(adj):
    """
    Compute average shortest path length across all connected pairs.

    ASPL = (1 / n_connected_pairs) * sum_{i≠j, d(i,j)<∞} d(i,j)

    Disconnected pairs (infinite path length) are excluded from the mean.
    If the graph is fully disconnected, returns np.nan.

    For weighted graphs, distance = 1/weight (strength → distance).
    For binary graphs, distance = hop count.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        float: average shortest path length, or np.nan if no connected pairs
    """
    N    = adj.shape[0]
    total     = 0.0
    n_pairs   = 0
    is_weighted = not np.all((adj == 0) | (adj == 1))

    for i in range(N):
        if is_weighted:
            dists = _dijkstra_distances(adj, i)
        else:
            dists = _bfs_distances(adj, i)

        finite = dists[dists < np.inf]
        # exclude self (distance 0)
        finite = finite[finite > 0]
        total   += finite.sum()
        n_pairs += len(finite)

    if n_pairs == 0:
        return np.nan
    return total / n_pairs


# --- BFS helpers (binary graphs) ---

def _bfs_efficiency(adj, source):
    """
    BFS from source. Returns (N,) array of 1/d for all targets.
    Unreachable nodes contribute 0.
    """
    N     = adj.shape[0]
    inv_d = np.zeros(N)
    dist  = np.full(N, -1, dtype=int)
    dist[source] = 0
    q = deque([source])

    while q:
        u = q.popleft()
        for v in np.nonzero(adj[u])[0]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                inv_d[v] = 1.0 / dist[v]
                q.append(v)

    inv_d[source] = 0.0
    return inv_d


def _bfs_distances(adj, source):
    """BFS from source. Returns (N,) float array of distances (inf = unreachable)."""
    N    = adj.shape[0]
    dist = np.full(N, np.inf)
    dist[source] = 0.0
    q = deque([source])

    while q:
        u = q.popleft()
        for v in np.nonzero(adj[u])[0]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1.0
                q.append(v)

    return dist


# --- Dijkstra helpers (weighted graphs) ---

def _dijkstra_efficiency(adj, source):
    """
    Dijkstra from source using strength-as-proximity (d = 1/weight).
    Returns (N,) array of 1/d. Unreachable nodes contribute 0.
    """
    dists = _dijkstra_distances(adj, source)
    inv_d = np.zeros_like(dists)
    reachable = (dists > 0) & (dists < np.inf)
    inv_d[reachable] = 1.0 / dists[reachable]
    return inv_d


def _dijkstra_distances(adj, source):
    """
    Dijkstra shortest paths where edge weight w → distance 1/w.
    Returns (N,) float array of distances (inf = unreachable).
    """
    import heapq
    N    = adj.shape[0]
    dist = np.full(N, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in np.nonzero(adj[u])[0]:
            w = adj[u, v]
            if w <= 0:
                continue
            nd = d + 1.0 / w   # strength → distance
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


# =============================================================================
# 7. PARCEL SIZE (atlas-aligned)
# =============================================================================

def compute_parcel_sizes_aligned(native_lh, native_rh, mapping):
    """
    Compute vertex count per parcel, reported under atlas IDs.

    Uses native label geometry (unmodified parcel shapes).
    After alignment via mapping, every atlas ID 1-200 receives
    exactly one vertex count.

    Args:
        native_lh: (n_valid_L,) native LH labels in valid-vertex space
        native_rh: (n_valid_R,) native RH labels in valid-vertex space
        mapping:   dict {native_global_id: atlas_id}
                   For SLIC, native_global_id for RH is already +100-offset.

    Returns:
        sizes: dict {atlas_id (1-200): n_vertices (int)}
               All 200 keys always present (0 if a parcel has no vertices,
               which should not happen after valid alignment).
    """
    # Detect SLIC per-hemisphere convention
    rh_unique = np.unique(native_rh[native_rh > 0])
    rh_per_hemi = bool(len(rh_unique) > 0 and rh_unique.max() <= 100)

    sizes = {i: 0 for i in range(1, 201)}

    # LH: native 1-100
    for pid in np.unique(native_lh[native_lh > 0]):
        n_v      = int(np.sum(native_lh == pid))
        atlas_id = mapping.get(int(pid))
        if atlas_id is not None:
            sizes[atlas_id] = n_v

    # RH
    for pid in np.unique(native_rh[native_rh > 0]):
        rh_key   = int(pid) + (100 if rh_per_hemi else 0)
        atlas_id = mapping.get(rh_key)
        if atlas_id is not None:
            n_v = int(np.sum(native_rh == pid))
            sizes[atlas_id] = n_v

    return sizes


# =============================================================================
# 8. FC SIMILARITY MATRIX
# =============================================================================

def fc_similarity_matrix(fc_vectors):
    """
    Compute subject-similarity matrix from FC vectors.

    Entry [i, j] = Pearson r between subject i's and subject j's
    lower-triangle FC vector (length 19,900).

    Args:
        fc_vectors: dict {subject_id: (19900,) array}

    Returns:
        sim_matrix: (n_subj, n_subj) float64 array
        subject_order: list of subject IDs in matrix row/col order
    """
    subject_order = sorted(fc_vectors.keys())
    n = len(subject_order)
    mat = np.zeros((n, n), dtype=np.float64)

    vecs = np.array([fc_vectors[s] for s in subject_order])  # (n, 19900)

    # Standardize each row
    means = vecs.mean(axis=1, keepdims=True)
    stds  = vecs.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    vecs_z = (vecs - means) / stds

    # Pearson r = dot product of z-scored rows / (n_features - 1)
    mat = (vecs_z @ vecs_z.T) / (vecs.shape[1] - 1)
    np.clip(mat, -1.0, 1.0, out=mat)

    return mat, subject_order