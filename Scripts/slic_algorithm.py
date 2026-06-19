# slic_algorithm.py
"""
Surface-based SLIC (Simple Linear Iterative Clustering) Algorithm

This implements the SLIC algorithm from Wang et al. (2016):
"Parcellating Whole Brain for Individuals by Simple Linear Iterative Clustering"

The algorithm clusters brain vertices based on a unified distance that combines:
    - Functional distance: How similar are the fMRI time series?
    - Spatial distance: How close are the vertices on the cortical surface?

Two modes are available:
    - 'spatial_functional' (SLIC-F / Method 3): Uses both distances
    - 'functional_only' (SLIC-C / Method 4): Uses only functional distance

Reference:
    Wang, J., Hu, Z., & Wang, H. (2016). Parcellating Whole Brain for Individuals
    by Simple Linear Iterative Clustering. ICONIP 2016, LNCS 9949, pp. 131-139.
"""

import numpy as np


class SurfaceSLIC:
    """
    SLIC clustering adapted for cortical surface analysis.
    
    Key difference from the original paper:
    - Paper uses Euclidean distance in 3D MNI space
    - We use geodesic distance along the cortical surface mesh
    - This respects cortical topology (doesn't "jump" across sulci)
    """
    
    def __init__(self, n_clusters, m=40.0, max_iter=20, mode='spatial_functional', random_seed=None):
        """
        Initialize the SLIC algorithm.
        
        Args:
            n_clusters (int): Number of parcels to create (K).
                              For your thesis: 100 per hemisphere (200 total brain).
            
            m (float): Compactness factor. Controls the balance between
                       spatial and functional distance. Higher m = more weight
                       on spatial compactness. Paper recommends 40.0.
            
            max_iter (int): Maximum number of iterations. Algorithm may
                            converge earlier. Paper uses 10-20.
            
            mode (str): Which distances to use:
                        - 'spatial_functional': Both spatial + functional (SLIC-F)
                        - 'functional_only': Only functional distance (SLIC-C)
            
            random_seed (int): For reproducibility. Same seed = same initial
                               cluster centers = same final result.
        """
        self.n_clusters = n_clusters
        self.m = float(m)
        self.max_iter = max_iter
        self.mode = mode
        self.random_seed = random_seed
        
        # These will be set after fitting
        self.labels_ = None              # Final cluster assignment for each vertex
        self.cluster_centers_func_ = None  # Functional center (mean time series) per cluster
        self.cluster_centers_idx_ = None   # Spatial center (vertex index) per cluster

    def fit(self, time_series, geo_matrix=None, surface_area=None):
        """
        Run the SLIC clustering algorithm.
        
        Args:
            time_series (np.ndarray): fMRI data with shape (N_vertices, N_timepoints).
                                      Should be z-scored (mean=0, std=1 per vertex).
            
            geo_matrix (np.ndarray): Geodesic distance matrix with shape (N_vertices, N_vertices).
                                     Entry [i,j] = geodesic distance between vertex i and j.
                                     Required for 'spatial_functional' mode, ignored otherwise.
            
            surface_area (float): Total cortical surface area in mmÂ² for this hemisphere.
                                  Used to compute S in the same units (mm) as geodesic
                                  distances. Required for 'spatial_functional' mode.
                                  Compute from mesh: sum of triangle areas for valid vertices.
        
        Returns:
            labels (np.ndarray): Cluster assignment for each vertex (0 to K-1).
        """
        n_vertices = time_series.shape[0]

        # =================================================================
        # STEP 1: INPUT VALIDATION
        # =================================================================
        # Make sure inputs are valid before we start
        
        if self.mode == 'spatial_functional':
            # SLIC-F requires the geodesic distance matrix
            if geo_matrix is None:
                raise ValueError("geo_matrix is required for 'spatial_functional' mode.")
            
            # SLIC-F requires surface area for correct S computation
            if surface_area is None:
                raise ValueError("surface_area is required for 'spatial_functional' mode.")
            
            # Check dimensions match
            if geo_matrix.shape[0] != n_vertices or geo_matrix.shape[1] != n_vertices:
                raise ValueError(
                    f"Dimension Mismatch!\n"
                    f"Time series has {n_vertices} vertices.\n"
                    f"Geo matrix has shape {geo_matrix.shape}.\n"
                    f"Both must have the same number of vertices."
                )
            
            # Check for invalid values
            if np.any(np.isnan(geo_matrix)) or np.any(np.isinf(geo_matrix)):
                raise ValueError("Geodesic matrix contains NaN or Inf values!")
            
            print(f"[SLIC] Geodesic distance range: [{np.min(geo_matrix):.3f}, {np.max(geo_matrix):.3f}]")

        # Check time series for invalid values
        if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
            raise ValueError("Time series contains NaN or Inf values!")

        # =================================================================
        # STEP 2: INITIALIZATION
        # =================================================================
        
        # Calculate expected cluster spacing (S) in mm
        # Paper: S = (N/K)^(1/3) for 3D volumes (in voxel units)
        # For 2D surfaces: S = sqrt(Area/K) where Area is in mmÂ²
        # This gives S in mm â€” the same units as geodesic distances,
        # so the ratio d_spatial/S is properly dimensionless.
        if self.mode == 'spatial_functional':
            S = np.sqrt(surface_area / self.n_clusters)
            print(f"[SLIC] Grid spacing S = {S:.2f} mm (from surface area {surface_area:.0f} mmÂ²)")
        else:
            # For functional_only mode, S is not used in the distance
            S = np.sqrt(n_vertices / self.n_clusters)
            print(f"[SLIC] Grid spacing S = {S:.2f} (functional-only mode, S not used)")
        
        # Randomly select initial cluster centers
        # Note: Paper suggests periodic grid, but random works well in practice
        rng = np.random.default_rng(self.random_seed)
        center_indices = rng.choice(n_vertices, self.n_clusters, replace=False)
        
        # Initialize cluster centers
        self.cluster_centers_idx_ = center_indices.copy()  # Spatial: vertex indices
        self.cluster_centers_func_ = time_series[center_indices, :].copy()  # Functional: time series
        
        # Initialize labels to -1 (unassigned)
        self.labels_ = -1 * np.ones(n_vertices, dtype=int)
        
        # =================================================================
        # STEP 3: ITERATIVE OPTIMIZATION
        # =================================================================
        # This is the main loop that alternates between:
        #   A) Assigning vertices to nearest cluster
        #   B) Updating cluster centers
        
        for iteration in range(self.max_iter):
            
            # ---------------------------------------------------------
            # STEP 3A: Calculate distances from all vertices to all cluster centers
            # ---------------------------------------------------------
            # d_combined[i, k] = distance from vertex i to cluster k
            d_combined = np.zeros((n_vertices, self.n_clusters), dtype=np.float32)
            
            for k in range(self.n_clusters):
                
                # Functional Distance: ||v_i - v_k||^2
                # This measures how different the time series are
                diff = time_series - self.cluster_centers_func_[k]
                d_func_sq = np.sum(diff**2, axis=1)  # Squared L2 norm

                if self.mode == 'functional_only':
                    # SLIC-C: Only use functional distance
                    d_combined[:, k] = np.sqrt(d_func_sq)
                    
                else:
                    # SLIC-F: Combine functional and spatial distance
                    # This is Equation 1 from the paper:
                    #   d = sqrt( ||v_i - v_j||^2 / m^2  +  ||u_i - u_j||^2 / S^2 )
                    
                    # Spatial Distance: geodesic distance to cluster center
                    center_node = self.cluster_centers_idx_[k]
                    d_spatial = geo_matrix[:, center_node]
                    
                    # Normalize each term
                    term_func = d_func_sq / (self.m ** 2)
                    term_spatial = (d_spatial ** 2) / (S ** 2)
                    
                    # Combined distance
                    d_combined[:, k] = np.sqrt(term_func + term_spatial)

            # ---------------------------------------------------------
            # STEP 3B: Assign each vertex to the nearest cluster
            # ---------------------------------------------------------
            new_labels = np.argmin(d_combined, axis=1)
            
            # Check for convergence (no vertices changed)
            changes = np.sum(new_labels != self.labels_)
            print(f"[SLIC] Iteration {iteration + 1}/{self.max_iter}: {changes} vertices changed labels")
            
            if np.array_equal(new_labels, self.labels_):
                print(f"[SLIC] Converged at iteration {iteration + 1}")
                break
                
            self.labels_ = new_labels

            # ---------------------------------------------------------
            # STEP 3C: Update cluster centers
            # ---------------------------------------------------------
            # Functional center = mean time series of all member vertices
            # Spatial center = spatial medoid (most central vertex on the surface)
            
            new_centers_func = np.zeros_like(self.cluster_centers_func_)
            new_centers_idx = np.zeros(self.n_clusters, dtype=int)
            empty_clusters = 0
            
            for k in range(self.n_clusters):
                # Find all vertices assigned to cluster k
                mask = (self.labels_ == k)
                cluster_size = np.sum(mask)
                
                if cluster_size > 0:
                    # Update functional center: mean of all member time series
                    new_centers_func[k] = np.mean(time_series[mask], axis=0)
                    
                    # Update spatial center
                    if self.mode == 'spatial_functional':
                        # Find vertex that is most centrally located in the cluster.
                        # Reference uses mean(x,y,z) â€” the geometric centroid.
                        # On a surface mesh we can't have a floating-point centroid
                        # (geodesic distances are only defined between vertices),
                        # so we use the spatial medoid: the vertex minimizing total
                        # geodesic distance to all other cluster members.
                        cluster_vertices = np.where(mask)[0]
                        geo_submatrix = geo_matrix[np.ix_(cluster_vertices, cluster_vertices)]
                        total_dists = geo_submatrix.sum(axis=1)
                        medoid_local = np.argmin(total_dists)
                        new_centers_idx[k] = cluster_vertices[medoid_local]
                    else:
                        # For functional_only mode, spatial index doesn't matter
                        new_centers_idx[k] = self.cluster_centers_idx_[k]
                else:
                    # Empty cluster: keep the previous center
                    # (This can happen if initialization is unlucky)
                    empty_clusters += 1
                    new_centers_func[k] = self.cluster_centers_func_[k]
                    new_centers_idx[k] = self.cluster_centers_idx_[k]
            
            if empty_clusters > 0:
                print(f"[SLIC] Warning: {empty_clusters} empty clusters detected")
            
            # Log cluster size statistics for monitoring
            unique_labels, counts = np.unique(self.labels_, return_counts=True)
            print(f"[SLIC]   Cluster sizes - Min: {np.min(counts)}, Max: {np.max(counts)}, "
                  f"Mean: {np.mean(counts):.1f}, Std: {np.std(counts):.1f}")
            
            # Apply the updated centers
            self.cluster_centers_func_ = new_centers_func
            self.cluster_centers_idx_ = new_centers_idx
        
        # =================================================================
        # DONE: Return the final labels
        # =================================================================
        return self.labels_