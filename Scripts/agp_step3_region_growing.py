import os
import numpy as np
import pickle
import time
import heapq
import config
from scipy.stats import zscore
from utils import get_valid_vertices, load_pickle, load_subject_data, load_and_filter_atlas

# ==========================================
# 2. THE REGION GROWING ENGINE
# ==========================================

def grow_regions_strict(data, seeds, adj_list, target_count):
    """
    Atlas-Guided Parcellation: Region Growing Algorithm
    
    This implements the exact algorithm from Li et al. (2022):
    "Atlas-guided parcellation: Individualized functionally-homogenous 
    parcellation in cerebral cortex"
    
    ALGORITHM OVERVIEW:
    1. Start with seed vertices (one per parcel from the atlas)
    2. For each parcel, find its best unassigned neighbor
       - "Best" = highest mean correlation with existing members
       - "Neighbor" = spatially adjacent (from mesh topology)
    3. Globally assign the vertex with the highest correlation
    4. Repeat until we've assigned the target number of vertices
    
    Args:
        data: Time series (Timepoints x Vertices), raw fMRI signal
        seeds: Dictionary mapping parcel_id -> seed_vertex_index
        adj_list: List of sets, where adj_list[i] = neighbors of vertex i
        target_count: Number of vertices to assign (matches atlas)
    
    Returns:
        labels: Array where labels[i] = parcel_id for vertex i (0 = unassigned)
    """
    
    # Get dimensions of our data
    n_timepoints, n_verts = data.shape
    
    # -------------------------------------------------------------------------
    # STEP 1: Z-SCORE NORMALIZATION
    # -------------------------------------------------------------------------
    # Why? After z-scoring, the dot product between two time series equals
    # their Pearson correlation coefficient. This is a mathematical identity.
    # 
    # Z-scoring means: (value - mean) / std_dev
    # After z-scoring: mean=0, std=1 for each vertex's time series
    #
    # axis=0 means we z-score along TIME (each vertex independently)
    print(" -> Z-scoring data for valid correlation math...")
    data_z = zscore(data, axis=0) 
    
    # -------------------------------------------------------------------------
    # STEP 2: INITIALIZE DATA STRUCTURES
    # -------------------------------------------------------------------------
    
    # labels[i] = which parcel does vertex i belong to? (0 = unassigned)
    labels = np.zeros(n_verts, dtype=np.int32)
    
    # parcel_members[p] = list of all vertex indices in parcel p
    # We need this to calculate mean correlations with ALL members
    parcel_members = {}
    
    print(f" -> Initializing {len(seeds)} seeds...")
    
    # Assign each seed vertex to its parcel
    for pid, seed_idx in seeds.items():
        labels[seed_idx] = pid           # Mark this vertex as belonging to parcel pid
        parcel_members[pid] = [seed_idx] # Start the member list with just the seed
    
    # Track how many vertices we've assigned (includes seeds)
    assigned_count = len(seeds)
    
    # -------------------------------------------------------------------------
    # STEP 3: PRIORITY QUEUE SETUP
    # -------------------------------------------------------------------------
    # Python's heapq implements a MIN-heap (smallest value first)
    # We want a MAX-heap (highest correlation first)
    # Solution: Store negative correlations, so max becomes min
    candidate_heap = []
    
    # -------------------------------------------------------------------------
    # STEP 4: THE CORE LOGIC - FINDING BEST CANDIDATE FOR A PARCEL
    # -------------------------------------------------------------------------
    def get_best_candidate(pid):
        """
        Finds the best unassigned neighbor vertex for parcel pid.
        
        "Best" means: highest MEAN correlation with ALL current members.
        "Neighbor" means: spatially adjacent to at least one current member.
        
        This is the heart of the paper's algorithm (Section 2.3, step i).
        
        Args:
            pid: The parcel ID we're finding a candidate for
        
        Returns:
            Tuple: (negative_correlation, vertex_index, parcel_id)
            OR None if this parcel has no valid neighbors
        """
        
        # Get all vertices currently in this parcel
        members = parcel_members[pid]
        
        # ---------------------------------------------------------------------
        # SUB-STEP 1: Find all unassigned neighbors (ADJACENCY CONSTRAINT)
        # ---------------------------------------------------------------------
        # Paper: "For each unassigned neighbor vertex of parcels..."
        
        potential_neighbors = set()  # Use set to avoid duplicates
        
        # For every vertex currently in the parcel
        for m in members:
            # Look at all its spatial neighbors (from mesh topology)
            for neighbor in adj_list[m]:
                # Only consider if it's unassigned (labels[i] = 0)
                if labels[neighbor] == 0:
                    potential_neighbors.add(neighbor)
        
        # If this parcel has no unassigned neighbors, it can't grow
        if not potential_neighbors:
            return None
        
        # Convert set to list for indexing
        candidates = list(potential_neighbors)
        
        # ---------------------------------------------------------------------
        # SUB-STEP 2: Calculate correlations (FUNCTIONAL SIMILARITY)
        # ---------------------------------------------------------------------
        # Paper: "the mean correlation with the vertices within the 
        #         corresponding parcel was calculated as the parcel profiles"
        
        # Extract time series for all candidates
        # Shape: (n_timepoints, n_candidates)
        cand_data = data_z[:, candidates]
        
        # Extract time series for all current members
        # Shape: (n_timepoints, n_members)
        member_data = data_z[:, members]
        
        # Calculate pairwise correlations between ALL members and ALL candidates
        # member_data.T = (n_members, n_timepoints)
        # cand_data = (n_timepoints, n_candidates)
        # Result: (n_members, n_candidates) - correlation between each pair
        #
        # Why divide by n_timepoints?
        # Because: correlation = (1/n) * sum(z_i * z_j) when data is z-scored
        pairwise_dots = np.dot(member_data.T, cand_data) / n_timepoints
        
        # ---------------------------------------------------------------------
        # SUB-STEP 3: Take MEAN across all members (PAPER'S KEY DECISION)
        # ---------------------------------------------------------------------
        # For each candidate, average its correlation with ALL members
        # This is what the paper calls "mean correlation"
        # 
        # axis=0 means: average down the rows (across all members)
        # Result: 1D array of length n_candidates
        mean_corrs = np.mean(pairwise_dots, axis=0)
        
        # ---------------------------------------------------------------------
        # SUB-STEP 4: Pick the winner
        # ---------------------------------------------------------------------
        # Find which candidate has the highest mean correlation
        best_idx = np.argmax(mean_corrs)
        best_score = mean_corrs[best_idx]
        best_vert = candidates[best_idx]
        
        # Return: (negative score for min-heap, vertex index, parcel id)
        # We negate the score because Python's heapq is a MIN-heap
        # So highest correlation becomes lowest negative value
        return (-best_score, best_vert, pid)
    
    # -------------------------------------------------------------------------
    # STEP 5: INITIALIZE THE QUEUE (ONE TICKET PER PARCEL)
    # -------------------------------------------------------------------------
    # Paper's approach: Each parcel gets ONE candidate in the queue at a time
    # This ensures "round-robin" fairness - no parcel dominates
    
    print(" -> Populating initial candidates (One per parcel)...")
    
    active_parcels = list(seeds.keys())
    
    # Get the first candidate for each parcel
    for pid in active_parcels:
        cand = get_best_candidate(pid)
        if cand:  # Only add if parcel has valid neighbors
            heapq.heappush(candidate_heap, cand)
    
    print(f" -> Starting expansion (Target: {target_count} vertices)...")
    iteration = 0
    
    # -------------------------------------------------------------------------
    # STEP 6: MAIN LOOP - THE REGION GROWING PROCESS
    # -------------------------------------------------------------------------
    # Continue until:
    #   - We've assigned the target number of vertices (PAPER REQUIREMENT), OR
    #   - No parcels have any more candidates (heap is empty)
    
    while candidate_heap and assigned_count < target_count:
        iteration += 1
        
        # ---------------------------------------------------------------------
        # 6A: Pop the globally best candidate
        # ---------------------------------------------------------------------
        # Paper: "The vertex with the highest parcel profiles will be 
        #         assigned to the corresponding parcel"
        #
        # heappop removes and returns the SMALLEST item (most negative = highest correlation)
        neg_score, vert_idx, pid = heapq.heappop(candidate_heap)
        score = -neg_score  # Convert back to positive for logging
        
        # ---------------------------------------------------------------------
        # 6B: Check if this vertex was "stolen"
        # ---------------------------------------------------------------------
        # Between when we added this candidate to the queue and now,
        # another parcel might have already claimed this vertex.
        # If so, we need to get a new candidate for this parcel.
        
        if labels[vert_idx] != 0:
            # Vertex already assigned - this candidate is stale
            # Get a fresh candidate for this parcel
            new_cand = get_best_candidate(pid)
            if new_cand:
                heapq.heappush(candidate_heap, new_cand)
            continue  # Skip to next iteration
        
        # ---------------------------------------------------------------------
        # 6C: Assign the vertex (THE ACTUAL GROWTH)
        # ---------------------------------------------------------------------
        labels[vert_idx] = pid                    # Update label array
        parcel_members[pid].append(vert_idx)      # Add to member list
        assigned_count += 1                       # Track progress
        
        # ---------------------------------------------------------------------
        # 6D: Get next candidate for this parcel (ROUND-ROBIN)
        # ---------------------------------------------------------------------
        # Paper: "Updating the neighbor vertices and parcel profiles"
        #
        # Since this parcel just grew, it might have NEW neighbors.
        # Get its next best candidate and put it back in the queue.
        # This ensures each parcel maintains one "ticket" in the queue.
        
        new_cand = get_best_candidate(pid)
        if new_cand:
            heapq.heappush(candidate_heap, new_cand)
        
        # Progress reporting every 2000 iterations
        if iteration % 2000 == 0:
            print(f"    Iter {iteration}: Assigned {assigned_count}/{target_count} vertices")
            print(f"              (Last: Vertex {vert_idx} -> Parcel {pid}, Corr: {score:.4f})")
    
    # -------------------------------------------------------------------------
    # STEP 7: FINAL REPORT
    # -------------------------------------------------------------------------
    print(f" -> Finished: Assigned {assigned_count}/{target_count} vertices")
    
    if assigned_count < target_count:
        print(f" -> WARNING: Could not reach target (some parcels had no more neighbors)")
    
    return labels

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 3: ATLAS-GUIDED PARCELLATION - REGION GROWING")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 3.1: SETUP - Load all necessary data
    # -------------------------------------------------------------------------
    
    # Use first subject to define the valid vertex structure
    # (All subjects in HCP are registered to same surface, so this applies to all)
    ref_subj = config.SUBJECT_IDS[0]
    ref_path = config.get_brain_path(ref_subj, config.RUN_IDS[0])
    
    # Get the indices of valid vertices (excludes medial wall)
    print("\n[1/5] Loading valid vertex masks...")
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    print(f"    Left:  {len(valid_L)} valid vertices")
    print(f"    Right: {len(valid_R)} valid vertices")
    
    # Load the atlas to determine TARGET COUNTS
    # This is the CRITICAL FIX - we need to know when to STOP growing
    print("\n[2/5] Loading atlas to determine target counts...")
    atlas_L, atlas_R = load_and_filter_atlas(config.SCHAEFER_200_FILE, valid_L, valid_R)
    
    # Count non-zero values = number of vertices that should be parcellated
    target_count_L = np.sum(atlas_L > 0)
    target_count_R = np.sum(atlas_R > 0)
    
    print(f"    Left Target:  {target_count_L} vertices to assign")
    print(f"    Right Target: {target_count_R} vertices to assign")
    
    # Load the adjacency graphs (spatial neighbors from mesh topology)
    print("\n[3/5] Loading adjacency graphs...")
    graph_L = load_pickle(config.METHOD_2_DIR / "Adjacency_Graph_L.pkl")
    graph_R = load_pickle(config.METHOD_2_DIR / "Adjacency_Graph_R.pkl")
    print(f"    Left graph:  {len(graph_L)} vertices")
    print(f"    Right graph: {len(graph_R)} vertices")
    
    # Load the seed vertices (geometric centers from Step 2)
    print("\n[4/5] Loading seed vertices...")
    seeds_L = load_pickle(config.METHOD_2_DIR / "Seeds_Geometric_L.pkl")
    seeds_R = load_pickle(config.METHOD_2_DIR / "Seeds_Geometric_R.pkl")
    print(f"    Left seeds:  {len(seeds_L)} parcels")
    print(f"    Right seeds: {len(seeds_R)} parcels")
    
    # -------------------------------------------------------------------------
    # 3.2: PROCESS ALL SUBJECTS
    # -------------------------------------------------------------------------
    print("\n[5/5] Processing subjects...")
    
    for subj in config.SUBJECT_IDS:
        print(f"\n{'=' * 70}")
        print(f"SUBJECT: {subj}")
        print(f"{'=' * 70}")
        
        # Define output paths
        out_L = config.METHOD_2_DIR / f"{subj}_Labels_L.npy"
        out_R = config.METHOD_2_DIR / f"{subj}_Labels_R.npy"
        
        # Load this subject's time series data
        try:
            ts_data = load_subject_data(subj)
        except Exception as e:
            print(f" -> ERROR loading data: {e}")
            continue
        
        # Split into left and right hemispheres
        # The data is organized as: [Left vertices | Right vertices | Subcortex]
        n_L = len(graph_L)
        n_R = len(graph_R)
        
        data_L = ts_data[:, :n_L]              # First n_L columns = Left
        data_R = ts_data[:, n_L : n_L + n_R]   # Next n_R columns = Right
        
        # Run region growing for LEFT hemisphere
        print(f"\n[LEFT HEMISPHERE]")
        labels_L = grow_regions_strict(data_L, seeds_L, graph_L, target_count_L)
        
        # Run region growing for RIGHT hemisphere
        print(f"\n[RIGHT HEMISPHERE]")
        labels_R = grow_regions_strict(data_R, seeds_R, graph_R, target_count_R)
        
        # =====================================================================
        # VALIDATION: Check if we assigned the correct number of vertices
        # =====================================================================
        filled_L = np.sum(labels_L > 0)
        filled_R = np.sum(labels_R > 0)
        
        print(f"\n{'=' * 70}")
        print(f"VALIDATION RESULTS FOR {subj}")
        print(f"{'=' * 70}")
        print(f"  Left Hemisphere:")
        print(f"    Generated: {filled_L} vertices")
        print(f"    Target:    {target_count_L} vertices")
        print(f"    Status:    {'✓ MATCH' if filled_L == target_count_L else '✗ MISMATCH'}")
        print(f"  Right Hemisphere:")
        print(f"    Generated: {filled_R} vertices")
        print(f"    Target:    {target_count_R} vertices")
        print(f"    Status:    {'✓ MATCH' if filled_R == target_count_R else '✗ MISMATCH'}")
        
        # Save the results
        np.save(out_L, labels_L)
        np.save(out_R, labels_R)
        print(f"\n✓ Saved: {out_L.name}")
        print(f"✓ Saved: {out_R.name}")
    
    print(f"\n{'=' * 70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 70}")
