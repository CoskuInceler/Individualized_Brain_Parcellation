import os
import numpy as np
import nibabel as nib
import pickle
import config  # Our configuration file
from utils import get_valid_vertices

def load_mesh_triangles(mesh_path):
    """
    Reads a GIFTI surface file (.surf.gii) and extracts the 'Triangles'.
    
    The Input: A .surf.gii file.
    The Content:
        - darrays[0]: The 3D Coordinates (X, Y, Z) of every vertex.
        - darrays[1]: The Topology (Triangles). A list of [Vertex_A, Vertex_B, Vertex_C].
    
    Returns:
        triangles: A numpy array of shape (N_Triangles, 3).
    """
    print(f"[Mesh] Loading: {mesh_path.name}")
    
    # 1. Load the file using Nibabel
    img = nib.load(mesh_path)
    
    # 2. Extract the Topology (Index 1 is usually the triangle list)
    triangles = img.darrays[1].data
    
    return triangles

def build_sparse_adjacency(triangles, valid_indices, n_mesh_vertices=32492):
    """
    Converts Mesh Topology (Triangles) into a Data-Space Graph.
    
    1. Removes Medial Wall vertices (gaps).
    2. Remaps indices so they match the Data Matrix (0 to 29k).
    """
    print(f"[Graph] Building Adjacency List (Valid Vertices: {len(valid_indices)})...")
    
    # --- Step A: Create Lookup Table (Mesh -> Data) ---
    # Initialize all as -1 (Invalid/Medial Wall)
    mesh_to_data = np.full(n_mesh_vertices, -1, dtype=int)
    
    # Fill in the valid ones
    # valid_indices[i] is the Mesh ID of the i-th Data Node
    for data_idx, mesh_idx in enumerate(valid_indices):
        mesh_to_data[mesh_idx] = data_idx
        
    # --- Step B: Build the Graph ---
    n_data_verts = len(valid_indices)
    adj_list = [set() for _ in range(n_data_verts)]
    
    # Loop through every triangle (A-B-C) in the mesh
    for v1, v2, v3 in triangles:
        # Convert Mesh IDs to Data IDs
        d1, d2, d3 = mesh_to_data[v1], mesh_to_data[v2], mesh_to_data[v3]
        
        # If A and B are both valid, they are neighbors
        if d1 != -1 and d2 != -1:
            adj_list[d1].add(d2)
            adj_list[d2].add(d1)
            
        # If B and C are both valid...
        if d2 != -1 and d3 != -1:
            adj_list[d2].add(d3)
            adj_list[d3].add(d2)
            
        # If C and A are both valid...
        if d3 != -1 and d1 != -1:
            adj_list[d3].add(d1)
            adj_list[d1].add(d3)
            
    return adj_list

if __name__ == "__main__":
    # 1. DEFINE PATHS
    # We use the first subject to define the standard graph (assuming all subjects are registered to fsaverage)
    test_subj = config.SUBJECT_IDS[0]
    test_run = config.RUN_IDS[0]
    data_path = config.get_brain_path(test_subj, test_run)
    
    # 2. LEFT HEMISPHERE
    print("\n--- Processing LEFT Hemisphere ---")
    tris_L = load_mesh_triangles(config.MESH_FILE_L)
    valid_L = get_valid_vertices(data_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    
    adj_L = build_sparse_adjacency(tris_L, valid_L)
    
    # Save L
    out_path_L = config.METHOD_2_DIR / "Adjacency_Graph_L.pkl"
    with open(out_path_L, 'wb') as f:
        pickle.dump(adj_L, f)
    print(f" -> Saved Left Graph to: {out_path_L}")
    
    # 3. RIGHT HEMISPHERE
    print("\n--- Processing RIGHT Hemisphere ---")
    tris_R = load_mesh_triangles(config.MESH_FILE_R)
    valid_R = get_valid_vertices(data_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    
    adj_R = build_sparse_adjacency(tris_R, valid_R)
    
    # Save R
    out_path_R = config.METHOD_2_DIR / "Adjacency_Graph_R.pkl"
    with open(out_path_R, 'wb') as f:
        pickle.dump(adj_R, f)
    print(f" -> Saved Right Graph to: {out_path_R}")

    print("\n[SUCCESS] Graphs are ready for Atlas-Guided Parcellation.")