"""
Shared helper functions.
"""

import numpy as np
import nibabel as nib

N_VERTICES_PER_HEMI = 32492  # fs_LR 32k mesh, one hemisphere


def get_valid_vertices(cifti_path, structure_name):
    """
    Return the mesh vertex indices that carry data in a CIFTI file.
    CIFTI files exclude the medial wall, so a hemisphere holds roughly
    29,700 of the 32,492 mesh vertices.

    Parameters
    ----------
    cifti_path : path to any .dtseries.nii file
    structure_name : 'CIFTI_STRUCTURE_CORTEX_LEFT' or
                     'CIFTI_STRUCTURE_CORTEX_RIGHT'
    """
    img = nib.load(str(cifti_path))
    brain_axis = img.header.get_axis(1)
    for name, _, model in brain_axis.iter_structures():
        if name == structure_name:
            return np.array(model.vertex)
    raise ValueError(f"Structure '{structure_name}' not found in {cifti_path}")


def load_and_filter_atlas(atlas_path, valid_L, valid_R):
    """
    Load a dlabel atlas defined on the full 32k mesh and restrict it to
    the vertices actually present in the data.
    Returns (atlas_L, atlas_R) in valid-vertex space, so that entry i
    corresponds to data column i of that hemisphere.
    """
    data = nib.load(str(atlas_path)).get_fdata().squeeze().astype(int)
    atlas_L = data[:N_VERTICES_PER_HEMI][valid_L]
    atlas_R = data[N_VERTICES_PER_HEMI:][valid_R]
    return atlas_L, atlas_R


def save_cifti(data_matrix, template_path, output_path):
    """
    Save a matrix as a CIFTI file, borrowing the header from a template.
    The template must have the same brain model axis as the data, so a
    dlabel file on the full 32k mesh is used for parcellation maps.

    Parameters
    ----------
    data_matrix   : (1, 64984) array for a full-mesh parcellation map
    template_path : path to a CIFTI file with a matching structure
    output_path   : where to write the result
    """
    template = nib.load(str(template_path))
    img = nib.Cifti2Image(data_matrix, template.header, template.nifti_header)
    nib.save(img, str(output_path))
