import trimesh
from trimesh import smoothing, remesh
import numpy as np
from skimage.measure import marching_cubes_lewiner


def matrix_to_mesh(three_d_matrix):
    """
    Convert a 3d binary matrix to a 3d Trimesh object, containing triangles.
    @param three_d_matrix: 3d binary numpy array.
    @return: Trimesh object.
    """
    vertices, faces, normals, values = marching_cubes_lewiner(three_d_matrix)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def smooth_mesh_laplacian(mesh, lamb=0.15, iter=100):
    """
    Smooth inplace a given mesh by the laplacian smoother, generally an aggressive smoother.
    @param mesh: Trimesh mesh.
    @param lamb: 0.0 - 1.0 float. The bigger - the more heavily smoothed the outcome.
    @param iter: number of smoothing iterations. The more iterations - the smoother.
    @return: Smoothed mesh.
    """
    smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iter)
    return mesh


def smooth_mesh_humphrey(mesh, alpha=0.1, beta=0.1, iter=200):
    """
    Smooth inplace a given mesh by the humphrey smoother.
    @param mesh: Trimesh mesh.
    @param alpha: 0.0 - 1.0 float. The smaller - the more heavily smoothed the outcome.
    @param beta: 0.0 - 1.0 float. The smaller - the more heavily smoothed the outcome.
    @param iter: number of smoothing iterations. The more iterations - the smoother.
    @return: Smoothed mesh.
    """
    smoothing.filter_humphrey(mesh, alpha=alpha, beta=beta, iterations=iter)
    return mesh


def export_mesh_to_stl(mesh, name):
    """
    Export given mesh to an stl file.
    @param mesh: Trimesh mesh object.
    @param name: file name.
    """
    mesh.export(name + ".stl")
