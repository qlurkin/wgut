import numpy.typing as npt
import numpy as np


class Mesh:
    def __init__(self, vertices: npt.NDArray, indices: npt.NDArray):
        self.__vertices = vertices
        self.__indices = indices

    def get_vertices(self):
        return self.__vertices

    def get_indices(self):
        return self.__indices

    def get_transformed_vertices(self, transformation_matrix: npt.NDArray):
        vertices = self.__vertices

        # TODO: use scipy.linalg.block_diag to do all in one multiplication without decompose vertices
        positions = vertices[:, :3]
        colors = vertices[:, 3:6]
        tex_coords = vertices[:, 6:8]
        normals = vertices[:, 8:11]
        tangents = vertices[:, 11:14]
        bitangents = vertices[:, 14:17]

        ones = np.ones((positions.shape[0], 1))
        pos_homogeneous = np.hstack([positions, ones])
        # TODO: simplifie all these transpositions
        transformed_positions = (transformation_matrix @ pos_homogeneous.T).T[:, :3]

        M3 = transformation_matrix[:3, :3]
        normal_matrix = np.linalg.inv(M3).T

        transformed_normals = (normal_matrix @ normals.T).T
        transformed_normals /= np.linalg.norm(
            transformed_normals, axis=1, keepdims=True
        )

        transformed_tangents = (normal_matrix @ tangents.T).T
        transformed_tangents /= np.linalg.norm(
            transformed_tangents, axis=1, keepdims=True
        )

        transformed_bitangents = (normal_matrix @ bitangents.T).T
        transformed_bitangents /= np.linalg.norm(
            transformed_bitangents, axis=1, keepdims=True
        )

        transformed_vertices = np.hstack(
            [
                transformed_positions,
                colors,
                tex_coords,
                transformed_normals,
                transformed_tangents,
                transformed_bitangents,
            ]
        )

        return transformed_vertices
