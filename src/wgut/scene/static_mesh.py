from __future__ import annotations


import numpy.typing as npt
import numpy as np
import scipy as sp


# TODO:
# - Add swap triangle orientation
# - Add swap normals


class StaticMesh:
    def __init__(self, vertices: npt.NDArray, indices: npt.NDArray):
        self.__vertices = vertices
        self.__indices = indices

    def get_vertices(self) -> npt.NDArray:
        return self.__vertices

    def get_indices(self) -> npt.NDArray:
        return self.__indices

    def get_transformed_vertices(
        self, transformation_matrix: npt.NDArray
    ) -> npt.NDArray:
        M3 = transformation_matrix[:3, :3]
        normal_matrix = np.linalg.inv(M3).T

        big_transform = sp.linalg.block_diag(
            transformation_matrix,
            np.identity(4, dtype=np.float32),
            np.identity(2, dtype=np.float32),
            normal_matrix,
            normal_matrix,
            normal_matrix,
        )

        # vertices are on rows so the product is transposed
        transformed_vertices = self.__vertices @ big_transform.T

        # normalize normals
        transformed_vertices[:, 10:13] /= np.linalg.norm(
            transformed_vertices[:, 10:13], axis=1, keepdims=True
        )

        # normalize tangents
        transformed_vertices[:, 13:16] /= np.linalg.norm(
            transformed_vertices[:, 13:16], axis=1, keepdims=True
        )

        # normalize bitangents
        transformed_vertices[:, 16:19] /= np.linalg.norm(
            transformed_vertices[:, 16:19], axis=1, keepdims=True
        )

        return transformed_vertices

    def __str__(self):
        return f"StaticMesh(vert={len(self.get_vertices())}, ind={len(self.get_indices())})"
