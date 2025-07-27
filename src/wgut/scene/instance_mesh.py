from typing import Callable
from wgut.scene.mesh import Mesh
import numpy.typing as npt
import numpy as np


class InstanceMesh:
    def __init__(
        self,
        mesh: Mesh,
        translations: Callable[[], npt.NDArray],
    ):
        self.__mesh = mesh
        self.__translations = translations
        self.__instance_count = 0
        self.__vertex_count = 0
        self.__index_count = 0
        self.__vertices = np.array([], dtype=np.float32)
        self.__indices = np.array([], dtype=np.int32)

    def get_transformed_vertices(
        self, transformation_matrix: npt.NDArray
    ) -> npt.NDArray:
        translations = self.__translations()
        self.__instance_count = len(translations)

        mesh_vertices = self.__mesh.get_transformed_vertices(transformation_matrix)
        mesh_indices = self.__mesh.get_indices()
        mesh_vertex_count = len(mesh_vertices)
        mesh_index_count = len(mesh_indices)

        vertex_count = self.__instance_count * mesh_vertex_count
        if vertex_count != self.__vertex_count:
            self.__vertices = np.zeros((vertex_count, 19), dtype=np.float32)

        index_count = self.__instance_count * mesh_index_count
        if index_count != self.__index_count:
            self.__indices = np.zeros(index_count, dtype=np.int32)

        for i, translation in enumerate(translations):
            self.__vertices[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count, :
            ] = mesh_vertices
            self.__vertices[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count, 0:4
            ] += translation
            self.__indices[
                i * mesh_index_count : i * mesh_index_count + mesh_index_count
            ] = mesh_indices + i * mesh_vertex_count

        return self.__vertices

    def get_indices(self) -> npt.NDArray:
        return self.__indices
