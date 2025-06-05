from __future__ import annotations

from pyglm.glm import (
    vec4,
    vec3,
    vec2,
    array,
    mat4,
    mat3,
    int32,
    inverseTranspose,
)


# TODO:
# - Add swap triangle orientation
# - Add swap normals


class StaticMesh:
    def __init__(
        self,
        positions: array[vec4],
        colors: array[vec4],
        uvs: array[vec2],
        normals: array[vec3],
        tangents: array[vec3],
        bitangents: array[vec3],
        indices: array[int32],
    ):
        assert len(positions) == len(colors), (
            "'colors' must be the same length as 'positions'"
        )
        assert len(positions) == len(uvs), (
            "'uvs' must be the same length as 'positions'"
        )
        assert len(positions) == len(normals), (
            "'normals' must be the same length as 'positions'"
        )
        assert len(positions) == len(tangents), (
            "'tangents' must be the same length as 'positions'"
        )
        assert len(positions) == len(bitangents), (
            "'bitangents' must be the same length as 'positions'"
        )

        self.__positions = positions
        self.__colors = colors
        self.__uvs = uvs
        self.__normals = normals
        self.__tangents = tangents
        self.__bitangents = bitangents
        self.__indices = indices

    def get_positions(self):
        return self.__positions

    def get_colors(self):
        return self.__colors

    def get_uvs(self):
        return self.__uvs

    def get_normals(self):
        return self.__normals

    def get_tangents(self):
        return self.__tangents

    def get_bitangents(self):
        return self.__bitangents

    def get_vertices(self):
        return (
            self.__positions,
            self.__colors,
            self.__uvs,
            self.__normals,
            self.__tangents,
            self.__bitangents,
        )

    def get_indices(self) -> array[int32]:
        return self.__indices

    def get_transformed_vertices(
        self, transformation_matrix: mat4
    ) -> tuple[
        array[vec4], array[vec4], array[vec2], array[vec3], array[vec3], array[vec3]
    ]:
        normal_matrix = inverseTranspose(mat3(transformation_matrix))

        return (
            transformation_matrix * self.__positions,
            self.__colors,
            self.__uvs,
            normal_matrix * self.__normals,
            normal_matrix * self.__tangents,
            normal_matrix * self.__bitangents,
        )  # type: ignore
