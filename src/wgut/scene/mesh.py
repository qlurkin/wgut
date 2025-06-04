from __future__ import annotations

import numpy.typing as npt
import numpy as np
import wgpu
from pyglm.glm import (
    cross,
    dot,
    length,
    normalize,
    vec4,
    vec3,
    vec2,
    array,
    mat4,
    mat3,
    int32,
    inverseTranspose,
)

from wgut.builders.vertexbufferdescriptorsbuilder import VertexBufferDescriptorsBuilder


def vertex(
    position: npt.NDArray,
    color: npt.NDArray | None = None,
    tex_coord: npt.NDArray | None = None,
    normal: npt.NDArray | None = None,
    tangent: npt.NDArray | None = None,
    bitangent: npt.NDArray | None = None,
) -> tuple[
    array[vec4], array[vec4], array[vec2], array[vec3], array[vec3], array[vec3]
]:
    if position.ndim == 1:
        position = position.reshape((1, len(position)))

    vertex_count = position.shape[0]

    if position.shape[1] == 3:
        position = np.hstack([position, np.ones((vertex_count, 1), dtype=np.float32)])

    if color is None:
        color = np.ones((vertex_count, 4))

    if color.ndim == 1:
        color = np.full((vertex_count, 4), color)

    if tex_coord is None:
        tex_coord = np.zeros((vertex_count, 2))

    if tex_coord.ndim == 1:
        tex_coord = np.full((vertex_count, 2), tex_coord)

    if normal is None:
        normal = np.zeros((vertex_count, 3))

    if normal.ndim == 1:
        normal = np.full((vertex_count, 3), normal)

    if tangent is None:
        tangent = np.zeros((vertex_count, 3))

    if tangent.ndim == 1:
        tangent = np.full((vertex_count, 3), tangent)

    if bitangent is None:
        bitangent = np.zeros((vertex_count, 3))

    if bitangent.ndim == 1:
        bitangent = np.full((vertex_count, 3), bitangent)

    return (
        array([vec4(p) for p in position]),
        array([vec4(c) for c in color]),
        array([vec2(u) for u in tex_coord]),
        array([vec3(n) for n in normal]),
        array([vec3(t) for t in tangent]),
        array([vec3(b) for b in bitangent]),
    )


def get_vertex_buffer_descriptors():
    return (
        VertexBufferDescriptorsBuilder()
        .with_vertex_descriptor(
            {
                "array_stride": 4 * 4,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x4,
                        "offset": 0,
                        "shader_location": 0,
                    }
                ],
            }
        )
        .with_vertex_descriptor(
            {
                "array_stride": 4 * 4,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x4,
                        "offset": 0,
                        "shader_location": 1,
                    }
                ],
            }
        )
        .with_vertex_descriptor(
            {
                "array_stride": 4 * 2,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 0,
                        "shader_location": 2,
                    }
                ],
            }
        )
        .with_vertex_descriptor(
            {
                "array_stride": 4 * 3,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x3,
                        "offset": 0,
                        "shader_location": 3,
                    }
                ],
            }
        )
        .with_vertex_descriptor(
            {
                "array_stride": 4 * 3,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x3,
                        "offset": 0,
                        "shader_location": 4,
                    }
                ],
            }
        )
        .with_vertex_descriptor(
            {
                "array_stride": 4 * 3,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x3,
                        "offset": 0,
                        "shader_location": 5,
                    }
                ],
            }
        )
        .with_vertex_descriptor(
            {
                "array_stride": 4,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32,
                        "offset": 0,
                        "shader_location": 6,
                    }
                ],
            }
        )
        .build()
    )


def compute_triangle_normal(p1: vec3, p2: vec3, p3: vec3) -> vec3:
    p_1_2 = p2 - p1
    p_1_3 = p3 - p1

    res = cross(p_1_2, p_1_3)
    return normalize(res)


def compute_triangle_tangent(
    p1: vec3,
    uv1: vec2,
    p2: vec3,
    uv2: vec2,
    p3: vec3,
    uv3: vec2,
) -> tuple[vec3, vec3]:
    p_1_2 = p2 - p1
    p_1_3 = p3 - p1

    tc_1_2 = uv2 - uv1
    tc_1_3 = uv3 - uv1

    tangent = tc_1_3.y * p_1_2 - tc_1_2.y * p_1_3
    tangent = normalize(tangent)

    bitangent = -tc_1_3.x * p_1_2 + tc_1_2.x * p_1_3
    bitangent = normalize(bitangent)

    return tangent, bitangent


def compute_normal_vectors(
    positions: array[vec3] | array[vec4], indices: array[int32]
) -> array[vec3]:
    normals = array(vec3(0.0)).repeat(len(positions))

    for i in range(0, len(indices), 3):
        index1 = indices[i]
        index2 = indices[i + 1]
        index3 = indices[i + 2]

        normal = compute_triangle_normal(
            positions[index1].xyz, positions[index2].xyz, positions[index3].xyz
        )

        normals[index1] += normal
        normals[index2] += normal
        normals[index3] += normal

    normals = normals.map(normalize)
    return normals


def compute_tangent_vectors(
    positions: array[vec3] | array[vec4],
    uvs: array[vec2],
    normals: array[vec3],
    indices: array[int32],
) -> array[vec3]:
    tangents = array(vec3(0.0)).repeat(len(positions))

    for i in range(0, len(indices), 3):
        index1 = indices[i]
        index2 = indices[i + 1]
        index3 = indices[i + 2]

        tangent, _ = compute_triangle_tangent(
            positions[index1].xyz,
            uvs[index1],
            positions[index2].xyz,
            uvs[index2],
            positions[index3].xyz,
            uvs[index3],
        )

        tangents[index1] += tangent
        tangents[index2] += tangent
        tangents[index3] += tangent

    for i in range(len(tangents)):
        t = tangents[i]
        norm = length(t)
        if norm != 0:
            t /= norm
            t -= dot(normals[i], t) * normals[i]
            t = normalize(t)
        else:
            print("undefined tangent at position", positions[i])
            t = normals[i]

        tangents[i] = t

    return tangents


def compute_bitangent_vectors(
    normals: array[vec3], tangents: array[vec3]
) -> array[vec3]:
    bitangents = []
    for i in range(len(normals)):
        if normals[i] == tangents[i]:
            bitangents.append(vec3(normals[i]))
            print(
                f"degenerate tangent space for normal {normals[i]} and tangent {tangents[i]}"
            )
        else:
            bitangents.append(cross(normals[i], tangents[i]))
    return array(bitangents)


def compute_line_list(triangle_list: array[int32]) -> array[int32]:
    lines = set()

    def add_line(i1: int, i2: int):
        a = min(i1, i2)
        b = max(i1, i2)

        lines.add((a, b))

    for i in range(0, len(triangle_list), 3):
        i1 = triangle_list[i]
        i2 = triangle_list[i + 1]
        i3 = triangle_list[i + 2]
        add_line(i1, i2)
        add_line(i2, i3)
        add_line(i3, i1)

    res = []
    for line in lines:
        res.append(line[0])
        res.append(line[1])

    return array(res)


# TODO:
# - Add swap triangle orientation
# - Add swap normals


class Mesh:
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

    def get_indices(self):
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
