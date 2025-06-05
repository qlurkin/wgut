from typing import Protocol

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
    int32,
)

from wgut.builders.vertexbufferdescriptorsbuilder import VertexBufferDescriptorsBuilder


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


class Mesh(Protocol):
    def get_transformed_vertices(
        self, transformation_matrix: mat4
    ) -> tuple[
        array[vec4], array[vec4], array[vec2], array[vec3], array[vec3], array[vec3]
    ]: ...

    def get_indices(self) -> array[int32]: ...
