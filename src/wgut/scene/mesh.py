from __future__ import annotations

import numpy.typing as npt
import numpy as np
import wgpu
import scipy as sp


def vertex(
    position: npt.NDArray,
    color: npt.NDArray | None = None,
    tex_coord: npt.NDArray | None = None,
    normal: npt.NDArray | None = None,
    tangent: npt.NDArray | None = None,
    bitangent: npt.NDArray | None = None,
) -> npt.NDArray:
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

    res = np.hstack([position, color, tex_coord, normal, tangent, bitangent])
    if res.dtype != np.float32:
        print("Warning: Vertex Data not in Float32 => Casting")
        res = np.array(res, dtype=np.float32)
    return res


def get_vertex_buffer_descriptor():
    return {
        "array_stride": 4 * (4 + 4 + 2 + 3 + 3 + 3),
        "step_mode": wgpu.VertexStepMode.vertex,
        "attributes": [
            {
                "format": wgpu.VertexFormat.float32x4,
                "offset": 0,
                "shader_location": 0,
            },
            {
                "format": wgpu.VertexFormat.float32x4,
                "offset": 3 * 4,
                "shader_location": 1,
            },
            {
                "format": wgpu.VertexFormat.float32x2,
                "offset": (3 + 3) * 4,
                "shader_location": 2,
            },
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": (3 + 3 + 2) * 4,
                "shader_location": 3,
            },
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": (3 + 3 + 2 + 3) * 4,
                "shader_location": 4,
            },
            {
                "format": wgpu.VertexFormat.float32x3,
                "offset": (3 + 3 + 2 + 3 + 3) * 4,
                "shader_location": 5,
            },
        ],
    }


def compute_triangle_normal(
    p1: npt.NDArray, p2: npt.NDArray, p3: npt.NDArray
) -> npt.NDArray:
    p_1_2 = p2 - p1
    p_1_3 = p3 - p1

    res = np.cross(p_1_2, p_1_3)
    return res / np.linalg.norm(res)


def compute_triangle_tangent(
    p1: npt.NDArray,
    uv1: npt.NDArray,
    p2: npt.NDArray,
    uv2: npt.NDArray,
    p3: npt.NDArray,
    uv3: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    p_1_2 = p2 - p1
    p_1_3 = p3 - p1

    tc_1_2 = uv2 - uv1
    tc_1_3 = uv3 - uv1

    tangent = tc_1_3[1] * p_1_2 - tc_1_2[1] * p_1_3
    tangent /= np.linalg.norm(tangent)

    bitangent = -tc_1_3[0] * p_1_2 + tc_1_2[0] * p_1_3
    bitangent /= np.linalg.norm(bitangent)

    return tangent, bitangent


def compute_normal_vectors(positions: npt.NDArray, indices: npt.NDArray) -> npt.NDArray:
    normals = np.zeros((positions.shape[0], 3))

    for i in range(0, len(indices), 3):
        index1 = indices[i]
        index2 = indices[i + 1]
        index3 = indices[i + 2]

        normal = compute_triangle_normal(
            positions[index1], positions[index2], positions[index3]
        )

        normals[index1] += normal
        normals[index2] += normal
        normals[index3] += normal

    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def compute_tangent_vectors(
    positions: npt.NDArray, uvs: npt.NDArray, normals: npt.NDArray, indices: npt.NDArray
) -> npt.NDArray:
    tangents = np.zeros((positions.shape[0], 3))

    for i in range(0, len(indices), 3):
        index1 = indices[i]
        index2 = indices[i + 1]
        index3 = indices[i + 2]

        tangent, _ = compute_triangle_tangent(
            positions[index1],
            uvs[index1],
            positions[index2],
            uvs[index2],
            positions[index3],
            uvs[index3],
        )

        tangents[index1] += tangent
        tangents[index2] += tangent
        tangents[index3] += tangent

    for i in range(len(tangents)):
        t = tangents[i]
        norm = np.linalg.norm(t)
        if norm != 0:
            t /= np.linalg.norm(t)
            t -= (normals[i] @ t) * normals[i]
            t /= np.linalg.norm(t)
        else:
            print("undefined tangent at position", positions[i])
            t = normals[i]

        tangents[i] = t

    return tangents


def compute_bitangent_vectors(
    normals: npt.NDArray, tangents: npt.NDArray
) -> npt.NDArray:
    bitangents = []
    for i in range(len(normals)):
        if np.array_equal(normals[i], tangents[i]):
            bitangents.append(normals[i].copy())
            print(
                f"degenerate tangent space for normal {normals[i]} and tangent {tangents[i]}"
            )
        else:
            bitangents.append(np.cross(normals[i], tangents[i]))
    return np.array(bitangents)


def compute_line_list(triangle_list: npt.NDArray) -> npt.NDArray:
    lines = set()

    def add_line(i1: int, i2: int):
        a = min(i1, i2)
        b = max(i1, i2)

        # Cantor's Pairing Function
        # key = ((a + b) * (a + b + 1) / 2) + a

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

    return np.array(res)


class Mesh:
    def __init__(self, vertices: npt.NDArray, indices: npt.NDArray):
        self.__vertices = vertices
        self.__indices = indices

    def get_vertices(self):
        return self.__vertices

    def get_indices(self):
        return self.__indices

    def get_transformed_vertices(self, transformation_matrix: npt.NDArray):
        M3 = transformation_matrix[:3, :3]
        normal_matrix = np.linalg.inv(M3).T

        big_transform = sp.linalg.block_diag(
            [
                transformation_matrix,
                np.identity(4),
                np.identity(2),
                normal_matrix,
                normal_matrix,
                normal_matrix,
            ]
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
