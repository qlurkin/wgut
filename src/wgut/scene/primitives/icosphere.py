import numpy as np
from math import pi, atan2, acos
from pyglm.glm import array, int32, normalize, vec2, vec3, vec4
from ..mesh import (
    Mesh,
    compute_tangent_vectors,
    compute_bitangent_vectors,
)


def icosphere_positions_and_indices(order: int) -> tuple[array[vec3], array[int32]]:
    f = (1.0 + np.sqrt(5.0)) / 2.0
    positions = [
        vec3(-1, f, 0),
        vec3(1, f, 0),
        vec3(-1, -f, 0),
        vec3(1, -f, 0),
        vec3(0, -1, f),
        vec3(0, 1, f),
        vec3(0, -1, -f),
        vec3(0, 1, -f),
        vec3(f, 0, -1),
        vec3(f, 0, 1),
        vec3(-f, 0, -1),
        vec3(-f, 0, 1),
    ]

    indices = [
        0,
        11,
        5,
        0,
        5,
        1,
        0,
        1,
        7,
        0,
        7,
        10,
        0,
        10,
        11,
        11,
        10,
        2,
        5,
        11,
        4,
        1,
        5,
        9,
        7,
        1,
        8,
        10,
        7,
        6,
        3,
        9,
        4,
        3,
        4,
        2,
        3,
        2,
        6,
        3,
        6,
        8,
        3,
        8,
        9,
        9,
        8,
        1,
        4,
        9,
        5,
        2,
        4,
        11,
        6,
        2,
        10,
        8,
        6,
        7,
    ]

    v = 12

    mid_cache = {}

    def add_mid_point(a: int, b: int) -> int:
        nonlocal v
        # Cantor's Pairing Function
        key = ((a + b) * (a + b + 1) / 2) + min(a, b)

        if key not in mid_cache:
            mid_cache[key] = v
            v += 1
            positions.append((positions[a] + positions[b]) / 2.0)

        return mid_cache[key]

    def subdivide(indices: list[int]) -> list[int]:
        res = []
        for k in range(0, len(indices), 3):
            v1 = indices[k]
            v2 = indices[k + 1]
            v3 = indices[k + 2]
            a = add_mid_point(v1, v2)
            b = add_mid_point(v2, v3)
            c = add_mid_point(v3, v1)
            res.append(v1)
            res.append(a)
            res.append(c)
            res.append(v2)
            res.append(b)
            res.append(a)
            res.append(v3)
            res.append(c)
            res.append(b)
            res.append(a)
            res.append(b)
            res.append(c)
        return res

    for _ in range(order):
        indices = subdivide(indices)

    # positions /= np.linalg.norm(positions, axis=1, keepdims=True)

    return array(positions).map(normalize), array.from_numbers(int32, *indices)


def icosphere_with_uv(order: int) -> tuple[array[vec3], array[vec2], array[int32]]:
    positions, indices = icosphere_positions_and_indices(order)

    def compute_uv(pos: vec3):
        x, y, z = pos
        u = (atan2(z, x) / (2 * pi)) % 1.0  # [0, 1)
        v = acos(y) / pi  # [0, 1]
        return vec2(u, v)

    vertex_map = {}
    final_positions = []
    final_uvs = []
    final_indices = []

    for i in range(0, len(indices), 3):
        tri = indices[i : i + 3]
        tri_indices = []
        tri_uvs = [compute_uv(positions[v]) for v in tri]

        us = [uv.x for uv in tri_uvs]
        if max(us) - min(us) > 0.5:
            for j in range(3):
                if us[j] < 0.5:
                    tri_uvs[j][0] += 1.0

        for j in range(3):
            v_idx = tri[j]
            uv = vec2(tri_uvs[j])
            key = (v_idx, uv)
            if key not in vertex_map:
                vertex_map[key] = len(final_positions)
                final_positions.append(positions[v_idx])
                final_uvs.append(uv)
            tri_indices.append(vertex_map[key])

        final_indices.extend(tri_indices)

    return (
        array(final_positions),
        array(final_uvs),
        array.from_numbers(int32, *final_indices),
    )


def icosphere(order: int) -> Mesh:
    positions, uvs, indices = icosphere_with_uv(order)

    normals = array(positions)
    tangents = compute_tangent_vectors(positions, uvs, normals, indices)
    bitangents = compute_bitangent_vectors(normals, tangents)

    return Mesh(
        positions.map(lambda p: vec4(p, 1.0)),  # type: ignore
        array(vec4(1.0)).repeat(len(positions)),
        uvs,
        normals,
        tangents,
        bitangents,
        indices,
    )
